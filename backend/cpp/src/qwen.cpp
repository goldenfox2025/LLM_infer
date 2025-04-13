#include "qwen.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "cudaOP.cuh"

// Debug 打印函数（保持不变）
template <typename T>
void debugPrintTensor(const Tensor<T>& tensor, const std::string& tensor_name,
                      size_t num_to_print) {
  std::cout << "[Debug] " << tensor_name << ":\n";
  std::cout << "  shape: [";
  for (auto s : tensor.sizes()) {
    std::cout << s << " ";
  }
  std::cout << "]\n";
  std::cout << "  strides: [";
  for (auto st : tensor.strides()) {
    std::cout << st << " ";
  }
  std::cout << "]\n";
  std::cout << "  device: "
            << (tensor.device() == Device::CUDA ? "CUDA" : "CPU") << "\n";
  size_t offset = 0;
  size_t total_elements = tensor.numel();
  size_t n_print = std::min(num_to_print, total_elements - offset);
  std::cout << "  elements from offset " << offset << " (" << n_print
            << " element(s)): ";
  if (tensor.device() == Device::CPU) {
    const T* ptr = tensor.data_ptr();
    for (size_t i = 0; i < n_print; i++) {
      std::cout << ptr[offset + i] << " ";
    }
    std::cout << "\n";
  } else {
    std::vector<T> host_buffer(n_print);
    cudaError_t err = cudaMemcpy(host_buffer.data(), tensor.data_ptr() + offset,
                                 n_print * sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      std::cout << "  [Error] cudaMemcpy failed\n";
      return;
    }
    for (size_t i = 0; i < n_print; i++) {
      std::cout << host_buffer[i] << " ";
    }
    std::cout << "\n";
  }
}

// -------------------------------
// QwenModel<T> 构造函数
// -------------------------------
template <typename T>
QwenModel<T>::QwenModel(
    const std::unordered_map<std::string, Tensor<T>>& params,
    const std::unordered_map<std::string, int>& config)
    : params_(params) {
  vocab_size_ = config.at("vocab_size");
  n_layers_ = config.at("n_layers");
  n_heads_ = config.at("n_heads");
  n_kv_heads_ = config.at("n_kv_heads");
  hidden_size_ = config.at("hidden_size");
  intermediate_size_ = config.at("intermediate_size");
  max_position_embeddings_ = config.at("max_position_embeddings");
  bos_token_id_ = static_cast<uint32_t>(config.at("bos_token_id"));
  eos_token_id_ = static_cast<uint32_t>(config.at("eos_token_id"));
  rms_norm_eps_ = static_cast<float>(config.at("rms_norm_eps"));
  rope_theta_ = static_cast<float>(config.at("rope_theta"));
  head_dim_ = hidden_size_ / n_heads_;

  device_ = Device::CUDA;
  for (int i = 0; i < kNumStreams; ++i) {
    cudaError_t err = cudaStreamCreate(&compute_streams_[i]);
    if (err != cudaSuccess) {
      throw std::runtime_error("Failed to create CUDA stream in constructor");
    }
  }
  forward_logits_ = Tensor<T>({1, vocab_size_}, Device::CUDA);
  // 预先分配 workspace，用于 capture 阶段复用，防止动态分配
  workspace_residual_ = Tensor<T>({1, hidden_size_}, Device::CUDA);
  workspace_hidden_ = Tensor<T>({1, hidden_size_}, Device::CUDA);
  cudaError_t err = cudaStreamCreate(&graph_stream_);
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "Failed to create CUDA graph stream in constructor");
  }
}

template <typename T>
QwenModel<T>::~QwenModel() {
  for (cudaStream_t stream : compute_streams_) {
    if (stream) {
      cudaStreamDestroy(stream);
    }
  }
  // 释放图执行实例
  cudaGraphExecDestroy(forward_graph_exec_);
  if (graph_stream_) {
    cudaStreamDestroy(graph_stream_);
  }
}

// -------------------------------
// 参数验证
// -------------------------------
template <typename T>
bool QwenModel<T>::verify_params() const {
  std::cout << "Not checking parameters" << std::endl;
  return true;
}

// -------------------------------
// 打印模型基本信息
// -------------------------------
template <typename T>
void QwenModel<T>::print_model_info() const {
  std::cout << "QwenModel Info:" << std::endl;
  std::cout << "  Vocab size: " << vocab_size_ << std::endl;
  std::cout << "  Layers: " << n_layers_ << std::endl;
  std::cout << "  Heads: " << n_heads_ << std::endl;
  std::cout << "  KV Heads: " << n_kv_heads_ << std::endl;
  std::cout << "  Hidden size: " << hidden_size_ << std::endl;
  std::cout << "  Intermediate size: " << intermediate_size_ << std::endl;
  std::cout << "  Max sequence length: " << max_position_embeddings_
            << std::endl;
  std::cout << "  RMS Norm eps: " << rms_norm_eps_ << std::endl;
  std::cout << "  RoPE theta: " << rope_theta_ << std::endl;
  std::cout << "  Head dim: " << head_dim_ << std::endl;
  std::cout << "  Device: " << (device_ == Device::CUDA ? "CUDA" : "CPU")
            << std::endl;
}

// -------------------------------
// forward_cuda: 单个 token 的 CUDA 前向传播，所有内核均使用 used_stream
// 当 used_stream 非空时，使用预分配 workspace 避免捕获期间动态分配
// -------------------------------
template <typename T>
Tensor<T> QwenModel<T>::forward_cuda(const Tensor<uint32_t>* input,
                                     KVCache<T>* kv_cache, Tensor<T>* p_output,
                                     cudaStream_t used_stream) {
  if (input->device() != Device::CUDA)
    throw std::runtime_error("Input tensor must be on CUDA device");
  const size_t seq_len = 1;
  size_t offset = 0;
  if (kv_cache) {
    if (kv_cache->device() != Device::CUDA)
      throw std::runtime_error("KVCache must be on CUDA device");
    offset = kv_cache->size() - seq_len;
  }

  // 若处于 capture 区（used_stream 非空），直接使用预分配
  // workspace，否则动态构造
  Tensor<T>* residual_ptr = nullptr;
  Tensor<T>* hidden_ptr = nullptr;
  if (used_stream) {
    residual_ptr = &workspace_residual_;
    hidden_ptr = &workspace_hidden_;
    // 可在此清零 workspace（若需要）
  } else {
    residual_ptr = new Tensor<T>({seq_len, hidden_size_}, Device::CUDA);
    hidden_ptr = new Tensor<T>({seq_len, hidden_size_}, Device::CUDA);
  }
  Tensor<T>& residual = *residual_ptr;
  Tensor<T>& hidden_states = *hidden_ptr;

  cuda_OP::gather(&residual, input, &params_.at("token_embeddings.weight"),
                  used_stream);

  for (size_t i = 0; i < n_layers_; i++) {
    std::string layer_prefix = "layers." + std::to_string(i) + ".";
    auto& attention_norm_weight =
        params_.at(layer_prefix + "input_layernorm.weight");
    cuda_OP::rms_norm(&hidden_states, &residual, &attention_norm_weight,
                      rms_norm_eps_, used_stream);

    auto& wq = params_.at(layer_prefix + "self_attn.q_proj.weight");
    auto& wk = params_.at(layer_prefix + "self_attn.k_proj.weight");
    auto& wv = params_.at(layer_prefix + "self_attn.v_proj.weight");
    auto& wo = params_.at(layer_prefix + "self_attn.o_proj.weight");

    const Tensor<T>* q_bias = nullptr;
    const Tensor<T>* k_bias = nullptr;
    const Tensor<T>* v_bias = nullptr;
    const Tensor<T>* o_bias = nullptr;
    try {
      q_bias = &params_.at(layer_prefix + "self_attn.q_proj.bias");
    } catch (...) {
    }
    try {
      k_bias = &params_.at(layer_prefix + "self_attn.k_proj.bias");
    } catch (...) {
    }
    try {
      v_bias = &params_.at(layer_prefix + "self_attn.v_proj.bias");
    } catch (...) {
    }

    Tensor<T> q_buf({seq_len, n_heads_ * head_dim_}, Device::CUDA);
    cuda_OP::matmul(hidden_states, wq, &q_buf, used_stream, q_bias);

    Tensor<T>& k_slice = kv_cache->k_cache(i, offset);
    Tensor<T>& v_slice = kv_cache->v_cache(i, offset);
    cuda_OP::matmul(hidden_states, wk, &k_slice, used_stream, k_bias);
    cuda_OP::matmul(hidden_states, wv, &v_slice, used_stream, v_bias);

    Tensor<T> q_buf_view = q_buf.view({seq_len, n_heads_, head_dim_});
    Tensor<T> k_buf_view = k_slice.view({seq_len, n_kv_heads_, head_dim_});
    Tensor<T> v_buf_view = v_slice.view({seq_len, n_kv_heads_, head_dim_});

    cuda_OP::rope(&q_buf_view, offset, rope_theta_, used_stream);
    cuda_OP::rope(&k_buf_view, offset, rope_theta_, used_stream);

    Tensor<T> total_K, total_V;
    size_t total_seq_len = seq_len;
    if (offset != 0) {
      auto [total_K1, total_V1] = kv_cache->get_contiguous_tensor(i);
      total_K = total_K1.view({offset + seq_len, n_kv_heads_, head_dim_});
      total_V = total_V1.view({offset + seq_len, n_kv_heads_, head_dim_});
    } else {
      total_K = k_buf_view;
      total_V = v_buf_view;
    }

    Tensor<T> att_heads({n_heads_, head_dim_}, Device::CUDA);
    cuda_OP::flash_attention(q_buf_view, total_K, total_V, att_heads,
                             used_stream);

    Tensor<T> att_heads_reshaped = att_heads.view({1, n_heads_ * head_dim_});
    Tensor<T> att_proj({1, hidden_size_}, Device::CUDA);
    cuda_OP::matmul(att_heads_reshaped, wo, &att_proj, used_stream, o_bias);

    cuda_OP::add(&residual, &residual, &att_proj, used_stream);

    auto& ffn_norm_weight =
        params_.at(layer_prefix + "post_attention_layernorm.weight");
    cuda_OP::rms_norm(&hidden_states, &residual, &ffn_norm_weight,
                      rms_norm_eps_, used_stream);

    auto& gate_weight = params_.at(layer_prefix + "mlp.gate_proj.weight");
    auto& up_weight = params_.at(layer_prefix + "mlp.up_proj.weight");
    auto& down_weight = params_.at(layer_prefix + "mlp.down_proj.weight");

    const Tensor<T>* gate_bias = nullptr;
    const Tensor<T>* up_bias = nullptr;
    const Tensor<T>* down_bias = nullptr;
    Tensor<T> gate_buf({seq_len, gate_weight.sizes()[1]}, Device::CUDA);
    cuda_OP::matmul(hidden_states, gate_weight, &gate_buf, used_stream,
                    gate_bias);
    Tensor<T> up_buf({seq_len, up_weight.sizes()[1]}, Device::CUDA);
    cuda_OP::matmul(hidden_states, up_weight, &up_buf, used_stream, up_bias);

    cuda_OP::silu(&gate_buf, &gate_buf, used_stream);
    cuda_OP::multiply(&gate_buf, &gate_buf, &up_buf, used_stream);

    Tensor<T> ffn_out({seq_len, down_weight.sizes()[1]}, Device::CUDA);
    cuda_OP::matmul(gate_buf, down_weight, &ffn_out, used_stream, down_bias);

    cuda_OP::add(&residual, &residual, &ffn_out, used_stream);
  }  // end for layers

  auto& norm_weight = params_.at("norm.weight");
  Tensor<T> final_h({seq_len, hidden_size_}, Device::CUDA);
  cuda_OP::rms_norm(&final_h, &residual, &norm_weight, rms_norm_eps_,
                    used_stream);

  auto& lm_head_weight = params_.at("lm_head");
  const Tensor<T>* lm_head_bias = nullptr;
  Tensor<T> logits_placeholder({seq_len, vocab_size_}, Device::CUDA);
  if (p_output) {
    cuda_OP::matmul(final_h, lm_head_weight, p_output, used_stream,
                    lm_head_bias);
    return *p_output;
  } else {
    cuda_OP::matmul(final_h, lm_head_weight, &logits_placeholder, used_stream,
                    lm_head_bias);
    return logits_placeholder;
  }
}

// -------------------------------
// prefill_cuda: 保持原样
// -------------------------------
template <typename T>
Tensor<T> QwenModel<T>::prefill_cuda(const Tensor<uint32_t>* input,
                                     KVCache<T>* kv_cache) {
  if (input->device() != Device::CUDA)
    throw std::runtime_error("Input tensor must be on CUDA device");
  const size_t seq_len = input->sizes()[0];
  size_t offset = 0;
  if (kv_cache) {
    if (kv_cache->device() != Device::CUDA)
      throw std::runtime_error("KVCache must be on CUDA device");
    offset = kv_cache->size() - seq_len;
  }
  kv_cache->resize(offset + seq_len);
  Tensor<T> residual({seq_len, hidden_size_}, Device::CUDA);
  Tensor<T> hidden_states({seq_len, hidden_size_}, Device::CUDA);

  cuda_OP::gather(&residual, input, &params_.at("token_embeddings.weight"));
  cudaStreamSynchronize(compute_streams_[3]);
  cudaStreamSynchronize(compute_streams_[4]);

  for (size_t i = 0; i < n_layers_; i++) {
    std::string layer_prefix = "layers." + std::to_string(i) + ".";
    auto& attention_norm_weight =
        params_.at(layer_prefix + "input_layernorm.weight");
    cuda_OP::rms_norm(&hidden_states, &residual, &attention_norm_weight,
                      rms_norm_eps_);
    auto& wq = params_.at(layer_prefix + "self_attn.q_proj.weight");
    auto& wk = params_.at(layer_prefix + "self_attn.k_proj.weight");
    auto& wv = params_.at(layer_prefix + "self_attn.v_proj.weight");
    auto& wo = params_.at(layer_prefix + "self_attn.o_proj.weight");

    const Tensor<T>* q_bias = nullptr;
    const Tensor<T>* k_bias = nullptr;
    const Tensor<T>* v_bias = nullptr;
    const Tensor<T>* o_bias = nullptr;
    try {
      q_bias = &params_.at(layer_prefix + "self_attn.q_proj.bias");
    } catch (...) {
    }
    try {
      k_bias = &params_.at(layer_prefix + "self_attn.k_proj.bias");
    } catch (...) {
    }
    try {
      v_bias = &params_.at(layer_prefix + "self_attn.v_proj.bias");
    } catch (...) {
    }

    Tensor<T> q_buf({seq_len, n_heads_ * head_dim_}, Device::CUDA);
    cuda_OP::matmul(hidden_states, wq, &q_buf, compute_streams_[0], q_bias);
    Tensor<T> k_buf({seq_len, n_kv_heads_ * head_dim_}, Device::CUDA);
    cuda_OP::matmul(hidden_states, wk, &k_buf, compute_streams_[1], k_bias);
    Tensor<T> v_buf({seq_len, n_kv_heads_ * head_dim_}, Device::CUDA);
    cuda_OP::matmul(hidden_states, wv, &v_buf, compute_streams_[2], v_bias);

    const size_t row_size = n_kv_heads_ * head_dim_;
    Tensor<T> q_buf_view = q_buf.view({seq_len, n_heads_, head_dim_});
    Tensor<T> k_buf_view = k_buf.view({seq_len, n_kv_heads_, head_dim_});
    Tensor<T> v_buf_view = v_buf.view({seq_len, n_kv_heads_, head_dim_});
    cuda_OP::rope(&q_buf_view, offset, rope_theta_, compute_streams_[0]);
    cuda_OP::rope(&k_buf_view, offset, rope_theta_, compute_streams_[1]);
    for (int j = 0; j < 3; ++j) {
      cudaStreamSynchronize(compute_streams_[j]);
    }
    for (size_t j = 0; j < seq_len; j++) {
      Tensor<T>& k_slice = kv_cache->k_cache(i, offset + j);
      Tensor<T>& v_slice = kv_cache->v_cache(i, offset + j);
      cudaMemcpyAsync(k_slice.data_ptr(), k_buf_view.data_ptr() + j * row_size,
                      row_size * sizeof(T), cudaMemcpyDeviceToDevice,
                      compute_streams_[3]);
      cudaMemcpyAsync(v_slice.data_ptr(), v_buf_view.data_ptr() + j * row_size,
                      row_size * sizeof(T), cudaMemcpyDeviceToDevice,
                      compute_streams_[4]);
    }
    Tensor<T> Q_3d = q_buf_view;
    Tensor<T> total_K, total_V;
    size_t total_seq_len = 0;
    if (offset > 0) {
      size_t cached_len = offset;
      total_seq_len = cached_len + seq_len;
      auto [total_K1, total_V1] = kv_cache->get_contiguous_tensor(i);
      total_K = total_K1.view({total_seq_len, n_kv_heads_, head_dim_});
      total_V = total_V1.view({total_seq_len, n_kv_heads_, head_dim_});
    } else {
      total_K = k_buf_view;
      total_V = v_buf_view;
      total_seq_len = seq_len;
    }
    Tensor<T> att_scores({seq_len, n_heads_, total_seq_len}, Device::CUDA);
    cuda_OP::compute_attention_scores_prefill(Q_3d, total_K, att_scores,
                                              head_dim_, compute_streams_[0]);
    cuda_OP::softmax(&att_scores, &att_scores, /*dim=*/2, true, offset,
                     compute_streams_[0]);
    Tensor<T> att_heads({seq_len, n_heads_, head_dim_}, Device::CUDA);
    cuda_OP::compute_att_output_prefill(att_scores, total_V, att_heads,
                                        n_heads_, head_dim_, total_seq_len,
                                        n_kv_heads_, compute_streams_[0]);
    Tensor<T> att_proj({seq_len, hidden_size_}, Device::CUDA);
    cuda_OP::matmul(att_heads.view({seq_len, n_heads_ * head_dim_}), wo,
                    &att_proj, compute_streams_[0], o_bias);
    cuda_OP::add(&residual, &residual, &att_proj, compute_streams_[0]);
    auto& ffn_norm_weight =
        params_.at(layer_prefix + "post_attention_layernorm.weight");
    cuda_OP::rms_norm(&hidden_states, &residual, &ffn_norm_weight,
                      rms_norm_eps_, compute_streams_[0]);
    auto& gate_weight = params_.at(layer_prefix + "mlp.gate_proj.weight");
    auto& up_weight = params_.at(layer_prefix + "mlp.up_proj.weight");
    auto& down_weight = params_.at(layer_prefix + "mlp.down_proj.weight");
    const Tensor<T>* gate_bias = nullptr;
    const Tensor<T>* up_bias = nullptr;
    const Tensor<T>* down_bias = nullptr;
    size_t ffn_hidden_size = gate_weight.sizes()[1];
    Tensor<T> gate_buf({seq_len, ffn_hidden_size}, Device::CUDA);
    cuda_OP::matmul(hidden_states, gate_weight, &gate_buf, compute_streams_[0],
                    gate_bias);
    Tensor<T> up_buf({seq_len, ffn_hidden_size}, Device::CUDA);
    cuda_OP::matmul(hidden_states, up_weight, &up_buf, compute_streams_[0],
                    up_bias);
    cuda_OP::silu(&gate_buf, &gate_buf, compute_streams_[0]);
    cuda_OP::multiply(&gate_buf, &gate_buf, &up_buf, compute_streams_[0]);
    size_t down_output_dim = down_weight.sizes()[1];
    Tensor<T> ffn_out({seq_len, down_output_dim}, Device::CUDA);
    cuda_OP::matmul(gate_buf, down_weight, &ffn_out, compute_streams_[0],
                    down_bias);
    cuda_OP::add(&residual, &residual, &ffn_out, compute_streams_[0]);
  }
  auto& norm_weight = params_.at("norm.weight");
  Tensor<T> final_h({seq_len, hidden_size_}, Device::CUDA);
  cuda_OP::rms_norm(&final_h, &residual, &norm_weight, rms_norm_eps_,
                    compute_streams_[0]);
  auto& lm_head_weight = params_.at("lm_head");
  const Tensor<T>* lm_head_bias = nullptr;
  Tensor<T> logits({seq_len, vocab_size_}, Device::CUDA);
  cuda_OP::matmul(final_h, lm_head_weight, &logits, compute_streams_[0],
                  lm_head_bias);
  return logits;
}

// -------------------------------
// cuda() 接口
// -------------------------------
template <typename T>
QwenModel<T>& QwenModel<T>::cuda() {
  for (auto& kv : params_) {
    if (kv.second.device() != Device::CUDA) {
      kv.second.cuda();
    }
  }
  device_ = Device::CUDA;
  return *this;
}

// -------------------------------
// cpu() 接口（仅支持 CUDA）
// -------------------------------
template <typename T>
QwenModel<T>& QwenModel<T>::cpu() {
  throw std::runtime_error("QwenModel only supports CUDA execution.");
  return *this;
}

// -------------------------------
// generate 接口（stub 实现）
// -------------------------------
template <typename T>
std::vector<uint32_t> QwenModel<T>::generate(
    const std::vector<uint32_t>& input_ids, size_t max_length,
    float temperature, float top_p, size_t top_k) {
  throw std::runtime_error("Token generation not implemented for QwenModel");
  return std::vector<uint32_t>();
}

// -------------------------------
// forward 对外接口，支持 CUDA 图优化（预热 + 每次 forward 重新捕获）
// -------------------------------
template <typename T>
uint32_t* QwenModel<T>::forward(const Tensor<uint32_t>* input,
                                ThreadPool& thread_pool, KVCacheBase* kv_cache,
                                size_t top_k, float temperature, float top_p,
                                curandState* d_states) {
  KVCache<T>* typed_cache = dynamic_cast<KVCache<T>*>(kv_cache);
  if (graph_enabled_) {
    // 预热：dummy 调用预热 cuBLAS 内部缓存，避免 capture 中动态分配
    {
      Tensor<T> dummy_A({1, 1}, Device::CUDA);
      Tensor<T> dummy_B({1, 1}, Device::CUDA);
      Tensor<T> dummy_C({1, 1}, Device::CUDA);
      // 显式转换 nullptr 为 (const Tensor<T>*)
      cuda_OP::matmul(dummy_A, dummy_B, &dummy_C, graph_stream_,
                      (const Tensor<T>*)(nullptr), 1);
      cudaStreamSynchronize(graph_stream_);
    }
    // 使用 relaxed 模式捕获
    cudaError_t err =
        cudaStreamBeginCapture(graph_stream_, cudaStreamCaptureModeRelaxed);
    if (err != cudaSuccess)
      throw std::runtime_error(
          "Failed to begin CUDA graph capture in forward()");
    forward_cuda(input, typed_cache, &forward_logits_, graph_stream_);
    err = cudaStreamEndCapture(graph_stream_, &forward_graph_);
    if (err != cudaSuccess)
      throw std::runtime_error("Failed to end CUDA graph capture in forward()");
    err = cudaGraphInstantiate(&forward_graph_exec_, forward_graph_, 0);
    if (err != cudaSuccess)
      throw std::runtime_error("Failed to instantiate CUDA graph in forward()");
    cudaGraphLaunch(forward_graph_exec_, graph_stream_);
    cudaStreamSynchronize(graph_stream_);
    uint32_t* result = cuda_OP::sample(std::move(forward_logits_), temperature,
                                       top_p, top_k, d_states);

    // 重新捕获下一轮图，保证最新的 offset 被使用
    err = cudaStreamBeginCapture(graph_stream_, cudaStreamCaptureModeRelaxed);
    if (err != cudaSuccess)
      throw std::runtime_error(
          "Failed to begin CUDA graph capture for next round");
    forward_cuda(input, typed_cache, &forward_logits_, graph_stream_);
    err = cudaStreamEndCapture(graph_stream_, &forward_graph_);
    if (err != cudaSuccess)
      throw std::runtime_error(
          "Failed to end CUDA graph capture for next round");
    // 销毁先前的图执行实例
    cudaGraphExecDestroy(forward_graph_exec_);
    err = cudaGraphInstantiate(&forward_graph_exec_, forward_graph_, 0);
    if (err != cudaSuccess)
      throw std::runtime_error(
          "Failed to instantiate CUDA graph for next round");

    return result;
  } else {
    Tensor<T> logits = forward_cuda(input, typed_cache, nullptr, nullptr);
    return cuda_OP::sample(std::move(logits), temperature, top_p, top_k,
                           d_states);
  }
}

// -------------------------------
// 辅助函数：FP32 到 __nv_bfloat16 权重转换
// -------------------------------
std::unordered_map<std::string, Tensor<__nv_bfloat16>> convert_weights_to_bf16(
    const std::unordered_map<std::string, Tensor<float>>& float_weights) {
  std::unordered_map<std::string, Tensor<__nv_bfloat16>> bf16_weights;
  for (const auto& kv : float_weights) {
    const std::string& key = kv.first;
    const Tensor<float>& tensor = kv.second;
    std::vector<__nv_bfloat16> bf16_data;
    bf16_data.reserve(tensor.numel());
    const float* data_ptr = tensor.data_ptr();
    for (size_t i = 0; i < tensor.numel(); ++i) {
      bf16_data.push_back(__nv_bfloat16(data_ptr[i]));
    }
    bf16_weights.emplace(
        key, Tensor<__nv_bfloat16>(std::move(bf16_data), tensor.sizes()));
  }
  return bf16_weights;
}

template class QwenModel<float>;
template class QwenModel<__nv_bfloat16>;
