// qwen.cpp
#include "qwen.hpp"

#include <cmath>
#include <iostream>

#include "cudaOP.cuh"

// Debug print function for tensors

// -------------------------------
// QwenModel<T> 构造函数
// -------------------------------
template <typename T>
QwenModel<T>::QwenModel(
    const std::unordered_map<std::string, Tensor<T>>& params,
    const std::unordered_map<std::string, int>& config)
    : params_(params) {
  // 从 config 中提取基本参数
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

  // Qwen 模型仅支持 CUDA 运行
  device_ = Device::CUDA;
}

// -------------------------------
// 参数验证：检查全局与层级关键参数是否存在
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
// forward_cuda: Qwen2 模型的单个 token CUDA 前向传播
// -------------------------------
template <typename T>
Tensor<T> QwenModel<T>::forward_cuda(const Tensor<uint32_t>* input,
                                     KVCache<T>* kv_cache) {
  // 确保输入在 CUDA 上
  if (input->device() != Device::CUDA) {
    throw std::runtime_error("Input tensor must be on CUDA device");
  }

  // 获取输入信息
  const size_t seq_len = 1;  // 前向传播时序列长度固定为1

  // 计算起始KV缓存位置
  size_t offset = 0;
  if (kv_cache) {
    if (kv_cache->device() != Device::CUDA) {
      throw std::runtime_error("KVCache must be on CUDA device");
    }
    offset = kv_cache->size() - seq_len;
  }

  // 创建residual和hidden_states张量(没有batch维度)
  Tensor<T> residual({seq_len, hidden_size_}, Device::CUDA);
  Tensor<T> hidden_states({seq_len, hidden_size_}, Device::CUDA);

  // Token嵌入 (从embedding_table中获取token嵌入)
  cuda_OP::gather(&residual, input, &params_.at("token_embeddings.weight"));

  // 主循环：遍历所有Transformer层
  for (size_t i = 0; i < n_layers_; i++) {
    std::string layer_prefix = "layers." + std::to_string(i) + ".";

    // 1. Input LayerNorm (RMSNorm)
    auto& attention_norm_weight =
        params_.at(layer_prefix + "input_layernorm.weight");
    cuda_OP::rms_norm(&hidden_states, &residual, &attention_norm_weight,
                      rms_norm_eps_);

    // 2. Self-Attention
    auto& wq = params_.at(layer_prefix + "self_attn.q_proj.weight");
    auto& wk = params_.at(layer_prefix + "self_attn.k_proj.weight");
    auto& wv = params_.at(layer_prefix + "self_attn.v_proj.weight");
    auto& wo = params_.at(layer_prefix + "self_attn.o_proj.weight");

    // 获取偏置项（如果存在）
    const Tensor<T>* q_bias = nullptr;
    const Tensor<T>* k_bias = nullptr;
    const Tensor<T>* v_bias = nullptr;
    const Tensor<T>* o_bias = nullptr;

    try {
      q_bias = &params_.at(layer_prefix + "self_attn.q_proj.bias");
      // std::cout << "Found q_bias for layer " << i << std::endl;
    } catch (const std::out_of_range&) {
      // 偏置不存在，保持为nullptr
    }

    try {
      k_bias = &params_.at(layer_prefix + "self_attn.k_proj.bias");
      // std::cout << "Found k_bias for layer " << i << std::endl;
    } catch (const std::out_of_range&) {
      // 偏置不存在，保持为nullptr
    }

    try {
      v_bias = &params_.at(layer_prefix + "self_attn.v_proj.bias");
      // std::cout << "Found v_bias for layer " << i << std::endl;
    } catch (const std::out_of_range&) {
      // 偏置不存在，保持为nullptr
    }

    try {
      o_bias = &params_.at(layer_prefix + "self_attn.o_proj.bias");
      // std::cout << "Found o_bias for layer " << i << std::endl;
    } catch (const std::out_of_range&) {
      // 偏置不存在，保持为nullptr
    }

    // 创建CUDA流以并行计算Q, K, V
    cudaStream_t streams[3];
    for (int j = 0; j < 3; j++) {
      cudaError_t err = cudaStreamCreate(&streams[j]);
      if (err != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA stream");
      }
    }

    // 计算Q, K, V投影（使用新的matmul接口，bias作为第四个参数）
    Tensor<T> q_buf = cuda_OP::matmul(hidden_states, wq, streams[0], q_bias);
    Tensor<T> k_buf = cuda_OP::matmul(hidden_states, wk, streams[1], k_bias);
    Tensor<T> v_buf = cuda_OP::matmul(hidden_states, wv, streams[2], v_bias);

    // 同步CUDA流
    for (int j = 0; j < 3; j++) {
      cudaStreamSynchronize(streams[j]);
      cudaStreamDestroy(streams[j]);
    }

    // 重塑张量，准备应用RoPE
    Tensor<T> q_buf_view = q_buf.view({seq_len, n_heads_, head_dim_});
    Tensor<T> k_buf_view = k_buf.view({seq_len, n_kv_heads_, head_dim_});
    Tensor<T> v_buf_view = v_buf.view({seq_len, n_kv_heads_, head_dim_});

    // 应用旋转位置编码 (RoPE)
    cuda_OP::rope(&q_buf_view, offset, rope_theta_);
    cuda_OP::rope(&k_buf_view, offset, rope_theta_);

    // 更新KV缓存
    size_t row_size = n_kv_heads_ * head_dim_;
    for (size_t j = 0; j < seq_len; j++) {
      Tensor<T> k_i({1, row_size}, Device::CUDA);
      Tensor<T> v_i({1, row_size}, Device::CUDA);

      cudaMemcpy(k_i.data_ptr(), k_buf_view.data_ptr() + j * row_size,
                 row_size * sizeof(T), cudaMemcpyDeviceToDevice);

      cudaMemcpy(v_i.data_ptr(), v_buf_view.data_ptr() + j * row_size,
                 row_size * sizeof(T), cudaMemcpyDeviceToDevice);

      kv_cache->k_cache(i, offset + j) = std::move(k_i);
      kv_cache->v_cache(i, offset + j) = std::move(v_i);
    }

    // 准备计算自注意力
    Tensor<T> Q_3d = q_buf_view;
    Tensor<T> total_K, total_V;
    size_t total_seq_len = seq_len;

    // 如果有缓存，拼接当前和缓存的K,V
    if (offset != 0) {
      size_t cached_len = offset;
      total_seq_len = cached_len + seq_len;

      total_K =
          Tensor<T>({total_seq_len, n_kv_heads_, head_dim_}, Device::CUDA);
      total_V =
          Tensor<T>({total_seq_len, n_kv_heads_, head_dim_}, Device::CUDA);

      // 拷贝缓存的K,V
      for (size_t pos = 0; pos < cached_len; pos++) {
        Tensor<T>& cached_k = kv_cache->k_cache(i, pos);
        Tensor<T>& cached_v = kv_cache->v_cache(i, pos);

        cudaMemcpy(total_K.data_ptr() + pos * row_size, cached_k.data_ptr(),
                   row_size * sizeof(T), cudaMemcpyDeviceToDevice);

        cudaMemcpy(total_V.data_ptr() + pos * row_size, cached_v.data_ptr(),
                   row_size * sizeof(T), cudaMemcpyDeviceToDevice);
      }

      // 拷贝当前的K,V
      cudaMemcpy(total_K.data_ptr() + cached_len * row_size,
                 k_buf_view.data_ptr(), seq_len * row_size * sizeof(T),
                 cudaMemcpyDeviceToDevice);

      cudaMemcpy(total_V.data_ptr() + cached_len * row_size,
                 v_buf_view.data_ptr(), seq_len * row_size * sizeof(T),
                 cudaMemcpyDeviceToDevice);
    } else {
      total_K = k_buf_view;
      total_V = v_buf_view;
    }

    // 计算注意力分数
    Tensor<T> att_scores({n_heads_, total_seq_len}, Device::CUDA);
    cuda_OP::compute_attention_scores(Q_3d, total_K, n_heads_, head_dim_,
                                      att_scores, n_kv_heads_);

    // Softmax处理注意力分数
    cuda_OP::softmax(&att_scores, &att_scores, /*dim=*/1, false, offset);

    // 计算注意力输出
    Tensor<T> att_heads({n_heads_, head_dim_}, Device::CUDA);
    cuda_OP::compute_att_output(att_scores, total_V, n_heads_, head_dim_,
                                att_heads, n_kv_heads_);

    // 投影回原始维度
    Tensor<T> att_proj = cuda_OP::matmul(
        att_heads.view({1, n_heads_ * head_dim_}), wo, nullptr, o_bias);

    // 残差连接
    residual = residual + att_proj;

    // 3. Post Attention LayerNorm (RMSNorm)
    auto& ffn_norm_weight =
        params_.at(layer_prefix + "post_attention_layernorm.weight");
    cuda_OP::rms_norm(&hidden_states, &residual, &ffn_norm_weight,
                      rms_norm_eps_);

    // 4. MLP (Feed Forward Network)
    auto& gate_weight = params_.at(layer_prefix + "mlp.gate_proj.weight");
    auto& up_weight = params_.at(layer_prefix + "mlp.up_proj.weight");
    auto& down_weight = params_.at(layer_prefix + "mlp.down_proj.weight");

    // 获取偏置项（如果存在）
    const Tensor<T>* gate_bias = nullptr;
    const Tensor<T>* up_bias = nullptr;
    const Tensor<T>* down_bias = nullptr;

    try {
      gate_bias = &params_.at(layer_prefix + "mlp.gate_proj.bias");
      // std::cout << "Found gate_bias for layer " << i << std::endl;
    } catch (const std::out_of_range&) {
      // 偏置不存在，保持为nullptr
    }

    try {
      up_bias = &params_.at(layer_prefix + "mlp.up_proj.bias");
      // std::cout << "Found up_bias for layer " << i << std::endl;
    } catch (const std::out_of_range&) {
      // 偏置不存在，保持为nullptr
    }

    try {
      down_bias = &params_.at(layer_prefix + "mlp.down_proj.bias");
      // std::cout << "Found down_bias for layer " << i << std::endl;
    } catch (const std::out_of_range&) {
      // 偏置不存在，保持为nullptr
    }

    // SwiGLU激活: (gate_proj * silu(up_proj))
    Tensor<T> gate_buf =
        cuda_OP::matmul(hidden_states, gate_weight, nullptr, gate_bias);
    Tensor<T> up_buf =
        cuda_OP::matmul(hidden_states, up_weight, nullptr, up_bias);

    cuda_OP::silu(&gate_buf, &gate_buf);               // SiLU激活
    cuda_OP::multiply(&gate_buf, &gate_buf, &up_buf);  // 逐元素相乘

    // 投影回原始维度
    Tensor<T> ffn_out =
        cuda_OP::matmul(gate_buf, down_weight, nullptr, down_bias);

    // 残差连接
    residual = residual + ffn_out;
  }

  // 最终的LayerNorm (RMSNorm)
  auto& norm_weight = params_.at("norm.weight");
  Tensor<T> final_h({seq_len, hidden_size_}, Device::CUDA);
  cuda_OP::rms_norm(&final_h, &residual, &norm_weight, rms_norm_eps_);

  // LM head投影到词汇表大小
  auto& lm_head_weight = params_.at("lm_head");

  // 检查是否存在lm_head的偏置
  const Tensor<T>* lm_head_bias = nullptr;
  try {
    lm_head_bias = &params_.at("lm_head_bias");
    std::cout << "Found lm_head_bias" << std::endl;
  } catch (const std::out_of_range&) {
    // 偏置不存在，保持为nullptr
  }

  Tensor<T> logits({seq_len, vocab_size_}, Device::CUDA);
  logits = cuda_OP::matmul(final_h, lm_head_weight, nullptr, lm_head_bias);

  // 返回最后一个token的logits
  return logits;
}

// -------------------------------
// prefill_cuda: Qwen2 模型的序列预填充 CUDA 实现
// -------------------------------
template <typename T>
Tensor<T> QwenModel<T>::prefill_cuda(const Tensor<uint32_t>* input,
                                     KVCache<T>* kv_cache) {
  // 确保输入在 CUDA 上
  if (input->device() != Device::CUDA) {
    throw std::runtime_error("Input tensor must be on CUDA device");
  }

  // 获取输入信息
  const size_t seq_len = input->sizes()[0];

  // 计算起始KV缓存位置
  size_t offset = 0;
  if (kv_cache) {
    if (kv_cache->device() != Device::CUDA) {
      throw std::runtime_error("KVCache must be on CUDA device");
    }
    offset = kv_cache->size() - seq_len;
  }

  // 重设KV缓存大小
  kv_cache->resize(offset + seq_len);

  // 创建residual和hidden_states张量
  Tensor<T> residual({seq_len, hidden_size_}, Device::CUDA);
  Tensor<T> hidden_states({seq_len, hidden_size_}, Device::CUDA);

  // Token嵌入 (从embedding_table中获取token嵌入)
  cuda_OP::gather(&residual, input, &params_.at("token_embeddings.weight"));

  // 主循环：遍历所有Transformer层
  for (size_t i = 0; i < n_layers_; i++) {
    std::string layer_prefix = "layers." + std::to_string(i) + ".";

    // 1. Input LayerNorm (RMSNorm)
    auto& attention_norm_weight =
        params_.at(layer_prefix + "input_layernorm.weight");
    cuda_OP::rms_norm(&hidden_states, &residual, &attention_norm_weight,
                      rms_norm_eps_);
    // debugPrintTensor(hidden_states,
    //  "hidden_states after input_layernorm" + std::to_string(i));
    // 2. Self-Attention
    auto& wq = params_.at(layer_prefix + "self_attn.q_proj.weight");
    auto& wk = params_.at(layer_prefix + "self_attn.k_proj.weight");
    auto& wv = params_.at(layer_prefix + "self_attn.v_proj.weight");
    auto& wo = params_.at(layer_prefix + "self_attn.o_proj.weight");
    // debugPrintTensor(wq, "wq after input_layernorm" +
    // std::to_string(i));

    // 获取偏置项（如果存在）
    const Tensor<T>* q_bias = nullptr;
    const Tensor<T>* k_bias = nullptr;
    const Tensor<T>* v_bias = nullptr;
    const Tensor<T>* o_bias = nullptr;

    try {
      q_bias = &params_.at(layer_prefix + "self_attn.q_proj.bias");
      // std::cout << "Found q_bias for layer " << i << std::endl;
    } catch (const std::out_of_range&) {
      // 偏置不存在，保持为nullptr
    }

    try {
      k_bias = &params_.at(layer_prefix + "self_attn.k_proj.bias");
      // std::cout << "Found k_bias for layer " << i << std::endl;
    } catch (const std::out_of_range&) {
      // 偏置不存在，保持为nullptr
    }

    try {
      v_bias = &params_.at(layer_prefix + "self_attn.v_proj.bias");
      // std::cout << "Found v_bias for layer " << i << std::endl;
    } catch (const std::out_of_range&) {
      // 偏置不存在，保持为nullptr
    }

    try {
      o_bias = &params_.at(layer_prefix + "self_attn.o_proj.bias");
      // std::cout << "Found o_bias for layer " << i << std::endl;
    } catch (const std::out_of_range&) {
      // 偏置不存在，保持为nullptr
    }

    // 创建CUDA流以并行计算Q, K, V
    cudaStream_t streams[3];
    for (int j = 0; j < 3; j++) {
      cudaError_t err = cudaStreamCreate(&streams[j]);
      if (err != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA stream");
      }
    }

    // 计算Q, K, V投影（使用新的matmul接口，bias作为第四个参数）
    Tensor<T> q_buf = cuda_OP::matmul(hidden_states, wq, streams[0], q_bias);
    Tensor<T> k_buf = cuda_OP::matmul(hidden_states, wk, streams[1], k_bias);
    Tensor<T> v_buf = cuda_OP::matmul(hidden_states, wv, streams[2], v_bias);
    // debugPrintTensor(q_buf, "q_buf after matmul" +
    // std::to_string(i)); debugPrintTensor(k_buf, "k_buf after
    // matmul" + std::to_string(i)); debugPrintTensor(v_buf,
    // "v_buf after matmul" + std::to_string(i));
    //  同步CUDA流
    for (int j = 0; j < 3; j++) {
      cudaStreamSynchronize(streams[j]);
      cudaStreamDestroy(streams[j]);
    }

    // 重塑张量，准备应用RoPE
    const size_t head_size = hidden_size_ / n_heads_;
    const size_t kv_head_size = hidden_size_ / n_kv_heads_;
    const size_t row_size = n_kv_heads_ * head_dim_;

    // 使用正确的维度进行重塑操作
    Tensor<T> q_buf_view = q_buf.view({seq_len, n_heads_, head_dim_});
    Tensor<T> k_buf_view = k_buf.view({seq_len, n_kv_heads_, head_dim_});
    Tensor<T> v_buf_view = v_buf.view({seq_len, n_kv_heads_, head_dim_});
    // debugPrintTensor(q_buf_view,
    //   "q_buf_view after reshape" + std::to_string(i));
    // debugPrintTensor(k_buf_view,
    //   "k_buf_view after reshape" + std::to_string(i));
    // debugPrintTensor(v_buf_view,
    //   "v_buf_view after reshape" + std::to_string(i));

    // 应用旋转位置编码 (RoPE)
    cuda_OP::rope(&q_buf_view, offset, rope_theta_);
    cuda_OP::rope(&k_buf_view, offset, rope_theta_);
    // debugPrintTensor(q_buf_view, "q_buf_view after rope" +
    // std::to_string(i)); debugPrintTensor(k_buf_view,
    // "k_buf_view after rope" + std::to_string(i));

    // 将K,V存储到缓存中
    for (size_t j = 0; j < seq_len; j++) {
      size_t pos = offset + j;
      Tensor<T> k_i({1, row_size}, Device::CUDA);
      Tensor<T> v_i({1, row_size}, Device::CUDA);

      // 修正：从k_buf和v_buf中取正确的数据（使用view确保使用正确的内存布局）
      cudaMemcpy(k_i.data_ptr(),
                 k_buf_view.data_ptr() + j * n_kv_heads_ * head_dim_,
                 row_size * sizeof(T), cudaMemcpyDeviceToDevice);

      cudaMemcpy(v_i.data_ptr(),
                 v_buf_view.data_ptr() + j * n_kv_heads_ * head_dim_,
                 row_size * sizeof(T), cudaMemcpyDeviceToDevice);

      kv_cache->k_cache(i, pos) = std::move(k_i);
      kv_cache->v_cache(i, pos) = std::move(v_i);
    }

    // 重新格式化Q用于注意力计算
    Tensor<T> Q_3d = q_buf_view;

    // 准备K和V张量用于注意力计算
    Tensor<T> total_K, total_V;
    size_t total_seq_len = 0;
    // debugPrintTensor(Q_3d, "Q_3d before attention" +
    // std::to_string(i));
    //  如果有缓存，拼接当前和缓存的K,V
    if (offset > 0) {
      size_t cached_len = offset;
      total_seq_len = cached_len + seq_len;

      // 分配足够大小的张量，使用正确的维度
      total_K =
          Tensor<T>({total_seq_len, n_kv_heads_, head_dim_}, Device::CUDA);
      total_V =
          Tensor<T>({total_seq_len, n_kv_heads_, head_dim_}, Device::CUDA);

      // 拷贝缓存的K,V - 确保正确处理内存布局
      for (size_t pos = 0; pos < cached_len; pos++) {
        const auto& cached_k = kv_cache->k_cache(i, pos);
        const auto& cached_v = kv_cache->v_cache(i, pos);
        // debugPrintTensor(cached_k, "cached_k before copy" +
        // std::to_string(i)); debugPrintTensor(cached_v,
        // "cached_v before copy" + std::to_string(i));

        // 确保从一维k_cache中正确拷贝到三维total_K
        cudaMemcpy(total_K.data_ptr() + pos * n_kv_heads_ * head_dim_,
                   cached_k.data_ptr(), n_kv_heads_ * head_dim_ * sizeof(T),
                   cudaMemcpyDeviceToDevice);

        cudaMemcpy(total_V.data_ptr() + pos * n_kv_heads_ * head_dim_,
                   cached_v.data_ptr(), n_kv_heads_ * head_dim_ * sizeof(T),
                   cudaMemcpyDeviceToDevice);
      }
      // debugPrintTensor(total_K, "total_K after copy" +
      // std::to_string(i)); debugPrintTensor(total_V, "total_V
      // after copy" + std::to_string(i));

      // 拷贝当前的K,V
      cudaMemcpy(total_K.data_ptr() + cached_len * n_kv_heads_ * head_dim_,
                 k_buf_view.data_ptr(),
                 seq_len * n_kv_heads_ * head_dim_ * sizeof(T),
                 cudaMemcpyDeviceToDevice);

      cudaMemcpy(total_V.data_ptr() + cached_len * n_kv_heads_ * head_dim_,
                 v_buf_view.data_ptr(),
                 seq_len * n_kv_heads_ * head_dim_ * sizeof(T),
                 cudaMemcpyDeviceToDevice);
      // debugPrintTensor(total_K, "total_K after copy" +
      // std::to_string(i)); debugPrintTensor(total_V, "total_V
      // after copy" + std::to_string(i));
    } else {
      total_K = k_buf_view;
      total_V = v_buf_view;
      total_seq_len = seq_len;
    }
    // debugPrintTensor(total_K, "total_K after attention" +
    // std::to_string(i)); debugPrintTensor(total_V, "total_V
    // after attention" + std::to_string(i));

    // 计算注意力分数 - prefill版本处理整个序列
    Tensor<T> att_scores({seq_len, n_heads_, total_seq_len}, Device::CUDA);
    cuda_OP::compute_attention_scores_prefill(Q_3d, total_K, att_scores,
                                              head_dim_);

    // Softmax处理注意力分数 - prefill版本需要设置mask=true
    cuda_OP::softmax(&att_scores, &att_scores, /*dim=*/2, true, offset);
    // debugPrintTensor(att_scores,
    //  "att_scores after softmax" + std::to_string(i));

    // 计算注意力输出 - prefill版本
    Tensor<T> att_heads({seq_len, n_heads_, head_dim_}, Device::CUDA);
    cuda_OP::compute_att_output_prefill(att_scores, total_V, att_heads,
                                        n_heads_, head_dim_, total_seq_len,
                                        n_kv_heads_);
    // debugPrintTensor(att_heads, "att_heads after
    // compute_att_output_prefill" +
    // std::to_string(i));

    // 投影回原始维度
    Tensor<T> att_proj = cuda_OP::matmul(
        att_heads.view({seq_len, n_heads_ * head_dim_}), wo, nullptr, o_bias);
    // debugPrintTensor(att_proj, "att_proj
    // after matmul" + std::to_string(i));

    // 残差连接
    residual = residual + att_proj;
    // debugPrintTensor(residual,
    //  "residual after residual connection" +
    //  std::to_string(i));

    // 3. Post Attention LayerNorm (RMSNorm)
    auto& ffn_norm_weight =
        params_.at(layer_prefix + "post_attention_layernorm.weight");
    cuda_OP::rms_norm(&hidden_states, &residual, &ffn_norm_weight,
                      rms_norm_eps_);
    // debugPrintTensor(
    //  hidden_states,
    //  "hidden_states after
    //  post_attention_layernorm" +
    //  std::to_string(i));

    // 4. MLP (Feed Forward Network)
    auto& gate_weight = params_.at(layer_prefix + "mlp.gate_proj.weight");
    auto& up_weight = params_.at(layer_prefix + "mlp.up_proj.weight");
    auto& down_weight = params_.at(layer_prefix + "mlp.down_proj.weight");
    // debugPrintTensor(gate_weight,
    // "gate_weight after mlp" +
    // std::to_string(i));
    // debugPrintTensor(up_weight, "up_weight
    // after mlp" + std::to_string(i));
    // debugPrintTensor(down_weight,
    // "down_weight after mlp" +
    // std::to_string(i));

    // 获取偏置项（如果存在）
    const Tensor<T>* gate_bias = nullptr;
    const Tensor<T>* up_bias = nullptr;
    const Tensor<T>* down_bias = nullptr;

    try {
      gate_bias = &params_.at(layer_prefix + "mlp.gate_proj.bias");
      // std::cout << "Found gate_bias for layer " << i << std::endl;
    } catch (const std::out_of_range&) {
      // 偏置不存在，保持为nullptr
    }

    try {
      up_bias = &params_.at(layer_prefix + "mlp.up_proj.bias");
      // std::cout << "Found up_bias for layer " << i << std::endl;
    } catch (const std::out_of_range&) {
      // 偏置不存在，保持为nullptr
    }

    try {
      down_bias = &params_.at(layer_prefix + "mlp.down_proj.bias");
      // std::cout << "Found down_bias for layer " << i << std::endl;
    } catch (const std::out_of_range&) {
      // 偏置不存在，保持为nullptr
    }

    // SwiGLU激活: (gate_proj * silu(up_proj))
    Tensor<T> gate_buf =
        cuda_OP::matmul(hidden_states, gate_weight, nullptr, gate_bias);
    Tensor<T> up_buf =
        cuda_OP::matmul(hidden_states, up_weight, nullptr, up_bias);

    cuda_OP::silu(&gate_buf,
                  &gate_buf);  // SiLU激活
    cuda_OP::multiply(&gate_buf, &gate_buf,
                      &up_buf);  // 逐元素相乘
    // debugPrintTensor(gate_buf, "gate_buf
    // after silu" + std::to_string(i));
    // debugPrintTensor(up_buf, "up_buf after
    // silu" + std::to_string(i));

    // 投影回原始维度
    Tensor<T> ffn_out =
        cuda_OP::matmul(gate_buf, down_weight, nullptr, down_bias);

    // 残差连接
    residual = residual + ffn_out;
    // debugPrintTensor(residual,
    //  "residual after residual connection" +
    //  std::to_string(i));
  }

  // 最终的LayerNorm (RMSNorm)
  auto& norm_weight = params_.at("norm.weight");
  Tensor<T> final_h({seq_len, hidden_size_}, Device::CUDA);
  cuda_OP::rms_norm(&final_h, &residual, &norm_weight, rms_norm_eps_);
  // debugPrintTensor(final_h, "final_h after rms_norm");

  // LM head投影到词汇表大小
  auto& lm_head_weight = params_.at("lm_head");

  // 检查是否存在lm_head的偏置
  const Tensor<T>* lm_head_bias = nullptr;
  try {
    lm_head_bias = &params_.at("lm_head_bias");
    std::cout << "Found lm_head_bias" << std::endl;
  } catch (const std::out_of_range&) {
    // 偏置不存在，保持为nullptr
  }

  Tensor<T> logits({seq_len, vocab_size_}, Device::CUDA);
  logits = cuda_OP::matmul(final_h, lm_head_weight, nullptr, lm_head_bias);

  // 返回最后一个token的logits
  return logits;
}

// -------------------------------
// cuda()：将所有参数移到 CUDA，并设置设备
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
// cpu()：Qwen 模型仅支持 CUDA，故调用 cpu() 抛出异常
// -------------------------------
template <typename T>
QwenModel<T>& QwenModel<T>::cpu() {
  throw std::runtime_error("QwenModel only supports CUDA execution.");
  return *this;
}

// -------------------------------
// generate: Token 生成接口，目前作为 stub
// -------------------------------
template <typename T>
std::vector<uint32_t> QwenModel<T>::generate(
    const std::vector<uint32_t>& input_ids, size_t max_length,
    float temperature, float top_p, size_t top_k) {
  // TODO: 实现 Qwen 模型的 token 生成逻辑
  throw std::runtime_error("Token generation not implemented for QwenModel");
  return std::vector<uint32_t>();
}

// -------------------------------
// 辅助函数：将 FP32 权重转换为 __nv_bfloat16 权重
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

// 显式模板实例化
template class QwenModel<float>;
template class QwenModel<__nv_bfloat16>;
