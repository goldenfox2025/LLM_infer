// qwen.cpp
#include "qwen.hpp"

#include <cmath>
#include <cmath>   // For std::fabs (for tolerance comparison if needed)
#include <cstring> // For memcmp (binary comparison)
#include <iostream>
#include <limits> // For std::numeric_limits (for tolerance comparison if needed)
#include <vector>

#include "cudaOP.cuh"

// Assuming Tensor class and Device enum are defined as before
// Assuming checkCudaError is available

template <typename T>
bool compareGpuTensors(const Tensor<T> &t1, const Tensor<T> &t2,
                       const std::string &t1_name = "Tensor1",
                       const std::string &t2_name = "Tensor2",
                       bool verbose = true) {
  // 1. 检查元数据
  if (t1.device() != Device::CUDA || t2.device() != Device::CUDA) {
    if (verbose)
      std::cerr << "[Compare Error] Both tensors must be on CUDA device.\n";
    return false;
  }
  if (t1.sizes() != t2.sizes()) {
    if (verbose)
      std::cerr << "[Compare Error] Tensor shapes mismatch: " << t1_name
                << " vs " << t2_name << "\n";
    return false;
  }
  if (t1.numel() == 0) {
    if (verbose)
      std::cout << "[Compare Info] Both tensors are empty, considered equal.\n";
    return true; // Empty tensors are equal
  }

  size_t num_elements = t1.numel();
  size_t n_bytes = num_elements * sizeof(T);

  // 2. 分配 Host 内存
  std::vector<T> h_buffer1(num_elements);
  std::vector<T> h_buffer2(num_elements);

  // 3. 拷贝数据 (使用默认流，确保拷贝完成)
  cudaError_t err1 = cudaMemcpy(h_buffer1.data(), t1.data_ptr(), n_bytes,
                                cudaMemcpyDeviceToHost);
  cudaError_t err2 = cudaMemcpy(h_buffer2.data(), t2.data_ptr(), n_bytes,
                                cudaMemcpyDeviceToHost);

  // --- 强制同步以确保拷贝完成 ---
  // 在比较前同步是安全的，尽管 cudaMemcpy 默认是同步的（对于默认流）
  // 但显式同步更清晰
  cudaError_t syncErr = cudaDeviceSynchronize();

  if (err1 != cudaSuccess || err2 != cudaSuccess || syncErr != cudaSuccess) {
    if (verbose) {
      std::cerr
          << "[Compare Error] cudaMemcpy or cudaDeviceSynchronize failed.\n";
      if (err1 != cudaSuccess)
        std::cerr << "  memcpy t1: " << cudaGetErrorString(err1) << std::endl;
      if (err2 != cudaSuccess)
        std::cerr << "  memcpy t2: " << cudaGetErrorString(err2) << std::endl;
      if (syncErr != cudaSuccess)
        std::cerr << "  sync: " << cudaGetErrorString(syncErr) << std::endl;
    }
    return false; // Treat copy error as inequality
  }

  // 4. 逐元素比较
  bool mismatch_found = false;
  size_t first_mismatch_idx = 0;
  T val1_at_mismatch = T(); // Default constructor
  T val2_at_mismatch = T();

  // --- 使用 memcmp 进行快速二进制比较 (推荐) ---
  if (memcmp(h_buffer1.data(), h_buffer2.data(), n_bytes) != 0) {
    // 如果二进制不匹配，再逐个查找第一个不同的元素用于报告
    mismatch_found = true;
    for (size_t i = 0; i < num_elements; ++i) {
      // 对于 bf16，直接比较可能不够精确，但可以先用 ==
      if constexpr (std::is_same_v<T, cuda_OP::nvbf16>) {
        // 转换为 float 比较更可靠
        if (static_cast<float>(h_buffer1[i]) !=
            static_cast<float>(h_buffer2[i])) {
          first_mismatch_idx = i;
          val1_at_mismatch = h_buffer1[i];
          val2_at_mismatch = h_buffer2[i];
          break;
        }
      } else { // For float or other types where == is reasonable initially
        if (h_buffer1[i] != h_buffer2[i]) {
          first_mismatch_idx = i;
          val1_at_mismatch = h_buffer1[i];
          val2_at_mismatch = h_buffer2[i];
          break;
        }
      }
      // --- 如果需要容差比较 (以 float 为例) ---
      // else if constexpr (std::is_same_v<T, float>) {
      //     float diff = std::fabs(h_buffer1[i] - h_buffer2[i]);
      //     float tolerance = 1e-6f; // 或者根据需要设置相对容差
      //     if (diff > tolerance) {
      //         mismatch_found = true;
      //         first_mismatch_idx = i;
      //         val1_at_mismatch = h_buffer1[i];
      //         val2_at_mismatch = h_buffer2[i];
      //         break;
      //     }
      // }
    }
  }

  // 5. 返回结果并打印信息
  if (mismatch_found) {
    if (verbose) {
      std::cerr << "[Compare Result] Tensors differ: " << t1_name << " vs "
                << t2_name << "\n";
      std::cerr << "  First mismatch at index " << first_mismatch_idx << ": "
                << val1_at_mismatch << " != " << val2_at_mismatch << "\n";
    }
    return false;
  } else {
    if (verbose) {
      std::cout << "[Compare Result] Tensors are identical: " << t1_name
                << " vs " << t2_name << "\n";
    }
    return true;
  }
}
// Debug print function for tensors
template <typename T>
void debugPrintTensor(const Tensor<T> &tensor, const std::string &tensor_name,
                      size_t num_to_print = 10) {
  std::cout << "[Debug] " << tensor_name << ":\n";

  // 1) Print shape
  std::cout << "  shape: [";
  for (auto s : tensor.sizes()) {
    std::cout << s << " ";
  }
  std::cout << "]\n";

  // 2) Print strides
  std::cout << "  strides: [";
  for (auto st : tensor.strides()) {
    std::cout << st << " ";
  }
  std::cout << "]\n";

  // 3) Print device
  std::cout << "  device: ";
  if (tensor.device() == Device::CPU) {
    std::cout << "CPU";
  } else if (tensor.device() == Device::CUDA) {
    std::cout << "CUDA";
  } else {
    std::cout << "UNKNOWN";
  }
  std::cout << "\n";

  // 4) Print elements starting from offset 0
  size_t offset = 1023; // 从开始处打印
  size_t total_elements = tensor.numel();
  size_t n_print = std::min(num_to_print, total_elements - offset);

  std::cout << "  elements from offset " << offset << " (" << n_print
            << " element(s)): ";
  if (tensor.device() == Device::CPU) {
    const T *ptr = tensor.data_ptr();
    for (size_t i = 0; i < n_print; i++) {
      std::cout << ptr[offset + i] << " ";
    }
    std::cout << "\n";
  } else {
    // Copy from GPU to CPU, then print
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
    const std::unordered_map<std::string, Tensor<T>> &params,
    const std::unordered_map<std::string, int> &config)
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

  // 检查是否有量化类型参数
  if (config.find("quant_type") != config.end()) {
    quant_type_ = config.at("quant_type");
  }

  // 检查是否有分组大小参数
  if (config.find("group_size") != config.end()) {
    group_size_ = config.at("group_size");
  }

  // Qwen 模型仅支持 CUDA 运行
  device_ = Device::CUDA;

  for (int i = 0; i < 5; ++i) {
    cudaError_t err = cudaStreamCreate(&compute_streams_[i]);
    if (err != cudaSuccess) {
      // 处理错误，可能需要清理已创建的流
      throw std::runtime_error("Failed to create CUDA stream in constructor");
    }
  }
  for (int i = 0; i < 3; ++i) {
    // 使用 cudaEventDisableTiming
    // 可以获得微小的性能提升，因为我们只关心完成状态，不测量时间
    cudaEventCreateWithFlags(&fa_done_events_[i], cudaEventDisableTiming);
  }
}

// 带量化参数的构造函数
template <typename T>
QwenModel<T>::QwenModel(
    const std::unordered_map<std::string, Tensor<T>> &params,
    const std::unordered_map<std::string, Tensor<int32_t>> &qweight_params,
    const std::unordered_map<std::string, Tensor<float>> &scales_params,
    const std::unordered_map<std::string, Tensor<int32_t>> &qzeros_params,
    const std::unordered_map<std::string, int> &config)
    : params_(params), qweight_params_(qweight_params),
      scales_params_(scales_params), qzeros_params_(qzeros_params) {
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

  // 设置量化类型为AWQ
  quant_type_ = 1;

  // 检查是否有分组大小参数
  if (config.find("group_size") != config.end()) {
    group_size_ = config.at("group_size");
  }
  // 创建非const副本并移动到CUDA
  for (const auto &kv : qweight_params) {
    if (kv.second.device() != Device::CUDA) {
      // 创建副本并移动到CUDA
      Tensor<int32_t> tensor_copy = kv.second;
      tensor_copy.cuda();
      qweight_params_[kv.first] = tensor_copy;
    }
  }
  for (const auto &kv : scales_params) {
    if (kv.second.device() != Device::CUDA) {
      // 创建副本并移动到CUDA
      Tensor<float> tensor_copy = kv.second;
      tensor_copy.cuda();
      scales_params_[kv.first] = tensor_copy;
    }
  }
  for (const auto &kv : qzeros_params) {
    if (kv.second.device() != Device::CUDA) {
      // 创建副本并移动到CUDA
      Tensor<int32_t> tensor_copy = kv.second;
      tensor_copy.cuda();
      qzeros_params_[kv.first] = tensor_copy;
    }
  }
  // Qwen 模型仅支持 CUDA 运行
  device_ = Device::CUDA;

  for (int i = 0; i < 5; ++i) {
    cudaError_t err = cudaStreamCreate(&compute_streams_[i]);
    if (err != cudaSuccess) {
      // 处理错误，可能需要清理已创建的流
      throw std::runtime_error("Failed to create CUDA stream in constructor");
    }
  }
  for (int i = 0; i < 3; ++i) {
    // 使用 cudaEventDisableTiming
    // 可以获得微小的性能提升，因为我们只关心完成状态，不测量时间
    cudaEventCreateWithFlags(&fa_done_events_[i], cudaEventDisableTiming);
  }
}

template <typename T> QwenModel<T>::~QwenModel() {
  for (cudaStream_t stream : compute_streams_) {
    if (stream) {
      // 最好在销毁流之前同步它，确保所有工作完成
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    }
  }

  for (int i = 0; i < 3; ++i) {
    if (fa_done_events_[i]) {
      cudaEventDestroy(fa_done_events_[i]);
    }
  }
}

// -------------------------------
// 参数验证：检查全局与层级关键参数是否存在
// -------------------------------
template <typename T> bool QwenModel<T>::verify_params() const {
  std::cout << "Not checking parameters" << std::endl;
  return true;
}

// -------------------------------
// 打印模型基本信息
// -------------------------------
template <typename T> void QwenModel<T>::print_model_info() const {
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
  std::cout << "  Quantization: "
            << (quant_type_ == 0 ? "None"
                                 : (quant_type_ == 1 ? "AWQ" : "Unknown"))
            << std::endl;
  if (quant_type_ != 0) {
    std::cout << "  Group size: " << group_size_ << std::endl;
    std::cout << "  Quantized weights count: " << qweight_params_.size()
              << std::endl;
    std::cout << "  Scales count: " << scales_params_.size() << std::endl;
    std::cout << "  Zeros count: " << qzeros_params_.size() << std::endl;
  }
}

// -------------------------------
// forward_cuda: Qwen2 模型的单个 token CUDA 前向传播
// -------------------------------

template <typename T>
Tensor<T> QwenModel<T>::forward_cuda(const Tensor<uint32_t> *input,
                                     KVCache<T> *kv_cache) {
  // 确保输入在 CUDA 上
  if (input->device() != Device::CUDA) {
    throw std::runtime_error("Input tensor must be on CUDA device");
  }

  // 获取输入信息，前向传播时序列长度固定为1
  const size_t seq_len = 1;

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

  // cudaStreamSynchronize(compute_streams_[3]);
  // cudaStreamSynchronize(compute_streams_[4]);
  // 主循环：遍历所有Transformer层
  std::string l = "layers." + std::to_string(0) + ".";
  auto &attention_norm_weight = params_.at(l + "input_layernorm.weight");
  cuda_OP::rms_norm(&hidden_states, &residual, &attention_norm_weight,
                    rms_norm_eps_);
  for (size_t i = 0; i < n_layers_; i++) {
    std::string layer_prefix = "layers." + std::to_string(i) + ".";

    // 2. Self-Attention
    Tensor<T> q_buf({seq_len, n_heads_ * head_dim_}, Device::CUDA);
    Tensor<T> &k_slice = kv_cache->k_cache(i, offset);
    Tensor<T> &v_slice = kv_cache->v_cache(i, offset);

    // 获取偏置项（如果存在）
    const Tensor<T> *q_bias = nullptr;
    const Tensor<T> *k_bias = nullptr;
    const Tensor<T> *v_bias = nullptr;
    const Tensor<T> *o_bias = nullptr;

    try {
      q_bias = &params_.at(layer_prefix + "self_attn.q_proj.bias");
    } catch (const std::out_of_range &) {
    }
    try {
      k_bias = &params_.at(layer_prefix + "self_attn.k_proj.bias");
    } catch (const std::out_of_range &) {
    }
    try {
      v_bias = &params_.at(layer_prefix + "self_attn.v_proj.bias");
    } catch (const std::out_of_range &) {
    }

    if (quant_type_ == 1) {
      // AWQ量化版本
      // 检查是否存在量化权重
      std::string q_weight_key = layer_prefix + "self_attn.q_proj";
      std::string k_weight_key = layer_prefix + "self_attn.k_proj";
      std::string v_weight_key = layer_prefix + "self_attn.v_proj";

      std::string q_qweight_key = q_weight_key + ".qweight";
      std::string q_scales_key = q_weight_key + ".scales";
      std::string q_qzeros_key = q_weight_key + ".qzeros";

      std::string k_qweight_key = k_weight_key + ".qweight";
      std::string k_scales_key = k_weight_key + ".scales";
      std::string k_qzeros_key = k_weight_key + ".qzeros";

      std::string v_qweight_key = v_weight_key + ".qweight";
      std::string v_scales_key = v_weight_key + ".scales";
      std::string v_qzeros_key = v_weight_key + ".qzeros";

      auto q_qweight_it = qweight_params_.find(q_qweight_key);
      auto k_qweight_it = qweight_params_.find(k_qweight_key);
      auto v_qweight_it = qweight_params_.find(v_qweight_key);

      if (q_qweight_it != qweight_params_.end() &&
          scales_params_.find(q_scales_key) != scales_params_.end() &&
          qzeros_params_.find(q_qzeros_key) != qzeros_params_.end()) {
        // 使用量化矩阵乘法
        cuda_OP::matmul_quantized(
            hidden_states, qweight_params_.at(q_qweight_key),
            scales_params_.at(q_scales_key), qzeros_params_.at(q_qzeros_key),
            group_size_, &q_buf, nullptr, q_bias);
      } else {
        // 回退到非量化版本
        auto &wq = params_.at(q_weight_key);
        cuda_OP::matmul(hidden_states, wq, &q_buf, nullptr, q_bias);
      }

      if (k_qweight_it != qweight_params_.end() &&
          scales_params_.find(k_scales_key) != scales_params_.end() &&
          qzeros_params_.find(k_qzeros_key) != qzeros_params_.end()) {
        // 使用量化矩阵乘法
        cuda_OP::matmul_quantized(
            hidden_states, qweight_params_.at(k_qweight_key),
            scales_params_.at(k_scales_key), qzeros_params_.at(k_qzeros_key),
            group_size_, &k_slice, nullptr, k_bias);
      } else {
        // 回退到非量化版本
        auto &wk = params_.at(k_weight_key);
        cuda_OP::matmul(hidden_states, wk, &k_slice, nullptr, k_bias);
      }

      if (v_qweight_it != qweight_params_.end() &&
          scales_params_.find(v_scales_key) != scales_params_.end() &&
          qzeros_params_.find(v_qzeros_key) != qzeros_params_.end()) {
        // 使用量化矩阵乘法
        cuda_OP::matmul_quantized(
            hidden_states, qweight_params_.at(v_qweight_key),
            scales_params_.at(v_scales_key), qzeros_params_.at(v_qzeros_key),
            group_size_, &v_slice, nullptr, v_bias);
      } else {
        // 回退到非量化版本
        auto &wv = params_.at(v_weight_key);
        cuda_OP::matmul(hidden_states, wv, &v_slice, nullptr, v_bias);
      }
    } else {
      // 非量化版本
      auto &wq = params_.at(layer_prefix + "self_attn.q_proj.weight");
      auto &wk = params_.at(layer_prefix + "self_attn.k_proj.weight");
      auto &wv = params_.at(layer_prefix + "self_attn.v_proj.weight");

      cuda_OP::matmul(hidden_states, wq, &q_buf, nullptr, q_bias);
      cuda_OP::matmul(hidden_states, wk, &k_slice, nullptr, k_bias);
      cuda_OP::matmul(hidden_states, wv, &v_slice, nullptr, v_bias);
    }

    // // 同步CUDA流并销毁
    // for (int j = 0; j < 3; j++) {
    //   cudaStreamSynchronize(streams[j]);
    //   cudaStreamDestroy(streams[j]);
    // }

    // 重塑张量，准备应用RoPE
    Tensor<T> q_buf_view = q_buf.view({seq_len, n_heads_, head_dim_});
    Tensor<T> k_buf_view = k_slice.view({seq_len, n_kv_heads_, head_dim_});
    Tensor<T> v_buf_view = v_slice.view({seq_len, n_kv_heads_, head_dim_});

    // for (size_t j = 0; j < seq_len; j++) {
    //   // 获取对应的 k 和 v slice

    //   Tensor<T>& k_slice = kv_cache->k_cache(i, offset + j);
    //   Tensor<T>& v_slice = kv_cache->v_cache(i, offset + j);
    //   // debugPrintTensor(k_slice, "k_slice");
    //   // debugPrintTensor(v_slice, "v_slice");
    //   // 将数据从 k_buf_view 和 v_buf_view 拷贝到对应 slice 的内存中
    //   cudaMemcpy(k_slice.data_ptr(), k_buf_view.data_ptr() + j * row_size,
    //              row_size * sizeof(T), cudaMemcpyDeviceToDevice);
    //   cudaMemcpy(v_slice.data_ptr(), v_buf_view.data_ptr() + j * row_size,
    //              row_size * sizeof(T), cudaMemcpyDeviceToDevice);
    //   // debugPrintTensor(k_slice, "k_slice");
    //   // debugPrintTensor(v_slice, "v_slice");
    // }

    // 应用旋转位置编码 (RoPE)
    cuda_OP::rope(&q_buf_view, offset, rope_theta_, nullptr);
    cuda_OP::rope(&k_buf_view, offset, rope_theta_, nullptr);

    // 更新KV缓存
    // size_t row_size = n_kv_heads_ * head_dim_;

    // for (size_t j = 0; j < seq_len; j++) {
    //   // 获取对应的 k 和 v slice
    //   Tensor<T>& k_slice = kv_cache->k_cache(i, offset + j);
    //   Tensor<T>& v_slice = kv_cache->v_cache(i, offset + j);

    //   // 异步拷贝：使用 cudaMemcpyAsync 替换同步版本
    //   cudaError_t err1 = cudaMemcpyAsync(
    //       k_slice.data_ptr(), k_buf_view.data_ptr() + j * row_size,
    //       row_size * sizeof(T), cudaMemcpyDeviceToDevice,
    //       compute_streams_[3]);
    //   cudaError_t err2 = cudaMemcpyAsync(
    //       v_slice.data_ptr(), v_buf_view.data_ptr() + j * row_size,
    //       row_size * sizeof(T), cudaMemcpyDeviceToDevice,
    //       compute_streams_[4]);
    // }

    // 准备计算自注意力
    Tensor<T> Q_3d = q_buf_view;
    Tensor<T> total_K, total_V;
    size_t total_seq_len = seq_len;

    // 如果有缓存，拼接当前和缓存的K,V
    if (offset != 0) {
      size_t cached_len = offset;
      total_seq_len = cached_len + seq_len;
      // total_K =
      //     Tensor<T>({total_seq_len, n_kv_heads_, head_dim_}, Device::CUDA);
      // // total_KX =
      // //     Tensor<T>({total_seq_len, n_kv_heads_, head_dim_},
      // Device::CUDA); total_V =
      //     Tensor<T>({total_seq_len, n_kv_heads_, head_dim_}, Device::CUDA);

      // // 拷贝缓存的K,V
      // for (size_t pos = 0; pos < cached_len; pos++) {
      //   Tensor<T>& cached_k = kv_cache->k_cache(i, pos);
      //   Tensor<T>& cached_v = kv_cache->v_cache(i, pos);
      //   cudaMemcpy(total_K.data_ptr() + pos * row_size, cached_k.data_ptr(),
      //              row_size * sizeof(T), cudaMemcpyDeviceToDevice);
      //   cudaMemcpy(total_V.data_ptr() + pos * row_size, cached_v.data_ptr(),
      //              row_size * sizeof(T), cudaMemcpyDeviceToDevice);
      //   // debugPrintTensor(cached_k, "cached_k");
      //   // debugPrintTensor(cached_v, "cached_v");
      // }
      // // 拷贝当前的K,V

      // cudaMemcpy(total_K.data_ptr() + cached_len * row_size,
      //            k_buf_view.data_ptr(), seq_len * row_size * sizeof(T),
      //            cudaMemcpyDeviceToDevice);
      // cudaMemcpy(total_V.data_ptr() + cached_len * row_size,
      //            v_buf_view.data_ptr(), seq_len * row_size * sizeof(T),
      //            cudaMemcpyDeviceToDevice);

      auto [total_K1, total_V1] = kv_cache->get_contiguous_tensor(i);
      // total_KX.view({total_seq_len, n_kv_heads_, head_dim_});
      // debugPrintTensor(total_KX, "total_KX");
      // debugPrintTensor(total_K, "total_K");

      total_K = total_K1.view({total_seq_len, n_kv_heads_, head_dim_});
      total_V = total_V1.view({total_seq_len, n_kv_heads_, head_dim_});

    } else {
      total_K = k_buf_view;
      total_V = v_buf_view;
    }

    Tensor<T> att_heads({n_heads_, head_dim_}, Device::CUDA);

    // Tensor<T> att_heads_1({n_heads_ * (head_dim_ + 2)}, Device::CUDA);

    // Tensor<T> att_heads_2({n_heads_ * (head_dim_ + 2)}, Device::CUDA);

    // Tensor<T> att_heads_3({n_heads_ * (head_dim_ + 2)}, Device::CUDA);

    // // Tensor<T> att_heads_4({n_heads_ * (head_dim_ + 2)}, Device::CUDA);

    // // Tensor<T> att_heads({n_heads_, head_dim_}, Device::CUDA);
    // size_t seq_divide = total_seq_len / 3;
    // int ratio = n_heads_ / n_kv_heads_;
    // // for (int j = 0; j < 3; ++j) {
    // //   cudaStreamSynchronize(compute_streams_[j]);
    // // }

    // cuda_OP::flash_attention(
    //     Q_3d, total_K.slice({0, 0, 0}, {seq_divide, n_kv_heads_, head_dim_}),
    //     total_K.slice({seq_divide, 0, 0},
    //                   {2 * seq_divide, n_kv_heads_, head_dim_}),
    //     total_K.slice({2 * seq_divide, 0, 0},
    //                   {total_seq_len, n_kv_heads_, head_dim_}),
    //     total_V.slice({0, 0, 0}, {seq_divide, n_kv_heads_, head_dim_}),
    //     total_V.slice({seq_divide, 0, 0},
    //                   {2 * seq_divide, n_kv_heads_, head_dim_}),
    //     total_V.slice({2 * seq_divide, 0, 0},
    //                   {total_seq_len, n_kv_heads_, head_dim_}),
    //     att_heads_1, att_heads_2, att_heads_3);

    // cuda_OP::gather_fa(att_heads_1, att_heads_2, att_heads_3, att_heads,
    //                    nullptr);

    // 使用新的动态分支数量的flash attention包装函数
    cuda_OP::dynamic_flash_attention_wrapper(Q_3d, total_K, total_V, att_heads,
                                             n_kv_heads_, nullptr);
    // Tensor<T> att_scores({n_heads_, total_seq_len}, Device::CUDA);
    // // // cuda_OP::compute_attention_scores(Q_3d, total_K, n_heads_,
    // head_dim_,
    // // //                                   att_scores, n_kv_heads_);

    // cuda_OP::launch_gemmv_scores(total_K.data_ptr(), Q_3d.data_ptr(),
    //                       att_scores.data_ptr(), n_heads_, ratio,
    //                       total_seq_len, head_dim_, head_dim_,
    //                       total_seq_len);

    // cuda_OP::softmax(&att_scores, &att_scores, /*dim=*/1, false, offset);

    // cuda_OP::compute_att_output(att_scores, total_V, n_heads_, head_dim_,
    //                             att_heads, n_kv_heads_);

    Tensor<T> att_heads_reshaped = att_heads.view({1, n_heads_ * head_dim_});
    Tensor<T> att_proj({seq_len, hidden_size_}, Device::CUDA);

    if (quant_type_ == 1) {
      // AWQ量化版本
      std::string o_weight_key = layer_prefix + "self_attn.o_proj";
      std::string o_qweight_key = o_weight_key + ".qweight";
      std::string o_scales_key = o_weight_key + ".scales";
      std::string o_qzeros_key = o_weight_key + ".qzeros";

      auto o_qweight_it = qweight_params_.find(o_qweight_key);

      if (o_qweight_it != qweight_params_.end() &&
          scales_params_.find(o_scales_key) != scales_params_.end() &&
          qzeros_params_.find(o_qzeros_key) != qzeros_params_.end()) {
        // 使用量化矩阵乘法
        cuda_OP::matmul_quantized(
            att_heads_reshaped, qweight_params_.at(o_qweight_key),
            scales_params_.at(o_scales_key), qzeros_params_.at(o_qzeros_key),
            group_size_, &att_proj, nullptr, o_bias);
      } else {
        // 回退到非量化版本
        auto &wo = params_.at(o_weight_key);
        cuda_OP::matmul(att_heads_reshaped, wo, &att_proj, nullptr, o_bias);
      }
    } else {
      // 非量化版本
      auto &wo = params_.at(layer_prefix + "self_attn.o_proj.weight");
      cuda_OP::matmul(att_heads_reshaped, wo, &att_proj, nullptr, o_bias);
    }

    // 残差连接
    // cudaDeviceSynchronize();
    auto &ffn_norm_weight =
        params_.at(layer_prefix + "post_attention_layernorm.weight");

    // cuda_OP::add(&residual, &residual, &att_proj);

    // cuda_OP::rms_norm(&hidden_states, &residual, &ffn_norm_weight,
    //                   rms_norm_eps_);

    cuda_OP::add_rms(&hidden_states, &residual, &att_proj, &ffn_norm_weight,
                     rms_norm_eps_);
    // debugPrintTensor(hidden_states, "hidden_states-after");

    // 获取偏置项（如果存在）
    const Tensor<T> *gate_bias = nullptr;
    const Tensor<T> *up_bias = nullptr;
    const Tensor<T> *down_bias = nullptr;

    // 预先分配输出张量
    size_t intermediate_size = intermediate_size_;
    if (quant_type_ == 0) {
      // 非量化版本，从权重获取维度
      auto &gate_weight = params_.at(layer_prefix + "mlp.gate_proj.weight");
      intermediate_size = gate_weight.sizes()[1];
    }

    Tensor<T> gate_buf({seq_len, intermediate_size}, Device::CUDA);
    Tensor<T> up_buf({seq_len, intermediate_size}, Device::CUDA);

    if (quant_type_ == 1) {
      // AWQ量化版本
      std::string gate_weight_key = layer_prefix + "mlp.gate_proj";
      std::string up_weight_key = layer_prefix + "mlp.up_proj";

      std::string gate_qweight_key = gate_weight_key + ".qweight";
      std::string gate_scales_key = gate_weight_key + ".scales";
      std::string gate_qzeros_key = gate_weight_key + ".qzeros";

      std::string up_qweight_key = up_weight_key + ".qweight";
      std::string up_scales_key = up_weight_key + ".scales";
      std::string up_qzeros_key = up_weight_key + ".qzeros";

      auto gate_qweight_it = qweight_params_.find(gate_qweight_key);
      auto up_qweight_it = qweight_params_.find(up_qweight_key);

      if (gate_qweight_it != qweight_params_.end() &&
          scales_params_.find(gate_scales_key) != scales_params_.end() &&
          qzeros_params_.find(gate_qzeros_key) != qzeros_params_.end()) {
        // 使用量化矩阵乘法
        cuda_OP::matmul_quantized(hidden_states,
                                  qweight_params_.at(gate_qweight_key),
                                  scales_params_.at(gate_scales_key),
                                  qzeros_params_.at(gate_qzeros_key),
                                  group_size_, &gate_buf, nullptr, gate_bias);
      } else {
        // 回退到非量化版本
        std::cout << "Some errors." << std::endl;
        auto &gate_weight = params_.at(gate_weight_key);
        cuda_OP::matmul(hidden_states, gate_weight, &gate_buf, nullptr,
                        gate_bias);
      }
      if (up_qweight_it != qweight_params_.end() &&
          scales_params_.find(up_scales_key) != scales_params_.end() &&
          qzeros_params_.find(up_qzeros_key) != qzeros_params_.end()) {
        // 使用量化矩阵乘法
        cuda_OP::matmul_quantized(
            hidden_states, qweight_params_.at(up_qweight_key),
            scales_params_.at(up_scales_key), qzeros_params_.at(up_qzeros_key),
            group_size_, &up_buf, nullptr, up_bias);
      } else {
        throw std::runtime_error("Some errors.");
      }
    } else {
      // 非量化版本
      auto &gate_weight = params_.at(layer_prefix + "mlp.gate_proj.weight");
      auto &up_weight = params_.at(layer_prefix + "mlp.up_proj.weight");

      cuda_OP::matmul(hidden_states, gate_weight, &gate_buf, nullptr,
                      gate_bias);
      cuda_OP::matmul(hidden_states, up_weight, &up_buf, nullptr, up_bias);
    }

    cuda_OP::silu(&gate_buf, &gate_buf);              // SiLU激活
    cuda_OP::multiply(&gate_buf, &gate_buf, &up_buf); // 逐元素相乘

    // 投影回原始维度
    Tensor<T> ffn_out({seq_len, hidden_size_}, Device::CUDA);

    if (quant_type_ == 1) {
      // AWQ量化版本
      std::string down_weight_key = layer_prefix + "mlp.down_proj";
      std::string down_qweight_key = down_weight_key + ".qweight";
      std::string down_scales_key = down_weight_key + ".scales";
      std::string down_qzeros_key = down_weight_key + ".qzeros";

      auto down_qweight_it = qweight_params_.find(down_qweight_key);

      if (down_qweight_it != qweight_params_.end() &&
          scales_params_.find(down_scales_key) != scales_params_.end() &&
          qzeros_params_.find(down_qzeros_key) != qzeros_params_.end()) {
        // 使用量化矩阵乘法
        cuda_OP::matmul_quantized(gate_buf,
                                  qweight_params_.at(down_qweight_key),
                                  scales_params_.at(down_scales_key),
                                  qzeros_params_.at(down_qzeros_key),
                                  group_size_, &ffn_out, nullptr, down_bias);
      } else {
        throw std::runtime_error("Some errors.");
      }
    } else {
      // 非量化版本
      auto &down_weight = params_.at(layer_prefix + "mlp.down_proj.weight");
      cuda_OP::matmul(gate_buf, down_weight, &ffn_out, nullptr, down_bias);
    }

    // 残差连接
    // cuda_OP::add(&residual, &residual, &ffn_out);
    if (i == n_layers_ - 1) {
      // 最后一层的残差连接
      cuda_OP::add(&residual, &residual, &ffn_out);
    } else {
      std::string lx = "layers." + std::to_string(i + 1) + ".";
      auto &attention_norm_weight =

          params_.at(lx + "input_layernorm.weight");
      cuda_OP::add_rms(&hidden_states, &residual, &ffn_out,
                       &attention_norm_weight, rms_norm_eps_);
    }
  }

  // 最终的LayerNorm (RMSNorm)
  auto &norm_weight = params_.at("norm.weight");
  Tensor<T> final_h({seq_len, hidden_size_}, Device::CUDA);
  cuda_OP::rms_norm(&final_h, &residual, &norm_weight, rms_norm_eps_);

  // LM head投影到词汇表大小
  auto &lm_head_weight = params_.at("lm_head");
  const Tensor<T> *lm_head_bias = nullptr;
  // try {
  //   lm_head_bias = &params_.at("lm_head_bias");
  //   std::cout << "Found lm_head_bias" << std::endl;
  // } catch (const std::out_of_range&) {
  // }

  Tensor<T> logits({seq_len, vocab_size_}, Device::CUDA);
  cuda_OP::matmul(final_h, lm_head_weight, &logits, nullptr, lm_head_bias);

  // 返回最后一个token的logits

  return logits;
}

// -------------------------------
// prefill_cuda: Qwen2 模型的序列预填充 CUDA 实现
// -------------------------------
template <typename T>
Tensor<T> QwenModel<T>::prefill_cuda(const Tensor<uint32_t> *input,
                                     KVCache<T> *kv_cache) {
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

  // 创建residual和hidden_states张量，在prefill阶段自动使用prefill buffer
  Tensor<T> residual({seq_len, hidden_size_}, Device::CUDA);
  Tensor<T> hidden_states({seq_len, hidden_size_}, Device::CUDA);

  // Token嵌入 (从embedding_table中获取token嵌入)
  cuda_OP::gather(&residual, input, &params_.at("token_embeddings.weight"));
  cudaStreamSynchronize(compute_streams_[3]);
  cudaStreamSynchronize(compute_streams_[4]);
  // 主循环：遍历所有Transformer层
  for (size_t i = 0; i < n_layers_; i++) {
    std::string layer_prefix = "layers." + std::to_string(i) + ".";

    auto &attention_norm_weight =
        params_.at(layer_prefix + "input_layernorm.weight");
    cuda_OP::rms_norm(&hidden_states, &residual, &attention_norm_weight,
                      rms_norm_eps_);

    // 获取偏置项（如果存在）
    const Tensor<T> *q_bias = nullptr;
    const Tensor<T> *k_bias = nullptr;
    const Tensor<T> *v_bias = nullptr;
    const Tensor<T> *o_bias = nullptr;

    try {
      q_bias = &params_.at(layer_prefix + "self_attn.q_proj.bias");
    } catch (const std::out_of_range &) {
    }

    try {
      k_bias = &params_.at(layer_prefix + "self_attn.k_proj.bias");
    } catch (const std::out_of_range &) {
    }

    try {
      v_bias = &params_.at(layer_prefix + "self_attn.v_proj.bias");
    } catch (const std::out_of_range &) {
    }

    Tensor<T> q_buf({seq_len, n_heads_ * head_dim_}, Device::CUDA);
    Tensor<T> k_buf({seq_len, n_kv_heads_ * head_dim_}, Device::CUDA);
    Tensor<T> v_buf({seq_len, n_kv_heads_ * head_dim_}, Device::CUDA);

    if (quant_type_ == 1) {
      // AWQ量化版本
      std::string q_weight_key = layer_prefix + "self_attn.q_proj";
      std::string k_weight_key = layer_prefix + "self_attn.k_proj";
      std::string v_weight_key = layer_prefix + "self_attn.v_proj";

      std::string q_qweight_key = q_weight_key + ".qweight";
      std::string q_scales_key = q_weight_key + ".scales";
      std::string q_qzeros_key = q_weight_key + ".qzeros";

      std::string k_qweight_key = k_weight_key + ".qweight";
      std::string k_scales_key = k_weight_key + ".scales";
      std::string k_qzeros_key = k_weight_key + ".qzeros";

      std::string v_qweight_key = v_weight_key + ".qweight";
      std::string v_scales_key = v_weight_key + ".scales";
      std::string v_qzeros_key = v_weight_key + ".qzeros";

      // auto print_shape =
      //     [](const std::string &name,
      //        const auto &sizes) { /* ... (printing code) ... */
      //                             std::cout << name << " shape: [";
      //                             if (sizes.empty()) {
      //                               std::cout << "<empty>";
      //                             } else {
      //                               for (size_t i = 0; i < sizes.size(); ++i) {
      //                                 std::cout
      //                                     << sizes[i]
      //                                     << (i == sizes.size() - 1 ? ""
      //                                                               : ", ");
      //                               }
      //                             }
      //                             std::cout << "]";
      //     };
      auto q_qweight_it = qweight_params_.find(q_qweight_key);
      auto k_qweight_it = qweight_params_.find(k_qweight_key);
      auto v_qweight_it = qweight_params_.find(v_qweight_key);
      // print_shape(k_qweight_it->first,
      // qweight_params_.at(k_qweight_key).sizes());

      if (q_qweight_it != qweight_params_.end() &&
          scales_params_.find(q_scales_key) != scales_params_.end() &&
          qzeros_params_.find(q_qzeros_key) != qzeros_params_.end()) {
        // 使用量化矩阵乘法

        // 使用量化矩阵乘法，确保传递正确的流
        cuda_OP::matmul_quantized(
            hidden_states, qweight_params_.at(q_qweight_key),
            scales_params_.at(q_scales_key), qzeros_params_.at(q_qzeros_key),
            group_size_, &q_buf, compute_streams_[0], q_bias);

        // 检查CUDA错误
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error in q_proj matmul_quantized: " << cudaGetErrorString(err) << std::endl;
        }
      } else {
        // 回退到非量化版本
        throw std::runtime_error("Some errors.");
      }

      if (k_qweight_it != qweight_params_.end() &&
          scales_params_.find(k_scales_key) != scales_params_.end() &&
          qzeros_params_.find(k_qzeros_key) != qzeros_params_.end()) {
        // 使用量化矩阵乘法
        // 使用量化矩阵乘法，确保传递正确的流
        cuda_OP::matmul_quantized(
            hidden_states, qweight_params_.at(k_qweight_key),
            scales_params_.at(k_scales_key), qzeros_params_.at(k_qzeros_key),
            group_size_, &k_buf, compute_streams_[1], k_bias);

        // 检查CUDA错误
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error in k_proj matmul_quantized: " << cudaGetErrorString(err) << std::endl;
        }
      } else {

        throw std::runtime_error("Some errors.");
      }

      if (v_qweight_it != qweight_params_.end() &&
          scales_params_.find(v_scales_key) != scales_params_.end() &&
          qzeros_params_.find(v_qzeros_key) != qzeros_params_.end()) {
        // 使用量化矩阵乘法
        // 使用量化矩阵乘法，确保传递正确的流
        cuda_OP::matmul_quantized(
            hidden_states, qweight_params_.at(v_qweight_key),
            scales_params_.at(v_scales_key), qzeros_params_.at(v_qzeros_key),
            group_size_, &v_buf, compute_streams_[2], v_bias);

        // 检查CUDA错误
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error in v_proj matmul_quantized: " << cudaGetErrorString(err) << std::endl;
        }
      } else {

        throw std::runtime_error("Some errors.");
      }
    } else {
      // 非量化版本
      auto &wq = params_.at(layer_prefix + "self_attn.q_proj.weight");
      auto &wk = params_.at(layer_prefix + "self_attn.k_proj.weight");
      auto &wv = params_.at(layer_prefix + "self_attn.v_proj.weight");

      cuda_OP::matmul(hidden_states, wq, &q_buf, compute_streams_[0], q_bias);
      cuda_OP::matmul(hidden_states, wk, &k_buf, compute_streams_[1], k_bias);
      cuda_OP::matmul(hidden_states, wv, &v_buf, compute_streams_[2], v_bias);
    }

    // 同步并销毁流
    // for (int j = 0; j < 3; j++) {
    //   cudaStreamSynchronize(streams[j]);
    //   cudaStreamDestroy(streams[j]);
    // }

    // 重塑张量，准备应用RoPE
    const size_t head_size = hidden_size_ / n_heads_;
    const size_t kv_head_size = hidden_size_ / n_kv_heads_;
    const size_t row_size = n_kv_heads_ * head_dim_;

    Tensor<T> q_buf_view = q_buf.view({seq_len, n_heads_, head_dim_});
    Tensor<T> k_buf_view = k_buf.view({seq_len, n_kv_heads_, head_dim_});
    Tensor<T> v_buf_view = v_buf.view({seq_len, n_kv_heads_, head_dim_});

    // 应用旋转位置编码 (RoPE)
    cuda_OP::rope(&q_buf_view, offset, rope_theta_, compute_streams_[0]);
    cuda_OP::rope(&k_buf_view, offset, rope_theta_, compute_streams_[1]);

    for (int j = 0; j < 3; ++j) {
      cudaStreamSynchronize(compute_streams_[j]);
    }

    // 将K,V存储到缓存中
    // for (size_t j = 0; j < seq_len; j++) {
    //   // 获取对应的 k 和 v slice
    //   Tensor<T>& k_slice = kv_cache->k_cache(i, offset + j);
    //   Tensor<T>& v_slice = kv_cache->v_cache(i, offset + j);

    //   // 将数据从 k_buf_view 和 v_buf_view 拷贝到对应 slice 的内存中
    //   cudaMemcpy(k_slice.data_ptr(), k_buf_view.data_ptr() + j * row_size,
    //              row_size * sizeof(T), cudaMemcpyDeviceToDevice);
    //   cudaMemcpy(v_slice.data_ptr(), v_buf_view.data_ptr() + j * row_size,
    //              row_size * sizeof(T), cudaMemcpyDeviceToDevice);
    // }

    for (size_t j = 0; j < seq_len; j++) {
      // 获取对应的 k 和 v slice
      Tensor<T> &k_slice = kv_cache->k_cache(i, offset + j);
      Tensor<T> &v_slice = kv_cache->v_cache(i, offset + j);

      // 异步拷贝：使用 cudaMemcpyAsync 替换同步版本
      cudaError_t err1 = cudaMemcpyAsync(
          k_slice.data_ptr(), k_buf_view.data_ptr() + j * row_size,
          row_size * sizeof(T), cudaMemcpyDeviceToDevice, compute_streams_[3]);
      cudaError_t err2 = cudaMemcpyAsync(
          v_slice.data_ptr(), v_buf_view.data_ptr() + j * row_size,
          row_size * sizeof(T), cudaMemcpyDeviceToDevice, compute_streams_[4]);
    }

    // 重新格式化Q用于注意力计算
    Tensor<T> Q_3d = q_buf_view;

    // 准备K和V张量用于注意力计算
    Tensor<T> total_K, total_V;
    size_t total_seq_len = 0;
    if (offset > 0) {
      size_t cached_len = offset;
      total_seq_len = cached_len + seq_len;

      // total_K =
      //     Tensor<T>({total_seq_len, n_kv_heads_, head_dim_}, Device::CUDA);
      // total_V =
      //     Tensor<T>({total_seq_len, n_kv_heads_, head_dim_}, Device::CUDA);

      // // 拷贝缓存的K,V
      // for (size_t pos = 0; pos < cached_len; pos++) {
      //   const auto& cached_k = kv_cache->k_cache(i, pos);
      //   const auto& cached_v = kv_cache->v_cache(i, pos);

      //   cudaMemcpy(total_K.data_ptr() + pos * n_kv_heads_ * head_dim_,
      //              cached_k.data_ptr(), n_kv_heads_ * head_dim_ * sizeof(T),
      //              cudaMemcpyDeviceToDevice);

      //   cudaMemcpy(total_V.data_ptr() + pos * n_kv_heads_ * head_dim_,
      //              cached_v.data_ptr(), n_kv_heads_ * head_dim_ * sizeof(T),
      //              cudaMemcpyDeviceToDevice);
      // }
      // // 拷贝当前的K, V

      // cudaMemcpy(total_K.data_ptr() + cached_len * n_kv_heads_ * head_dim_,
      //            k_buf_view.data_ptr(),
      //            seq_len * n_kv_heads_ * head_dim_ * sizeof(T),
      //            cudaMemcpyDeviceToDevice);

      // cudaMemcpy(total_V.data_ptr() + cached_len * n_kv_heads_ * head_dim_,
      //            v_buf_view.data_ptr(),
      //            seq_len * n_kv_heads_ * head_dim_ * sizeof(T),
      //            cudaMemcpyDeviceToDevice);
      auto [total_K1, total_V1] = kv_cache->get_contiguous_tensor(i);
      // total_KX.view({total_seq_len, n_kv_heads_, head_dim_});
      // debugPrintTensor(total_KX, "total_KX");
      // debugPrintTensor(total_K, "total_K");

      total_K = total_K1.view({total_seq_len, n_kv_heads_, head_dim_});
      // debugPrintTensor(total_K, "total_K");
      total_V = total_V1.view({total_seq_len, n_kv_heads_, head_dim_});
    } else {
      total_K = k_buf_view;
      total_V = v_buf_view;
      total_seq_len = seq_len;
    }

    // 计算注意力分数
    Tensor<T> att_scores({seq_len, n_heads_, total_seq_len}, Device::CUDA);
    cuda_OP::compute_attention_scores_prefill(Q_3d, total_K, att_scores,
                                              head_dim_);

    // Softmax处理注意力分数（prefill版本需要设置mask=true）
    cuda_OP::softmax(&att_scores, &att_scores, /*dim=*/2, true, offset);

    // 计算注意力输出（prefill版本）
    Tensor<T> att_heads({seq_len, n_heads_, head_dim_}, Device::CUDA);
    cuda_OP::compute_att_output_prefill(att_scores, total_V, att_heads,
                                        n_heads_, head_dim_, total_seq_len,
                                        n_kv_heads_);

    // 将注意力输出投影回原始维度
    Tensor<T> att_proj({seq_len, hidden_size_}, Device::CUDA);

    if (quant_type_ == 1) {
      // AWQ量化版本
      std::string o_weight_key = layer_prefix + "self_attn.o_proj";
      std::string o_qweight_key = o_weight_key + ".qweight";
      std::string o_scales_key = o_weight_key + ".scales";
      std::string o_qzeros_key = o_weight_key + ".qzeros";

      auto o_qweight_it = qweight_params_.find(o_qweight_key);

      if (o_qweight_it != qweight_params_.end() &&
          scales_params_.find(o_scales_key) != scales_params_.end() &&
          qzeros_params_.find(o_qzeros_key) != qzeros_params_.end()) {
        // 使用量化矩阵乘法
        cuda_OP::matmul_quantized(
            att_heads.view({seq_len, n_heads_ * head_dim_}),
            qweight_params_.at(o_qweight_key), scales_params_.at(o_scales_key),
            qzeros_params_.at(o_qzeros_key), group_size_, &att_proj, nullptr,
            o_bias);
      } else {
        // 回退到非量化版本
        auto &wo = params_.at(o_weight_key);
        cuda_OP::matmul(att_heads.view({seq_len, n_heads_ * head_dim_}), wo,
                        &att_proj, nullptr, o_bias);
      }
    } else {
      // 非量化版本
      auto &wo = params_.at(layer_prefix + "self_attn.o_proj.weight");
      cuda_OP::matmul(att_heads.view({seq_len, n_heads_ * head_dim_}), wo,
                      &att_proj, nullptr, o_bias);
    }

    // 残差连接

    cuda_OP::add(&residual, &residual, &att_proj);
    // 3. Post Attention LayerNorm (RMSNorm)
    auto &ffn_norm_weight =
        params_.at(layer_prefix + "post_attention_layernorm.weight");
    cuda_OP::rms_norm(&hidden_states, &residual, &ffn_norm_weight,
                      rms_norm_eps_);

    // 4. MLP (Feed Forward Network)
    // 获取偏置项（如果存在）
    const Tensor<T> *gate_bias = nullptr;
    const Tensor<T> *up_bias = nullptr;
    const Tensor<T> *down_bias = nullptr;

    // 预先分配输出张量
    size_t ffn_hidden_size = intermediate_size_;
    if (quant_type_ == 0) {
      // 非量化版本，从权重获取维度
      auto &gate_weight = params_.at(layer_prefix + "mlp.gate_proj.weight");
      ffn_hidden_size = gate_weight.sizes()[1];
    }

    Tensor<T> gate_buf({seq_len, ffn_hidden_size}, Device::CUDA);
    Tensor<T> up_buf({seq_len, ffn_hidden_size}, Device::CUDA);

    if (quant_type_ == 1) {
      // AWQ量化版本
      std::string gate_weight_key = layer_prefix + "mlp.gate_proj";
      std::string up_weight_key = layer_prefix + "mlp.up_proj";

      std::string gate_qweight_key = gate_weight_key + ".qweight";
      std::string gate_scales_key = gate_weight_key + ".scales";
      std::string gate_qzeros_key = gate_weight_key + ".qzeros";

      std::string up_qweight_key = up_weight_key + ".qweight";
      std::string up_scales_key = up_weight_key + ".scales";
      std::string up_qzeros_key = up_weight_key + ".qzeros";

      auto gate_qweight_it = qweight_params_.find(gate_qweight_key);
      auto up_qweight_it = qweight_params_.find(up_qweight_key);

      if (gate_qweight_it != qweight_params_.end() &&
          scales_params_.find(gate_scales_key) != scales_params_.end() &&
          qzeros_params_.find(gate_qzeros_key) != qzeros_params_.end()) {
        // 使用量化矩阵乘法
        cuda_OP::matmul_quantized(hidden_states,
                                  qweight_params_.at(gate_qweight_key),
                                  scales_params_.at(gate_scales_key),
                                  qzeros_params_.at(gate_qzeros_key),
                                  group_size_, &gate_buf, nullptr, gate_bias);
      } else {
        // 回退到非量化版本
        auto &gate_weight = params_.at(gate_weight_key);
        cuda_OP::matmul(hidden_states, gate_weight, &gate_buf, nullptr,
                        gate_bias);
      }

      if (up_qweight_it != qweight_params_.end() &&
          scales_params_.find(up_scales_key) != scales_params_.end() &&
          qzeros_params_.find(up_qzeros_key) != qzeros_params_.end()) {
        // 使用量化矩阵乘法
        cuda_OP::matmul_quantized(
            hidden_states, qweight_params_.at(up_qweight_key),
            scales_params_.at(up_scales_key), qzeros_params_.at(up_qzeros_key),
            group_size_, &up_buf, nullptr, up_bias);
      } else {
        // 回退到非量化版本
        auto &up_weight = params_.at(up_weight_key);
        cuda_OP::matmul(hidden_states, up_weight, &up_buf, nullptr, up_bias);
      }
    } else {
      // 非量化版本
      auto &gate_weight = params_.at(layer_prefix + "mlp.gate_proj.weight");
      auto &up_weight = params_.at(layer_prefix + "mlp.up_proj.weight");

      cuda_OP::matmul(hidden_states, gate_weight, &gate_buf, nullptr,
                      gate_bias);
      cuda_OP::matmul(hidden_states, up_weight, &up_buf, nullptr, up_bias);
    }

    cuda_OP::silu(&gate_buf, &gate_buf);              // SiLU激活
    cuda_OP::multiply(&gate_buf, &gate_buf, &up_buf); // 逐元素相乘

    // 投影回原始维度
    Tensor<T> ffn_out({seq_len, hidden_size_}, Device::CUDA);

    if (quant_type_ == 1) {
      // AWQ量化版本
      std::string down_weight_key = layer_prefix + "mlp.down_proj";
      std::string down_qweight_key = down_weight_key + ".qweight";
      std::string down_scales_key = down_weight_key + ".scales";
      std::string down_qzeros_key = down_weight_key + ".qzeros";

      auto down_qweight_it = qweight_params_.find(down_qweight_key);

      if (down_qweight_it != qweight_params_.end() &&
          scales_params_.find(down_scales_key) != scales_params_.end() &&
          qzeros_params_.find(down_qzeros_key) != qzeros_params_.end()) {
        // 使用量化矩阵乘法
        cuda_OP::matmul_quantized(gate_buf,
                                  qweight_params_.at(down_qweight_key),
                                  scales_params_.at(down_scales_key),
                                  qzeros_params_.at(down_qzeros_key),
                                  group_size_, &ffn_out, nullptr, down_bias);
      } else {
        // 回退到非量化版本
        auto &down_weight = params_.at(down_weight_key);
        cuda_OP::matmul(gate_buf, down_weight, &ffn_out, nullptr, down_bias);
      }
    } else {
      // 非量化版本
      auto &down_weight = params_.at(layer_prefix + "mlp.down_proj.weight");
      cuda_OP::matmul(gate_buf, down_weight, &ffn_out, nullptr, down_bias);
    }

    // 残差连接
    cuda_OP::add(&residual, &residual, &ffn_out);
  }

  // 最终的LayerNorm (RMSNorm)
  auto &norm_weight = params_.at("norm.weight");
  Tensor<T> final_h({seq_len, hidden_size_}, Device::CUDA);
  cuda_OP::rms_norm(&final_h, &residual, &norm_weight, rms_norm_eps_);

  // LM head投影到词汇表大小
  auto &lm_head_weight = params_.at("lm_head");

  const Tensor<T> *lm_head_bias = nullptr;
  // try {
  //   lm_head_bias = &params_.at("lm_head_bias");
  //   std::cout << "Found lm_head_bias" << std::endl;
  // } catch (const std::out_of_range&) {
  // }

  Tensor<T> logits({seq_len, vocab_size_}, Device::CUDA);
  cuda_OP::matmul(final_h, lm_head_weight, &logits, nullptr, lm_head_bias);

  return logits;
}

// -------------------------------
// cuda()：将所有参数移到 CUDA，并设置设备
// -------------------------------
template <typename T> QwenModel<T> &QwenModel<T>::cuda() {
  for (auto &kv : params_) {
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
template <typename T> QwenModel<T> &QwenModel<T>::cpu() {
  throw std::runtime_error("QwenModel only supports CUDA execution.");
  return *this;
}

// -------------------------------
// generate: Token 生成接口，目前作为 stub
// -------------------------------
template <typename T>
std::vector<uint32_t>
QwenModel<T>::generate(const std::vector<uint32_t> &input_ids,
                       size_t max_length, float temperature, float top_p,
                       size_t top_k) {
  // TODO: 实现 Qwen 模型的 token 生成逻辑
  throw std::runtime_error("Token generation not implemented for QwenModel");
  return std::vector<uint32_t>();
}

// -------------------------------
// 辅助函数：将 FP32 权重转换为 __nv_bfloat16 权重
// -------------------------------
std::unordered_map<std::string, Tensor<__nv_bfloat16>> convert_weights_to_bf16(
    const std::unordered_map<std::string, Tensor<float>> &float_weights) {
  std::unordered_map<std::string, Tensor<__nv_bfloat16>> bf16_weights;
  for (const auto &kv : float_weights) {
    const std::string &key = kv.first;
    const Tensor<float> &tensor = kv.second;
    std::vector<__nv_bfloat16> bf16_data;
    bf16_data.reserve(tensor.numel());
    const float *data_ptr = tensor.data_ptr();
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
