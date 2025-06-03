// qwen.cpp
#include "qwen.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

#include "cudaOP.cuh"

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
    return true;  // Empty tensors are equal
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
    return false;  // Treat copy error as inequality
  }

  // 4. 逐元素比较
  bool mismatch_found = false;
  size_t first_mismatch_idx = 0;
  T val1_at_mismatch = T();  // Default constructor
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
      } else {  // For float or other types where == is reasonable initially
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
  size_t offset = 1023;  // 从开始处打印
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

  // 确保CUDA设备可用
  int device_count;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    throw std::runtime_error("No CUDA devices available: " +
                             std::string(cudaGetErrorString(err)));
  }

  // 设置CUDA设备
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to set CUDA device: " +
                             std::string(cudaGetErrorString(err)));
  }

  // 初始化算子接口
  operators_ = std::make_unique<op::UnifiedOperators<T>>(Device::CUDA);

  for (int i = 0; i < 5; ++i) {
    err = cudaStreamCreate(&compute_streams_[i]);
    if (err != cudaSuccess) {
      // 处理错误，可能需要清理已创建的流
      throw std::runtime_error("Failed to create CUDA stream in constructor: " +
                               std::string(cudaGetErrorString(err)));
    }
  }
  for (int i = 0; i < 3; ++i) {
    // 使用 cudaEventDisableTiming
    // 可以获得微小的性能提升，因为我们只关心完成状态，不测量时间
    cudaEventCreateWithFlags(&fa_done_events_[i], cudaEventDisableTiming);
  }

  // 初始化 CUDA 图相关成员
  cuda_graph_ = nullptr;
  graph_exec_ = nullptr;
  graph_stream_ = nullptr;
  graph_initialized_ = false;
  d_rope_offset_ = nullptr;

  // 初始化异步预准备优化相关成员
  next_execution_prepared_ = false;
  prep_stream_ = nullptr;
  prep_done_event_ = nullptr;
  last_kv_cache_size_ = 0;
  kv_params_cached_ = false;

  // 创建图执行专用流
  err = cudaStreamCreate(&graph_stream_);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to create graph stream in constructor: " +
                             std::string(cudaGetErrorString(err)));
  }

  // 创建预准备专用流
  err = cudaStreamCreate(&prep_stream_);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to create prep stream in constructor: " +
                             std::string(cudaGetErrorString(err)));
  }

  // 创建预准备完成事件
  err = cudaEventCreateWithFlags(&prep_done_event_, cudaEventDisableTiming);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to create prep event in constructor: " +
                             std::string(cudaGetErrorString(err)));
  }
}

// 带量化参数的构造函数
template <typename T>
QwenModel<T>::QwenModel(
    const std::unordered_map<std::string, Tensor<T>> &params,
    const std::unordered_map<std::string, Tensor<int32_t>> &qweight_params,
    const std::unordered_map<std::string, Tensor<T>> &scales_params,
    const std::unordered_map<std::string, Tensor<int32_t>> &qzeros_params,
    const std::unordered_map<std::string, int> &config)
    : params_(params),
      qweight_params_(qweight_params),
      scales_params_(scales_params),
      qzeros_params_(qzeros_params) {
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
      Tensor<T> tensor_copy = kv.second;
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

  // 确保CUDA设备可用
  int device_count;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    throw std::runtime_error("No CUDA devices available: " +
                             std::string(cudaGetErrorString(err)));
  }

  // 设置CUDA设备
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to set CUDA device: " +
                             std::string(cudaGetErrorString(err)));
  }

  // 初始化算子接口
  operators_ = std::make_unique<op::UnifiedOperators<T>>(Device::CUDA);

  for (int i = 0; i < 5; ++i) {
    err = cudaStreamCreate(&compute_streams_[i]);
    if (err != cudaSuccess) {
      // 处理错误，可能需要清理已创建的流
      throw std::runtime_error("Failed to create CUDA stream in constructor: " +
                               std::string(cudaGetErrorString(err)));
    }
  }
  for (int i = 0; i < 3; ++i) {
    // 使用 cudaEventDisableTiming
    // 可以获得微小的性能提升，因为我们只关心完成状态，不测量时间
    cudaEventCreateWithFlags(&fa_done_events_[i], cudaEventDisableTiming);
  }

  // 初始化 CUDA 图相关成员
  cuda_graph_ = nullptr;
  graph_exec_ = nullptr;
  graph_stream_ = nullptr;
  graph_initialized_ = false;
  d_rope_offset_ = nullptr;

  // 初始化异步预准备优化相关成员
  next_execution_prepared_ = false;
  prep_stream_ = nullptr;
  prep_done_event_ = nullptr;
  last_kv_cache_size_ = 0;

  // 创建图执行专用流
  err = cudaStreamCreate(&graph_stream_);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to create graph stream in constructor: " +
                             std::string(cudaGetErrorString(err)));
  }

  // 创建预准备专用流
  err = cudaStreamCreate(&prep_stream_);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to create prep stream in constructor: " +
                             std::string(cudaGetErrorString(err)));
  }

  // 创建预准备完成事件
  err = cudaEventCreateWithFlags(&prep_done_event_, cudaEventDisableTiming);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to create prep event in constructor: " +
                             std::string(cudaGetErrorString(err)));
  }
}

template <typename T>
QwenModel<T>::~QwenModel() {
  // 清理 CUDA 图相关资源
  cleanup_graph_fixed_memory();

  if (graph_exec_) {
    cudaGraphExecDestroy(graph_exec_);
  }
  if (cuda_graph_) {
    cudaGraphDestroy(cuda_graph_);
  }
  if (graph_stream_) {
    cudaStreamSynchronize(graph_stream_);
    cudaStreamDestroy(graph_stream_);
  }

  // 清理异步预准备相关资源
  if (prep_stream_) {
    cudaStreamSynchronize(prep_stream_);
    cudaStreamDestroy(prep_stream_);
  }
  if (prep_done_event_) {
    cudaEventDestroy(prep_done_event_);
  }

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
template <typename T>
bool QwenModel<T>::verify_params() const {
  // 禁用
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
  std::cout << "  Quantization: "
            << (quant_type_ == 0 ? "None"
                                 : (quant_type_ == 1 ? "AWQ" : "Unknown"))
            << std::endl;
  std::cout << "  CUDA Graph: " << (use_cuda_graph_ ? "Enabled" : "Disabled")
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
                                     KVCache<T> *kv_cache,
                                     const std::string &save_prefix) {
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

  // 创建residual和hidden_states张量(没有batch维度)，使用标签固定内存
  Tensor<T> residual({seq_len, hidden_size_}, Device::CUDA, false, "residual");
  Tensor<T> hidden_states({seq_len, hidden_size_}, Device::CUDA, false,
                          "hidden_states");

  //   // 保存输入token（调试用）
  //   std::cout << "forward_cuda: save_prefix = '" << save_prefix << "'"
  //             << std::endl;
  //   if (!save_prefix.empty()) {
  //     std::cout << "保存输入token到: " << save_prefix +
  //     "_cuda_input_token.bin"
  //               << std::endl;
  //     save_uint32_tensor_to_binary(*input, save_prefix +
  //     "_cuda_input_token.bin");

  //     // 调试：打印输入token值和embedding权重信息
  //     std::vector<uint32_t> host_input(input->numel());
  //     cudaMemcpy(host_input.data(), input->data_ptr(),
  //                input->numel() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  //     std::cout << "CUDA推理输入token: " << host_input[0] << std::endl;

  //     // 打印embedding权重的地址和前几个值
  //     auto &embedding_weight = params_.at("token_embeddings.weight");
  //     std::cout << "CUDA推理embedding权重地址: " <<
  //     embedding_weight.data_ptr()
  //               << std::endl;
  //     std::cout << "CUDA推理embedding权重形状: [" <<
  //     embedding_weight.sizes()[0]
  //               << ", " << embedding_weight.sizes()[1] << "]" << std::endl;

  //     // 保存当前token对应的embedding向量（用于对比）
  //     uint32_t token_id = host_input[0];
  //     if (token_id < embedding_weight.sizes()[0]) {
  //       // 创建一个临时张量来保存这个token的embedding
  //       Tensor<T> token_embedding(
  //           {1, static_cast<size_t>(embedding_weight.sizes()[1])},
  //           Device::CUDA, false);
  //       size_t offset = token_id * embedding_weight.sizes()[1];
  //       cudaMemcpy(token_embedding.data_ptr(),
  //                  static_cast<const T *>(embedding_weight.data_ptr()) +
  //                  offset, embedding_weight.sizes()[1] * sizeof(T),
  //                  cudaMemcpyDeviceToDevice);
  //       save_tensor_to_binary(token_embedding, save_prefix + "_cuda_token_" +
  //                                                  std::to_string(token_id) +
  //                                                  "_embedding.bin");
  //       std::cout << "保存了token " << token_id << " 的embedding向量"
  //                 << std::endl;
  //     }
  //   } else {
  //     std::cout << "save_prefix为空，跳过保存输入token" << std::endl;
  //   }

  // Token嵌入 (从embedding_table中获取token嵌入)
  cuda_OP::gather(&residual, input, &params_.at("token_embeddings.weight"));

  // 保存初始嵌入
  if (!save_prefix.empty()) {
    save_tensor_to_binary(residual, save_prefix + "_cuda_embedding.bin");
  }

  // cudaStreamSynchronize(compute_streams_[3]);
  // cudaStreamSynchronize(compute_streams_[4]);
  // 主循环：遍历所有Transformer层
  std::string l = "layers." + std::to_string(0) + ".";
  auto &attention_norm_weight = params_.at(l + "input_layernorm.weight");
  // 使用新的算子抽象层
  operators_->rms_norm(&hidden_states, &residual, &attention_norm_weight,
                       rms_norm_eps_);

  if (!save_prefix.empty()) {
    save_tensor_to_binary(hidden_states,
                          save_prefix + "_cuda_layer_0_input_norm.bin");
  }
  // 旧算子
  // cuda_OP::rms_norm(&hidden_states, &residual, &attention_norm_weight,
  //                  rms_norm_eps_);
  for (size_t i = 0; i < n_layers_; i++) {
    std::string layer_prefix = "layers." + std::to_string(i) + ".";

    // 2. Self-Attention
    std::string q_tag = "q_buf_" + std::to_string(i);
    Tensor<T> q_buf({seq_len, n_heads_ * head_dim_}, Device::CUDA, false,
                    q_tag);
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

    auto q_weight = get_weight(layer_prefix + "self_attn.q_proj");
    auto k_weight = get_weight(layer_prefix + "self_attn.k_proj");
    auto v_weight = get_weight(layer_prefix + "self_attn.v_proj");

    operators_->matmul(&q_buf, &hidden_states, q_weight, q_bias);
    operators_->matmul(&k_slice, &hidden_states, k_weight, k_bias);
    operators_->matmul(&v_slice, &hidden_states, v_weight, v_bias);

    // // 保存QKV投影结果
    // if (!save_prefix.empty()) {
    //   save_tensor_to_binary(q_buf, save_prefix + "_cuda_layer_" +
    //                                    std::to_string(i) + "_q_proj.bin");
    //   save_tensor_to_binary(k_slice, save_prefix + "_cuda_layer_" +
    //                                      std::to_string(i) + "_k_proj.bin");
    //   save_tensor_to_binary(v_slice, save_prefix + "_cuda_layer_" +
    //                                      std::to_string(i) + "_v_proj.bin");
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
    // 使用新的算子抽象层（二重指针版本）
    size_t offset_q = offset;
    size_t offset_k = offset;
    operators_->rope(&q_buf_view, offset_q, rope_theta_, nullptr);
    operators_->rope(&k_buf_view, offset_k, rope_theta_, nullptr);

    // // 保存RoPE后的结果
    // if (!save_prefix.empty()) {
    //   save_tensor_to_binary(q_buf_view, save_prefix + "_cuda_layer_" +
    //                                         std::to_string(i) +
    //                                         "_q_rope.bin");
    //   save_tensor_to_binary(k_buf_view, save_prefix + "_cuda_layer_" +
    //                                         std::to_string(i) +
    //                                         "_k_rope.bin");
    // }

    // 旧算子
    // cuda_OP::rope(&q_buf_view, offset, rope_theta_, nullptr);
    // cuda_OP::rope(&k_buf_view, offset, rope_theta_, nullptr);

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

    std::string att_heads_tag = "att_heads_" + std::to_string(i);
    Tensor<T> att_heads({n_heads_, head_dim_}, Device::CUDA, false,
                        att_heads_tag);

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

    // 保存完整的KV
    // if (!save_prefix.empty()) {
    //   save_tensor_to_binary(total_K, save_prefix + "_cuda_layer_" +
    //                                      std::to_string(i) + "_total_k.bin");
    //   save_tensor_to_binary(total_V, save_prefix + "_cuda_layer_" +
    //                                      std::to_string(i) + "_total_v.bin");
    // }

    // 使用新的动态分支数量的flash attention包装函数
    cuda_OP::dynamic_flash_attention_wrapper(Q_3d, total_K, total_V, att_heads,
                                             n_kv_heads_, nullptr);

    // 保存attention输出
    // if (!save_prefix.empty()) {
    //   save_tensor_to_binary(
    //       att_heads,
    //       save_prefix + "_cuda_layer_" + std::to_string(i) +
    //       "_att_heads.bin");
    // }
    // Tensor<T> att_scores({n_heads_, total_seq_len}, Device::CUDA);
    // // // cuda_OP::compute_attention_scores(Q_3d, total_K, n_heads_,
    // head_dim_,
    // // //                                   att_scores, n_kv_heads_);

    // cuda_OP::launch_gemv_scores(total_K.data_ptr(), Q_3d.data_ptr(),
    //                       att_scores.data_ptr(), n_heads_, ratio,
    //                       total_seq_len, head_dim_, head_dim_,
    //                       total_seq_len);

    // cuda_OP::softmax(&att_scores, &att_scores, /*dim=*/1, false, offset);

    // cuda_OP::compute_att_output(att_scores, total_V, n_heads_, head_dim_,
    //                             att_heads, n_kv_heads_);

    Tensor<T> att_heads_reshaped = att_heads.view({1, n_heads_ * head_dim_});
    std::string att_proj_tag = "att_proj_" + std::to_string(i);
    Tensor<T> att_proj({seq_len, hidden_size_}, Device::CUDA, false,
                       att_proj_tag);

    auto o_weight = get_weight(layer_prefix + "self_attn.o_proj");

    // 执行矩阵乘法（统一接口）
    operators_->matmul(&att_proj, &att_heads_reshaped, o_weight, o_bias);

    // 保存attention投影
    if (!save_prefix.empty()) {
      save_tensor_to_binary(att_proj, save_prefix + "_cuda_layer_" +
                                          std::to_string(i) + "_att_proj.bin");
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

    // 保存post attention状态
    if (!save_prefix.empty()) {
      save_tensor_to_binary(residual, save_prefix + "_cuda_layer_" +
                                          std::to_string(i) +
                                          "_post_att_residual.bin");
      save_tensor_to_binary(hidden_states, save_prefix + "_cuda_layer_" +
                                               std::to_string(i) +
                                               "_post_att_norm.bin");
    }
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

    std::string gate_buf_tag = "gate_buf_" + std::to_string(i);
    std::string up_buf_tag = "up_buf_" + std::to_string(i);
    Tensor<T> gate_buf({seq_len, intermediate_size}, Device::CUDA, false,
                       gate_buf_tag);
    Tensor<T> up_buf({seq_len, intermediate_size}, Device::CUDA, false,
                     up_buf_tag);

    auto gate_weight = get_weight(layer_prefix + "mlp.gate_proj");
    auto up_weight = get_weight(layer_prefix + "mlp.up_proj");

    // 执行矩阵乘法
    operators_->matmul(&gate_buf, &hidden_states, gate_weight, gate_bias);
    operators_->matmul(&up_buf, &hidden_states, up_weight, up_bias);

    // // 保存MLP投影
    // if (!save_prefix.empty()) {
    //   save_tensor_to_binary(gate_buf, save_prefix + "_cuda_layer_" +
    //                                       std::to_string(i) +
    //                                       "_gate_proj.bin");
    //   save_tensor_to_binary(up_buf, save_prefix + "_cuda_layer_" +
    //                                     std::to_string(i) + "_up_proj.bin");
    // }

    // cuda_OP::silu(&gate_buf, &gate_buf);               // SiLU激活
    // cuda_OP::multiply(&gate_buf, &gate_buf, &up_buf);  // 逐元素相乘

    // 使用新的算子抽象层
    operators_->silu(&gate_buf, &gate_buf);               // SiLU激活
    operators_->multiply(&gate_buf, &gate_buf, &up_buf);  // 逐元素相乘

    // 保存激活后的结果
    if (!save_prefix.empty()) {
      save_tensor_to_binary(gate_buf, save_prefix + "_cuda_layer_" +
                                          std::to_string(i) + "_silu_mul.bin");
    }

    // 投影回原始维度
    std::string ffn_out_tag = "ffn_out_" + std::to_string(i);
    Tensor<T> ffn_out({seq_len, hidden_size_}, Device::CUDA, false,
                      ffn_out_tag);

    // 改为:
    // 获取权重（自动处理量化与非量化情况）
    auto down_weight = get_weight(layer_prefix + "mlp.down_proj");

    // 执行矩阵乘法（统一接口）
    operators_->matmul(&ffn_out, &gate_buf, down_weight, down_bias);

    // 保存down投影
    if (!save_prefix.empty()) {
      save_tensor_to_binary(ffn_out, save_prefix + "_cuda_layer_" +
                                         std::to_string(i) + "_down_proj.bin");
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

    // 保存层结束后的状态
    // if (!save_prefix.empty()) {
    //   save_tensor_to_binary(residual, save_prefix + "_cuda_layer_" +
    //                                       std::to_string(i) +
    //                                       "_final_residual.bin");
    //   if (i < n_layers_ - 1) {
    //     save_tensor_to_binary(hidden_states, save_prefix + "_cuda_layer_" +
    //                                              std::to_string(i) +
    //                                              "_final_hidden.bin");
    //   }
    // }
  }

  // 最终的LayerNorm (RMSNorm)
  auto &norm_weight = params_.at("norm.weight");
  Tensor<T> final_h({seq_len, hidden_size_}, Device::CUDA, false, "final_h");
  // 使用新的算子抽象层
  operators_->rms_norm(&final_h, &residual, &norm_weight, rms_norm_eps_);
  // 旧算子
  // cuda_OP::rms_norm(&final_h, &residual, &norm_weight, rms_norm_eps_);

  // 保存最终norm
  if (!save_prefix.empty()) {
    save_tensor_to_binary(final_h, save_prefix + "_cuda_final_norm.bin");
  }

  // LM head投影到词汇表大小
  auto lm_head_weight = get_weight("lm_head");

  const Tensor<T> *lm_head_bias = nullptr;
  // try {
  //   lm_head_bias = &params_.at("lm_head_bias");
  //   std::cout << "Found lm_head_bias" << std::endl;
  // } catch (const std::out_of_range&) {
  // }

  Tensor<T> logits({seq_len, vocab_size_}, Device::CUDA, false, "logits");
  operators_->matmul(&logits, &final_h, lm_head_weight, lm_head_bias);

  // 保存最终logits
  if (!save_prefix.empty()) {
    save_tensor_to_binary(logits, save_prefix + "_cuda_final_logits.bin");
  }

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
    // 使用新的算子抽象层
    operators_->rms_norm(&hidden_states, &residual, &attention_norm_weight,
                         rms_norm_eps_);
    // 旧算子
    // cuda_OP::rms_norm(&hidden_states, &residual, &attention_norm_weight,
    //                  rms_norm_eps_);

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

    auto q_weight = get_weight(layer_prefix + "self_attn.q_proj");
    auto k_weight = get_weight(layer_prefix + "self_attn.k_proj");
    auto v_weight = get_weight(layer_prefix + "self_attn.v_proj");

    operators_->matmul(&q_buf, &hidden_states, q_weight, q_bias,
                       compute_streams_[0]);
    operators_->matmul(&k_buf, &hidden_states, k_weight, k_bias,
                       compute_streams_[1]);
    operators_->matmul(&v_buf, &hidden_states, v_weight, v_bias,
                       compute_streams_[2]);

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
    // 使用新的算子抽象层（二重指针版本）
    size_t offset_q = offset;
    size_t offset_k = offset;
    operators_->rope(&q_buf_view, offset_q, rope_theta_, compute_streams_[0]);
    operators_->rope(&k_buf_view, offset_k, rope_theta_, compute_streams_[1]);

    // 旧算子
    // cuda_OP::rope(&q_buf_view, offset, rope_theta_, compute_streams_[0]);
    // cuda_OP::rope(&k_buf_view, offset, rope_theta_, compute_streams_[1]);

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

    // 获取权重（自动处理量化与非量化情况）
    auto o_weight = get_weight(layer_prefix + "self_attn.o_proj");

    // 执行矩阵乘法（统一接口）
    operators_->matmul(&att_proj,
                       &att_heads.view({seq_len, n_heads_ * head_dim_}),
                       o_weight, o_bias);

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

    Tensor<T> gate_buf({seq_len, ffn_hidden_size}, Device::CUDA);
    Tensor<T> up_buf({seq_len, ffn_hidden_size}, Device::CUDA);

    // 获取权重（自动处理量化与非量化情况）
    auto gate_weight = get_weight(layer_prefix + "mlp.gate_proj");
    auto up_weight = get_weight(layer_prefix + "mlp.up_proj");

    // 执行矩阵乘法（内部自动选择合适的实现）
    operators_->matmul(&gate_buf, &hidden_states, gate_weight, gate_bias);
    operators_->matmul(&up_buf, &hidden_states, up_weight, up_bias);

    cuda_OP::silu(&gate_buf, &gate_buf);               // SiLU激活
    cuda_OP::multiply(&gate_buf, &gate_buf, &up_buf);  // 逐元素相乘

    // 投影回原始维度
    Tensor<T> ffn_out({seq_len, hidden_size_}, Device::CUDA);

    // 获取down_proj权重（自动处理量化与非量化情况）
    auto down_weight = get_weight(layer_prefix + "mlp.down_proj");

    // 执行矩阵乘法（内部自动选择合适的实现）
    operators_->matmul(&ffn_out, &gate_buf, down_weight, down_bias);

    // 残差连接
    operators_->add(&residual, &residual, &ffn_out);
    // cuda_OP::add(&residual, &residual, &ffn_out);
  }

  // 最终的LayerNorm (RMSNorm)
  auto &norm_weight = params_.at("norm.weight");
  Tensor<T> final_h({seq_len, hidden_size_}, Device::CUDA);
  cuda_OP::rms_norm(&final_h, &residual, &norm_weight, rms_norm_eps_);

  // LM head投影到词汇表大小
  auto lm_head_weight = get_weight("lm_head");

  const Tensor<T> *lm_head_bias = nullptr;
  // try {
  //   lm_head_bias = &params_.at("lm_head_bias");
  //   std::cout << "Found lm_head_bias" << std::endl;
  // } catch (const std::out_of_range&) {
  // }

  Tensor<T> logits({seq_len, vocab_size_}, Device::CUDA);
  operators_->matmul(&logits, &final_h, lm_head_weight, lm_head_bias);

  return logits;
}

// -------------------------------
// cuda()：将所有参数移到 CUDA，并设置设备
// -------------------------------
template <typename T>
QwenModel<T> &QwenModel<T>::cuda() {
  for (auto &kv : params_) {
    if (kv.second.device() != Device::CUDA) {
      kv.second.cuda();
    }
  }
  device_ = Device::CUDA;

  // 更新算子接口
  if (operators_) {
    operators_->cuda();
  } else {
    operators_ = std::make_unique<op::UnifiedOperators<T>>(Device::CUDA);
  }

  // CUDA图初始化现在延迟到第一次forward调用时进行
  // 这样可以使用真实的KV cache

  return *this;
}

// -------------------------------
// cpu()：Qwen 模型仅支持 CUDA，故调用 cpu() 抛出异常
// -------------------------------
template <typename T>
QwenModel<T> &QwenModel<T>::cpu() {
  // 更新算子接口（虽然会抛出异常，但保持一致性）
  if (operators_) {
    operators_->cpu();
  }

  throw std::runtime_error("QwenModel only supports CUDA execution.");
  return *this;
}

// -------------------------------
// generate: Token 生成接口，目前作为 stub
// -------------------------------
template <typename T>
std::vector<uint32_t> QwenModel<T>::generate(
    const std::vector<uint32_t> &input_ids, size_t max_length,
    float temperature, float top_p, size_t top_k) {
  // TODO: 实现 Qwen 模型的 token 生成逻辑
  throw std::runtime_error("Token generation not implemented for QwenModel");
  return std::vector<uint32_t>();
}

// -------------------------------
// forward_for_graph: Qwen2 模型的简化前向传播，用于图优化开发
// -------------------------------
template <typename T>
Tensor<T> QwenModel<T>::forward_for_graph(const Tensor<uint32_t> *input,
                                          KVCache<T> *kv_cache,
                                          cudaStream_t stream) {
  // 基本输入检查
  if (input->device() != Device::CUDA) {
    throw std::runtime_error("Input tensor must be on CUDA device");
  }

  const size_t seq_len = 1;  // 单token推理

  // 创建基本张量 - 使用tagged memory
  Tensor<T> residual({seq_len, hidden_size_}, Device::CUDA, false,
                     "graph_residual");
  Tensor<T> hidden_states({seq_len, hidden_size_}, Device::CUDA, false,
                          "graph_hidden_states");

  // 1. Token嵌入 - 关键修复：使用固定的输入张量，确保地址一致性
  // 在图执行前，graph_input_tensor_的内容已经通过cudaMemcpyAsync更新到正确的token值

  // 调试：打印gather操作的参数地址
  std::cout << "gather操作参数地址:" << std::endl;
  std::cout << "  residual地址: " << residual.data_ptr() << std::endl;
  std::cout << "  graph_input_tensor_地址: " << graph_input_tensor_.data_ptr()
            << std::endl;
  std::cout << "  embedding权重地址: "
            << params_.at("token_embeddings.weight").data_ptr() << std::endl;

  cuda_OP::gather(&residual, &graph_input_tensor_,
                  &params_.at("token_embeddings.weight"), stream);
  std::string l = "layers." + std::to_string(0) + ".";
  auto &attention_norm_weight = params_.at(l + "input_layernorm.weight");
  // 使用新的算子抽象层
  operators_->rms_norm(&hidden_states, &residual, &attention_norm_weight,
                       rms_norm_eps_, stream);

  // 2. Transformer层循环
  for (size_t i = 0; i < n_layers_; i++) {
    std::string layer_prefix = "layers." + std::to_string(i) + ".";

    // // 2.1 输入层归一化
    // auto &attention_norm_weight = params_.at(layer_prefix +
    // "input_layernorm.weight"); operators_->rms_norm(&hidden_states,
    // &residual, &attention_norm_weight, rms_norm_eps_, stream);

    // 2.2 自注意力机制
    // 关键修复：直接使用固定缓冲区作为matmul的输出，确保K和V写入正确位置
    std::string q_tag = "graph_q_buf_" + std::to_string(i);
    Tensor<T> q_buf({seq_len, n_heads_ * head_dim_}, Device::CUDA, false,
                    q_tag);

    // 关键修复：直接使用固定的K/V缓冲区作为matmul输出
    Tensor<T> &k_buf = fixed_k_buffers_[i];
    Tensor<T> &v_buf = fixed_v_buffers_[i];

    // 获取偏置项（如果存在）- 仿照普通forward的bias读取
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

    auto q_weight = get_weight(layer_prefix + "self_attn.q_proj");
    auto k_weight = get_weight(layer_prefix + "self_attn.k_proj");
    auto v_weight = get_weight(layer_prefix + "self_attn.v_proj");

    // matmul写入固定内存，包含bias
    operators_->matmul(&q_buf, &hidden_states, q_weight, q_bias, stream);
    operators_->matmul(&k_buf, &hidden_states, k_weight, k_bias, stream);
    operators_->matmul(&v_buf, &hidden_states, v_weight, v_bias, stream);

    // 关键修复：确保K和V的计算结果写入到固定缓冲区
    // 这样图执行后可以正确复制到KV cache
    // 注意：这里不能直接写入KV cache，因为地址会变化，但必须写入固定缓冲区

    // 重塑为3D张量
    Tensor<T> q_3d = q_buf.view({seq_len, n_heads_, head_dim_});
    Tensor<T> k_3d = k_buf.view({seq_len, n_kv_heads_, head_dim_});
    Tensor<T> v_3d = v_buf.view({seq_len, n_kv_heads_, head_dim_});

    // 问题1解决方案：RoPE使用固定设备内存的offset
    // 注意：offset的更新必须在图执行之前完成，这里只是读取固定内存中的值
    cuda_OP::rope_with_device_offset(&q_3d, d_rope_offset_, rope_theta_,
                                     stream);
    cuda_OP::rope_with_device_offset(&k_3d, d_rope_offset_, rope_theta_,
                                     stream);

    // 关键修复：先写入KV cache，然后读取完整数据，与forward_cuda保持一致
    // KV缓存必须存在，否则是错误的
    if (!kv_cache) {
      throw std::runtime_error("forward_for_graph requires KV cache");
    }

    // 方案：使用图节点更新来处理动态KV地址
    // 在图中直接写入KV cache，然后通过图节点更新来改变目标地址

    // 获取当前token在KV cache中的位置（图捕获时使用默认位置）
    size_t current_offset = kv_cache->size() - 1;

    // 将经过RoPE处理的K和V写入KV cache（这些memcpy节点将被更新）
    Tensor<T> &k_slice = kv_cache->k_cache(i, current_offset);
    Tensor<T> &v_slice = kv_cache->v_cache(i, current_offset);

    // 关键修复：写入经过RoPE处理的K和V数据
    // 注意：k_3d和v_3d已经经过RoPE处理，应该写入这些数据而不是原始的k_buf/v_buf
    cudaMemcpyAsync(k_slice.data_ptr(), k_3d.data_ptr(),
                    k_3d.numel() * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(v_slice.data_ptr(), v_3d.data_ptr(),
                    v_3d.numel() * sizeof(T), cudaMemcpyDeviceToDevice, stream);

    // 关键修复：在写入KV cache之后再读取完整数据，与forward_cuda保持一致
    // 获取包含当前token的完整KV cache数据
    auto [total_K_flat, total_V_flat] = kv_cache->get_contiguous_tensor(i);

    // 问题3解决方案：使用固定内存地址的flash attention
    std::string att_tag = "graph_att_heads_" + std::to_string(i);
    Tensor<T> att_heads({n_heads_, head_dim_}, Device::CUDA, false, att_tag);

    // 关键修复：KV cache返回的格式是[seq_len, n_kv_heads * dqkv]
    // 但Flash Attention期望的格式是[seq_len, n_kv_heads, dqkv]
    // 需要正确reshape
    size_t total_seq_len = kv_cache->size();

    // 关键修复：与普通推理保持一致，进行reshape
    // KV cache返回格式：[seq_len, head_dim] 其中 head_dim = n_kv_heads * dqkv
    // Flash Attention期望：[seq_len, n_kv_heads, dqkv]
    Tensor<T> total_K =
        total_K_flat.view({total_seq_len, n_kv_heads_, head_dim_});
    Tensor<T> total_V =
        total_V_flat.view({total_seq_len, n_kv_heads_, head_dim_});

    // 使用图优化的flash attention
    cuda_OP::flash_attention_graph_fixed(q_3d, total_K, total_V, d_output_ptrs_,
                                         d_segment_info_, n_kv_heads_, stream);

    // 使用图优化的gather_fa（固定内存版本）
    cuda_OP::gather_fa_graph_fixed(d_output_ptrs_, att_heads, d_segment_info_,
                                   stream);

    // 输出投影 - 使用tagged memory
    std::string att_proj_tag = "graph_att_proj_" + std::to_string(i);
    Tensor<T> att_proj({seq_len, hidden_size_}, Device::CUDA, false,
                       att_proj_tag);
    auto o_weight = get_weight(layer_prefix + "self_attn.o_proj");
    Tensor<T> att_heads_reshaped =
        att_heads.view({seq_len, n_heads_ * head_dim_});
    operators_->matmul(&att_proj, &att_heads_reshaped, o_weight, o_bias,
                       stream);

    // 2.3 后注意力层归一化和残差连接 - 完全仿照普通模式
    auto &ffn_norm_weight =
        params_.at(layer_prefix + "post_attention_layernorm.weight");
    cuda_OP::add_rms(&hidden_states, &residual, &att_proj, &ffn_norm_weight,
                     rms_norm_eps_, stream);

    // 2.4 MLP前馈网络 - 使用tagged memory
    std::string gate_tag = "graph_gate_buf_" + std::to_string(i);
    std::string up_tag = "graph_up_buf_" + std::to_string(i);
    std::string ffn_tag = "graph_ffn_out_" + std::to_string(i);
    Tensor<T> gate_buf({seq_len, intermediate_size_}, Device::CUDA, false,
                       gate_tag);
    Tensor<T> up_buf({seq_len, intermediate_size_}, Device::CUDA, false,
                     up_tag);

    auto gate_weight = get_weight(layer_prefix + "mlp.gate_proj");
    auto up_weight = get_weight(layer_prefix + "mlp.up_proj");

    operators_->matmul(&gate_buf, &hidden_states, gate_weight, nullptr, stream);
    operators_->matmul(&up_buf, &hidden_states, up_weight, nullptr, stream);

    // SiLU激活和逐元素相乘
    operators_->silu(&gate_buf, &gate_buf, stream);
    operators_->multiply(&gate_buf, &gate_buf, &up_buf, stream);

    // 下投影 - 使用tagged memory
    Tensor<T> ffn_out({seq_len, hidden_size_}, Device::CUDA, false, ffn_tag);
    auto down_weight = get_weight(layer_prefix + "mlp.down_proj");
    operators_->matmul(&ffn_out, &gate_buf, down_weight, nullptr, stream);

    // 残差连接和下一层的input_layernorm - 仿照普通模式的逻辑
    if (i == n_layers_ - 1) {
      // 最后一层只做残差连接
      operators_->add(&residual, &residual, &ffn_out, stream);
    } else {
      // 非最后一层：残差连接 + 下一层的input_layernorm
      std::string next_layer_prefix = "layers." + std::to_string(i + 1) + ".";
      auto &next_attention_norm_weight =
          params_.at(next_layer_prefix + "input_layernorm.weight");
      cuda_OP::add_rms(&hidden_states, &residual, &ffn_out,
                       &next_attention_norm_weight, rms_norm_eps_, stream);
    }
  }

  // 3. 最终层归一化 - 使用tagged memory
  auto &norm_weight = params_.at("norm.weight");
  Tensor<T> final_h({seq_len, hidden_size_}, Device::CUDA, false,
                    "graph_final_h");
  operators_->rms_norm(&final_h, &residual, &norm_weight, rms_norm_eps_,
                       stream);

  // 4. LM Head投影 - 使用固定tag确保地址一致
  auto lm_head_weight = get_weight("lm_head");
  Tensor<T> logits({seq_len, vocab_size_}, Device::CUDA, false,
                   "graph_fixed_logits");
  operators_->matmul(&logits, &final_h, lm_head_weight, nullptr, stream);

  return logits;
}

// -------------------------------
// CUDA图优化相关方法实现
// -------------------------------

template <typename T>
void QwenModel<T>::initialize_graph_fixed_memory() {
  // 问题1解决方案：分配RoPE offset的固定设备内存
  cudaError_t err = cudaMalloc(&d_rope_offset_, sizeof(size_t));
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "Failed to allocate device memory for RoPE offset: " +
        std::string(cudaGetErrorString(err)));
  }

  fixed_k_buffers_.clear();
  fixed_v_buffers_.clear();
  fixed_k_buffers_.reserve(n_layers_);
  fixed_v_buffers_.reserve(n_layers_);

  for (size_t i = 0; i < n_layers_; i++) {
    std::string k_tag = "graph_fixed_k_buf_" + std::to_string(i);
    std::string v_tag = "graph_fixed_v_buf_" + std::to_string(i);

    Tensor<T> k_buf({1, n_kv_heads_ * head_dim_}, Device::CUDA, false, k_tag);
    Tensor<T> v_buf({1, n_kv_heads_ * head_dim_}, Device::CUDA, false, v_tag);

    fixed_k_buffers_.push_back(std::move(k_buf));
    fixed_v_buffers_.push_back(std::move(v_buf));
  }

  // 初始化KV复制节点列表
  kv_copy_nodes_.clear();

  // 问题3解决方案：分配flash attention的固定内存和分段信息
  // 使用tagged memory分配分段信息内存：[total_seq_len, branch_count,
  // branch_lengths...]
  const int max_branches = 3;
  Tensor<int> segment_info_tensor({2 + max_branches}, Device::CUDA, false,
                                  "graph_segment_info");
  d_segment_info_ = segment_info_tensor.data_ptr();

  // 只需要输出指针数组，不再需要K/V切片指针
  Tensor<T *> output_ptrs_tensor({max_branches}, Device::CUDA, false,
                                 "graph_output_ptrs");
  d_output_ptrs_ = output_ptrs_tensor.data_ptr();

  // 为每层分配flash attention输出缓冲区
  fixed_fa_outputs_.clear();
  fixed_fa_outputs_.reserve(max_branches);

  for (int i = 0; i < max_branches; i++) {
    std::string output_tag = "graph_fa_output_" + std::to_string(i);
    Tensor<T> output({n_heads_, head_dim_ + 2}, Device::CUDA, false,
                     output_tag);  // +2 for m,l
    fixed_fa_outputs_.push_back(std::move(output));
  }

  std::cout << "Graph fixed memory initialized successfully!" << std::endl;
}

template <typename T>
void QwenModel<T>::cleanup_graph_fixed_memory() {
  // 清理RoPE offset设备内存（这个仍然使用cudaMalloc分配的）
  if (d_rope_offset_) {
    cudaFree(d_rope_offset_);
    d_rope_offset_ = nullptr;
  }

  // 清理KV复制节点列表
  kv_copy_nodes_.clear();

  // 其他内存都是tagged memory，会自动清理
  // 只需要将指针置空
  d_segment_info_ = nullptr;
  d_output_ptrs_ = nullptr;

  // 清理固定缓冲区（Tensor析构函数会自动处理CUDA内存）
  fixed_k_buffers_.clear();
  fixed_v_buffers_.clear();
  fixed_fa_outputs_.clear();
}

template <typename T>
void QwenModel<T>::update_rope_offset(size_t offset) {
  if (d_rope_offset_) {
    // 关键修复：使用异步内存复制，避免在图执行前产生同步
    cudaError_t err = cudaMemcpyAsync(d_rope_offset_, &offset, sizeof(size_t),
                                      cudaMemcpyHostToDevice, graph_stream_);
    if (err != cudaSuccess) {
      throw std::runtime_error("Failed to update RoPE offset: " +
                               std::string(cudaGetErrorString(err)));
    }
  }
}

template <typename T>
void QwenModel<T>::extract_updateable_nodes() {
  if (!cuda_graph_) return;

  // 获取图中的所有节点
  size_t num_nodes = 0;
  cudaGraphGetNodes(cuda_graph_, nullptr, &num_nodes);

  std::vector<cudaGraphNode_t> nodes(num_nodes);
  cudaGraphGetNodes(cuda_graph_, nodes.data(), &num_nodes);

  // 清空之前的记录
  kv_copy_nodes_.clear();

  // 遍历所有节点，找到memcpy节点
  for (size_t i = 0; i < num_nodes; i++) {
    cudaGraphNodeType node_type;
    cudaGraphNodeGetType(nodes[i], &node_type);

    if (node_type == cudaGraphNodeTypeMemcpy) {
      // 获取memcpy参数
      cudaMemcpy3DParms params;
      cudaGraphMemcpyNodeGetParams(nodes[i], &params);

      // 检查是否是我们的KV复制节点（通过源地址判断）
      // 关键修复：现在源地址是k_3d和v_3d，不再是fixed_k_buffers_
      // 我们需要检查源地址是否指向我们的K/V缓冲区
      for (size_t layer = 0; layer < n_layers_; layer++) {
        // 获取当前层的K和V缓冲区地址（这些是图中实际使用的源地址）
        Tensor<T> k_buf =
            fixed_k_buffers_[layer].view({1, n_kv_heads_, head_dim_});
        Tensor<T> v_buf =
            fixed_v_buffers_[layer].view({1, n_kv_heads_, head_dim_});

        if (params.srcPtr.ptr == k_buf.data_ptr() ||
            params.srcPtr.ptr == v_buf.data_ptr()) {
          kv_copy_nodes_.push_back(nodes[i]);
          std::cout << "找到Layer " << layer << " 的KV复制节点" << std::endl;
          break;
        }
      }
    }
  }

  std::cout << "总共找到 " << kv_copy_nodes_.size() << " 个KV复制节点"
            << std::endl;
}
template <typename T>
void QwenModel<T>::update_graph_kv_addresses(KVCache<T> *kv_cache,
                                             size_t offset) {
  if (!graph_initialized_ || !graph_exec_ || !kv_cache ||
      kv_copy_nodes_.empty()) {  // Added graph_exec_ check for safety
    // Optional: Log a warning or handle if called when not ready
    // std::cout << "Skipping KV address update: Graph not initialized or no
    // nodes to update." << std::endl;
    return;
  }

  // std::cout << "更新图节点地址，offset=" << offset << ", 节点数=" <<
  // kv_copy_nodes_.size() << std::endl;

  // 为每个KV复制节点更新目标地址
  size_t node_idx = 0;
  for (size_t layer = 0; layer < n_layers_ && node_idx < kv_copy_nodes_.size();
       layer++) {
    // 更新K复制节点
    if (node_idx < kv_copy_nodes_.size()) {
      cudaMemcpy3DParms k_params;
      // It's generally safer to get params from the original graph node if
      // that's what kv_copy_nodes_ stores. If kv_copy_nodes_ could somehow
      // store references to nodes within graph_exec_, then getting params from
      // graph_exec_ would be an option, but typically you identify based on
      // original graph.
      cudaError_t get_param_err =
          cudaGraphMemcpyNodeGetParams(kv_copy_nodes_[node_idx], &k_params);
      if (get_param_err != cudaSuccess) {
        std::cerr << "获取K节点参数失败 for layer " << layer << ": "
                  << cudaGetErrorString(get_param_err) << std::endl;
        node_idx++;  // Attempt to process next node even if this one fails to
                     // get params
        if (node_idx < kv_copy_nodes_.size() &&
            layer <
                n_layers_ - 1) {  // Check if there's a V node for this layer
          node_idx++;  // Skip corresponding V node as well if K failed for this
                       // layer
        }
        continue;
      }

      Tensor<T> &k_slice = kv_cache->k_cache(layer, offset);
      k_params.dstPtr.ptr =
          k_slice.data_ptr();  // Update the destination pointer

      // --- THIS IS THE KEY CORRECTION ---
      cudaError_t result = cudaGraphExecMemcpyNodeSetParams(
          graph_exec_,               // Target the executable graph
          kv_copy_nodes_[node_idx],  // The original node handle
          &k_params);                // The updated parameters
      if (result != cudaSuccess) {
        std::cerr << "更新K节点失败 for layer " << layer << " at offset "
                  << offset << " to address " << k_slice.data_ptr() << ": "
                  << cudaGetErrorString(result) << std::endl;
      }
      // else {
      //     std::cout << "成功更新Layer " << layer << " K节点地址: " <<
      //     k_slice.data_ptr() << std::endl;
      // }
      node_idx++;
    }

    // 更新V复制节点
    if (node_idx < kv_copy_nodes_.size()) {
      cudaMemcpy3DParms v_params;
      cudaError_t get_param_err =
          cudaGraphMemcpyNodeGetParams(kv_copy_nodes_[node_idx], &v_params);
      if (get_param_err != cudaSuccess) {
        std::cerr << "获取V节点参数失败 for layer " << layer << ": "
                  << cudaGetErrorString(get_param_err) << std::endl;
        node_idx++;
        continue;
      }

      Tensor<T> &v_slice = kv_cache->v_cache(layer, offset);
      v_params.dstPtr.ptr =
          v_slice.data_ptr();  // Update the destination pointer

      // --- THIS IS THE KEY CORRECTION ---
      cudaError_t result = cudaGraphExecMemcpyNodeSetParams(
          graph_exec_,               // Target the executable graph
          kv_copy_nodes_[node_idx],  // The original node handle
          &v_params);                // The updated parameters
      if (result != cudaSuccess) {
        std::cerr << "更新V节点失败 for layer " << layer << " at offset "
                  << offset << " to address " << v_slice.data_ptr() << ": "
                  << cudaGetErrorString(result) << std::endl;
      }
      // else {
      //     std::cout << "成功更新Layer " << layer << " V节点地址: " <<
      //     v_slice.data_ptr() << std::endl;
      // }
      node_idx++;
    }
  }
  // Safety check if not all nodes were processed as expected
  if (node_idx != kv_copy_nodes_.size()) {
    std::cerr << "[Warning] update_graph_kv_addresses: Processed " << node_idx
              << " nodes, but expected to process " << kv_copy_nodes_.size()
              << " based on n_layers." << std::endl;
  }
}

template <typename T>
void QwenModel<T>::update_graph_kv_addresses_async_for_next(
    KVCache<T> *kv_cache, size_t next_offset) {
  if (!graph_initialized_ || !graph_exec_ || !kv_cache ||
      kv_copy_nodes_.empty()) {
    return;
  }

  // 异步更新图节点参数为下一次执行准备
  // 这个函数在图执行期间被调用，为下一次执行预准备KV地址

  size_t node_idx = 0;
  for (size_t layer = 0; layer < n_layers_ && node_idx < kv_copy_nodes_.size();
       layer++) {
    // 异步更新K复制节点
    if (node_idx < kv_copy_nodes_.size()) {
      cudaMemcpy3DParms k_params;
      cudaError_t get_param_err =
          cudaGraphMemcpyNodeGetParams(kv_copy_nodes_[node_idx], &k_params);
      if (get_param_err == cudaSuccess) {
        // 获取下一次执行的K cache地址
        Tensor<T> &next_k_slice = kv_cache->k_cache(layer, next_offset);
        k_params.dstPtr.ptr = next_k_slice.data_ptr();

        // 异步更新图节点参数
        cudaError_t result = cudaGraphExecMemcpyNodeSetParams(
            graph_exec_, kv_copy_nodes_[node_idx], &k_params);

        // 注意：这里不打印错误，因为是异步操作
        // 错误会在下次图执行时体现
      }
      node_idx++;
    }

    // 异步更新V复制节点
    if (node_idx < kv_copy_nodes_.size()) {
      cudaMemcpy3DParms v_params;
      cudaError_t get_param_err =
          cudaGraphMemcpyNodeGetParams(kv_copy_nodes_[node_idx], &v_params);
      if (get_param_err == cudaSuccess) {
        // 获取下一次执行的V cache地址
        Tensor<T> &next_v_slice = kv_cache->v_cache(layer, next_offset);
        v_params.dstPtr.ptr = next_v_slice.data_ptr();

        // 异步更新图节点参数
        cudaError_t result = cudaGraphExecMemcpyNodeSetParams(
            graph_exec_, kv_copy_nodes_[node_idx], &v_params);
      }
      node_idx++;
    }
  }

  // 注意：这里不需要同步，因为是异步预准备
  // 下次执行时会检查是否已经预准备完成
}


template <typename T>
void QwenModel<T>::update_segment_info(size_t total_seq_len, int layer_idx) {
  if (!d_segment_info_) return;

  // 计算分段信息 - 强制使用3分支模式，类似flash_attention.cu
  const int max_branches = 3;
  const int FIXED_BRANCHES = 3;  // 强制使用3分支

  // 简化分段逻辑：始终使用3分支，让kernel内部处理分段细节
  int branches_needed = FIXED_BRANCHES;

  // 准备主机端数据
  std::vector<int> h_segment_info(2 + max_branches);
  h_segment_info[0] = static_cast<int>(total_seq_len);
  h_segment_info[1] = branches_needed;

  // 简化分段逻辑：让kernel内部处理分段，这里只传递基本信息
  // 不再预先计算分支长度，因为kernel会根据total_seq_len和FIXED_BRANCHES自动计算
  for (int i = 0; i < max_branches; i++) {
    h_segment_info[2 + i] = 0;  // 设为0，让kernel自己计算
  }

  // // 调试输出分段信息
  // std::cout << "更新分段信息: total_seq_len=" << total_seq_len
  //           << ", branches_needed=" << branches_needed << std::endl;
  // for (int i = 0; i < max_branches; i++) {
  //     std::cout << "  分支" << i << ": 长度=" << h_segment_info[2 + i] <<
  //     std::endl;
  // }

  // 关键修复：使用异步内存复制，避免在图执行前产生同步
  cudaMemcpyAsync(d_segment_info_, h_segment_info.data(),
                  (2 + max_branches) * sizeof(int), cudaMemcpyHostToDevice,
                  graph_stream_);

  // 更新输出指针数组（只需要输出指针，不再需要K/V切片指针）
  std::vector<T *> h_output_ptrs(max_branches);

  for (int i = 0; i < max_branches; i++) {
    h_output_ptrs[i] = fixed_fa_outputs_[i].data_ptr();
  }

  // 关键修复：使用异步内存复制，避免在图执行前产生同步
  cudaMemcpyAsync(d_output_ptrs_, h_output_ptrs.data(),
                  max_branches * sizeof(T *), cudaMemcpyHostToDevice,
                  graph_stream_);
}

template <typename T>
void QwenModel<T>::prepare_graph_execution(size_t rope_offset,
                                           size_t total_seq_len, int layer_idx,
                                           KVCache<T> *kv_cache) {
  // 在图执行前更新所有动态数据
  update_rope_offset(rope_offset);

  // 关键：更新图中KV复制节点的目标地址
  update_graph_kv_addresses(kv_cache, rope_offset);

  update_segment_info(total_seq_len, layer_idx);

  // 关键修复：确保所有异步更新完成后再执行图
  // 添加超时检查，避免无限等待
  cudaError_t sync_result = cudaStreamSynchronize(graph_stream_);
  if (sync_result != cudaSuccess) {
    throw std::runtime_error("Graph stream synchronization failed: " +
                             std::string(cudaGetErrorString(sync_result)));
  }
}

template <typename T>
void QwenModel<T>::prepare_next_graph_execution_async(size_t next_rope_offset,
                                                      size_t next_total_seq_len,
                                                      int layer_idx,
                                                      KVCache<T> *kv_cache) {
  // 异步准备下一次图执行的动态数据
  // 使用专用的prep_stream_来避免与主执行流冲突
  // 注意：KV地址更新已经在图启动后立即完成，这里只处理其他参数

  // 1. 异步更新RoPE offset
  if (d_rope_offset_) {
    cudaMemcpyAsync(d_rope_offset_, &next_rope_offset, sizeof(size_t),
                    cudaMemcpyHostToDevice, prep_stream_);
  }

  // 2. 异步更新flash attention分段信息
  if (d_segment_info_) {
    std::vector<int> h_segment_info = {
        static_cast<int>(next_total_seq_len),  // total_seq_len
        1,                                     // branch_count (单分支)
        static_cast<int>(next_total_seq_len)   // branch_length
    };

    cudaMemcpyAsync(d_segment_info_, h_segment_info.data(),
                    h_segment_info.size() * sizeof(int), cudaMemcpyHostToDevice,
                    prep_stream_);
  }

  // 4. 记录完成事件，但不同步
  cudaEventRecord(prep_done_event_, prep_stream_);

  // 5. 标记预准备完成
  next_execution_prepared_ = true;
}

template <typename T>
void QwenModel<T>::reset_async_preparation_state() {
  // 重置异步预准备状态，通常在新对话开始时调用
  next_execution_prepared_ = false;
  last_kv_cache_size_ = 0;  // 重置KV cache大小记录

  //   // 如果有正在进行的异步预准备，等待其完成
  //   if (prep_stream_) {
  //     cudaStreamSynchronize(prep_stream_);
  //   }

  std::cout << "异步预准备状态已重置" << std::endl;
}

template <typename T>
void QwenModel<T>::initialize_cuda_graph_with_kv_cache(KVCache<T> *kv_cache) {
  if (graph_initialized_) {
    return;  // 已经初始化过了
  }

  std::cout << "使用真实KV cache初始化CUDA图..." << std::endl;

  // 初始化图执行所需的固定内存
  initialize_graph_fixed_memory();

  // 创建固定的输入输出张量 - 关键修复：使用tagged memory确保地址一致性
  graph_input_tensor_ = Tensor<uint32_t>(
      {1}, Device::CUDA, false,
      "sample_output");  // 单token输入，与sample输出使用相同地址
  graph_output_tensor_ =
      Tensor<T>({1, vocab_size_}, Device::CUDA, false,
                "graph_fixed_logits");  // 输出也使用tagged memory

  uint32_t init_token = 9707;  // 使用一个有效的token ID
  cudaMemcpy(graph_input_tensor_.data_ptr(), &init_token, sizeof(uint32_t),
             cudaMemcpyHostToDevice);
  std::cout << "初始化graph_input_tensor_为token: " << init_token << std::endl;

  // 预热运行：先运行一次forward_for_graph来分配所有tagged memory
  std::cout << "使用真实KV cache进行预热运行..." << std::endl;
  try {
    // 关键修复：图捕获时使用与图执行时相同的offset计算逻辑
    // 图执行时：rope_offset = typed_cache->size() - 1（当前token位置）
    // 图执行时：total_seq_len = typed_cache->size()（包含当前token的完整数据）
    size_t default_offset = kv_cache->size() - 1;     // 与图执行时保持一致
    size_t default_total_seq_len = kv_cache->size();  // 与图执行时保持一致

    // 关键修复：更新所有动态数据
    // 注意：这里我们需要使用一个通用的分段信息模式，确保图捕获和执行时一致
    update_rope_offset(default_offset);
    update_segment_info(default_total_seq_len, 0);
    cudaDeviceSynchronize();  // 确保所有更新完成

    // 使用默认流进行预热运行
    Tensor<T> warmup_output =
        forward_for_graph(&graph_input_tensor_, kv_cache, nullptr);
    cudaDeviceSynchronize();  // 确保所有操作完成
    std::cout << "预热运行完成！" << std::endl;
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed during warmup run: " +
                             std::string(e.what()));
  }

  // 开始图捕获
  std::cout << "开始CUDA图捕获..." << std::endl;
  std::cout << "图捕获前graph_input_tensor_地址: "
            << graph_input_tensor_.data_ptr() << std::endl;

  init_token = 4;  // 使用一个有效的token ID
  cudaMemcpy(graph_input_tensor_.data_ptr(), &init_token, sizeof(uint32_t),
             cudaMemcpyHostToDevice);

  std::cout << "初始化graph_input_tensor_为token: " << init_token << std::endl;

  cudaError_t result =
      cudaStreamBeginCapture(graph_stream_, cudaStreamCaptureModeGlobal);
  if (result != cudaSuccess) {
    throw std::runtime_error("Failed to begin CUDA graph capture: " +
                             std::string(cudaGetErrorString(result)));
  }

  try {
    graph_output_tensor_ =
        Tensor<T>({1, vocab_size_}, Device::CUDA, false,
                  "graph_fixed_logits");  // 输出也使用tagged memory

    graph_output_tensor_ =
        forward_for_graph(&graph_input_tensor_, kv_cache, graph_stream_);
    // 结束图捕获
    result = cudaStreamEndCapture(graph_stream_, &cuda_graph_);
    if (result != cudaSuccess) {
      throw std::runtime_error("Failed to end CUDA graph capture: " +
                               std::string(cudaGetErrorString(result)));
    }

    extract_updateable_nodes();

    result =
        cudaGraphInstantiate(&graph_exec_, cuda_graph_, nullptr, nullptr, 0);
    if (result != cudaSuccess) {
      throw std::runtime_error("Failed to instantiate CUDA graph: " +
                               std::string(cudaGetErrorString(result)));
    }

    graph_initialized_ = true;
    std::cout << "图捕获后graph_input_tensor_地址: "
              << graph_input_tensor_.data_ptr() << std::endl;
    std::cout << "CUDA图初始化成功！找到 " << kv_copy_nodes_.size()
              << " 个可更新节点" << std::endl;

  } catch (const std::exception &e) {
    // 如果图捕获失败，尝试恢复流状态
    cudaStreamEndCapture(graph_stream_, &cuda_graph_);
    throw std::runtime_error("Failed to capture CUDA graph: " +
                             std::string(e.what()));
  }
}

// -------------------------------
// 逐层输出保存功能：保存中间张量到二进制文件用于分析
// -------------------------------
template <typename T>
void QwenModel<T>::save_tensor_to_binary(const Tensor<T> &tensor,
                                         const std::string &filename) {
  // 创建目录（如果不存在）
  //     std::string dir_path = filename.substr(0, filename.find_last_of('/'));
  //     if (!dir_path.empty()) {
  //         std::string mkdir_cmd = "mkdir -p " + dir_path;
  //         system(mkdir_cmd.c_str());
  //     }

  //     // 将GPU张量数据复制到CPU
  //     std::vector<T> host_data(tensor.numel());
  //     cudaMemcpy(host_data.data(), tensor.data_ptr(), tensor.numel() *
  //     sizeof(T), cudaMemcpyDeviceToHost); cudaDeviceSynchronize();

  //     // 保存到二进制文件
  //     std::ofstream file(filename, std::ios::binary);
  //     if (!file.is_open()) {
  //         throw std::runtime_error("Failed to open file for writing: " +
  //         filename);
  //     }

  //     // 写入张量形状信息
  //     size_t ndim = tensor.sizes().size();
  //     file.write(reinterpret_cast<const char*>(&ndim), sizeof(size_t));
  //     for (size_t dim : tensor.sizes()) {
  //         file.write(reinterpret_cast<const char*>(&dim), sizeof(size_t));
  //     }

  //     // 写入数据类型大小
  //     size_t dtype_size = sizeof(T);
  //     file.write(reinterpret_cast<const char*>(&dtype_size), sizeof(size_t));

  //     // 写入张量数据
  //     file.write(reinterpret_cast<const char*>(host_data.data()),
  //     tensor.numel() * sizeof(T)); file.close();

  //     std::cout << "已保存张量到: " << filename << " (形状: [";
  //     for (size_t i = 0; i < tensor.sizes().size(); i++) {
  //         std::cout << tensor.sizes()[i];
  //         if (i < tensor.sizes().size() - 1) std::cout << ", ";
  //     }
  //     std::cout << "], 元素数: " << tensor.numel() << ")" << std::endl;
}

// 保存uint32_t张量的专用函数
template <typename T>
void QwenModel<T>::save_uint32_tensor_to_binary(const Tensor<uint32_t> &tensor,
                                                const std::string &filename) {
  //   // 创建目录（如果不存在）
  //   std::string dir_path = filename.substr(0, filename.find_last_of('/'));
  //   if (!dir_path.empty()) {
  //     std::string mkdir_cmd = "mkdir -p " + dir_path;
  //     system(mkdir_cmd.c_str());
  //   }

  //   // 将GPU张量数据复制到CPU
  //   std::vector<uint32_t> host_data(tensor.numel());
  //   cudaMemcpy(host_data.data(), tensor.data_ptr(),
  //              tensor.numel() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  //   cudaDeviceSynchronize();

  //   // 保存到二进制文件
  //   std::ofstream file(filename, std::ios::binary);
  //   if (!file.is_open()) {
  //     throw std::runtime_error("Failed to open file for writing: " +
  //     filename);
  //   }

  //   // 写入张量形状信息
  //   size_t ndim = tensor.sizes().size();
  //   file.write(reinterpret_cast<const char *>(&ndim), sizeof(size_t));
  //   for (size_t dim : tensor.sizes()) {
  //     file.write(reinterpret_cast<const char *>(&dim), sizeof(size_t));
  //   }

  //   // 写入数据类型大小
  //   size_t dtype_size = sizeof(uint32_t);
  //   file.write(reinterpret_cast<const char *>(&dtype_size), sizeof(size_t));

  //   // 写入张量数据
  //   file.write(reinterpret_cast<const char *>(host_data.data()),
  //              tensor.numel() * sizeof(uint32_t));
  //   file.close();

  //   std::cout << "已保存uint32张量到: " << filename << " (形状: [";
  //   for (size_t i = 0; i < tensor.sizes().size(); i++) {
  //     std::cout << tensor.sizes()[i];
  //     if (i < tensor.sizes().size() - 1) std::cout << ", ";
  //   }
  //   std::cout << "], 元素数: " << tensor.numel() << ")" << std::endl;
}

// -------------------------------
// 图执行后保存重要张量：通过tagged memory访问图执行后的结果
// 注意：每层的tag都是唯一的，不会相互覆盖
// -------------------------------
template <typename T>
void QwenModel<T>::save_graph_tensors_after_execution(
    const std::string &save_prefix) {
  if (save_prefix.empty()) return;

  //   std::cout << "保存图执行后的重要张量..." << std::endl;

  // 调试：打印图推理的输入token和embedding权重信息
  std::vector<uint32_t> host_graph_input(graph_input_tensor_.numel());
  cudaMemcpy(host_graph_input.data(), graph_input_tensor_.data_ptr(),
             graph_input_tensor_.numel() * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);
  //   std::cout << "图推理输入token: " << host_graph_input[0] << std::endl;

  // 打印embedding权重的地址和形状
  auto &embedding_weight = params_.at("token_embeddings.weight");
  //   std::cout << "图推理embedding权重地址: " << embedding_weight.data_ptr()
  //             << std::endl;
  //   std::cout << "图推理embedding权重形状: [" << embedding_weight.sizes()[0]
  //             << ", " << embedding_weight.sizes()[1] << "]" << std::endl;

  // 保存当前token对应的embedding向量（用于对比）
  uint32_t token_id = host_graph_input[0];
  if (token_id < embedding_weight.sizes()[0]) {
    // 创建一个临时张量来保存这个token的embedding
    Tensor<T> token_embedding(
        {1, static_cast<size_t>(embedding_weight.sizes()[1])}, Device::CUDA,
        false);
    size_t offset = token_id * embedding_weight.sizes()[1];
    cudaMemcpy(token_embedding.data_ptr(),
               static_cast<const T *>(embedding_weight.data_ptr()) + offset,
               embedding_weight.sizes()[1] * sizeof(T),
               cudaMemcpyDeviceToDevice);
    save_tensor_to_binary(token_embedding, save_prefix + "_graph_token_" +
                                               std::to_string(token_id) +
                                               "_embedding.bin");
    std::cout << "保存了token " << token_id << " 的embedding向量" << std::endl;
  }

  // 保存输入token（从图的固定输入张量读取）
  save_uint32_tensor_to_binary(graph_input_tensor_,
                               save_prefix + "_graph_input_token.bin");

  // 保存嵌入结果（从graph_residual张量读取）
  Tensor<T> residual({1, hidden_size_}, Device::CUDA, false, "graph_residual");
  save_tensor_to_binary(residual, save_prefix + "_graph_embedding.bin");

  // 保存第一层的input_norm
  Tensor<T> hidden_states({1, hidden_size_}, Device::CUDA, false,
                          "graph_hidden_states");
  save_tensor_to_binary(hidden_states,
                        save_prefix + "_graph_layer_0_input_norm.bin");

  // 逐层保存中间结果
  for (size_t i = 0; i < n_layers_; i++) {
    // 保存QKV投影结果（使用与forward_for_graph中相同的tag）
    std::string q_tag = "graph_q_buf_" + std::to_string(i);
    Tensor<T> q_buf({1, n_heads_ * head_dim_}, Device::CUDA, false, q_tag);
    save_tensor_to_binary(q_buf, save_prefix + "_graph_layer_" +
                                     std::to_string(i) + "_q_proj.bin");

    // 保存固定K/V缓冲区的结果（这些是图中实际使用的缓冲区）
    // 注意：fixed_k_buffers_和fixed_v_buffers_使用的tag是"graph_fixed_k_buf_i"和"graph_fixed_v_buf_i"
    Tensor<T> &k_buf = fixed_k_buffers_[i];
    Tensor<T> &v_buf = fixed_v_buffers_[i];
    save_tensor_to_binary(k_buf, save_prefix + "_graph_layer_" +
                                     std::to_string(i) + "_k_proj.bin");
    save_tensor_to_binary(v_buf, save_prefix + "_graph_layer_" +
                                     std::to_string(i) + "_v_proj.bin");

    // 保存RoPE后的结果（从固定缓冲区的view读取，这些已经被RoPE修改）
    Tensor<T> k_3d = k_buf.view({1, n_kv_heads_, head_dim_});
    Tensor<T> v_3d = v_buf.view({1, n_kv_heads_, head_dim_});
    Tensor<T> q_3d = q_buf.view({1, n_heads_, head_dim_});
    save_tensor_to_binary(q_3d, save_prefix + "_graph_layer_" +
                                    std::to_string(i) + "_q_rope.bin");
    save_tensor_to_binary(k_3d, save_prefix + "_graph_layer_" +
                                    std::to_string(i) + "_k_rope.bin");

    // 保存attention输出（使用与forward_for_graph中相同的tag）
    std::string att_tag = "graph_att_heads_" + std::to_string(i);
    Tensor<T> att_heads({n_heads_, head_dim_}, Device::CUDA, false, att_tag);
    save_tensor_to_binary(att_heads, save_prefix + "_graph_layer_" +
                                         std::to_string(i) + "_att_heads.bin");

    // 保存attention投影（使用与forward_for_graph中相同的tag）
    std::string att_proj_tag = "graph_att_proj_" + std::to_string(i);
    Tensor<T> att_proj({1, hidden_size_}, Device::CUDA, false, att_proj_tag);
    save_tensor_to_binary(att_proj, save_prefix + "_graph_layer_" +
                                        std::to_string(i) + "_att_proj.bin");

    // 保存MLP中间结果（使用与forward_for_graph中相同的tag）
    std::string gate_tag = "graph_gate_buf_" + std::to_string(i);
    std::string up_tag = "graph_up_buf_" + std::to_string(i);
    std::string ffn_tag = "graph_ffn_out_" + std::to_string(i);

    Tensor<T> gate_buf({1, intermediate_size_}, Device::CUDA, false, gate_tag);
    Tensor<T> up_buf({1, intermediate_size_}, Device::CUDA, false, up_tag);
    Tensor<T> ffn_out({1, hidden_size_}, Device::CUDA, false, ffn_tag);

    // 保存MLP投影结果
    save_tensor_to_binary(gate_buf, save_prefix + "_graph_layer_" +
                                        std::to_string(i) + "_gate_proj.bin");
    save_tensor_to_binary(up_buf, save_prefix + "_graph_layer_" +
                                      std::to_string(i) + "_up_proj.bin");

    // 注意：gate_buf在图执行过程中已经被silu和multiply操作修改
    // 所以这里保存的实际上是silu_mul的结果
    save_tensor_to_binary(gate_buf, save_prefix + "_graph_layer_" +
                                        std::to_string(i) + "_silu_mul.bin");

    // 保存down投影
    save_tensor_to_binary(ffn_out, save_prefix + "_graph_layer_" +
                                       std::to_string(i) + "_down_proj.bin");

    // 注意：residual和hidden_states在图执行过程中会被不断修改
    // 这里保存的是当前层结束后的状态，但由于图的特性，
    // 我们无法保存每层结束时的精确状态，只能保存最终状态
    // 如果需要每层的精确状态，需要为每层创建独立的residual和hidden_states张量
  }

  // 保存最终norm
  Tensor<T> final_h({1, hidden_size_}, Device::CUDA, false, "graph_final_h");
  save_tensor_to_binary(final_h, save_prefix + "_graph_final_norm.bin");

  // 保存最终logits（使用graph_output_tensor_，它的tag是"graph_fixed_logits"）
  save_tensor_to_binary(graph_output_tensor_,
                        save_prefix + "_graph_final_logits.bin");

  std::cout << "图执行后的张量保存完成!" << std::endl;
}

// -------------------------------
// 显式模板实例化
// -------------------------------
template class QwenModel<float>;
template class QwenModel<__nv_bfloat16>;
