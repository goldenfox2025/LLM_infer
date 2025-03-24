#include "qwen.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

#include "cudaOP.cuh"
#include "operators.hpp"

// 前向声明转换函数
std::unordered_map<std::string, Tensor<__nv_bfloat16>> convert_weights_to_bf16(
    const std::unordered_map<std::string, Tensor<float>>& float_weights);

// 模板类的实现
template <typename T>
QwenModel<T>::QwenModel(
    const std::unordered_map<std::string, Tensor<T>>& params,
    const std::unordered_map<std::string, int>& config)
    : params_(params),
      device_(Device::CPU) {
  // 从config中读取基本参数
  vocab_size_ = config.at("vocab_size");
  n_layers_ = config.at("n_layers");
  n_heads_ = config.at("n_heads");
  n_kv_heads_ = config.at("n_kv_heads");
  hidden_size_ = config.at("hidden_size");
  intermediate_size_ = config.at("intermediate_size");
  max_position_embeddings_ = config.at("max_position_embeddings");
  head_dim_ = hidden_size_ / n_heads_;
  bos_token_id_ = config.at("bos_token_id");
  eos_token_id_ = config.at("eos_token_id");
  rms_norm_eps_ = config.at("rms_norm_eps") / 1e6f;  // 这里可能需要转换单位
  rope_theta_ = config.at("rope_theta");
  
  // 对于BF16类型，默认使用CUDA
  if constexpr(std::is_same_v<T, __nv_bfloat16>) {
    device_ = Device::CUDA;
  }
}

template <typename T>
bool QwenModel<T>::verify_params() const {
  // 验证参数有效性
  bool valid = true;
  
  // 检查必要的张量是否存在
  std::vector<std::string> required_keys = {
    "token_embeddings.weight", 
    "norm.weight"
    // 更多检查可以在这里添加
  };
  
  // 检查所有层的参数
  for (size_t i = 0; i < n_layers_; i++) {
    std::string prefix = "layers." + std::to_string(i) + ".";
    required_keys.push_back(prefix + "attention.wq.weight");
    required_keys.push_back(prefix + "attention.wk.weight");
    required_keys.push_back(prefix + "attention.wv.weight");
    required_keys.push_back(prefix + "attention.wo.weight");
    required_keys.push_back(prefix + "feed_forward.w1.weight");
    required_keys.push_back(prefix + "feed_forward.w2.weight");
    required_keys.push_back(prefix + "attention_norm.weight");
    required_keys.push_back(prefix + "ffn_norm.weight");
  }
  
  for (const auto& key : required_keys) {
    if (params_.find(key) == params_.end()) {
      std::cerr << "Missing parameter: " << key << std::endl;
      valid = false;
    }
  }
  
  return valid;
}

template <typename T>
void QwenModel<T>::print_model_info() const {
  std::cout << "QwenModel Information:" << std::endl;
  std::cout << "  Vocabulary Size: " << vocab_size_ << std::endl;
  std::cout << "  Layers: " << n_layers_ << std::endl;
  std::cout << "  Attention Heads: " << n_heads_ << std::endl;
  std::cout << "  KV Heads: " << n_kv_heads_ << std::endl;
  std::cout << "  Hidden Size: " << hidden_size_ << std::endl;
  std::cout << "  Head Dimension: " << head_dim_ << std::endl;
  std::cout << "  Intermediate Size: " << intermediate_size_ << std::endl;
  std::cout << "  Max Sequence Length: " << max_position_embeddings_ << std::endl;
  std::cout << "  RMS Norm Epsilon: " << rms_norm_eps_ << std::endl;
  std::cout << "  RoPE Theta: " << rope_theta_ << std::endl;
  std::cout << "  Device: " << (device_ == Device::CUDA ? "CUDA" : "CPU") << std::endl;
  
  if constexpr(std::is_same_v<T, __nv_bfloat16>) {
    std::cout << "  Precision: BF16" << std::endl;
  } else {
    std::cout << "  Precision: FP32" << std::endl;
  }
}

// BaseModel interface implementations
template <typename T>
Tensor<float> QwenModel<T>::forward(const Tensor<uint32_t>* input,
                                   ThreadPool& thread_pool,
                                   KVCache* kv_cache) {
  // 对于float类型直接返回内部实现的结果
  if constexpr(std::is_same_v<T, float>) {
    return forward_internal(input, thread_pool, kv_cache);
  } 
  // 对于BF16类型需要转换回float
  else if constexpr(std::is_same_v<T, __nv_bfloat16>) {
    Tensor<__nv_bfloat16> bf16_output = forward_internal(input, thread_pool, kv_cache);
    return bf16_output.to_float();
  }
}

template <typename T>
Tensor<float> QwenModel<T>::prefill(const Tensor<uint32_t>* input,
                                   ThreadPool& thread_pool,
                                   KVCache* kv_cache) {
  // 对于float类型直接返回内部实现的结果
  if constexpr(std::is_same_v<T, float>) {
    return prefill_internal(input, thread_pool, kv_cache);
  } 
  // 对于BF16类型需要转换回float
  else if constexpr(std::is_same_v<T, __nv_bfloat16>) {
    Tensor<__nv_bfloat16> bf16_output = prefill_internal(input, thread_pool, kv_cache);
    return bf16_output.to_float();
  }
}

// 内部实现 - forward相关
template <typename T>
Tensor<T> QwenModel<T>::forward_internal(
    const Tensor<uint32_t>* input, ThreadPool& thread_pool, KVCache* kv_cache) {
  if (device_ == Device::CPU) {
    return forward_cpu(input, thread_pool, kv_cache);
  } else {
    return forward_cuda(input, kv_cache);
  }
}

template <typename T>
Tensor<T> QwenModel<T>::forward_cpu(
    const Tensor<uint32_t>* input, ThreadPool& thread_pool, KVCache* kv_cache) {
  // BF16类型不支持CPU
  if constexpr(std::is_same_v<T, __nv_bfloat16>) {
    throw std::runtime_error("QwenModel<__nv_bfloat16>::forward_cpu is not supported");
  }
  
  // Placeholder implementation - should be replaced with actual code
  // This is very similar to LlamaModel's forward_cpu but with Qwen's specific architecture
  throw std::runtime_error("QwenModel<float>::forward_cpu not implemented yet");
}

template <typename T>
Tensor<T> QwenModel<T>::forward_cuda(
    const Tensor<uint32_t>* input, KVCache* kv_cache) {
  // Placeholder implementation
  throw std::runtime_error("QwenModel::forward_cuda not implemented yet");
}

// 内部实现 - prefill相关
template <typename T>
Tensor<T> QwenModel<T>::prefill_internal(
    const Tensor<uint32_t>* input, ThreadPool& thread_pool, KVCache* kv_cache) {
  if (device_ == Device::CPU) {
    return prefill_cpu(input, kv_cache, thread_pool);
  } else {
    return prefill_cuda(input, kv_cache);
  }
}

template <typename T>
Tensor<T> QwenModel<T>::prefill_cpu(
    const Tensor<uint32_t>* input, KVCache* kv_cache, ThreadPool& thread_pool) {
  // BF16类型不支持CPU
  if constexpr(std::is_same_v<T, __nv_bfloat16>) {
    throw std::runtime_error("QwenModel<__nv_bfloat16>::prefill_cpu is not supported");
  }
  
  // Placeholder implementation
  throw std::runtime_error("QwenModel<float>::prefill_cpu not implemented yet");
}

template <typename T>
Tensor<T> QwenModel<T>::prefill_cuda(
    const Tensor<uint32_t>* input, KVCache* kv_cache) {
  // Placeholder implementation
  throw std::runtime_error("QwenModel::prefill_cuda not implemented yet");
}

// 设备管理
template <typename T>
QwenModel<T>& QwenModel<T>::cuda() {
  if (device_ == Device::CUDA) return *this;
  
  std::cout << "[QwenModel::cuda] Moving model to CUDA." << std::endl;
  device_ = Device::CUDA;
  
  // Convert all parameters to CUDA
  for (auto& param : params_) {
    param.second.cuda();
  }
  
  return *this;
}

template <typename T>
QwenModel<T>& QwenModel<T>::cpu() {
  // BF16类型不支持CPU
  if constexpr(std::is_same_v<T, __nv_bfloat16>) {
    throw std::runtime_error("QwenModel<__nv_bfloat16>::cpu is not supported, BF16 requires CUDA");
  }
  
  if (device_ == Device::CPU) return *this;
  
  std::cout << "[QwenModel::cpu] Moving model to CPU." << std::endl;
  device_ = Device::CPU;
  
  // Convert all parameters to CPU
  for (auto& param : params_) {
    param.second.cpu();
  }
  
  return *this;
}

// Helper function to convert weights from float to __nv_bfloat16
std::unordered_map<std::string, Tensor<__nv_bfloat16>> convert_weights_to_bf16(
    const std::unordered_map<std::string, Tensor<float>>& float_weights) {
  std::unordered_map<std::string, Tensor<__nv_bfloat16>> bf16_weights;
  
  for (const auto& [name, tensor] : float_weights) {
    // Get shape and size information
    std::vector<size_t> shape = tensor.sizes();
    size_t total_size = tensor.numel();
    
    // Create a buffer for bf16 data
    std::vector<__nv_bfloat16> bf16_data(total_size);
    
    // Get raw pointers
    const float* float_ptr = tensor.data_ptr();
    
    // Convert each element from float to bf16
    for (size_t i = 0; i < total_size; ++i) {
      bf16_data[i] = __nv_bfloat16(float_ptr[i]);
    }
    
    // Create new tensor with bf16 data
    Tensor<__nv_bfloat16> bf16_tensor(std::move(bf16_data), shape);
    
    // If the original tensor was on CUDA, move the new one to CUDA too
    if (tensor.device() == Device::CUDA) {
      bf16_tensor.cuda();
    }
    
    // Add to the result map
    bf16_weights[name] = std::move(bf16_tensor);
  }
  
  return bf16_weights;
}

// 显式实例化模板类，这样在其他文件中可以使用它们
template class QwenModel<float>;
template class QwenModel<__nv_bfloat16>;
