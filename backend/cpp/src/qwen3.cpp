#include "qwen3.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "cudaOP.cuh"
#include "operators/unified_operators.hpp"
#include "tensor.hpp"



// -------------------------------
// Qwen3Model<T> 构造函数
// -------------------------------
template <typename T>
Qwen3Model<T>::Qwen3Model(
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
  rms_norm_eps_ = static_cast<float>(config.at("rms_norm_eps")) / 1000000.0f;
  rope_theta_ = static_cast<float>(config.at("rope_theta"));
  head_dim_ = hidden_size_ / n_heads_;
  device_ = Device::CPU;  // 默认在CPU上初始化

  // 初始化算子接口
  operators_ = std::make_unique<op::UnifiedOperators<T>>(device_);

  // 初始化CUDA流和事件
  for (int i = 0; i < kNumStreams; ++i) {
    cudaStreamCreate(&compute_streams_[i]);
  }

  for (int i = 0; i < 3; ++i) {
    // 使用 cudaEventDisableTiming
    // 可以获得微小的性能提升，因为我们只关心完成状态，不测量时间
    cudaEventCreateWithFlags(&fa_done_events_[i], cudaEventDisableTiming);
  }
}

// 带量化参数的构造函数
template <typename T>
Qwen3Model<T>::Qwen3Model(
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
  rms_norm_eps_ = static_cast<float>(config.at("rms_norm_eps")) / 1000000.0f;
  rope_theta_ = static_cast<float>(config.at("rope_theta"));
  head_dim_ = hidden_size_ / n_heads_;
  device_ = Device::CPU;  // 默认在CPU上初始化

  // 设置量化类型和分组大小
  quant_type_ = config.at("quant_type");
  if (config.find("group_size") != config.end()) {
    group_size_ = config.at("group_size");
  }

  // 初始化算子接口
  operators_ = std::make_unique<op::UnifiedOperators<T>>(device_);

  // 初始化CUDA流和事件
  for (int i = 0; i < kNumStreams; ++i) {
    cudaStreamCreate(&compute_streams_[i]);
  }

  for (int i = 0; i < 3; ++i) {
    cudaEventCreateWithFlags(&fa_done_events_[i], cudaEventDisableTiming);
  }
}

template <typename T>
Qwen3Model<T>::~Qwen3Model() {
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
bool Qwen3Model<T>::verify_params() const {
  // 禁用
  std::cout << "Not checking parameters" << std::endl;
  return true;
}

// -------------------------------
// 打印模型信息
// -------------------------------
template <typename T>
void Qwen3Model<T>::print_model_info() const {
  std::cout << "\n=== Qwen3 Model Information ===" << std::endl;
  std::cout << "Vocab Size: " << vocab_size_ << std::endl;
  std::cout << "Hidden Size: " << hidden_size_ << std::endl;
  std::cout << "Num Layers: " << n_layers_ << std::endl;
  std::cout << "Num Attention Heads: " << n_heads_ << std::endl;
  std::cout << "Num KV Heads: " << n_kv_heads_ << std::endl;
  std::cout << "Head Dimension: " << head_dim_ << std::endl;
  std::cout << "Intermediate Size: " << intermediate_size_ << std::endl;
  std::cout << "Max Position Embeddings: " << max_position_embeddings_
            << std::endl;
  std::cout << "RMS Norm Epsilon: " << rms_norm_eps_ << std::endl;
  std::cout << "RoPE Theta: " << rope_theta_ << std::endl;

  if (quant_type_ > 0) {
    std::cout << "Quantization: AWQ (group_size=" << group_size_ << ")"
              << std::endl;
  } else {
    std::cout << "Quantization: None" << std::endl;
  }

  std::cout << "Device: " << (device_ == Device::CUDA ? "CUDA" : "CPU")
            << std::endl;
  std::cout << "================================\n" << std::endl;
}

// -------------------------------
// cuda()：将所有参数移到 CUDA，并设置设备
// -------------------------------
template <typename T>
Qwen3Model<T> &Qwen3Model<T>::cuda() {
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

  return *this;
}

// -------------------------------
// cpu()：Qwen3 模型仅支持 CUDA，故调用 cpu() 抛出异常
// -------------------------------
template <typename T>
Qwen3Model<T> &Qwen3Model<T>::cpu() {
  // 更新算子接口（虽然会抛出异常，但保持一致性）
  if (operators_) {
    operators_->cpu();
  }

  throw std::runtime_error("Qwen3Model only supports CUDA execution.");
  return *this;
}

// 显式实例化模板类
template class Qwen3Model<__nv_bfloat16>;
