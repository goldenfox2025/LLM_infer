#pragma once
#include <cuda_bf16.h>
#include <curand_kernel.h>  // 用于设备端随机数生成

#include <chrono>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "cudaOP.cuh"
#include "kvcache_base.hpp"
#include "tensor.hpp"
#include "thread_pool.hpp"

// Forward declaration of KVCache
template <typename T>
class KVCache;
constexpr int kNumStreams = 5;
// Base model class that will be used for both LlamaModel and QwenModel
class BaseModel {
 public:
  virtual ~BaseModel() = default;

  // Core inference methods that must be implemented by derived classes
  virtual uint32_t* forward(const Tensor<uint32_t>* input,
                            ThreadPool& thread_pool, KVCacheBase* kv_cache,
                            size_t top_k, float temperature, float top_p,
                            curandState* d_states = nullptr) = 0;
  virtual uint32_t* prefill(const Tensor<uint32_t>* input,
                            ThreadPool& thread_pool, KVCacheBase* kv_cache,
                            size_t top_k, float temperature, float top_p,
                            curandState* d_states = nullptr) = 0;

  // Common methods for all model types
  virtual bool verify_params() const = 0;
  virtual void print_model_info() const = 0;

  // Device management
  virtual BaseModel& cuda() = 0;
  virtual BaseModel& cpu() = 0;
  virtual Device device() const = 0;

  // Getters for model properties
  virtual size_t get_n_layers() const = 0;
  virtual size_t get_max_seq_len() const = 0;
  virtual size_t get_head_dim() const = 0;
  virtual size_t get_n_kv_heads() const = 0;
  virtual uint32_t get_eos_token_id() const = 0;
};
