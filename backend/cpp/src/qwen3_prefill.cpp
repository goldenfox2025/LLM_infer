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
// prefill: 预填充接口
// -------------------------------
template <typename T>
uint32_t* Qwen3Model<T>::prefill(const Tensor<uint32_t>* input,
                                ThreadPool& thread_pool, KVCacheBase* kv_cache,
                                size_t top_k, float temperature, float top_p,
                                curandState* d_states) {
  KVCache<T>* typed_cache = dynamic_cast<KVCache<T>*>(kv_cache);

  return cuda_OP::sample(prefill_cuda(input, typed_cache), temperature, top_p,
                         top_k, d_states);
}

// -------------------------------
// prefill_cuda: CUDA 版本的预填充实现
// -------------------------------
template <typename T>
Tensor<T> Qwen3Model<T>::prefill_cuda(const Tensor<uint32_t>* input,
                                     KVCache<T>* kv_cache) {
  // 这里是预填充的实现，暂时留空
  // 将在后续实现
  throw std::runtime_error("Qwen3Model::prefill_cuda not implemented yet");
  return Tensor<T>();
}

// 显式实例化模板函数
template uint32_t* Qwen3Model<__nv_bfloat16>::prefill(
    const Tensor<uint32_t>* input, ThreadPool& thread_pool,
    KVCacheBase* kv_cache, size_t top_k, float temperature, float top_p,
    curandState* d_states);

template Tensor<__nv_bfloat16> Qwen3Model<__nv_bfloat16>::prefill_cuda(
    const Tensor<uint32_t>* input, KVCache<__nv_bfloat16>* kv_cache);
