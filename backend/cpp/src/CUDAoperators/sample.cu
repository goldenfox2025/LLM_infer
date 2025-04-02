#ifndef CUDA_OP_CUH
#define CUDA_OP_CUH

#include <cuda_bf16.h>  // For __nv_bfloat16
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <float.h>  // for FLT_MAX

#include <limits>     // 用于 numeric_limits
#include <stdexcept>  // 用于 runtime_error
#include <vector>     // 用于 host 端 vector (如果需要)

#include "CudaMemoryPool.hpp"  // 用于调试输出
#include "cudaOP.cuh"
#include "tensor.hpp"  // 假设你的 Tensor 类头文件
#define max_topk 1024
// Simple CUDA error checking macro
#define CUDA_CHECK(call)                                                \
  do {                                                                  \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
      fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                 \
      /* 考虑抛出异常而不是退出 */                           \
      throw std::runtime_error(cudaGetErrorString(err));                \
    }                                                                   \
  } while (0)

namespace cuda_OP {
template <typename T = float>
__global__ void sample_kernel(
    size_t top_k,  // 需要选择的 TopK 数量
    float top_p,   // 暂未使用
    T* probabilities,  // 概率数组（需要可写，因为要“清除”已选项）
    uint32_t* d_sampled_index,  // 最终输出：从 TopK 中按权重随机采样的候选索引
    curandState* states,  // 随机状态（这里仅使用 states[0]）
    size_t vocab_size     // 词表大小
) {
  const int THREADS_PER_BLOCK = blockDim.x;
  const int tid = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;

  curandState localState = states[0];
  __shared__ T shared_probs[max_topk];
  __shared__ int shared_indices[max_topk];
  T top_candidate_probs[max_topk];
  int top_candidate_idx[max_topk];
  for (int idx = 0; idx < top_k; ++idx) {
    T local_max = -FLT_MAX;
    int local_idx = -1;
    for (int i = tid; i < vocab_size; i += THREADS_PER_BLOCK) {
      const T prob = __ldg(&probabilities[i]);
      if (prob > local_max) {
        local_max = prob;
        local_idx = i;
      }
    }
    shared_probs[tid] = local_max;
    shared_indices[tid] = local_idx;
    __syncthreads();
    for (int i = THREADS_PER_BLOCK >> 1; i > 0; i >>= 1) {
      if (tid < i) {
        const T prob = shared_probs[tid + i];
        if (prob > shared_probs[tid]) {
          shared_probs[tid] = prob;
          shared_indices[tid] = shared_indices[tid + i];
        }
      }
      __syncthreads();
    }
    if (tid == 0) {
      probabilities[shared_indices[0]] = T(-FLT_MAX);  // 清除已选项
      top_candidate_probs[idx] = shared_probs[0];
      top_candidate_idx[idx] = shared_indices[0];
    }
    __syncthreads();
  }
  if (tid == 0) {
    T total = 0;
    for (int i = 0; i < top_k; i++) {
      total += top_candidate_probs[i];
    }
    T r = static_cast<T>(curand_uniform(&localState)) * total;
    T cumulative = 0;
    uint32_t selected = top_candidate_idx[0];
    for (int i = 0; i < top_k; i++) {
      cumulative += top_candidate_probs[i];
      if (cumulative >= r) {
        selected = top_candidate_idx[i];
        break;
      }
    }
    *d_sampled_index = selected;
    states[0] = localState;
  }
}
template <typename T>
uint32_t sample(Tensor<T>&& logits, float temperature, float top_p,
                size_t top_k, curandState* d_states) {
  if (logits.device() != Device::CUDA) {
    throw std::runtime_error(
        "Input tensor for cuda_OP::sample must be on CUDA device");
  }
  if (top_k > 1024) {
    top_k = 1024;
  }
  if (top_k < 1) {
    top_k = 1;
  }

  const auto& shape = logits.sizes();
  if (shape.size() != 2) {  // 只检查维度是否为 2
    std::cerr << "Input tensor shape: [";
    for (size_t i = 0; i < shape.size(); ++i)
      std::cerr << shape[i] << (i == shape.size() - 1 ? "" : ", ");
    std::cerr << "]" << std::endl;
    throw std::runtime_error(
        "Input tensor must have 2 dimensions: [seq_len, vocab_size]");
  }

  const size_t seq_len = shape[0];
  const size_t vocab_size = shape[1];

  if (seq_len == 0 || vocab_size == 0) {
    std::cerr << "Input tensor shape: [" << seq_len << ", " << vocab_size << "]"
              << std::endl;
    throw std::runtime_error(
        "Sequence length and vocab size must be greater than zero");
  }

  if (seq_len > 1) {
    logits.slice_inplace({seq_len - 1, 0}, {seq_len, vocab_size});
  }

  Tensor<T> d_probabilities({vocab_size}, Device::CUDA);
  softmax<T>(&d_probabilities, &logits, 1, false, 0, temperature);

  uint32_t* d_sampled_index = static_cast<uint32_t*>(
      GlobalCudaMemoryPool::instance().allocate(sizeof(uint32_t)));
  sample_kernel<T><<<1, 1024>>>(top_k, top_p, d_probabilities.data_ptr(),
                                d_sampled_index, d_states, vocab_size);
  CUDA_CHECK(cudaGetLastError());
  uint32_t res = 0;
  cudaMemcpy(&res, d_sampled_index, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  GlobalCudaMemoryPool::instance().free(d_sampled_index);
  return res;
}

template uint32_t sample<float>(Tensor<float>&&, float, float, size_t,
                                curandState*);
template uint32_t sample<__nv_bfloat16>(Tensor<__nv_bfloat16>&&, float, float,
                                        size_t, curandState*);

}  // namespace cuda_OP

#endif  // CUDA_OP_CUH