#ifndef CUDA_OP_CUH
#define CUDA_OP_CUH

#include <cuda_bf16.h>  // For __nv_bfloat16
#include <cuda_runtime.h>

#include <limits>     // 用于 numeric_limits
#include <stdexcept>  // 用于 runtime_error
#include <vector>     // 用于 host 端 vector (如果需要)

#include "CudaMemoryPool.hpp"  // 用于调试输出
#include "cudaOP.cuh"
#include "tensor.hpp"  // 假设你的 Tensor 类头文件

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
#include <curand_kernel.h>
#include <float.h>  // for FLT_MAX

// 假设 blockDim.x 为 1024
template <typename T_prob = float>
__global__ void sample_basic_kernel(
    size_t top_k,  // 需要选取的 TopK 数量
    float top_p,   // 暂未使用
    T_prob*
        probabilities,  // 原始概率数组（必须可写，因为我们需要清除已经选出的值）
    uint32_t* d_sampled_index,  // 输出数组，存放 TopK
                                // 候选的索引（数组长度至少为 top_k）
    curandState* states,  // 随机状态（这里只使用 states[0]）
    size_t vocab_size     // 词表大小
) {
  const int THREADS = blockDim.x;  // 1024
  int tid = threadIdx.x;

  // 初始化局部随机状态
  curandState localState = states[0];

  // 安全检查：若 top_k 为 0，则直接返回
  if (top_k == 0) {
    if (tid == 0) {
      d_sampled_index[0] = 0;
    }
    return;
  }
  // 如果 top_k 超过词表大小，则置为词表大小
  if (top_k > vocab_size) {
    top_k = vocab_size;
  }

  // 循环迭代，每轮找出一个最大值
  for (int candidate = 0; candidate < top_k; candidate++) {
    // 每个线程扫描全局数组，计算自己负责区间内的局部最大值及其索引
    T_prob local_max = -FLT_MAX;
    int local_idx = -1;
    for (int i = tid; i < vocab_size; i += THREADS) {
      T_prob prob = probabilities[i];
      if (prob > local_max) {
        local_max = prob;
        local_idx = i;
      }
    }

    // 使用共享内存做线程块归约
    __shared__ T_prob sdata[1024];  // 存放局部最大值
    __shared__ int sindex[1024];    // 存放对应索引
    sdata[tid] = local_max;
    sindex[tid] = local_idx;
    __syncthreads();

    // 归约：采用二分法将 1024 个线程的结果归约成一个全局最大值
    for (int s = THREADS / 2; s > 0; s >>= 1) {
      if (tid < s) {
        if (sdata[tid] < sdata[tid + s]) {
          sdata[tid] = sdata[tid + s];
          sindex[tid] = sindex[tid + s];
        }
      }
      __syncthreads();
    }

    // 线程 0 得到本轮全局最大值及其索引
    if (tid == 0) {
      int max_idx = sindex[0];
      // 保存该候选的索引到输出数组中
      d_sampled_index[candidate] = max_idx;
      // 清除选中的最大值：置为 -FLT_MAX，保证后续轮次不再选中
      probabilities[max_idx] = -FLT_MAX;
    }
    __syncthreads();  // 等待所有线程看到更新后的 probabilities 数组
  }

  // 更新随机状态（本例中未在采样中使用随机数，但若后续需要可保留）
  if (tid == 0) {
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
  sample_basic_kernel<T><<<1, 1024>>>(top_k, top_p, d_probabilities.data_ptr(),
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