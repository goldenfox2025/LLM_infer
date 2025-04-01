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

template <typename T_prob = float>
__global__ void sample_kernel(
    size_t top_k,  // 需要选择的 TopK 数量
    float top_p,   // 暂未使用
    T_prob* probabilities,  // 概率数组（需要可写，因为要“清除”已选项）
    uint32_t* d_sampled_index,  // 最终输出：从 TopK 中按权重随机采样的候选索引
    curandState* states,  // 随机状态（这里仅使用 states[0]）
    size_t vocab_size     // 词表大小
) {
  const int THREADS = blockDim.x;
  int tid = threadIdx.x;
  curandState localState = states[0];
  // 安全检查：若 top_k 为 0 或大于词表大小，则调整

  // 仅线程 0 用于存储每轮选出的候选概率和索引
  const int MAX_TOPK = 1024;
  T_prob top_candidate_probs[MAX_TOPK];
  uint32_t top_candidate_indices[MAX_TOPK];
  // 迭代 top_k 轮，每轮找出当前全局最大值
  for (int candidate = 0; candidate < top_k; candidate++) {
    // 每个线程遍历自己负责的部分，找出局部最大值及其索引
    T_prob local_max = -FLT_MAX;
    int local_idx = -1;
    for (int i = tid; i < vocab_size; i += THREADS) {
      T_prob prob = probabilities[i];
      if (prob > local_max) {
        local_max = prob;
        local_idx = i;
      }
    }
    // 利用共享内存进行线程块归约，得到全局最大值
    __shared__ T_prob sdata[1024];  // 存放局部最大值
    __shared__ int sindex[1024];    // 存放对应的索引
    sdata[tid] = local_max;
    sindex[tid] = local_idx;
    __syncthreads();

    // 采用二分归约
    for (int s = THREADS / 2; s > 0; s >>= 1) {
      if (tid < s) {
        if (sdata[tid] < sdata[tid + s]) {
          sdata[tid] = sdata[tid + s];
          sindex[tid] = sindex[tid + s];
        }
      }
      __syncthreads();
    }

    // 归约结束后，线程 0 得到当前全局最大值及其索引
    if (tid == 0) {
      int max_idx = sindex[0];
      T_prob max_prob = sdata[0];
      // 保存当前候选信息
      top_candidate_probs[candidate] = max_prob;
      top_candidate_indices[candidate] = max_idx;
      // 将该候选从原数组中“清除”，以免下轮再次选中
      probabilities[max_idx] = -FLT_MAX;
    }
    __syncthreads();  // 等待所有线程更新后再进入下一轮
  }
  // 采样：由线程 0 对这 top_k 个候选进行加权随机采样
  if (tid == 0) {
    // 计算 top_k 候选的概率总和
    T_prob total = 0;
    for (int i = 0; i < top_k; i++) {
      total += top_candidate_probs[i];
    }
    // 生成 [0, total) 范围内的随机数
    T_prob r = static_cast<T_prob>(curand_uniform(&localState)) * total;
    T_prob cumulative = 0;
    uint32_t selected = top_candidate_indices[0];
    for (int i = 0; i < top_k; i++) {
      cumulative += top_candidate_probs[i];
      if (cumulative >= r) {
        selected = top_candidate_indices[i];
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