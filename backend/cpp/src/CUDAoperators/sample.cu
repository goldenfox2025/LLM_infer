#ifndef CUDA_OP_CUH
#define CUDA_OP_CUH

#include <cuda_bf16.h>  // For __nv_bfloat16
#include <cuda_runtime.h>

#include <cub/block/block_radix_sort.cuh>
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
__global__ void sample_basic_kernel(size_t top_k,
                                    float top_p,  // top_p 暂未使用
                                    const T_prob* __restrict__ probabilities,
                                    uint32_t* __restrict__ d_sampled_index,
                                    curandState* __restrict__ states,
                                    size_t vocab_size) {
  // 仅线程 0 执行采样逻辑

  curandState localState = states[0];

  // 安全检查：top_k 为 0 或大于词表大小时调整
  if (top_k == 0) {
    *d_sampled_index = 0;
    return;
  }
  if (top_k > vocab_size) {
    top_k = vocab_size;
  }

  // 用固定大小的数组保存 top-k 候选（假设 top_k <= 2048）
  const int MAX_TOPK = 2048;
  T_prob top_probs[MAX_TOPK];
  uint32_t top_indices[MAX_TOPK];
  int count = 0;

  // 遍历整个词表，选出概率最高的 top_k 候选项
  for (int i = 0; i < vocab_size; i++) {
    T_prob prob = probabilities[i];
    if (count < top_k) {
      // 前 top_k 个直接填入候选数组
      top_probs[count] = prob;
      top_indices[count] = i;
      count++;
    } else {
      // 在已有候选中寻找最小概率的候选
      int min_idx = 0;
      for (int j = 1; j < top_k; j++) {
        if (top_probs[j] < top_probs[min_idx]) {
          min_idx = j;
        }
      }
      // 如果当前概率大于候选中最小的，则替换之
      if (prob > top_probs[min_idx]) {
        top_probs[min_idx] = prob;
        top_indices[min_idx] = i;
      }
    }
  }

  // 计算 top-k 候选的概率总和
  T_prob total_prob = 0;
  for (int i = 0; i < top_k; i++) {
    total_prob += top_probs[i];
  }

  // // 生成一个随机数，并映射到 [0, total_prob) 区间
  T_prob r = static_cast<T_prob>(curand_uniform(&localState)) * total_prob;

  // 根据 r 进行累积采样，找到落在随机阈值中的候选项
  T_prob cumulative = 0;
  uint32_t selected = top_indices[0];  // 默认选择第一个候选项
  for (int i = 0; i < top_k; i++) {
    cumulative += top_probs[i];
    if (T_prob(0.2) < cumulative) {
      selected = top_indices[i];
      break;
    }
  }

  // 写回采样结果，并更新随机状态
  *d_sampled_index = selected;
  states[0] = localState;
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
  sample_basic_kernel<T><<<1, 1>>>(top_k, top_p, d_probabilities.data_ptr(),
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