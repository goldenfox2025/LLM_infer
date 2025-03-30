#include <cuda_runtime.h>
#include <math.h>

#include <cstdio>
#include <iostream>
#include <vector>

#include "cudaOP.cuh"

namespace cuda_OP {

// --------------------------------------------------
// 模板化的逐元素加法内核实现（使用 __ldg 进行只读内存加载优化）
// --------------------------------------------------
template <typename T>
__global__ void add_kernel(T *A, const T *B, T *out, int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) {
    out[idx] = __ldg(&A[idx]) + __ldg(&B[idx]);
  }
}

// --------------------------------------------------
// 模板化的 host 端逐元素加法函数
// 输入张量 A 与 B 均为 Tensor<T> 类型，输出张量 output 用于存储结果。
// --------------------------------------------------
template <typename T>
void add(Tensor<T> *output, Tensor<T> *A, const Tensor<T> *B) {
  // 确认张量尺寸匹配
  if (A->numel() != B->numel()) {
    throw std::runtime_error("Tensor shapes do not match for addition");
  }

  // 确保所有张量都在CUDA设备上
  if (A->device() != Device::CUDA || B->device() != Device::CUDA ||
      output->device() != Device::CUDA) {
    throw std::runtime_error("All tensors must be on CUDA device for addition");
  }

  size_t total = A->numel();
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  add_kernel<T><<<blocks, threads>>>(A->data_ptr(), B->data_ptr(),
                                     output->data_ptr(), total);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaDeviceSynchronize());
}

// 显式实例化模板函数
template void add<float>(Tensor<float> *, Tensor<float> *,
                         const Tensor<float> *);

template void add<nvbf16>(Tensor<nvbf16> *, Tensor<nvbf16> *,
                          const Tensor<nvbf16> *);
}  // namespace cuda_OP
