#include <cuda_runtime.h>
#include <math.h>

#include <cstdio>
#include <iostream>
#include <vector>

#include "cudaOP.cuh"

namespace cuda_OP {

// --------------------------------------------------
// 模板化的逐元素乘法内核实现（版本 v2，使用 __ldg 进行只读内存加载优化）
// --------------------------------------------------
template <typename T>
__global__ void multiply_kernel_v2(const T *A, const T *B, T *out, int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) {
    out[idx] = __ldg(&A[idx]) * __ldg(&B[idx]);
  }
}

// --------------------------------------------------
// 模板化的逐元素乘法内核实现（版本 v1，不使用 __ldg，仅做对比）
// --------------------------------------------------
template <typename T>
__global__ void multiply_kernel_v1(const T *A, const T *B, T *out, int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) {
    out[idx] = A[idx] * B[idx];
  }
}

// --------------------------------------------------
// 模板化的 host 端逐元素乘法函数
// 输入张量 A 与 B 均为 Tensor<T> 类型，输出张量 output 用于存储结果。
// 这里调用的是版本 v2 内核函数
// --------------------------------------------------
template <typename T>
void multiply(Tensor<T> *output, const Tensor<T> *A, const Tensor<T> *B) {
  size_t total = A->numel();
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  multiply_kernel_v2<T><<<blocks, threads>>>(A->data_ptr(), B->data_ptr(),
                                             output->data_ptr(), total);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaDeviceSynchronize());
}
template void multiply<float>(Tensor<float> *, const Tensor<float> *,
                              const Tensor<float> *);

template void multiply<nvbf16>(Tensor<nvbf16> *, const Tensor<nvbf16> *,
                               const Tensor<nvbf16> *);
}  // namespace cuda_OP
