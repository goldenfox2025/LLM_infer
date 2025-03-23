#include "cudaOP.cuh"
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <vector>
// --------------------------------------------------
// multiply 算子实现（逐元素乘法）
// --------------------------------------------------
namespace cuda_OP {

__global__ void multiply_kernel(const float *A, const float *B, float *out,
                                int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) {
    out[idx] = A[idx] * B[idx];
  }
}

void multiply(Tensor<float> *output, const Tensor<float> *A,
              const Tensor<float> *B) {
  size_t total = A->numel();
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  multiply_kernel<<<blocks, threads>>>(A->data_ptr(), B->data_ptr(),
                                       output->data_ptr(), total);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaDeviceSynchronize());
}
} // namespace cuda_OP