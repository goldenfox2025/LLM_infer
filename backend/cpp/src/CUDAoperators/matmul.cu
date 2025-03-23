
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math.h>

#include <cstdio>  // //printf
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cudaOP.cuh"
namespace cuda_OP {
// --------------------------------------------------
// matmul 算子实现
// --------------------------------------------------
__global__ void matmul_kernel(const float *A, const float *B, float *C, int M,
                              int K, int N) {
  __shared__ float As[16][16];
  __shared__ float Bs[16][16];
  int row = blockIdx.y * 16 + threadIdx.y;
  int col = blockIdx.x * 16 + threadIdx.x;
  float sum = 0.0f;
  for (int t = 0; t < (16 + K - 1) / 16; ++t) {
    int A_col = t * 16 + threadIdx.x;
    if (row < M && A_col < K) {
      As[threadIdx.y][threadIdx.x] = A[row * K + A_col];
    } else {
      As[threadIdx.y][threadIdx.x] = 0.0f;
    }
    int B_row = t * 16 + threadIdx.y;
    if (col < N && B_row < K) {
      Bs[threadIdx.y][threadIdx.x] = B[col * K + B_row];
    } else {
      Bs[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();
    for (int k = 0; k < 16; ++k) {
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

Tensor<float> matmul(const Tensor<float> &A, const Tensor<float> &B,
                     cudaStream_t stream) {
  const std::vector<size_t> &A_shape = A.sizes();
  const std::vector<size_t> &B_shape = B.sizes();

  size_t m = A_shape[0];
  size_t k = A_shape[1];
  size_t n = B_shape[1];
  Tensor<float> C({m, n}, Device::CUDA);

  // 启动 CUDA 核

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

  // 计算矩阵乘法 C = A * B^T
  matmul_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
      A.data_ptr(), B.data_ptr(), C.data_ptr(), m, k, n);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA kernel launch failed: " +
                             std::string(cudaGetErrorString(err)));
  }

  return C;
}
}  // namespace cuda_OP