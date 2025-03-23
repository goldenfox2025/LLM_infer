#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>  // printf
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cudaOP.cuh"

namespace cuda_OP {

// --------------------------------------------------
// --------------------------------------------------
template <typename T>
__global__ void matmul_kernel(const T *A, const T *B, T *C, int M, int K,
                              int N) {
  __shared__ T As[16][16];
  __shared__ T Bs[16][16];
  int row = blockIdx.y * 16 + threadIdx.y;
  int col = blockIdx.x * 16 + threadIdx.x;
  T sum = T(0);
  // 计算需要的 tile 数量
  int numTiles = (K + 16 - 1) / 16;
  for (int t = 0; t < numTiles; ++t) {
    int A_col = t * 16 + threadIdx.x;
    if (row < M && A_col < K) {
      As[threadIdx.y][threadIdx.x] = A[row * K + A_col];
    } else {
      As[threadIdx.y][threadIdx.x] = T(0);
    }
    int B_row = t * 16 + threadIdx.y;
   
    if (col < N && B_row < K) {
      Bs[threadIdx.y][threadIdx.x] = B[col * K + B_row];
    } else {
      Bs[threadIdx.y][threadIdx.x] = T(0);
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

// --------------------------------------------------
// --------------------------------------------------
template <typename T>
Tensor<T> matmul(const Tensor<T> &A, const Tensor<T> &B, cudaStream_t stream) {
  const std::vector<size_t> &A_shape = A.sizes();
  const std::vector<size_t> &B_shape = B.sizes();

  // A: [M, K], B: [N, K]（保证 A 的第二维与 B 的第二维一致）
  size_t M = A_shape[0];
  size_t K = A_shape[1];
  size_t n = B_shape[1];  
  Tensor<T> C({M, n}, Device::CUDA);

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

  matmul_kernel<T><<<numBlocks, threadsPerBlock, 0, stream>>>(
      A.data_ptr(), B.data_ptr(), C.data_ptr(), M, K, n);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA kernel launch failed: " +
                             std::string(cudaGetErrorString(err)));
  }

  return C;
}
template Tensor<float> matmul<float>(const Tensor<float>&, const Tensor<float>&,
  cudaStream_t);
  template Tensor<nvbf16> matmul<nvbf16>(const Tensor<nvbf16>&,
     const Tensor<nvbf16>&, cudaStream_t);

}  // namespace cuda_OP
