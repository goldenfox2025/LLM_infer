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
// 带偏置的矩阵乘法kernel
// --------------------------------------------------
template <typename T>
__global__ void matmul_with_bias_kernel(const T *A, const T *B, const T *bias,
                                        T *C, int M, int K, int N) {
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
    // 在结果中加上偏置
    C[row * N + col] = sum + bias[col];
  }
}

// --------------------------------------------------
// --------------------------------------------------
template <typename T>
void matmul(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> *C,
            cudaStream_t stream, const Tensor<T> *bias) {
  const std::vector<size_t> &A_shape = A.sizes();
  const std::vector<size_t> &B_shape = B.sizes();

  // A: [M, K], B: [N, K]（保证 A 的第二维与 B 的第二维一致）
  size_t M = A_shape[0];
  size_t K = A_shape[1];
  size_t N = B_shape[1];

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

  if (bias == nullptr) {
    // 使用无偏置版本的kernel
    matmul_kernel<T><<<numBlocks, threadsPerBlock, 0, stream>>>(
        A.data_ptr(), B.data_ptr(), C->data_ptr(), M, K, N);
  } else {
    // 检查偏置形状
    const std::vector<size_t> &bias_shape = bias->sizes();
    if (bias_shape.size() != 1) {
      throw std::runtime_error("Bias must be a 1D tensor");
    }
    if (bias_shape[0] != N) {
      throw std::runtime_error("Bias size must match output column dimension");
    }

    // 使用带偏置版本的kernel
    matmul_with_bias_kernel<T><<<numBlocks, threadsPerBlock, 0, stream>>>(
        A.data_ptr(), B.data_ptr(), bias->data_ptr(), C->data_ptr(), M, K, N);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA kernel launch failed: " +
                             std::string(cudaGetErrorString(err)));
  }

  return;
}

template void matmul<float>(const Tensor<float> &, const Tensor<float> &,
                            Tensor<float> *, cudaStream_t,
                            const Tensor<float> *);
template void matmul<nvbf16>(const Tensor<nvbf16> &, const Tensor<nvbf16> &,
                             Tensor<nvbf16> *, cudaStream_t,
                             const Tensor<nvbf16> *);

}  // namespace cuda_OP
