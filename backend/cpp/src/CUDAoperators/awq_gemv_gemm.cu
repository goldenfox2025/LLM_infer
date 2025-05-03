#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "cudaOP.cuh"

template <typename T>
std::string vec_to_string(const std::vector<T>& vec) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    ss << vec[i] << (i == vec.size() - 1 ? "" : ", ");
  }
  ss << "]";
  return ss.str();
}

namespace cuda_OP {

constexpr int BITS = 4;
constexpr int PACK_FACTOR = 32 / BITS;

template <typename T, typename S, int TILE_K = 32, int BLOCK_N = 256>
__global__ void matmul_awq_gemv_kernel(const T* __restrict__ inp,
                                       const int32_t* __restrict__ qwt,
                                       const S* __restrict__ scl,
                                       const int32_t* __restrict__ zos,
                                       T* __restrict__ out, int M, int K, int N,
                                       int group_size,
                                       const T* __restrict__ bias) {
  int m = blockIdx.y;
  int tid = threadIdx.x;
  int n = blockIdx.x * BLOCK_N + tid;

  if (m >= M || n >= N) {
    return;
  }

  __shared__ T sh_inp[TILE_K];
  constexpr int K_PACKED_PER_TILE = TILE_K / PACK_FACTOR;
  __shared__ int32_t sh_qwt[BLOCK_N][K_PACKED_PER_TILE];

  const int G = K / group_size;
  const int G_PACKED = (G + PACK_FACTOR - 1) / PACK_FACTOR;
  const int G_PADDED = G_PACKED * PACK_FACTOR;
  const int K_PACKED = K / PACK_FACTOR;

  float acc = 0.0f;

  for (int kb = 0; kb < K; kb += TILE_K) {
    __syncthreads();

    if (tid < TILE_K) {
      int k_idx = kb + tid;
      sh_inp[tid] = (k_idx < K) ? inp[m * K + k_idx] : T(0.0f);
    }

    for (int k_packed_offset = 0; k_packed_offset < K_PACKED_PER_TILE;
         ++k_packed_offset) {
      int global_packed_k = (kb / PACK_FACTOR) + k_packed_offset;

      if (global_packed_k < K_PACKED) {
        sh_qwt[tid][k_packed_offset] = qwt[n * K_PACKED + global_packed_k];
      } else {
        sh_qwt[tid][k_packed_offset] = 0;
      }
    }
    __syncthreads();

    for (int t = 0; t < TILE_K; ++t) {
      int current_k = kb + t;
      float iv = float(sh_inp[t]);

      int g = current_k / group_size;

      S cur_s = (g < G) ? scl[n * G_PADDED + g] : S(0.0f);

      int packed_g = g / PACK_FACTOR;
      int inner_g = g % PACK_FACTOR;
      int shift_z = inner_g * BITS;

      int32_t cur_z_packed = zos[n * G_PACKED + packed_g];

      uint32_t z = (cur_z_packed >> shift_z) & 0xF;

      int packed_k_in_tile = t / PACK_FACTOR;
      int32_t pw = sh_qwt[tid][packed_k_in_tile];

      int inner_k = current_k % PACK_FACTOR;
      int shift_w = inner_k * BITS;
      uint32_t w = (pw >> shift_w) & 0xF;

      acc = __fmaf_rn(iv, (float(w) - float(z)) * float(cur_s), acc);
    }
  }

  if (bias) {
    acc += float(bias[n]);
  }
  out[m * N + n] = T(acc);
}

template <typename T>
void matmul_quantized_gemv(const Tensor<T>& input,
                           const Tensor<int32_t>& qweight,
                           const Tensor<float>& scales,
                           const Tensor<int32_t>& zeros, int group_size,
                           Tensor<T>* output, cudaStream_t stream,
                           const Tensor<T>* bias) {
  if (input.sizes().size() != 2) throw std::runtime_error("Input must be 2D");
  int M = input.sizes()[0];
  int K = input.sizes()[1];

  if (qweight.sizes().size() != 2)
    throw std::runtime_error("QWeight must be 2D");
  int N = qweight.sizes()[0];
  int K_PACKED_w = qweight.sizes()[1];

  if (scales.sizes().size() != 2) throw std::runtime_error("Scales must be 2D");
  if (scales.sizes()[0] != N)
    throw std::runtime_error("Scales N dimension mismatch");
  int G_PADDED_s = scales.sizes()[1];

  if (zeros.sizes().size() != 2) throw std::runtime_error("Zeros must be 2D");
  if (zeros.sizes()[0] != N)
    throw std::runtime_error("Zeros N dimension mismatch");
  int G_PACKED_z = zeros.sizes()[1];

  if (bias && (bias->sizes().size() != 1 || bias->sizes()[0] != N)) {
    throw std::runtime_error("Bias must be 1D with size N (" +
                             std::to_string(N) + ")");
  }

  if (group_size <= 0) {
    throw std::runtime_error("group_size must be positive");
  }
  if (K % group_size != 0) {
    throw std::runtime_error("K (" + std::to_string(K) +
                             ") must be divisible by group_size (" +
                             std::to_string(group_size) + ")");
  }
  if (K % PACK_FACTOR != 0) {
    throw std::runtime_error("K (" + std::to_string(K) +
                             ") must be divisible by PACK_FACTOR (" +
                             std::to_string(PACK_FACTOR) + ")");
  }
  if (N % PACK_FACTOR != 0) {
    throw std::runtime_error("N (" + std::to_string(N) +
                             ") must be divisible by PACK_FACTOR (" +
                             std::to_string(PACK_FACTOR) + ")");
  }

  // int G = K / group_size;
  int K_PACKED = K / PACK_FACTOR;

  if (K_PACKED_w != K_PACKED) {
    throw std::runtime_error(
        "QWeight K/8 dimension (" + std::to_string(K_PACKED_w) +
        ") mismatch. Expected " + std::to_string(K_PACKED));
  }

  constexpr int BLOCK_N = 256;
  constexpr int TILE_K = 32;
  dim3 threads_per_block(BLOCK_N);
  dim3 num_blocks((N + BLOCK_N - 1) / BLOCK_N, M);

  using S = float;

  // #ifndef NDEBUG
  //   std::cout << "--- matmul_quantized_gemv ---" << std::endl;
  //   std::cout << "Input sizes: " << vec_to_string(input.sizes()) <<
  //   std::endl; std::cout << "QWeight sizes: " <<
  //   vec_to_string(qweight.sizes()) << std::endl; std::cout << "Scales sizes:
  //   " << vec_to_string(scales.sizes()) << std::endl; std::cout << "Zeros
  //   sizes: " << vec_to_string(zeros.sizes()) << std::endl; if (bias)
  //     std::cout << "Bias sizes: " << vec_to_string(bias->sizes()) <<
  //     std::endl;
  //   std::cout << "M=" << M << ", K=" << K << ", N=" << N
  //             << ", group_size=" << group_size << ", G=" << G
  //             << ", K/8=" << K_PACKED << ", G/8=" << G_PACKED
  //             << ", G_padded=" << G_PADDED << std::endl;
  //   std::cout << "Launching Kernel with Grid: (" << num_blocks.x << ", "
  //             << num_blocks.y << "), Block: (" << threads_per_block.x << ")"
  //             << std::endl;
  // #endif

  matmul_awq_gemv_kernel<T, S, TILE_K, BLOCK_N>
      <<<num_blocks, threads_per_block, 0, stream>>>(
          input.data_ptr(), qweight.data_ptr(), scales.data_ptr(),
          zeros.data_ptr(), output->data_ptr(), M, K, N, group_size,
          bias ? bias->data_ptr() : nullptr);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::string error_msg =
        "CUDA kernel launch failed (matmul_awq_gemv_kernel): " +
        std::string(cudaGetErrorString(err));
    error_msg += " [M=" + std::to_string(M) + ", K=" + std::to_string(K) +
                 ", N=" + std::to_string(N) +
                 ", gs=" + std::to_string(group_size) + "]";
    throw std::runtime_error(error_msg);
  }
}

template void matmul_quantized_gemv<float>(const Tensor<float>&,
                                           const Tensor<int32_t>&,
                                           const Tensor<float>&,
                                           const Tensor<int32_t>&, int,
                                           Tensor<float>*, cudaStream_t,
                                           const Tensor<float>*);

template void matmul_quantized_gemv<__nv_bfloat16>(
    const Tensor<__nv_bfloat16>&, const Tensor<int32_t>&, const Tensor<float>&,
    const Tensor<int32_t>&, int, Tensor<__nv_bfloat16>*, cudaStream_t,
    const Tensor<__nv_bfloat16>*);

template void matmul_quantized_gemv<__half>(const Tensor<__half>&,
                                            const Tensor<int32_t>&,
                                            const Tensor<float>&,
                                            const Tensor<int32_t>&, int,
                                            Tensor<__half>*, cudaStream_t,
                                            const Tensor<__half>*);

}  // namespace cuda_OP
