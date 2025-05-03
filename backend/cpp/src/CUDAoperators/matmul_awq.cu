#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cudaOP.cuh"

namespace cuda_OP {

inline std::string format_sizes(const std::vector<size_t> &sizes) {
  std::stringstream ss;
  ss << "(";
  for (size_t i = 0; i < sizes.size(); ++i) {
    ss << sizes[i] << (i == sizes.size() - 1 ? "" : ", ");
  }
  ss << ")";
  return ss.str();
}

template <typename T>
void debug_print_tensor(const Tensor<T> &tensor, const std::string &name,
                        int max_elements = 10) {
#ifdef DEBUG_AWQ
  std::cout << "Tensor " << name << " " << format_sizes(tensor.sizes()) << ":"
            << std::endl;
  std::vector<T> host_data(tensor.numel());

  cudaMemcpy(host_data.data(), tensor.data_ptr(), tensor.numel() * sizeof(T),
             cudaMemcpyDeviceToHost);
  int num_to_print = std::min(static_cast<int>(tensor.numel()), max_elements);
  std::cout << std::fixed << std::setprecision(4);
  for (int i = 0; i < num_to_print; ++i) {
    float val;
    if constexpr (std::is_same_v<T, float>) {
      val = host_data[i];
    } else if constexpr (std::is_same_v<T, __half>) {
      val = __half2float(host_data[i]);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      val = __bfloat162float(host_data[i]);
    } else {
      val = static_cast<float>(host_data[i]);
    }
    std::cout << "  [" << i << "] = " << val << std::endl;
  }
  if (tensor.numel() > max_elements) {
    std::cout << "  ... (还有 " << tensor.numel() - max_elements << " 个元素)"
              << std::endl;
  }
  std::cout << std::defaultfloat << std::setprecision(6);
#endif
}

constexpr int BITS = 4;
constexpr int PACK_FACTOR = 32 / BITS;

__constant__ const int LOGICAL_TO_PHYSICAL_INNER_IDX[PACK_FACTOR] = {
    0, 4, 1, 5, 2, 6, 3, 7};

template <typename T, typename ScaleType, int BK_GEMV = 32, int BN_GEMV = 128,
          int THREADS_PER_BLOCK_GEMV = 128>
__global__ void matmul_awq_gemv_kernel_v2(const T *__restrict__ inp,
                                          const int32_t *__restrict__ qwt,
                                          const ScaleType *__restrict__ scl,
                                          const int32_t *__restrict__ zos,
                                          T *__restrict__ out, const int K,
                                          const int N, const int group_size,
                                          const T *__restrict__ bias) {
  int B_block_start = blockIdx.y * BN_GEMV;
  constexpr int patch_N = BN_GEMV / PACK_FACTOR;
  static_assert(BN_GEMV % PACK_FACTOR == 0,
                "GEMV BN must be multiple of PACK_FACTOR");
  static_assert(THREADS_PER_BLOCK_GEMV >= (BN_GEMV / PACK_FACTOR),
                "Need enough threads for packed N");

  constexpr int PADDING_A_GEMV = 1;
  __shared__ T smemA[BK_GEMV + PADDING_A_GEMV];

  constexpr int PADDING_B_GEMV = 0;
  __shared__ uint32_t smemB[BK_GEMV * (patch_N + PADDING_B_GEMV)];

  constexpr int PADDING_Z_GEMV = 0;
  __shared__ uint32_t smemZeros[patch_N + PADDING_Z_GEMV];

  constexpr int PADDING_S2D_GEMV = 1;
  __shared__ ScaleType smemScales[PACK_FACTOR][patch_N + PADDING_S2D_GEMV];

  float acc = 0.0f;

  int tid = threadIdx.x;
  int n_local = tid;

  for (int tile_k_base = 0; tile_k_base < K; tile_k_base += BK_GEMV) {
#pragma unroll
    for (int k_offset = tid; k_offset < BK_GEMV;
         k_offset += THREADS_PER_BLOCK_GEMV) {
      int k_global = tile_k_base + k_offset;
      if (k_global < K) {
        smemA[k_offset] = inp[k_global];
      } else {
        smemA[k_offset] = static_cast<T>(0.0f);
      }
    }

#pragma unroll
    for (int load_idx = tid; load_idx < BK_GEMV * patch_N;
         load_idx += THREADS_PER_BLOCK_GEMV) {
      int load_row = load_idx / patch_N;
      int load_col = load_idx % patch_N;
      int global_n_packed = B_block_start / PACK_FACTOR + load_col;
      int k_global = tile_k_base + load_row;
      uint32_t qwt_val = 0;
      if (k_global < K && global_n_packed < (N / PACK_FACTOR)) {
        qwt_val = qwt[k_global * (N / PACK_FACTOR) + global_n_packed];
      }
      smemB[load_row * patch_N + load_col] = qwt_val;
    }
    int current_tile_group_idx = tile_k_base / group_size;
    const int num_groups = (K + group_size - 1) / group_size;
#pragma unroll
    for (int i = tid; i < patch_N; i += THREADS_PER_BLOCK_GEMV) {
      int n_packed_global = B_block_start / PACK_FACTOR + i;
      uint32_t zos_val = 0;
      if (current_tile_group_idx < num_groups &&
          n_packed_global < (N / PACK_FACTOR)) {
        zos_val =
            zos[current_tile_group_idx * (N / PACK_FACTOR) + n_packed_global];
      }
      smemZeros[i] = zos_val;
    }
#pragma unroll
    for (int i = tid; i < BN_GEMV; i += THREADS_PER_BLOCK_GEMV) {
      int n_global = B_block_start + i;
      ScaleType scl_val = static_cast<ScaleType>(0.0f);
      if (current_tile_group_idx < num_groups && n_global < N) {
        scl_val = scl[current_tile_group_idx * N + n_global];
      }
      int scale_row = i % PACK_FACTOR;
      int scale_col = i / PACK_FACTOR;
      if (scale_col < patch_N) {
        smemScales[scale_row][scale_col] = scl_val;
      }
    }

    __syncthreads();
#pragma unroll(4)
    for (int bk = 0; bk < BK_GEMV; ++bk) {
      float inp_val = static_cast<float>(smemA[bk]);

      int n_packed_local = n_local / PACK_FACTOR;
      int j_inner = n_local % PACK_FACTOR;
      // if (n_packed_local < patch_N) {
      //   uint32_t regB_packed = smemB[bk * patch_N + n_packed_local];
      //   uint32_t regZ_packed = smemZeros[n_packed_local];
      //   ScaleType regS = smemScales[j_inner][n_packed_local];

      //   int physical_inner_col_idx = LOGICAL_TO_PHYSICAL_INNER_IDX[j_inner];
      //   int bit_shift = physical_inner_col_idx * BITS;
      //   uint32_t q_w = (regB_packed >> bit_shift) & 0x0F;
      //   uint32_t q_z = (regZ_packed >> bit_shift) & 0x0F;

      //   float scale_float = static_cast<float>(regS);
      //   float dequant_w =
      //       (static_cast<float>(q_w) - static_cast<float>(q_z)) *
      //       scale_float;

      //   acc = __fmaf_rn(inp_val, dequant_w, acc);
      // }

      uint32_t regB_packed =
          (n_packed_local < patch_N)
              ? smemB[bk * (patch_N + PADDING_B_GEMV) + n_packed_local]
              : 0;
      uint32_t regZ_packed =
          (n_packed_local < patch_N) ? smemZeros[n_packed_local] : 0;

      ScaleType regS = (n_packed_local < patch_N)
                           ? smemScales[j_inner][n_packed_local]
                           : static_cast<ScaleType>(0.0f);

      int physical_inner_col_idx = LOGICAL_TO_PHYSICAL_INNER_IDX[j_inner];
      int bit_shift = physical_inner_col_idx * BITS;
      uint32_t q_w = (regB_packed >> bit_shift) & 0x0F;
      uint32_t q_z = (regZ_packed >> bit_shift) & 0x0F;
      float scale_float = static_cast<float>(regS);
      float dequant_w =
          (static_cast<float>(q_w) - static_cast<float>(q_z)) * scale_float;

      acc = __fmaf_rn(inp_val, dequant_w, acc);
    }
  }
  __syncthreads();

  int n_global = B_block_start + n_local;
  if (n_global < N) {
    float final_val = acc;
    if (bias) {
      final_val += static_cast<float>(bias[n_global]);
    }
    out[n_global] = static_cast<T>(final_val);
  }
}

template <typename T, typename ScaleType, int BM = 64, int BN = 64, int BK = 32,
          int WMMA_M = 16, int WMMA_N = 16, int WMMA_K = 16>
__global__ void matmul_awq_kernel_prefill_wmma_v1(
    const T *__restrict__ inp, const int32_t *__restrict__ qwt,
    const ScaleType *__restrict__ scl, const int32_t *__restrict__ zos,
    T *__restrict__ out, const int M, const int K, const int N,
    const int group_size, const T *__restrict__ bias) {
  static_assert(
      std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>,
      "WMMA kernel currently supports only half or bfloat16 inputs/outputs.");

  static_assert(BM % WMMA_M == 0, "BM must be a multiple of WMMA_M");
  static_assert(BN % WMMA_N == 0, "BN must be a multiple of WMMA_N");
  static_assert(BK % WMMA_K == 0, "BK must be a multiple of WMMA_K");

  using namespace nvcuda;
  using FragmentA = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T,
                                   wmma::row_major>;
  using FragmentB = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T,
                                   wmma::col_major>;
  using FragmentC =
      wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

  constexpr int SMEM_A_PADDING = 8;
  __shared__ T smemA[BM][BK + SMEM_A_PADDING];

  constexpr int SMEM_B_PADDING = 8;
  __shared__ T smemB_dequant[BN][BK + SMEM_B_PADDING];

  constexpr int patch_N = BN / PACK_FACTOR;
  __shared__ uint32_t smemB_packed[BK * patch_N];
  __shared__ uint32_t smemZeros[patch_N];
  __shared__ ScaleType smemScales[BN];

  __shared__ float smem_out[BM][BN];

  const int A_block_start = blockIdx.x * BM;
  const int B_block_start = blockIdx.y * BN;

  const int warp_id = threadIdx.x / 32;
  // const int lane_id = threadIdx.x % 32;

  // const int warps_per_block_m = BM / WMMA_M;
  const int warps_per_block_n = BN / WMMA_N;

  const int warp_m_id = warp_id / warps_per_block_n;
  const int warp_n_id = warp_id % warps_per_block_n;

  FragmentC fragC;
  wmma::fill_fragment(fragC, 0.0f);

  for (int tile_k_base = 0; tile_k_base < K; tile_k_base += BK) {
#pragma unroll
    for (int load_idx = threadIdx.x; load_idx < BM * BK;
         load_idx += blockDim.x) {
      int load_row = load_idx / BK;
      int load_col = load_idx % BK;
      int global_row = A_block_start + load_row;
      int global_col = tile_k_base + load_col;

      if (global_row < M && global_col < K) {
        smemA[load_row][load_col] = inp[global_row * K + global_col];
      } else {
        smemA[load_row][load_col] = static_cast<T>(0.0f);
      }
    }

#pragma unroll
    for (int load_idx = threadIdx.x; load_idx < BK * patch_N;
         load_idx += blockDim.x) {
      int load_row_k = load_idx / patch_N;
      int load_col_n_packed = load_idx % patch_N;
      int global_n_packed = B_block_start / PACK_FACTOR + load_col_n_packed;
      int k_global = tile_k_base + load_row_k;

      uint32_t qwt_val = 0;
      if (k_global < K && global_n_packed < (N / PACK_FACTOR)) {
        qwt_val = qwt[k_global * (N / PACK_FACTOR) + global_n_packed];
      }

      smemB_packed[load_row_k * patch_N + load_col_n_packed] = qwt_val;
    }

    int current_tile_group_idx = tile_k_base / group_size;
    const int num_groups = (K + group_size - 1) / group_size;

#pragma unroll
    for (int i = threadIdx.x; i < patch_N; i += blockDim.x) {
      int n_packed_global = B_block_start / PACK_FACTOR + i;
      uint32_t zos_val = 0;
      if (current_tile_group_idx < num_groups &&
          n_packed_global < (N / PACK_FACTOR)) {
        zos_val =
            zos[current_tile_group_idx * (N / PACK_FACTOR) + n_packed_global];
      }
      smemZeros[i] = zos_val;
    }

#pragma unroll
    for (int i = threadIdx.x; i < BN; i += blockDim.x) {
      int n_global = B_block_start + i;
      ScaleType scl_val = static_cast<ScaleType>(0.0f);
      if (current_tile_group_idx < num_groups && n_global < N) {
        scl_val = scl[current_tile_group_idx * N + n_global];
      }
      smemScales[i] = scl_val;
    }

    __syncthreads();

#pragma unroll
    for (int dq_idx = threadIdx.x; dq_idx < BN * BK; dq_idx += blockDim.x) {
      int n_local = dq_idx / BK;
      int k_local = dq_idx % BK;

      int n_packed_local = n_local / PACK_FACTOR;
      int n_inner = n_local % PACK_FACTOR;

      uint32_t qwt_packed = smemB_packed[k_local * patch_N + n_packed_local];
      uint32_t zos_packed = smemZeros[n_packed_local];

      ScaleType scale_val = smemScales[n_local];

      int physical_inner_col_idx = LOGICAL_TO_PHYSICAL_INNER_IDX[n_inner];
      int bit_shift = physical_inner_col_idx * BITS;

      uint32_t q_w = (qwt_packed >> bit_shift) & 0x0F;
      uint32_t q_z = (zos_packed >> bit_shift) & 0x0F;

      float scale_float = static_cast<float>(scale_val);
      float dequant_w =
          (static_cast<float>(q_w) - static_cast<float>(q_z)) * scale_float;

      smemB_dequant[n_local][k_local] = static_cast<T>(dequant_w);
    }

    __syncthreads();

#pragma unroll
    for (int k_step = 0; k_step < BK; k_step += WMMA_K) {
      FragmentA fragA;
      FragmentB fragB;

      const T *smemA_ptr = &smemA[warp_m_id * WMMA_M][k_step];
      wmma::load_matrix_sync(fragA, smemA_ptr, BK + SMEM_A_PADDING);

      const T *smemB_ptr = &smemB_dequant[warp_n_id * WMMA_N][k_step];
      wmma::load_matrix_sync(fragB, smemB_ptr, BK + SMEM_B_PADDING);

      wmma::mma_sync(fragC, fragA, fragB, fragC);
    }

    __syncthreads();
  }

  const int out_m_base = warp_m_id * WMMA_M;
  const int out_n_base = warp_n_id * WMMA_N;

  wmma::store_matrix_sync(&smem_out[out_m_base][out_n_base], fragC, BN,
                          wmma::mem_row_major);

  __syncthreads();

#pragma unroll
  for (int write_idx = threadIdx.x; write_idx < BM * BN;
       write_idx += blockDim.x) {
    int m_local = write_idx / BN;
    int n_local = write_idx % BN;

    int m_global = A_block_start + m_local;
    int n_global = B_block_start + n_local;

    if (m_global < M && n_global < N) {
      float result = smem_out[m_local][n_local];
      if (bias) {
        result += static_cast<float>(bias[n_global]);
      }

      out[m_global * N + n_global] = static_cast<T>(result);
    }
  }
}

template <typename T>
void matmul_quantized(const Tensor<T> &input, const Tensor<int32_t> &qweight,
                      const Tensor<float> &scales_input,
                      const Tensor<int32_t> &zeros_input, int group_size,
                      Tensor<T> *output, cudaStream_t stream,
                      const Tensor<T> *bias) {
  int M = static_cast<int>(input.sizes()[0]);
  int K = static_cast<int>(input.sizes()[1]);
  int N = 0;

  if (output->sizes().size() >= 2) {
    N = static_cast<int>(output->sizes().back());
  } else if (scales_input.sizes().size() == 2) {
    N = static_cast<int>(scales_input.sizes()[1]);
  } else if (qweight.sizes().size() == 2) {
    N = static_cast<int>(qweight.sizes()[1]) * PACK_FACTOR;
  } else if (bias && bias->sizes().size() >= 1) {
    N = static_cast<int>(bias->sizes().back());
  } else {
    throw std::runtime_error(
        "Cannot determine dimension N from provided tensors (output, scales, "
        "qweight, bias).");
  }

  if (K != static_cast<int>(qweight.sizes()[0])) {
    throw std::runtime_error("Dimension K mismatch between input (" +
                             std::to_string(K) + ") and qweight (" +
                             std::to_string(qweight.sizes()[0]) + ")");
  }
  int packed_N_dim = (N + PACK_FACTOR - 1) / PACK_FACTOR;
  if (static_cast<int>(qweight.sizes()[1]) != packed_N_dim) {
    throw std::runtime_error(
        "Dimension N/PACK_FACTOR mismatch between inferred N (" +
        std::to_string(N) + " -> " + std::to_string(packed_N_dim) +
        ") and qweight (" + std::to_string(qweight.sizes()[1]) + ")");
  }

  using ScaleType = float;

  cudaError_t launch_err = cudaSuccess;

  if (M == 1) {
    constexpr int BK_GEMV = 32;
    constexpr int BN_GEMV = 128;
    constexpr int THREADS_PER_BLOCK_GEMV = 128;

    dim3 block_decode(THREADS_PER_BLOCK_GEMV);
    dim3 grid_decode(1, (N + BN_GEMV - 1) / BN_GEMV);

    matmul_awq_gemv_kernel_v2<T, ScaleType, BK_GEMV, BN_GEMV,
                              THREADS_PER_BLOCK_GEMV>
        <<<grid_decode, block_decode, 0, stream>>>(
            input.data_ptr(), qweight.data_ptr(), scales_input.data_ptr(),
            zeros_input.data_ptr(), output->data_ptr(), K, N, group_size,
            bias ? bias->data_ptr() : nullptr);
    launch_err = cudaGetLastError();

  } else {
    if constexpr (std::is_same_v<T, __half> ||
                  std::is_same_v<T, __nv_bfloat16>) {
      constexpr int BM_WMMA = 64;
      constexpr int BN_WMMA = 64;
      constexpr int BK_WMMA = 32;
      constexpr int WMMA_M = 16;
      constexpr int WMMA_N = 16;
      constexpr int WMMA_K = 16;

      constexpr int WARPS_M = BM_WMMA / WMMA_M;
      constexpr int WARPS_N = BN_WMMA / WMMA_N;
      constexpr int THREADS_PER_BLOCK_WMMA = WARPS_M * WARPS_N * 32;

      dim3 block_wmma(THREADS_PER_BLOCK_WMMA);
      dim3 grid_wmma((M + BM_WMMA - 1) / BM_WMMA, (N + BN_WMMA - 1) / BN_WMMA);

      // #ifdef DEBUG_AWQ
      //       std::cout << "Launching Prefill WMMA Kernel v1 (M=" << M << ")"
      //                 << std::endl;
      //       std::cout << "Using BM=" << BM_WMMA << ", BN=" << BN_WMMA
      //                 << ", BK=" << BK_WMMA << std::endl;
      //       std::cout << "Grid: (" << grid_wmma.x << ", " << grid_wmma.y <<
      //       ", "
      //                 << grid_wmma.z << "), Block: (" << block_wmma.x << ", "
      //                 << block_wmma.y << ", " << block_wmma.z << ")" <<
      //                 std::endl;
      //       std::cout << "M=" << M << ", K=" << K << ", N=" << N
      //                 << ", group_size=" << group_size << std::endl;
      //       debug_print_tensor(input, "input (WMMA)");
      //       debug_print_tensor(qweight, "qweight (WMMA)");
      //       debug_print_tensor(scales_input, "scales (WMMA)");
      //       debug_print_tensor(zeros_input, "zeros (WMMA)");
      //       if (bias) debug_print_tensor(*bias, "bias (WMMA)");
      // #endif

      matmul_awq_kernel_prefill_wmma_v1<T, ScaleType, BM_WMMA, BN_WMMA, BK_WMMA,
                                        WMMA_M, WMMA_N, WMMA_K>
          <<<grid_wmma, block_wmma, 0, stream>>>(
              input.data_ptr(), qweight.data_ptr(), scales_input.data_ptr(),
              zeros_input.data_ptr(), output->data_ptr(), M, K, N, group_size,
              bias ? bias->data_ptr() : nullptr);
      launch_err = cudaGetLastError();

    } else {
      std::cerr << "Warning: matmul_quantized WMMA kernel does not support "
                   "input type "
                << typeid(T).name()
                << ". Prefill stage (M > 1) will not execute." << std::endl;

      throw std::runtime_error(
          "matmul_quantized WMMA kernel currently only supports half/bfloat16 "
          "input/output types for M > 1.");
    }
  }

  if (launch_err != cudaSuccess) {
    std::cerr << "CUDA kernel launch error in matmul_quantized: "
              << cudaGetErrorString(launch_err) << std::endl;

    throw std::runtime_error("CUDA kernel launch failed in matmul_quantized");
  }
}

template void matmul_quantized<float>(const Tensor<float> &,
                                      const Tensor<int32_t> &,
                                      const Tensor<float> &,
                                      const Tensor<int32_t> &, int,
                                      Tensor<float> *, cudaStream_t,
                                      const Tensor<float> *);
template void matmul_quantized<__nv_bfloat16>(
    const Tensor<__nv_bfloat16> &, const Tensor<int32_t> &,
    const Tensor<float> &, const Tensor<int32_t> &, int,
    Tensor<__nv_bfloat16> *, cudaStream_t, const Tensor<__nv_bfloat16> *);
template void matmul_quantized<__half>(const Tensor<__half> &,
                                       const Tensor<int32_t> &,
                                       const Tensor<float> &,
                                       const Tensor<int32_t> &, int,
                                       Tensor<__half> *, cudaStream_t,
                                       const Tensor<__half> *);

}  // namespace cuda_OP