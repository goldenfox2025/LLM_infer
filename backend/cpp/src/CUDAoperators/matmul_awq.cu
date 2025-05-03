#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
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

template <typename T, typename ScaleType, int TM = 8, int TN = 8, int BK = 32,
          int BN = 64, int BM = 64>
__global__ void matmul_awq_kernel_prefill_v1(
    const T *__restrict__ inp, const int32_t *__restrict__ qwt,
    const ScaleType *__restrict__ scl, const int32_t *__restrict__ zos,
    T *__restrict__ out, const int M, const int K, const int N,
    const int group_size, const T *__restrict__ bias) {
  int A_stride_1 = K;

  int A_block_start = blockIdx.x * BM;
  int B_block_start = blockIdx.y * BN;
  constexpr int patch_N = BN / PACK_FACTOR;

  constexpr int PADDING_A = 1;
  __shared__ T smemA[BM * (BK + PADDING_A)];

  constexpr int PADDING_B = 0;
  __shared__ uint32_t smemB[BK * (patch_N + PADDING_B)];

  constexpr int PADDING_Z = 0;
  __shared__ uint32_t smemZeros[patch_N + PADDING_Z];

  constexpr int PADDING_S2D = 1;
  __shared__ ScaleType smemScales[PACK_FACTOR][patch_N + PADDING_S2D];

  constexpr int vec_unit = (sizeof(T) <= 8) ? (16 / sizeof(T)) : 1;
  Vec<T, vec_unit> va;

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      acc[i][j] = 0.0f;
    }
  }

  constexpr int THREADS_N = BN / TN;
  int thread_m_in_block = threadIdx.x / THREADS_N;
  int thread_n_in_block = threadIdx.x % THREADS_N;

  for (int tile_k_base = 0; tile_k_base < K; tile_k_base += BK) {
#pragma unroll
    for (int load_idx = threadIdx.x; load_idx < BM * BK / vec_unit;
         load_idx += blockDim.x) {
      int load_row = load_idx / (BK / vec_unit);
      int load_col_vec = load_idx % (BK / vec_unit);
      int load_col_base = load_col_vec * vec_unit;
      int global_row = A_block_start + load_row;
      int global_col_base = tile_k_base + load_col_base;

      const T *inp_ptr = &inp[global_row * A_stride_1 + global_col_base];
      T local_a[vec_unit];

      if (global_row < M) {
        if (global_col_base + vec_unit <= K) {
          *reinterpret_cast<Vec<T, vec_unit> *>(local_a) =
              *reinterpret_cast<const Vec<T, vec_unit> *>(inp_ptr);
        } else {
#pragma unroll
          for (int v = 0; v < vec_unit; ++v) {
            if (global_col_base + v < K) {
              local_a[v] = inp[global_row * A_stride_1 + (global_col_base + v)];
            } else {
              local_a[v] = static_cast<T>(0.0f);
            }
          }
        }
      } else {
#pragma unroll
        for (int v = 0; v < vec_unit; ++v) {
          local_a[v] = static_cast<T>(0.0f);
        }
      }

#pragma unroll
      for (int v = 0; v < vec_unit; ++v) {
        if (load_col_base + v < BK) {
          smemA[load_row * (BK + PADDING_A) + load_col_base + v] = local_a[v];
        }
      }
    }

#pragma unroll
    for (int load_tid_idx = threadIdx.x; load_tid_idx < BK * patch_N;
         load_tid_idx += blockDim.x) {
      int load_row = load_tid_idx / patch_N;
      int load_col = load_tid_idx % patch_N;
      int global_n_packed = B_block_start / PACK_FACTOR + load_col;
      int k_global = tile_k_base + load_row;
      uint32_t qwt_val = 0;
      if (k_global < K && global_n_packed < (N / PACK_FACTOR)) {
        qwt_val = qwt[k_global * (N / PACK_FACTOR) + global_n_packed];
      }
      smemB[load_row * (patch_N + PADDING_B) + load_col] = qwt_val;
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
      int scale_row = i % PACK_FACTOR;
      int scale_col = i / PACK_FACTOR;
      if (scale_col < patch_N) {
        smemScales[scale_row][scale_col] = scl_val;
      }
    }

    __syncthreads();

    ScaleType regS_unpacked[PACK_FACTOR];

#pragma unroll(4)
    for (int bk = 0; bk < BK; ++bk) {
      T regA[TM];
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        int m_local = thread_m_in_block * TM + i;
        regA[i] = (m_local < BM) ? smemA[m_local * (BK + PADDING_A) + bk]
                                 : static_cast<T>(0.0f);
      }

#pragma unroll
      for (int j_packed_offset = 0; j_packed_offset < TN / PACK_FACTOR;
           ++j_packed_offset) {
        int n_packed_local =
            thread_n_in_block * (TN / PACK_FACTOR) + j_packed_offset;

        uint32_t regB_packed =
            (n_packed_local < patch_N)
                ? smemB[bk * (patch_N + PADDING_B) + n_packed_local]
                : 0;
        uint32_t regZ_packed =
            (n_packed_local < patch_N) ? smemZeros[n_packed_local] : 0;

#pragma unroll
        for (int j_inner_load = 0; j_inner_load < PACK_FACTOR; ++j_inner_load) {
          if (n_packed_local < patch_N) {
            regS_unpacked[j_inner_load] =
                smemScales[j_inner_load][n_packed_local];
          } else {
            regS_unpacked[j_inner_load] = static_cast<ScaleType>(0.0f);
          }
        }

#pragma unroll
        for (int j_inner = 0; j_inner < PACK_FACTOR; ++j_inner) {
          ScaleType regS = regS_unpacked[j_inner];

          int physical_inner_col_idx = LOGICAL_TO_PHYSICAL_INNER_IDX[j_inner];
          int bit_shift = physical_inner_col_idx * BITS;
          uint32_t q_w = (regB_packed >> bit_shift) & 0x0F;
          uint32_t q_z = (regZ_packed >> bit_shift) & 0x0F;
          float scale_float = static_cast<float>(regS);
          float dequant_w =
              (static_cast<float>(q_w) - static_cast<float>(q_z)) * scale_float;

          int acc_j_idx = j_packed_offset * PACK_FACTOR + j_inner;

#pragma unroll
          for (int i = 0; i < TM; ++i) {
            float inp_val = static_cast<float>(regA[i]);
            acc[i][acc_j_idx] =
                __fmaf_rn(inp_val, dequant_w, acc[i][acc_j_idx]);
          }
        }
      }
    }

    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    int m_global = A_block_start + thread_m_in_block * TM + i;
    if (m_global < M) {
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        int n_global = B_block_start + thread_n_in_block * TN + j;
        if (n_global < N) {
          float acc_val = acc[i][j];
          if (bias) {
            acc_val += static_cast<float>(bias[n_global]);
          }
          out[m_global * N + n_global] = static_cast<T>(acc_val);
        }
      }
    }
  }
}

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

  constexpr int N_PER_THREAD = BN_GEMV / THREADS_PER_BLOCK_GEMV;
  static_assert(N_PER_THREAD == 1,
                "This kernel version assumes 1 N per thread");
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
      smemB[load_row * (patch_N + PADDING_B_GEMV) + load_col] = qwt_val;
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

    __syncthreads();
  }

  int n_global = B_block_start + n_local;
  if (n_global < N) {
    float final_val = acc;
    if (bias) {
      final_val += static_cast<float>(bias[n_global]);
    }
    out[n_global] = static_cast<T>(final_val);
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
  if (scales_input.sizes().size() == 2)
    N = static_cast<int>(scales_input.sizes()[1]);
  else if (output->sizes().size() == 2)
    N = static_cast<int>(output->sizes()[1]);
  else if (qweight.sizes().size() == 2)
    N = static_cast<int>(qweight.sizes()[1]) * PACK_FACTOR;
  else
    throw std::runtime_error("无法确定维度 N");

  using ScaleType = float;

  if (M == 1) {
    constexpr int BK_GEMV = 32;
    constexpr int BN_GEMV = 128;
    constexpr int THREADS_PER_BLOCK_GEMV = 128;

    dim3 block_decode(THREADS_PER_BLOCK_GEMV);
    dim3 grid_decode(1, (N + BN_GEMV - 1) / BN_GEMV);

#ifdef DEBUG_AWQ
    std::cout << "Launching GEMV Kernel v2 (M=1)" << std::endl;
    std::cout << "Grid: (" << grid_decode.x << ", " << grid_decode.y
              << "), Block: (" << block_decode.x << ")" << std::endl;
#endif

    matmul_awq_gemv_kernel_v2<T, ScaleType, BK_GEMV, BN_GEMV,
                              THREADS_PER_BLOCK_GEMV>
        <<<grid_decode, block_decode, 0, stream>>>(
            input.data_ptr(), qweight.data_ptr(), scales_input.data_ptr(),
            zeros_input.data_ptr(), output->data_ptr(), K, N, group_size,
            bias ? bias->data_ptr() : nullptr);

  } else {
    constexpr int TM_PREFILL = 8;
    constexpr int TN_PREFILL = 8;
    constexpr int BK_PREFILL = 32;
    constexpr int BN_PREFILL = 64;
    constexpr int BM_PREFILL = 64;
    constexpr int THREADS_PER_BLOCK_PREFILL_ORIG =
        (BN_PREFILL / TN_PREFILL) * (BM_PREFILL / TM_PREFILL);

    dim3 block_prefill(THREADS_PER_BLOCK_PREFILL_ORIG);
    dim3 grid_prefill((M + BM_PREFILL - 1) / BM_PREFILL,
                      (N + BN_PREFILL - 1) / BN_PREFILL);

#ifdef DEBUG_AWQ
    std::cout << "Launching Prefill Kernel v1 (M=" << M << ")" << std::endl;
    std::cout << "Grid: (" << grid_prefill.x << ", " << grid_prefill.y
              << "), Block: (" << block_prefill.x << ")" << std::endl;
#endif

    matmul_awq_kernel_prefill_v1<T, ScaleType, TM_PREFILL, TN_PREFILL,
                                 BK_PREFILL, BN_PREFILL, BM_PREFILL>
        <<<grid_prefill, block_prefill, 0, stream>>>(
            input.data_ptr(), qweight.data_ptr(), scales_input.data_ptr(),
            zeros_input.data_ptr(), output->data_ptr(), M, K, N, group_size,
            bias ? bias->data_ptr() : nullptr);
  }

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    std::cerr << "CUDA kernel 启动错误: " << cudaGetErrorString(launch_err)
              << std::endl;
    throw std::runtime_error("CUDA kernel 启动错误");
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