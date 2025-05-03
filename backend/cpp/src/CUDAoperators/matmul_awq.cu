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
  // (Debug print 实现保持不变)
  std::cout << "Tensor " << name << " " << format_sizes(tensor.sizes()) << ":"
            << std::endl;
  // ... (rest of the debug print logic)
#endif
}

// --- 常量定义 ---
constexpr int BITS = 4;
constexpr int PACK_FACTOR = 32 / BITS;  // = 8

__constant__ const int LOGICAL_TO_PHYSICAL_INNER_IDX[PACK_FACTOR] = {
    0, 4, 1, 5, 2, 6, 3, 7};

//======================================================================
// Prefill Kernel (M > 1)
// (包含 smemB/smemZ padding 和轻度 bk unroll)
//======================================================================
template <typename T, typename ScaleType, int TM = 8, int TN = 8, int BK = 32,
          int BN = 64, int BM = 64>
__global__ void matmul_awq_kernel_prefill_v1(  // 重命名以区分
    const T *__restrict__ inp,                 // 输入激活: [M, K]
    const int32_t
        *__restrict__ qwt,  // 量化权重: [K, N / PACK_FACTOR] (AWQ Order)
    const ScaleType
        *__restrict__ scl,  // 缩放因子: [NumGroups, N] (Logical Order)
    const int32_t *__restrict__ zos,  // 量化零点: [NumGroups, N / PACK_FACTOR]
                                      // (AWQ Order)
    T *__restrict__ out,  // 输出结果: [M, N]
    const int M, const int K, const int N, const int group_size,
    const T *__restrict__ bias) {
  // --- 维度和步长计算 ---
  int A_stride_0 = 1;
  int A_stride_1 = K;

  // --- Block 索引计算 ---
  int A_block_start = blockIdx.x * BM;
  int B_block_start = blockIdx.y * BN;
  constexpr int patch_N = BN / PACK_FACTOR;

  // --- Shared Memory 定义 (带 Padding) ---
  constexpr int PADDING_A = 1;
  __shared__ T smemA[BM * (BK + PADDING_A)];
  constexpr int PADDING_B = 1;
  __shared__ uint32_t smemB[BK * (patch_N + PADDING_B)];
  constexpr int PADDING_Z = 1;
  __shared__ uint32_t smemZeros[patch_N + PADDING_Z];
  constexpr int PADDING_S2D = 1;
  __shared__ ScaleType smemScales[PACK_FACTOR][patch_N + PADDING_S2D];

  // --- 向量化加载和寄存器定义 ---
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

  // --- 线程索引计算 ---
  constexpr int THREADS_N = BN / TN;
  int thread_m_in_block = threadIdx.x / THREADS_N;
  int thread_n_in_block = threadIdx.x % THREADS_N;

  // --- 外层 K 维度 Tiling 循环 ---
  for (int tile_k_base = 0; tile_k_base < K; tile_k_base += BK) {
    // --- 加载激活值 (inp -> smemA) (使用之前的优化加载逻辑) ---
#pragma unroll
    for (int load_idx = threadIdx.x; load_idx < BM * BK / vec_unit;
         load_idx += blockDim.x) {
      int load_row = load_idx / (BK / vec_unit);
      int load_col_vec = load_idx % (BK / vec_unit);
      int load_col_base = load_col_vec * vec_unit;
      int global_row = A_block_start + load_row;
      int global_col_base = tile_k_base + load_col_base;
      const T *inp_ptr =
          &inp[global_row * A_stride_1 + global_col_base * A_stride_0];
      T local_a[vec_unit];
      if (global_row < M) {
        if (global_col_base + vec_unit <= K) {
          *reinterpret_cast<Vec<T, vec_unit> *>(local_a) =
              *reinterpret_cast<const Vec<T, vec_unit> *>(inp_ptr);
        } else {
#pragma unroll
          for (int v = 0; v < vec_unit; ++v) {
            if (global_col_base + v < K) {
              local_a[v] = inp[global_row * A_stride_1 +
                               (global_col_base + v) * A_stride_0];
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

    // --- 加载量化权重 (qwt -> smemB) (带 Padding) ---
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
      smemB[load_row * (patch_N + PADDING_B) + load_col] =
          qwt_val;  // 使用 Padding 索引
    }

    // --- 加载零点 (zos -> smemZeros) (带 Padding) ---
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
      smemZeros[i] = zos_val;  // 访问索引是 i，但写入了带 Padding 的内存
    }

    // --- 加载缩放因子 (scl -> smemScales 2D) ---
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

    // --- 计算循环 ---
    ScaleType regS_unpacked[PACK_FACTOR];
#pragma unroll(4)  // 保留轻度展开
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
                : 0;  // 使用 Padding
        uint32_t regZ_packed = (n_packed_local < patch_N)
                                   ? smemZeros[n_packed_local]
                                   : 0;  // 使用 Padding
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
        }  // end j_inner
      }  // end j_packed_offset
    }  // end bk
    __syncthreads();
  }  // end tile_k_base

// --- 写回结果 (out) ---
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

//======================================================================
// Decode/GEMV Kernel (M = 1) - v2: 更大线程块，更合理的工作分配
//======================================================================
template <typename T, typename ScaleType,
          int BK_GEMV = 32,  // K-tile size (影响 smemA 大小)
          int BN_GEMV = 128,  // Block 处理的 N 范围 (选择 128，配合 Block Size)
          int THREADS_PER_BLOCK_GEMV = 128>  // 线程块大小 (选择 128)
__global__ void matmul_awq_gemv_kernel_v2(   // 重命名
    const T *__restrict__ inp,               // 输入激活: [K] (逻辑上)
    const int32_t *__restrict__ qwt,    // 量化权重: [K, N / PACK_FACTOR]
    const ScaleType *__restrict__ scl,  // 缩放因子: [NumGroups, N]
    const int32_t *__restrict__ zos,  // 量化零点: [NumGroups, N / PACK_FACTOR]
    T *__restrict__ out,              // 输出结果: [N] (逻辑上)
    const int K, const int N, const int group_size,
    const T *__restrict__ bias) {  // Bias: [N]

  // --- Block 索引和常量 ---
  int B_block_start = blockIdx.y * BN_GEMV;  // Block 在 N 维度的起始位置
  constexpr int patch_N =
      BN_GEMV / PACK_FACTOR;  // N 维度 packed 向量数 per Block
  static_assert(BN_GEMV % PACK_FACTOR == 0,
                "GEMV BN must be multiple of PACK_FACTOR");
  static_assert(THREADS_PER_BLOCK_GEMV >= patch_N,
                "Need enough threads to cover N range");  // 基本要求

  // --- Shared Memory 定义 ---
  constexpr int PADDING_A_GEMV = 1;
  __shared__ T smemA[BK_GEMV + PADDING_A_GEMV];  // 存储当前 K tile 的输入

  constexpr int PADDING_B_GEMV = 1;
  __shared__ uint32_t smemB[BK_GEMV * (patch_N + PADDING_B_GEMV)];  // 存储 B
  constexpr int PADDING_Z_GEMV = 1;
  __shared__ uint32_t smemZeros[patch_N + PADDING_Z_GEMV];  // 存储 Z
  constexpr int PADDING_S2D_GEMV = 1;
  __shared__ ScaleType
      smemScales[PACK_FACTOR][patch_N + PADDING_S2D_GEMV];  // 存储 S

  // --- 累加器 ---
  // 每个线程负责计算 BN_GEMV / THREADS_PER_BLOCK_GEMV 个输出元素
  // 在这个配置下 (128/128 = 1)，每个线程负责 1 个输出元素
  constexpr int N_PER_THREAD = BN_GEMV / THREADS_PER_BLOCK_GEMV;
  static_assert(N_PER_THREAD == 1,
                "This kernel version assumes 1 N per thread");
  float acc = 0.0f;  // 每个线程一个累加器

  // --- 线程索引 ---
  int tid = threadIdx.x;  // 线程在 Block 内的 ID (0 to 127)
  int n_local = tid;      // 线程负责的 N 索引 (在 Block 内, 0 to 127)

  // --- 外层 K 维度 Tiling 循环 ---
  for (int tile_k_base = 0; tile_k_base < K; tile_k_base += BK_GEMV) {
    // --- 加载激活值 (inp[K] -> smemA[BK]) ---
    // 所有 128 个线程协作加载
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

    // --- 加载 B, Z, S (所有 128 个线程协作) ---
    // 计算加载范围，注意 patch_N = 128/8 = 16
    // 加载 smemB
#pragma unroll
    for (int load_idx = tid; load_idx < BK_GEMV * patch_N;
         load_idx += THREADS_PER_BLOCK_GEMV) {
      int load_row = load_idx / patch_N;  // k (0..BK-1)
      int load_col = load_idx % patch_N;  // n_packed (0..patch_N-1 = 0..15)
      int global_n_packed = B_block_start / PACK_FACTOR + load_col;
      int k_global = tile_k_base + load_row;
      uint32_t qwt_val = 0;
      if (k_global < K && global_n_packed < (N / PACK_FACTOR)) {
        qwt_val = qwt[k_global * (N / PACK_FACTOR) + global_n_packed];
      }
      smemB[load_row * (patch_N + PADDING_B_GEMV) + load_col] = qwt_val;
    }
    // 加载 smemZeros
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
    // 加载 smemScales (2D)
#pragma unroll
    for (int i = tid; i < BN_GEMV; i += THREADS_PER_BLOCK_GEMV) {
      int n_global = B_block_start + i;
      ScaleType scl_val = static_cast<ScaleType>(0.0f);
      if (current_tile_group_idx < num_groups && n_global < N) {
        scl_val = scl[current_tile_group_idx * N + n_global];
      }
      int scale_row = i % PACK_FACTOR;
      int scale_col = i / PACK_FACTOR;  // i/8
      if (scale_col < patch_N) {        // scale_col < 16
        smemScales[scale_row][scale_col] = scl_val;
      }
    }

    __syncthreads();  // 等待加载完成

// --- 计算循环 ---
// 遍历当前 K tile (BK_GEMV)
#pragma unroll(4)
    for (int bk = 0; bk < BK_GEMV; ++bk) {
      // 读取当前激活值 (所有线程相同)
      float inp_val = static_cast<float>(smemA[bk]);

      // 每个线程计算自己负责的 n_local (0..127)
      // 计算对应的 packed 索引 (0..15)
      int n_packed_local = n_local / PACK_FACTOR;  // = tid / 8

      // 读取对应的 B 和 Z
      uint32_t regB_packed =
          (n_packed_local < patch_N)
              ? smemB[bk * (patch_N + PADDING_B_GEMV) + n_packed_local]
              : 0;
      uint32_t regZ_packed =
          (n_packed_local < patch_N) ? smemZeros[n_packed_local] : 0;

      // 读取对应的 Scale
      // j_inner = n_local % PACK_FACTOR = tid % 8
      int j_inner = n_local % PACK_FACTOR;
      ScaleType regS = (n_packed_local < patch_N)
                           ? smemScales[j_inner][n_packed_local]
                           : static_cast<ScaleType>(0.0f);

      // 反量化 (只计算当前线程需要的那个权重)
      int physical_inner_col_idx = LOGICAL_TO_PHYSICAL_INNER_IDX[j_inner];
      int bit_shift = physical_inner_col_idx * BITS;
      uint32_t q_w = (regB_packed >> bit_shift) & 0x0F;
      uint32_t q_z = (regZ_packed >> bit_shift) & 0x0F;
      float scale_float = static_cast<float>(regS);
      float dequant_w =
          (static_cast<float>(q_w) - static_cast<float>(q_z)) * scale_float;

      // FMA 累加
      acc = __fmaf_rn(inp_val, dequant_w, acc);

    }  // end bk loop

    __syncthreads();  // 等待 K tile 计算完成
  }  // end tile_k_base loop

  // --- 写回结果 (out) ---
  int n_global = B_block_start + n_local;  // n_local is tid
  if (n_global < N) {
    float final_val = acc;
    if (bias) {
      final_val += static_cast<float>(bias[n_global]);
    }
    out[n_global] = static_cast<T>(final_val);
  }
}

//======================================================================
// Host Function - 根据 M 选择 Kernel
//======================================================================
template <typename T>
void matmul_quantized(const Tensor<T> &input, const Tensor<int32_t> &qweight,
                      const Tensor<float> &scales_input,
                      const Tensor<int32_t> &zeros_input, int group_size,
                      Tensor<T> *output, cudaStream_t stream,
                      const Tensor<T> *bias) {
  // --- 维度检查与获取 (保持不变) ---
  int M = static_cast<int>(input.sizes()[0]);
  int K = static_cast<int>(input.sizes()[1]);
  int N = 0;  // 推断 N 的逻辑保持不变
  if (scales_input.sizes().size() == 2)
    N = static_cast<int>(scales_input.sizes()[1]);
  else if (output->sizes().size() == 2)
    N = static_cast<int>(output->sizes()[1]);
  else if (qweight.sizes().size() == 2)
    N = static_cast<int>(qweight.sizes()[1]) * PACK_FACTOR;
  else
    throw std::runtime_error("无法确定维度 N");

  // 其他输入形状检查 (保持不变)
  if (K <= 0 || N <= 0 || M <= 0 || group_size <= 0 || K % group_size != 0)
    throw std::runtime_error("无效的维度或 group_size. M=" + std::to_string(M) +
                             ", K=" + std::to_string(K) +
                             ", N=" + std::to_string(N) +
                             ", group_size=" + std::to_string(group_size));
  int NumGroups = K / group_size;
  // ... (其他形状检查)

  // --- Kernel Launch ---
  using ScaleType = float;

  if (M == 1) {
    // --- Decode/GEMV 分支 (M=1) ---
    constexpr int BK_GEMV = 32;                  // K tile size
    constexpr int BN_GEMV = 128;                 // Block 处理的 N 范围
    constexpr int THREADS_PER_BLOCK_GEMV = 128;  // 每个 Block 的线程数

    dim3 block_decode(THREADS_PER_BLOCK_GEMV);
    dim3 grid_decode(1, (N + BN_GEMV - 1) / BN_GEMV);  // Grid 在 N 维度划分

#ifdef DEBUG_AWQ
    std::cout << "Launching GEMV Kernel v2 (M=1)" << std::endl;
    std::cout << "Grid: (" << grid_decode.x << ", " << grid_decode.y
              << "), Block: (" << block_decode.x << ")" << std::endl;
    // ...
#endif

    // 调用新的 GEMV Kernel
    matmul_awq_gemv_kernel_v2<T, ScaleType, BK_GEMV, BN_GEMV,
                              THREADS_PER_BLOCK_GEMV>
        <<<grid_decode, block_decode, 0, stream>>>(
            input.data_ptr(),  // [K]
            qweight.data_ptr(), scales_input.data_ptr(), zeros_input.data_ptr(),
            output->data_ptr(),  // [N]
            K, N, group_size, bias ? bias->data_ptr() : nullptr);

  } else {
    // --- Prefill 分支 (M > 1) ---
    constexpr int TM_PREFILL = 8;
    constexpr int TN_PREFILL = 8;
    constexpr int BK_PREFILL = 32;
    constexpr int BN_PREFILL = 64;  // 可以根据 Prefill 性能调整
    constexpr int BM_PREFILL = 64;  // 可以根据 Prefill 性能调整
    // 推荐使用较大的线程块，例如 128 或 256，如果寄存器允许
    constexpr int THREADS_PER_BLOCK_PREFILL = 128;  // 或者 256
    // 需要确保 THREADS_PER_BLOCK_PREFILL == (BN_PREFILL / TN_PREFILL) *
    // (BM_PREFILL / TM_PREFILL) 如果用 128 线程:
    //  - BN=64, TN=8 => 8 threads for N
    //  - BM=128, TM=8 => 16 threads for M. 8 * 16 = 128.
    //  - 或者 BN=128, TN=8 => 16 threads for N
    //  -        BM=64, TM=8 => 8 threads for M. 16 * 8 = 128.
    // 让我们调整一下 Prefill 参数以匹配 128 线程块:
    // constexpr int BN_PREFILL = 128; // 处理更大的 N tile
    // constexpr int BM_PREFILL = 64;
    // constexpr int THREADS_PER_BLOCK_PREFILL = (128/8)*(64/8) = 16 * 8 = 128;

    // 保持原始参数 (64x64) 对应的 64
    // 线程块，或者如果用户之前用的是256，则保持256
    // 为了保持“意外变快”的版本，我们假设它基于原始 Tiling 但用了更大的线程块
    // 或者就用原始的 64 线程配置，如果那确实更快
    constexpr int THREADS_PER_BLOCK_PREFILL_ORIG =
        (BN_PREFILL / TN_PREFILL) * (BM_PREFILL / TM_PREFILL);  // 64

    dim3 block_prefill(THREADS_PER_BLOCK_PREFILL_ORIG);  // 使用 64 线程配置
    dim3 grid_prefill((M + BM_PREFILL - 1) / BM_PREFILL,
                      (N + BN_PREFILL - 1) / BN_PREFILL);

#ifdef DEBUG_AWQ
    std::cout << "Launching Prefill Kernel v1 (M=" << M << ")" << std::endl;
    std::cout << "Grid: (" << grid_prefill.x << ", " << grid_prefill.y
              << "), Block: (" << block_prefill.x << ")" << std::endl;
    // ...
#endif

    // 调用保留的 Prefill Kernel
    matmul_awq_kernel_prefill_v1<T, ScaleType, TM_PREFILL, TN_PREFILL,
                                 BK_PREFILL, BN_PREFILL, BM_PREFILL>
        <<<grid_prefill, block_prefill, 0, stream>>>(
            input.data_ptr(), qweight.data_ptr(), scales_input.data_ptr(),
            zeros_input.data_ptr(), output->data_ptr(), M, K, N, group_size,
            bias ? bias->data_ptr() : nullptr);
  }

  // --- Kernel 启动错误检查 ---
  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    std::cerr << "CUDA kernel 启动错误: " << cudaGetErrorString(launch_err)
              << std::endl;
    throw std::runtime_error("CUDA kernel 启动错误");
  }
}

// --- 模板实例化 (保持不变) ---
// ... (instantiations for float, bfloat16, half) ...
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