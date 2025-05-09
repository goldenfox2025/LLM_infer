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
void debug_print_tensor(const Tensor<T> &tensor, const std::string &name, int max_elements = 10) {
#ifdef DEBUG_AWQ
    std::cout << "Tensor " << name << " " << format_sizes(tensor.sizes()) << ":" << std::endl;
    std::vector<T> host_data(tensor.numel());

    cudaMemcpy(host_data.data(), tensor.data_ptr(), tensor.numel() * sizeof(T), cudaMemcpyDeviceToHost);
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
        std::cout << "  ... (还有 " << tensor.numel() - max_elements << " 个元素)" << std::endl;
    }
    std::cout << std::defaultfloat << std::setprecision(6);
#endif
}

constexpr int BITS = 4;
constexpr int PACK_FACTOR = 32 / BITS;
constexpr int WARP_SIZE = 32;
// --- 用于 Warp Reduce 的辅助函数 (需要 float 版本) ---
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}
__constant__ const int LOGICAL_TO_PHYSICAL_INNER_IDX[PACK_FACTOR] = {0, 4, 1, 5, 2, 6, 3, 7};

template <typename T,           // 输入/输出类型 (half 或 __nv_bfloat16)
          typename ScaleType,   // Scale 类型 (half, __nv_bfloat16, 或 float)
          int BK_GEMV,          // K 分块大小
          int VEC_SIZE,         // K 维度向量化大小 (例如 2)
          int N_PER_WARP,       // 每个 Warp 处理的 N 维度数量 (例如 4)
          int WARPS_PER_BLOCK>  // 每个 Block 的 Warp 数量
__launch_bounds__(WARPS_PER_BLOCK *WARP_SIZE) __global__
    void awq_gemv_warp_vectorized_n(  // Renamed for clarity (n for N_PER_WARP)
        const T *__restrict__ inp, const int32_t *__restrict__ qwt, const ScaleType *__restrict__ scl,
        const int32_t *__restrict__ zos, T *__restrict__ out, const int K, const int N, const int group_size,
        const T *__restrict__ bias) {
    // --- 类型和常量定义 ---
    using TVec =
        typename std::conditional<VEC_SIZE == 2 && std::is_same<T, half>::value, half2,
                                  typename std::conditional<VEC_SIZE == 2 && std::is_same<T, __nv_bfloat16>::value,
                                                            __nv_bfloat162, T>::type>::type;

    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
    constexpr int K_PER_ITERATION = WARP_SIZE * VEC_SIZE;  // 每个 Warp 在内层循环处理的 K 数量

    // --- 块和线程索引 ---
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp_id_in_block = threadIdx.x / WARP_SIZE;  // Block 内的 Warp ID (0 to WARPS_PER_BLOCK-1)
    const int block_warp_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;  // 全局 Warp ID

    // --- N 相关索引计算 (修改) ---
    // 每个 Warp 处理 N_PER_WARP 个输出列
    const int warp_n_base = block_warp_idx * N_PER_WARP;  // 这个 Warp 负责的起始 n_global

    // 提前计算 Warp 负责的 N_PER_WARP 个列的 packed 索引和内部索引信息
    // 这些信息在反量化时需要用到
    int n_packed_global[N_PER_WARP];
    int physical_inner_col_idx[N_PER_WARP];
    int bit_shift[N_PER_WARP];
    const uint32_t dequant_mask = (1 << BITS) - 1;

#pragma unroll
    for (int n_offset = 0; n_offset < N_PER_WARP; ++n_offset) {
        int n_global = warp_n_base + n_offset;
        if (n_global < N) {
            n_packed_global[n_offset] = n_global / PACK_FACTOR;
            int j_inner = n_global % PACK_FACTOR;
            physical_inner_col_idx[n_offset] = LOGICAL_TO_PHYSICAL_INNER_IDX[j_inner];
            bit_shift[n_offset] = physical_inner_col_idx[n_offset] * BITS;
        } else {
            // 标记为无效，防止后续加载和计算
            n_packed_global[n_offset] = -1;  // Or some indicator of invalidity
            bit_shift[n_offset] = 0;         // Default value
        }
    }

    // 如果 Warp 处理的所有 N 都超出范围，则提前退出
    // 注意：这里检查的是第一个 n_global，更精确的检查可能需要看所有 n_global
    // 是否都 >= N 但通常 Warp 处理的 N 是一起有效的或无效的（除非 N 不是
    // N_PER_WARP 的倍数）
    if (warp_n_base >= N) {
        return;
    }

    // --- 共享内存 (修改) ---
    __shared__ T smemA[BK_GEMV];  // 输入 Activation (Block 共享)
    // 每个 Warp 独立拥有一块 smemB, smemS, smemZ
    __shared__ uint32_t smemB[WARPS_PER_BLOCK][N_PER_WARP][BK_GEMV];  // Packed 权重
    __shared__ ScaleType smemS[WARPS_PER_BLOCK][N_PER_WARP];          // Scales
    __shared__ uint32_t smemZ[WARPS_PER_BLOCK][N_PER_WARP];           // Packed Zeros

    // --- 累加器 (修改) ---
    float thread_sum[N_PER_WARP];
#pragma unroll
    for (int n_offset = 0; n_offset < N_PER_WARP; ++n_offset) {
        thread_sum[n_offset] = 0.0f;
    }

    // --- K 分块循环 ---
    const int num_groups = (K + group_size - 1) / group_size;
    const int N_PACKED = (N + PACK_FACTOR - 1) / PACK_FACTOR;  // Total packed columns

    for (int tile_k_base = 0; tile_k_base < K; tile_k_base += BK_GEMV) {
// --- 加载 smemA (Block 协作，不变) ---
// 使用 threadIdx.x (Block 内线程 ID) 协作加载
#pragma unroll
        for (int load_idx = threadIdx.x * VEC_SIZE; load_idx < BK_GEMV; load_idx += THREADS_PER_BLOCK * VEC_SIZE) {
            int k_tile = load_idx;  // k_tile 是 Block 内共享内存 smemA 的索引
            int k_global = tile_k_base + k_tile;

            // Check bounds for VEC_SIZE elements together
            if (k_tile + VEC_SIZE <= BK_GEMV) {  // Ensure we don't write past smemA boundary
                if (k_global + VEC_SIZE <= K) {  // Check if global K is fully within bounds
                    // Vectorized load if possible
                    *reinterpret_cast<TVec *>(&smemA[k_tile]) = *reinterpret_cast<const TVec *>(&inp[k_global]);
                } else {
                    // Handle boundary case for K (scalar load)
                    T *smem_T_ptr = &smemA[k_tile];
                    for (int i = 0; i < VEC_SIZE; ++i) {
                        smem_T_ptr[i] = (k_global + i < K) ? inp[k_global + i] : static_cast<T>(0.0f);
                    }
                }
            }
            // Note: If BK_GEMV is not a multiple of THREADS_PER_BLOCK * VEC_SIZE,
            // the last few elements might need scalar handling if VEC_SIZE > 1.
            // This simplified version assumes alignment or handles potential
            // over-read by padding.
        }

        // --- 加载 smemB (Warp 协作，优化访存模式) ---
        // 每个线程负责加载 BK_GEMV / WARP_SIZE 行，每行加载 N_PER_WARP 个 packed
        // weights 目标: smemB[warp_id_in_block][n_offset][k_tile] = qwt[k_global *
        // stride + n_packed] 优化策略：让线程 lane 负责 k_tile = lane,
        // lane+WARP_SIZE, ... 然后在内层循环加载 N_PER_WARP 个列，实现对 qwt
        // 行内数据的合并访问
        const int qwt_stride = N_PACKED;  // stride in qwt matrix
#pragma unroll
        for (int k_tile_offset = 0; k_tile_offset < BK_GEMV; k_tile_offset += WARP_SIZE) {
            int k_tile = k_tile_offset + lane;  // smem 内部的行索引
            if (k_tile < BK_GEMV) {             // 检查 smem 行边界
                int k_global = tile_k_base + k_tile;
                if (k_global < K) {  // 检查全局 K 边界
#pragma unroll
                    for (int n_offset = 0; n_offset < N_PER_WARP; ++n_offset) {
                        int current_n_packed = n_packed_global[n_offset];
                        if (current_n_packed >= 0) {  // Check if this n_global is valid
                            smemB[warp_id_in_block][n_offset][k_tile] = qwt[k_global * qwt_stride + current_n_packed];
                        } else {
                            smemB[warp_id_in_block][n_offset][k_tile] = 0;  // Pad with 0 for invalid N
                        }
                    }
                } else {
// K 超出边界，用 0 填充 smemB 对应行
#pragma unroll
                    for (int n_offset = 0; n_offset < N_PER_WARP; ++n_offset) {
                        smemB[warp_id_in_block][n_offset][k_tile] = 0;
                    }
                }
            }
        }

        // --- 加载 smemS 和 smemZ (Warp 协作) ---
        // 计算当前 K-tile 所属的 group index (不变)
        int current_group_idx = tile_k_base / group_size;
        bool valid_group = (current_group_idx < num_groups);

        // 让 Warp 的前 N_PER_WARP 个线程分别加载各自负责的 Scale 和 Zero
        if (lane < N_PER_WARP) {
            int n_global = warp_n_base + lane;             // n_offset == lane here
            int current_n_packed = n_packed_global[lane];  // Get precomputed packed index

            // 加载 Scale
            bool valid_s = valid_group && (n_global < N);
            smemS[warp_id_in_block][lane] = valid_s ? scl[current_group_idx * N + n_global]
                                                    : static_cast<ScaleType>(1.0f);  // Use 1.0f for invalid scale? Or
                                                                                     // 0? Check AWQ logic. Assume 1.0f.

            // 加载 Packed Zero-point
            bool valid_z = valid_group && (current_n_packed >= 0);  // Check if packed index is valid
            smemZ[warp_id_in_block][lane] =
                valid_z ? zos[current_group_idx * N_PACKED + current_n_packed] : 0;  // Pad with 0
        }
        // 不需要 __syncwarp() 因为后面有 __syncthreads()

        __syncthreads();  // 等待 smemA, smemB, smemS, smemZ 全部加载完成

        // --- K 块内计算 (修改) ---
        const int num_k_iterations = BK_GEMV / K_PER_ITERATION;  // = BK_GEMV / (WARP_SIZE * VEC_SIZE)

#pragma unroll
        for (int w = 0; w < num_k_iterations; ++w) {
            // 每个线程处理 VEC_SIZE 个 K 元素
            int k_base_in_tile = (w * WARP_SIZE + lane) * VEC_SIZE;

            // 加载 VEC_SIZE 个输入 activation
            TVec inp_vec = *reinterpret_cast<TVec *>(&smemA[k_base_in_tile]);
            T *inp_vals = reinterpret_cast<T *>(&inp_vec);  // size VEC_SIZE

// 循环处理 VEC_SIZE 个 K 元素
#pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
                int k_tile = k_base_in_tile + i;  // smem 内部行索引
                float current_inp_float = static_cast<float>(inp_vals[i]);

// 对 N_PER_WARP 个输出列进行计算
#pragma unroll
                for (int n_offset = 0; n_offset < N_PER_WARP; ++n_offset) {
                    if (n_packed_global[n_offset] < 0)
                        continue;  // Skip invalid N columns

                    // 从共享内存读取 packed weight, scale, packed zero
                    uint32_t regB_packed = smemB[warp_id_in_block][n_offset][k_tile];
                    ScaleType scale_val = smemS[warp_id_in_block][n_offset];
                    uint32_t regZ_packed = smemZ[warp_id_in_block][n_offset];

                    // 提取 q_w 和 q_z
                    // 使用预计算好的 bit_shift[n_offset]
                    uint32_t q_w = (regB_packed >> bit_shift[n_offset]) & dequant_mask;
                    uint32_t q_z = (regZ_packed >> bit_shift[n_offset]) & dequant_mask;

                    // 反量化
                    float scale_float = static_cast<float>(scale_val);
                    float z_float = static_cast<float>(q_z);
                    float dequant_w = (static_cast<float>(q_w) - z_float) * scale_float;

                    // 累加 (FMA)
                    thread_sum[n_offset] = fmaf(current_inp_float, dequant_w, thread_sum[n_offset]);
                }  // 结束 n_offset 循环
            }  // 结束 VEC_SIZE 循环 (i)
        }  // 结束 K 块内迭代循环 (w)

        __syncthreads();  // 确保当前 K-tile 计算完成，再开始下一轮 K-tile 加载

    }  // 结束 K 分块循环 (tile_k_base)

// --- Warp Reduce (修改) ---
// 对每个 N 的累加结果分别进行 Warp Reduce
#pragma unroll
    for (int n_offset = 0; n_offset < N_PER_WARP; ++n_offset) {
        if (n_packed_global[n_offset] >= 0) {  // Only reduce valid columns
            thread_sum[n_offset] = warp_reduce_sum_f32<WARP_SIZE>(thread_sum[n_offset]);
        }
    }

    // --- 写回结果 (修改) ---
    // Warp 的前 N_PER_WARP 个线程负责写回 N_PER_WARP 个结果
    if (lane < N_PER_WARP) {
        int n_global = warp_n_base + lane;  // n_offset == lane here
        if (n_global < N) {                 // 检查 N 边界
            float final_val = thread_sum[lane];
            if (bias) {
                final_val += static_cast<float>(bias[n_global]);
            }
            out[n_global] = static_cast<T>(final_val);
        }
    }
}

template <typename T, typename ScaleType, int BK_GEMV = 32, int BN_GEMV = 128, int THREADS_PER_BLOCK_GEMV = 128>
__global__ void matmul_awq_gemv_kernel_v2(const T *__restrict__ inp, const int32_t *__restrict__ qwt,
                                          const ScaleType *__restrict__ scl, const int32_t *__restrict__ zos,
                                          T *__restrict__ out, const int K, const int N, const int group_size,
                                          const T *__restrict__ bias) {
    int B_block_start = blockIdx.y * BN_GEMV;
    constexpr int patch_N = BN_GEMV / PACK_FACTOR;
    static_assert(BN_GEMV % PACK_FACTOR == 0, "GEMV BN must be multiple of PACK_FACTOR");
    static_assert(THREADS_PER_BLOCK_GEMV >= (BN_GEMV / PACK_FACTOR), "Need enough threads for packed N");

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
        for (int k_offset = tid; k_offset < BK_GEMV; k_offset += THREADS_PER_BLOCK_GEMV) {
            int k_global = tile_k_base + k_offset;
            if (k_global < K) {
                smemA[k_offset] = inp[k_global];
            } else {
                smemA[k_offset] = static_cast<T>(0.0f);
            }
        }

#pragma unroll
        for (int load_idx = tid; load_idx < BK_GEMV * patch_N; load_idx += THREADS_PER_BLOCK_GEMV) {
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
            if (current_tile_group_idx < num_groups && n_packed_global < (N / PACK_FACTOR)) {
                zos_val = zos[current_tile_group_idx * (N / PACK_FACTOR) + n_packed_global];
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
                (n_packed_local < patch_N) ? smemB[bk * (patch_N + PADDING_B_GEMV) + n_packed_local] : 0;
            uint32_t regZ_packed = (n_packed_local < patch_N) ? smemZeros[n_packed_local] : 0;

            ScaleType regS =
                (n_packed_local < patch_N) ? smemScales[j_inner][n_packed_local] : static_cast<ScaleType>(0.0f);

            int physical_inner_col_idx = LOGICAL_TO_PHYSICAL_INNER_IDX[j_inner];
            int bit_shift = physical_inner_col_idx * BITS;
            uint32_t q_w = (regB_packed >> bit_shift) & 0x0F;
            uint32_t q_z = (regZ_packed >> bit_shift) & 0x0F;
            float scale_float = static_cast<float>(regS);
            float dequant_w = (static_cast<float>(q_w) - static_cast<float>(q_z)) * scale_float;

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

template <typename T, typename ScaleType, int BM = 64, int BN = 64, int BK = 32, int WMMA_M = 16, int WMMA_N = 16,
          int WMMA_K = 16>
__global__ void matmul_awq_kernel_prefill_wmma_v1(const T *__restrict__ inp, const int32_t *__restrict__ qwt,
                                                  const ScaleType *__restrict__ scl, const int32_t *__restrict__ zos,
                                                  T *__restrict__ out, const int M, const int K, const int N,
                                                  const int group_size, const T *__restrict__ bias) {
    static_assert(std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>,
                  "WMMA kernel currently supports only half or bfloat16 inputs/outputs.");

    static_assert(BM % WMMA_M == 0, "BM must be a multiple of WMMA_M");
    static_assert(BN % WMMA_N == 0, "BN must be a multiple of WMMA_N");
    static_assert(BK % WMMA_K == 0, "BK must be a multiple of WMMA_K");

    using namespace nvcuda;
    using FragmentA = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T, wmma::row_major>;
    using FragmentB = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T, wmma::col_major>;
    using FragmentC = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

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
        for (int load_idx = threadIdx.x; load_idx < BM * BK; load_idx += blockDim.x) {
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
        for (int load_idx = threadIdx.x; load_idx < BK * patch_N; load_idx += blockDim.x) {
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
            if (current_tile_group_idx < num_groups && n_packed_global < (N / PACK_FACTOR)) {
                zos_val = zos[current_tile_group_idx * (N / PACK_FACTOR) + n_packed_global];
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
            float dequant_w = (static_cast<float>(q_w) - static_cast<float>(q_z)) * scale_float;

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

    wmma::store_matrix_sync(&smem_out[out_m_base][out_n_base], fragC, BN, wmma::mem_row_major);

    __syncthreads();

#pragma unroll
    for (int write_idx = threadIdx.x; write_idx < BM * BN; write_idx += blockDim.x) {
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

template <typename T, typename ScaleType>
void matmul_quantized(const Tensor<T> &input, const Tensor<int32_t> &qweight, const Tensor<ScaleType> &scales_input,
                      const Tensor<int32_t> &zeros_input, int group_size, Tensor<T> *output, cudaStream_t stream,
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
        throw std::runtime_error("Dimension K mismatch between input (" + std::to_string(K) + ") and qweight (" +
                                 std::to_string(qweight.sizes()[0]) + ")");
    }
    int packed_N_dim = (N + PACK_FACTOR - 1) / PACK_FACTOR;
    if (static_cast<int>(qweight.sizes()[1]) != packed_N_dim) {
        throw std::runtime_error("Dimension N/PACK_FACTOR mismatch between inferred N (" + std::to_string(N) + " -> " +
                                 std::to_string(packed_N_dim) + ") and qweight (" + std::to_string(qweight.sizes()[1]) +
                                 ")");
    }

    cudaError_t launch_err = cudaSuccess;

    if (M == 1) {                                // Assuming M == 1 is the GEMV case
        constexpr int BK_GEMV = 64;              // Or 64, must be <= group_size and multiple of K_PER_ITERATION
        constexpr int VEC_SIZE_GEMV = 2;         // Example vectorization
        constexpr int N_PER_WARP_GEMV = 8;       // <<< 每个 Warp 处理 4 个 N
        constexpr int WARPS_PER_BLOCK_GEMV = 8;  // <<< 调整 Block 内 Warp 数量，例如 8 个 Warp (256 threads/block)
                                                 // 你原来的 WARPS_PER_BLOCK=32 (1024 threads) 可能太大了，需要根据
                                                 // GPU 调整

        // 确保 WARPS_PER_BLOCK * WARP_SIZE <= maxThreadsPerBlock (通常是 1024)
        static_assert(WARPS_PER_BLOCK_GEMV * WARP_SIZE <= 1024, "Too many threads per block");

        // 计算需要的总 Warp 数量
        const int total_n_tasks = (N + N_PER_WARP_GEMV - 1) / N_PER_WARP_GEMV;
        // 计算 Grid 维度
        const dim3 grid_gemv((total_n_tasks + WARPS_PER_BLOCK_GEMV - 1) / WARPS_PER_BLOCK_GEMV);
        // 计算 Block 维度
        const dim3 block_gemv(WARPS_PER_BLOCK_GEMV * WARP_SIZE);

        // 计算需要的共享内存大小
        // smemA: BK_GEMV * sizeof(T)
        // smemB: WARPS_PER_BLOCK_GEMV * N_PER_WARP_GEMV * BK_GEMV * sizeof(int32_t)
        // smemS: WARPS_PER_BLOCK_GEMV * N_PER_WARP_GEMV * sizeof(ScaleType)
        // smemZ: WARPS_PER_BLOCK_GEMV * N_PER_WARP_GEMV * sizeof(int32_t)
        // 注意：这里共享内存是按 Warp 分配的，所以总大小是 per-warp *
        // WARPS_PER_BLOCK_GEMV
        size_t smem_size = BK_GEMV * sizeof(T);  // smemA is shared by block implicitly in declaration
        smem_size += WARPS_PER_BLOCK_GEMV * N_PER_WARP_GEMV * BK_GEMV * sizeof(int32_t);  // smemB
        smem_size += WARPS_PER_BLOCK_GEMV * N_PER_WARP_GEMV * sizeof(ScaleType);          // smemS
        smem_size += WARPS_PER_BLOCK_GEMV * N_PER_WARP_GEMV * sizeof(int32_t);            // smemZ

        // std::cout << "Required shared memory: " << smem_size << " bytes"
        //           << std::endl;
        // 你可能需要检查 smem_size 是否超过设备限制，并可能需要通过
        // cudaFuncSetAttribute 设置

        // 调用修改后的 Kernel
        awq_gemv_warp_vectorized_n<T, ScaleType, BK_GEMV, VEC_SIZE_GEMV, N_PER_WARP_GEMV, WARPS_PER_BLOCK_GEMV>
            <<<grid_gemv, block_gemv, smem_size, stream>>>(  // Pass smem_size if
                                                             // needed, or ensure
                                                             // declarations fit
                input.data_ptr(), qweight.data_ptr(), scales_input.data_ptr(), zeros_input.data_ptr(),
                output->data_ptr(), K, N, group_size, bias ? bias->data_ptr() : nullptr);

        launch_err = cudaGetLastError();
        // Check launch_err
    } else {
        if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
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

            matmul_awq_kernel_prefill_wmma_v1<T, ScaleType, BM_WMMA, BN_WMMA, BK_WMMA, WMMA_M, WMMA_N, WMMA_K>
                <<<grid_wmma, block_wmma, 0, stream>>>(input.data_ptr(), qweight.data_ptr(), scales_input.data_ptr(),
                                                       zeros_input.data_ptr(), output->data_ptr(), M, K, N, group_size,
                                                       bias ? bias->data_ptr() : nullptr);
            launch_err = cudaGetLastError();

        } else {
            std::cerr << "Warning: matmul_quantized WMMA kernel does not support "
                         "input type "
                      << typeid(T).name() << ". Prefill stage (M > 1) will not execute." << std::endl;

            throw std::runtime_error(
                "matmul_quantized WMMA kernel currently only supports "
                "half/bfloat16 "
                "input/output types for M > 1.");
        }
    }

    if (launch_err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error in matmul_quantized: " << cudaGetErrorString(launch_err) << std::endl;

        throw std::runtime_error("CUDA kernel launch failed in matmul_quantized");
    }
}

// 原始的float scales版本
template void matmul_quantized<float, float>(const Tensor<float> &, const Tensor<int32_t> &, const Tensor<float> &,
                                             const Tensor<int32_t> &, int, Tensor<float> *, cudaStream_t,
                                             const Tensor<float> *);

// bf16输入和bf16 scales版本
template void matmul_quantized<__nv_bfloat16, __nv_bfloat16>(const Tensor<__nv_bfloat16> &, const Tensor<int32_t> &,
                                                             const Tensor<__nv_bfloat16> &, const Tensor<int32_t> &,
                                                             int, Tensor<__nv_bfloat16> *, cudaStream_t,
                                                             const Tensor<__nv_bfloat16> *);

// bf16输入和float scales版本 (兼容旧代码)
template void matmul_quantized<__nv_bfloat16, float>(const Tensor<__nv_bfloat16> &, const Tensor<int32_t> &,
                                                     const Tensor<float> &, const Tensor<int32_t> &, int,
                                                     Tensor<__nv_bfloat16> *, cudaStream_t,
                                                     const Tensor<__nv_bfloat16> *);

// half输入和half scales版本
template void matmul_quantized<__half, __half>(const Tensor<__half> &, const Tensor<int32_t> &, const Tensor<__half> &,
                                               const Tensor<int32_t> &, int, Tensor<__half> *, cudaStream_t,
                                               const Tensor<__half> *);

// half输入和float scales版本 (兼容旧代码)
template void matmul_quantized<__half, float>(const Tensor<__half> &, const Tensor<int32_t> &, const Tensor<float> &,
                                              const Tensor<int32_t> &, int, Tensor<__half> *, cudaStream_t,
                                              const Tensor<__half> *);

}  // namespace cuda_OP