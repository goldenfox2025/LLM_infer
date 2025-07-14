#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// 添加CuTe库的头文件支持
#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>
#include <type_traits>

#include "ptx_common.h"  // 使用正确的路径

// 添加与SICore兼容的类型别名
using nv_bfloat16 = __nv_bfloat16;
using nv_bfloat162 = __nv_bfloat162;

namespace advanced_kernels {

// kernel5: 使用16x8x16 MMA指令
template <typename T, int BM, int BN, int BK, int WMMA_M, int WMMA_N, int WMMA_K, int WAPR_NUM, int K_STAGE,
          int WARP_TILE_M, int WARP_TILE_N>
__global__ void kernel5(const T *A, const T *B, T *C, int M, int N, int K) {
    int warp_id = threadIdx.x / 32;
    constexpr int WARP_N_NUM = BN / (WMMA_N * WARP_TILE_N);
    int warp_n_id = warp_id % WARP_N_NUM;
    int warp_m_id = warp_id / WARP_N_NUM;
    int global_m_base = blockIdx.x * BM;
    int global_n_base = blockIdx.y * BN;
    constexpr int SA_SIZE = BM * BK;
    constexpr int SB_SIZE = BN * BK;
    const int lane_id = threadIdx.x % 32;

    __shared__ T smemA[K_STAGE * BM * BK];
    __shared__ T smemB[K_STAGE * BN * BK];
    uint32_t smem_a_base_ptr = __cvta_generic_to_shared(smemA);
    uint32_t smem_b_base_ptr = __cvta_generic_to_shared(smemB);

    constexpr int vec_size = sizeof(float4) / sizeof(T);

    uint32_t RC[WARP_TILE_M][WARP_TILE_N][4];
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            RC[i][j][0] = 0;
            RC[i][j][1] = 0;
            RC[i][j][2] = 0;
            RC[i][j][3] = 0;
        }
    }

    // 加载除最后阶段外的数据
    for (int k_load_stage = 0; k_load_stage < (K_STAGE - 1); ++k_load_stage) {
        // 加载A矩阵数据
        for (int load_idx = threadIdx.x * vec_size; load_idx < BM * BK; load_idx += blockDim.x * vec_size) {
            int smem_row = load_idx / (BK);
            int smem_col = load_idx % (BK);
            int global_row = global_m_base + smem_row;
            int global_col = k_load_stage * BK + smem_col;

            if (global_row < M && (global_col + vec_size - 1) < K) {
                int load_gmem_a_addr = global_row * K + global_col;
                int swizzled_col = swizzle_permuted_A_j(smem_row, smem_col);
                int swizzled_idx = smem_row * BK + swizzled_col;
                uint32_t load_smem_a_ptr = smem_a_base_ptr + (swizzled_idx + k_load_stage * SA_SIZE) * sizeof(T);
                CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
            }
        }

        // 加载B矩阵数据
        for (int load_idx = threadIdx.x * vec_size; load_idx < BN * BK; load_idx += blockDim.x * vec_size) {
            int smem_row = load_idx / (BK);
            int smem_col = load_idx % (BK);
            int global_row = global_n_base + smem_row;
            int global_col = k_load_stage * BK + smem_col;

            if (global_col + vec_size - 1 < K && global_row < N) {
                int load_gmem_b_addr = global_row * K + global_col;
                int swizzled_col = swizzle_permuted_B_j(smem_row, smem_col);
                int swizzled_idx = smem_row * BK + swizzled_col;
                uint32_t load_smem_b_ptr = smem_b_base_ptr + (swizzled_idx + k_load_stage * SB_SIZE) * sizeof(T);
                CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
            }
        }
        CP_ASYNC_COMMIT_GROUP();
    }
    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();

    // 主循环
    uint32_t RA[WARP_TILE_M][4];
    uint32_t RB[WARP_TILE_N][2];
    for (int k_load_base = (K_STAGE - 1) * BK; k_load_base < K; k_load_base += BK) {
        const int k_load_stage = k_load_base / BK;
        int smem_sel = (k_load_stage + 1) % K_STAGE;
        int smem_sel_next = k_load_stage % K_STAGE;

        // 加载A矩阵数据
        for (int load_idx = threadIdx.x * vec_size; load_idx < BM * BK; load_idx += blockDim.x * vec_size) {
            int smem_row = load_idx / (BK);
            int smem_col = load_idx % (BK);
            int global_row = global_m_base + smem_row;
            int global_col = k_load_base + smem_col;

            if (global_row < M && (global_col + vec_size - 1) < K) {
                int load_gmem_a_addr = global_row * K + global_col;
                int swizzled_col = swizzle_permuted_A_j(smem_row, smem_col);
                int swizzled_idx = smem_row * BK + swizzled_col;
                uint32_t load_smem_a_ptr = smem_a_base_ptr + (swizzled_idx + smem_sel_next * SA_SIZE) * sizeof(T);
                CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
            }
        }

        // 加载B矩阵数据
        for (int load_idx = threadIdx.x * vec_size; load_idx < BN * BK; load_idx += blockDim.x * vec_size) {
            int smem_row = load_idx / (BK);
            int smem_col = load_idx % (BK);
            int global_row = global_n_base + smem_row;
            int global_col = k_load_base + smem_col;

            if (global_col + vec_size - 1 < K && global_row < N) {
                int load_gmem_b_addr = global_row * K + global_col;
                int swizzled_col = swizzle_permuted_B_j(smem_row, smem_col);
                int swizzled_idx = smem_row * BK + swizzled_col;
                uint32_t load_smem_b_ptr = smem_b_base_ptr + (swizzled_idx + smem_sel_next * SB_SIZE) * sizeof(T);
                CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
            }
        }
        CP_ASYNC_COMMIT_GROUP();

        // 计算部分
        for (int TILE_K = 0; TILE_K < BK; TILE_K += WMMA_K) {
            // 读取A矩阵数据
            for (int i = 0; i < WARP_TILE_M; ++i) {
                int warp_smem_a_m = warp_m_id * WMMA_M * WARP_TILE_M + i * WMMA_M;
                int warp_smem_a_k = TILE_K;

                int base_row = warp_smem_a_m + (lane_id % 16);
                int base_col = warp_smem_a_k + (lane_id / 16) * vec_size;

                int swizzled_col = swizzle_permuted_A_j(base_row, base_col);
                T *lane_smem_a_ptr = smemA + base_row * BK + swizzled_col + smem_sel * SA_SIZE;

                uint32_t ptr = __cvta_generic_to_shared(lane_smem_a_ptr);
                LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], ptr);
            }

            // 读取B矩阵数据
            for (int i = 0; i < WARP_TILE_N; ++i) {
                int warp_smem_b_n = warp_n_id * WMMA_N * WARP_TILE_N + i * WMMA_N;
                int warp_smem_b_k = TILE_K;

                int base_row = warp_smem_b_n + (lane_id % 8);
                int base_col = warp_smem_b_k + (lane_id / 8) * vec_size;

                int swizzled_col = swizzle_permuted_B_j(base_row, base_col);
                T *lane_smem_b_ptr = smemB + base_row * BK + swizzled_col + smem_sel * SB_SIZE;

                uint32_t ptr = __cvta_generic_to_shared(lane_smem_b_ptr);
                LDMATRIX_X2(RB[i][0], RB[i][1], ptr);
            }

            // 执行矩阵乘法
            for (int i = 0; i < WARP_TILE_M; ++i) {
                for (int j = 0; j < WARP_TILE_N; ++j) {
                    MMA16816_BF16(RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3], RA[i][0], RA[i][1], RA[i][2],
                                  RA[i][3], RB[j][0], RB[j][1], RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3]);
                }
            }
        }

        CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
        __syncthreads();
    }

    // 主循环结束
    if ((K_STAGE - 2) > 0) {
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }

    // 计算剩余阶段
    for (int k_load = 0; k_load < K_STAGE - 1; ++k_load) {
        const int stage_sel = ((K / BK - (K_STAGE - 1) + k_load) % K_STAGE);
        for (int TILE_K = 0; TILE_K < BK; TILE_K += WMMA_K) {
            // 读取A矩阵数据
            for (int i = 0; i < WARP_TILE_M; ++i) {
                int warp_smem_a_m = warp_m_id * WMMA_M * WARP_TILE_M + i * WMMA_M;
                int warp_smem_a_k = TILE_K;

                int base_row = warp_smem_a_m + (lane_id % 16);
                int base_col = warp_smem_a_k + (lane_id / 16) * vec_size;

                int swizzled_col = swizzle_permuted_A_j(base_row, base_col);
                T *lane_smem_a_ptr = smemA + base_row * BK + swizzled_col + stage_sel * SA_SIZE;

                uint32_t ptr = __cvta_generic_to_shared(lane_smem_a_ptr);
                LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], ptr);
            }

            // 读取B矩阵数据
            for (int i = 0; i < WARP_TILE_N; ++i) {
                int warp_smem_b_n = warp_n_id * WMMA_N * WARP_TILE_N + i * WMMA_N;
                int warp_smem_b_k = TILE_K;

                int base_row = warp_smem_b_n + (lane_id % 8);
                int base_col = warp_smem_b_k + (lane_id / 8) * vec_size;

                int swizzled_col = swizzle_permuted_B_j(base_row, base_col);
                T *lane_smem_b_ptr = smemB + base_row * BK + swizzled_col + stage_sel * SB_SIZE;

                uint32_t ptr = __cvta_generic_to_shared(lane_smem_b_ptr);
                LDMATRIX_X2(RB[i][0], RB[i][1], ptr);
            }

            // 执行矩阵乘法
            for (int i = 0; i < WARP_TILE_M; ++i) {
                for (int j = 0; j < WARP_TILE_N; ++j) {
                    MMA16816_BF16(RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3], RA[i][0], RA[i][1], RA[i][2],
                                  RA[i][3], RB[j][0], RB[j][1], RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3]);
                }
            }
        }
    }

    // 写回结果
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            const int tile_m0 = global_m_base + (warp_m_id * WMMA_M * WARP_TILE_M) + i * WMMA_M;
            const int tile_n0 = global_n_base + (warp_n_id * WMMA_N * WARP_TILE_N) + j * WMMA_N;

            int group = lane_id >> 2;
            int tid4 = lane_id & 3;
            int row0 = group;
            int row1 = group + 8;
            int col0 = 2 * tid4;
            int col1 = 2 * tid4 + 1;

            float v0 = reinterpret_cast<float *>(&RC[i][j][0])[0];
            float v1 = reinterpret_cast<float *>(&RC[i][j][1])[0];
            float v2 = reinterpret_cast<float *>(&RC[i][j][2])[0];
            float v3 = reinterpret_cast<float *>(&RC[i][j][3])[0];

            if ((tile_m0 + row0) < M && (tile_n0 + col0) < N)
                C[(tile_m0 + row0) * N + (tile_n0 + col0)] = v0;
            if ((tile_m0 + row0) < M && (tile_n0 + col1) < N)
                C[(tile_m0 + row0) * N + (tile_n0 + col1)] = v1;
            if ((tile_m0 + row1) < M && (tile_n0 + col0) < N)
                C[(tile_m0 + row1) * N + (tile_n0 + col0)] = v2;
            if ((tile_m0 + row1) < M && (tile_n0 + col1) < N)
                C[(tile_m0 + row1) * N + (tile_n0 + col1)] = v3;
        }
    }
}

// kernel5 with bias
template <typename T, int BM, int BN, int BK, int WMMA_M, int WMMA_N, int WMMA_K, int WAPR_NUM, int K_STAGE,
          int WARP_TILE_M, int WARP_TILE_N>
__global__ void kernel5_with_bias(const T *A, const T *B, const T *bias, T *C, int M, int N, int K) {
    int warp_id = threadIdx.x / 32;
    constexpr int WARP_N_NUM = BN / (WMMA_N * WARP_TILE_N);
    int warp_n_id = warp_id % WARP_N_NUM;
    int warp_m_id = warp_id / WARP_N_NUM;
    int global_m_base = blockIdx.x * BM;
    int global_n_base = blockIdx.y * BN;
    constexpr int SA_SIZE = BM * BK;
    constexpr int SB_SIZE = BN * BK;
    const int lane_id = threadIdx.x % 32;

    __shared__ T smemA[K_STAGE * BM * BK];
    __shared__ T smemB[K_STAGE * BN * BK];
    uint32_t smem_a_base_ptr = __cvta_generic_to_shared(smemA);
    uint32_t smem_b_base_ptr = __cvta_generic_to_shared(smemB);

    constexpr int vec_size = sizeof(float4) / sizeof(T);

    uint32_t RC[WARP_TILE_M][WARP_TILE_N][4];
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            RC[i][j][0] = 0;
            RC[i][j][1] = 0;
            RC[i][j][2] = 0;
            RC[i][j][3] = 0;
        }
    }

    // 加载除最后阶段外的数据
    for (int k_load_stage = 0; k_load_stage < (K_STAGE - 1); ++k_load_stage) {
        // 加载A矩阵数据
        for (int load_idx = threadIdx.x * vec_size; load_idx < BM * BK; load_idx += blockDim.x * vec_size) {
            int smem_row = load_idx / (BK);
            int smem_col = load_idx % (BK);
            int global_row = global_m_base + smem_row;
            int global_col = k_load_stage * BK + smem_col;

            if (global_row < M && (global_col + vec_size - 1) < K) {
                int load_gmem_a_addr = global_row * K + global_col;
                int swizzled_col = swizzle_permuted_A_j(smem_row, smem_col);
                int swizzled_idx = smem_row * BK + swizzled_col;
                uint32_t load_smem_a_ptr = smem_a_base_ptr + (swizzled_idx + k_load_stage * SA_SIZE) * sizeof(T);
                CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
            }
        }

        // 加载B矩阵数据
        for (int load_idx = threadIdx.x * vec_size; load_idx < BN * BK; load_idx += blockDim.x * vec_size) {
            int smem_row = load_idx / (BK);
            int smem_col = load_idx % (BK);
            int global_row = global_n_base + smem_row;
            int global_col = k_load_stage * BK + smem_col;

            if (global_col + vec_size - 1 < K && global_row < N) {
                int load_gmem_b_addr = global_row * K + global_col;
                int swizzled_col = swizzle_permuted_B_j(smem_row, smem_col);
                int swizzled_idx = smem_row * BK + swizzled_col;
                uint32_t load_smem_b_ptr = smem_b_base_ptr + (swizzled_idx + k_load_stage * SB_SIZE) * sizeof(T);
                CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
            }
        }
        CP_ASYNC_COMMIT_GROUP();
    }
    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();

    // 主循环
    uint32_t RA[WARP_TILE_M][4];
    uint32_t RB[WARP_TILE_N][2];
    for (int k_load_base = (K_STAGE - 1) * BK; k_load_base < K; k_load_base += BK) {
        const int k_load_stage = k_load_base / BK;
        int smem_sel = (k_load_stage + 1) % K_STAGE;
        int smem_sel_next = k_load_stage % K_STAGE;

        // 加载A矩阵数据
        for (int load_idx = threadIdx.x * vec_size; load_idx < BM * BK; load_idx += blockDim.x * vec_size) {
            int smem_row = load_idx / (BK);
            int smem_col = load_idx % (BK);
            int global_row = global_m_base + smem_row;
            int global_col = k_load_base + smem_col;

            if (global_row < M && (global_col + vec_size - 1) < K) {
                int load_gmem_a_addr = global_row * K + global_col;
                int swizzled_col = swizzle_permuted_A_j(smem_row, smem_col);
                int swizzled_idx = smem_row * BK + swizzled_col;
                uint32_t load_smem_a_ptr = smem_a_base_ptr + (swizzled_idx + smem_sel_next * SA_SIZE) * sizeof(T);
                CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
            }
        }

        // 加载B矩阵数据
        for (int load_idx = threadIdx.x * vec_size; load_idx < BN * BK; load_idx += blockDim.x * vec_size) {
            int smem_row = load_idx / (BK);
            int smem_col = load_idx % (BK);
            int global_row = global_n_base + smem_row;
            int global_col = k_load_base + smem_col;

            if (global_col + vec_size - 1 < K && global_row < N) {
                int load_gmem_b_addr = global_row * K + global_col;
                int swizzled_col = swizzle_permuted_B_j(smem_row, smem_col);
                int swizzled_idx = smem_row * BK + swizzled_col;
                uint32_t load_smem_b_ptr = smem_b_base_ptr + (swizzled_idx + smem_sel_next * SB_SIZE) * sizeof(T);
                CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
            }
        }
        CP_ASYNC_COMMIT_GROUP();

        // 计算部分
        for (int TILE_K = 0; TILE_K < BK; TILE_K += WMMA_K) {
            // 读取A矩阵数据
            for (int i = 0; i < WARP_TILE_M; ++i) {
                int warp_smem_a_m = warp_m_id * WMMA_M * WARP_TILE_M + i * WMMA_M;
                int warp_smem_a_k = TILE_K;

                int base_row = warp_smem_a_m + (lane_id % 16);
                int base_col = warp_smem_a_k + (lane_id / 16) * vec_size;

                int swizzled_col = swizzle_permuted_A_j(base_row, base_col);
                T *lane_smem_a_ptr = smemA + base_row * BK + swizzled_col + smem_sel * SA_SIZE;

                uint32_t ptr = __cvta_generic_to_shared(lane_smem_a_ptr);
                LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], ptr);
            }

            // 读取B矩阵数据
            for (int i = 0; i < WARP_TILE_N; ++i) {
                int warp_smem_b_n = warp_n_id * WMMA_N * WARP_TILE_N + i * WMMA_N;
                int warp_smem_b_k = TILE_K;

                int base_row = warp_smem_b_n + (lane_id % 8);
                int base_col = warp_smem_b_k + (lane_id / 8) * vec_size;

                int swizzled_col = swizzle_permuted_B_j(base_row, base_col);
                T *lane_smem_b_ptr = smemB + base_row * BK + swizzled_col + smem_sel * SB_SIZE;

                uint32_t ptr = __cvta_generic_to_shared(lane_smem_b_ptr);
                LDMATRIX_X2(RB[i][0], RB[i][1], ptr);
            }

            // 执行矩阵乘法
            for (int i = 0; i < WARP_TILE_M; ++i) {
                for (int j = 0; j < WARP_TILE_N; ++j) {
                    MMA16816_BF16(RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3], RA[i][0], RA[i][1], RA[i][2],
                                  RA[i][3], RB[j][0], RB[j][1], RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3]);
                }
            }
        }

        CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
        __syncthreads();
    }

    // 主循环结束
    if ((K_STAGE - 2) > 0) {
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }

    // 计算剩余阶段
    for (int k_load = 0; k_load < K_STAGE - 1; ++k_load) {
        const int stage_sel = ((K / BK - (K_STAGE - 1) + k_load) % K_STAGE);
        for (int TILE_K = 0; TILE_K < BK; TILE_K += WMMA_K) {
            // 读取A矩阵数据
            for (int i = 0; i < WARP_TILE_M; ++i) {
                int warp_smem_a_m = warp_m_id * WMMA_M * WARP_TILE_M + i * WMMA_M;
                int warp_smem_a_k = TILE_K;

                int base_row = warp_smem_a_m + (lane_id % 16);
                int base_col = warp_smem_a_k + (lane_id / 16) * vec_size;

                int swizzled_col = swizzle_permuted_A_j(base_row, base_col);
                T *lane_smem_a_ptr = smemA + base_row * BK + swizzled_col + stage_sel * SA_SIZE;

                uint32_t ptr = __cvta_generic_to_shared(lane_smem_a_ptr);
                LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], ptr);
            }

            // 读取B矩阵数据
            for (int i = 0; i < WARP_TILE_N; ++i) {
                int warp_smem_b_n = warp_n_id * WMMA_N * WARP_TILE_N + i * WMMA_N;
                int warp_smem_b_k = TILE_K;

                int base_row = warp_smem_b_n + (lane_id % 8);
                int base_col = warp_smem_b_k + (lane_id / 8) * vec_size;

                int swizzled_col = swizzle_permuted_B_j(base_row, base_col);
                T *lane_smem_b_ptr = smemB + base_row * BK + swizzled_col + stage_sel * SB_SIZE;

                uint32_t ptr = __cvta_generic_to_shared(lane_smem_b_ptr);
                LDMATRIX_X2(RB[i][0], RB[i][1], ptr);
            }

            // 执行矩阵乘法
            for (int i = 0; i < WARP_TILE_M; ++i) {
                for (int j = 0; j < WARP_TILE_N; ++j) {
                    MMA16816_BF16(RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3], RA[i][0], RA[i][1], RA[i][2],
                                  RA[i][3], RB[j][0], RB[j][1], RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3]);
                }
            }
        }
    }

    // 写回结果并加上bias
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            const int tile_m0 = global_m_base + (warp_m_id * WMMA_M * WARP_TILE_M) + i * WMMA_M;
            const int tile_n0 = global_n_base + (warp_n_id * WMMA_N * WARP_TILE_N) + j * WMMA_N;

            int group = lane_id >> 2;
            int tid4 = lane_id & 3;
            int row0 = group;
            int row1 = group + 8;

            // 对于16x16，列偏移需要覆盖16列，分为两组8列
            int col0 = 2 * tid4;
            int col1 = 2 * tid4 + 1;

            // 取出8个累加结果（FP32）
            float v0 = reinterpret_cast<float *>(&RC[i][j][0])[0];
            float v1 = reinterpret_cast<float *>(&RC[i][j][1])[0];
            float v2 = reinterpret_cast<float *>(&RC[i][j][2])[0];
            float v3 = reinterpret_cast<float *>(&RC[i][j][3])[0];

            // 写回全局内存并加上bias
            if ((tile_m0 + row0) < M && (tile_n0 + col0) < N)
                C[(tile_m0 + row0) * N + (tile_n0 + col0)] = static_cast<T>(v0) + bias[tile_n0 + col0];
            if ((tile_m0 + row0) < M && (tile_n0 + col1) < N)
                C[(tile_m0 + row0) * N + (tile_n0 + col1)] = static_cast<T>(v1) + bias[tile_n0 + col1];
            if ((tile_m0 + row1) < M && (tile_n0 + col0) < N)
                C[(tile_m0 + row1) * N + (tile_n0 + col0)] = static_cast<T>(v2) + bias[tile_n0 + col0];
            if ((tile_m0 + row1) < M && (tile_n0 + col1) < N)
                C[(tile_m0 + row1) * N + (tile_n0 + col1)] = static_cast<T>(v3) + bias[tile_n0 + col1];
        }
    }
}

// 继续添加kernel6...
// kernel6: 使用16x16x16 MMA指令，基于kernel5但使用16x16x16维度
template <typename T, int BM, int BN, int BK, int WMMA_M, int WMMA_N, int WMMA_K, int WAPR_NUM, int K_STAGE,
          int WARP_TILE_M, int WARP_TILE_N>
__global__ void kernel6(const T *A, const T *B, T *C, int M, int N, int K) {
    int warp_id = threadIdx.x / 32;
    constexpr int WARP_N_NUM = BN / (WMMA_N * WARP_TILE_N);
    int warp_n_id = warp_id % WARP_N_NUM;
    int warp_m_id = warp_id / WARP_N_NUM;
    int global_m_base = blockIdx.x * BM;
    int global_n_base = blockIdx.y * BN;
    constexpr int SA_SIZE = BM * BK;
    constexpr int SB_SIZE = BN * BK;
    const int lane_id = threadIdx.x % 32;

    __shared__ T smemA[K_STAGE * BM * BK];
    __shared__ T smemB[K_STAGE * BN * BK];
    uint32_t smem_a_base_ptr = __cvta_generic_to_shared(smemA);
    uint32_t smem_b_base_ptr = __cvta_generic_to_shared(smemB);

    constexpr int vec_size = sizeof(float4) / sizeof(T);

    // 对于16x16x16，我们有8个输出寄存器
    uint32_t RC[WARP_TILE_M][WARP_TILE_N][8];
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            RC[i][j][0] = 0;
            RC[i][j][1] = 0;
            RC[i][j][2] = 0;
            RC[i][j][3] = 0;
            RC[i][j][4] = 0;
            RC[i][j][5] = 0;
            RC[i][j][6] = 0;
            RC[i][j][7] = 0;
        }
    }

    // 加载初始阶段数据
    for (int k_load_stage = 0; k_load_stage < (K_STAGE - 1); ++k_load_stage) {
        // 加载A矩阵数据
        for (int load_idx = threadIdx.x * vec_size; load_idx < BM * BK; load_idx += blockDim.x * vec_size) {
            int smem_row = load_idx / (BK);
            int smem_col = load_idx % (BK);
            int global_row = global_m_base + smem_row;
            int global_col = k_load_stage * BK + smem_col;

            if (global_row < M && (global_col + vec_size - 1) < K) {
                int load_gmem_a_addr = global_row * K + global_col;
                int swizzled_col = swizzle_permuted_A_j(smem_row, smem_col);
                int swizzled_idx = smem_row * BK + swizzled_col;
                uint32_t load_smem_a_ptr = smem_a_base_ptr + (swizzled_idx + k_load_stage * SA_SIZE) * sizeof(T);
                CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
            }
        }

        // 加载B矩阵数据
        for (int load_idx = threadIdx.x * vec_size; load_idx < BN * BK; load_idx += blockDim.x * vec_size) {
            int smem_row = load_idx / (BK);
            int smem_col = load_idx % (BK);
            int global_row = global_n_base + smem_row;
            int global_col = k_load_stage * BK + smem_col;

            if (global_col + vec_size - 1 < K && global_row < N) {
                int load_gmem_b_addr = global_row * K + global_col;
                int swizzled_col = swizzle_permuted_B_j(smem_row, smem_col);
                int swizzled_idx = smem_row * BK + swizzled_col;
                uint32_t load_smem_b_ptr = smem_b_base_ptr + (swizzled_idx + k_load_stage * SB_SIZE) * sizeof(T);
                CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
            }
        }
        CP_ASYNC_COMMIT_GROUP();
    }
    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();

    // 主循环
    uint32_t RA[WARP_TILE_M][4];  // A矩阵保持4个寄存器
    uint32_t RB[WARP_TILE_N][4];  // B矩阵需要4个寄存器来支持16x16
    for (int k_load_base = (K_STAGE - 1) * BK; k_load_base < K; k_load_base += BK) {
        const int k_load_stage = k_load_base / BK;
        int smem_sel = (k_load_stage + 1) % K_STAGE;
        int smem_sel_next = k_load_stage % K_STAGE;

        // 加载A矩阵数据
        for (int load_idx = threadIdx.x * vec_size; load_idx < BM * BK; load_idx += blockDim.x * vec_size) {
            int smem_row = load_idx / (BK);
            int smem_col = load_idx % (BK);
            int global_row = global_m_base + smem_row;
            int global_col = k_load_base + smem_col;

            if (global_row < M && (global_col + vec_size - 1) < K) {
                int load_gmem_a_addr = global_row * K + global_col;
                int swizzled_col = swizzle_permuted_A_j(smem_row, smem_col);
                int swizzled_idx = smem_row * BK + swizzled_col;
                uint32_t load_smem_a_ptr = smem_a_base_ptr + (swizzled_idx + smem_sel_next * SA_SIZE) * sizeof(T);
                CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
            }
        }

        // 加载B矩阵数据
        for (int load_idx = threadIdx.x * vec_size; load_idx < BN * BK; load_idx += blockDim.x * vec_size) {
            int smem_row = load_idx / (BK);
            int smem_col = load_idx % (BK);
            int global_row = global_n_base + smem_row;
            int global_col = k_load_base + smem_col;

            if (global_col + vec_size - 1 < K && global_row < N) {
                int load_gmem_b_addr = global_row * K + global_col;
                int swizzled_col = swizzle_permuted_B_j(smem_row, smem_col);
                int swizzled_idx = smem_row * BK + swizzled_col;
                uint32_t load_smem_b_ptr = smem_b_base_ptr + (swizzled_idx + smem_sel_next * SB_SIZE) * sizeof(T);
                CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
            }
        }
        CP_ASYNC_COMMIT_GROUP();

        // 计算部分
        for (int TILE_K = 0; TILE_K < BK; TILE_K += WMMA_K) {
            // 读取A矩阵数据
            for (int i = 0; i < WARP_TILE_M; ++i) {
                int warp_smem_a_m = warp_m_id * WMMA_M * WARP_TILE_M + i * WMMA_M;
                int warp_smem_a_k = TILE_K;

                int base_row = warp_smem_a_m + (lane_id % 16);
                int base_col = warp_smem_a_k + (lane_id / 16) * vec_size;

                int swizzled_col = swizzle_permuted_A_j(base_row, base_col);
                T *lane_smem_a_ptr = smemA + base_row * BK + swizzled_col + smem_sel * SA_SIZE;

                uint32_t ptr = __cvta_generic_to_shared(lane_smem_a_ptr);
                LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], ptr);
            }

            // 读取B矩阵数据 - 对于16x16x16 MMA，需要读取两组8列的数据
            for (int i = 0; i < WARP_TILE_N; ++i) {
                int warp_smem_b_n = warp_n_id * WMMA_N * WARP_TILE_N + i * WMMA_N;
                int warp_smem_b_k = TILE_K;

                // 第一组8列（0-7列）
                int base_row1 = warp_smem_b_n + (lane_id % 8);
                int base_col1 = warp_smem_b_k + (lane_id / 8) * vec_size;
                int swizzled_col1 = swizzle_permuted_B_j(base_row1, base_col1);
                T *lane_smem_b_ptr1 = smemB + base_row1 * BK + swizzled_col1 + smem_sel * SB_SIZE;
                uint32_t ptr1 = __cvta_generic_to_shared(lane_smem_b_ptr1);
                LDMATRIX_X2(RB[i][0], RB[i][1], ptr1);

                // 第二组8列（8-15列）
                int base_row2 = warp_smem_b_n + 8 + (lane_id % 8);
                int base_col2 = warp_smem_b_k + (lane_id / 8) * vec_size;
                int swizzled_col2 = swizzle_permuted_B_j(base_row2, base_col2);
                T *lane_smem_b_ptr2 = smemB + base_row2 * BK + swizzled_col2 + smem_sel * SB_SIZE;
                uint32_t ptr2 = __cvta_generic_to_shared(lane_smem_b_ptr2);
                LDMATRIX_X2(RB[i][2], RB[i][3], ptr2);
            }

            // 执行16x16x16矩阵乘法
            for (int i = 0; i < WARP_TILE_M; ++i) {
                for (int j = 0; j < WARP_TILE_N; ++j) {
                    MMA161616_BF16(RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3], RC[i][j][4], RC[i][j][5],
                                   RC[i][j][6], RC[i][j][7], RA[i][0], RA[i][1], RA[i][2], RA[i][3], RB[j][0], RB[j][1],
                                   RB[j][2], RB[j][3], RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3], RC[i][j][4],
                                   RC[i][j][5], RC[i][j][6], RC[i][j][7]);
                }
            }
        }

        CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
        __syncthreads();
    }

    // 主循环结束
    if ((K_STAGE - 2) > 0) {
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }

    // 计算剩余阶段
    for (int k_load = 0; k_load < K_STAGE - 1; ++k_load) {
        const int stage_sel = ((K / BK - (K_STAGE - 1) + k_load) % K_STAGE);
        for (int TILE_K = 0; TILE_K < BK; TILE_K += WMMA_K) {
            // 读取A矩阵数据
            for (int i = 0; i < WARP_TILE_M; ++i) {
                int warp_smem_a_m = warp_m_id * WMMA_M * WARP_TILE_M + i * WMMA_M;
                int warp_smem_a_k = TILE_K;

                int base_row = warp_smem_a_m + (lane_id % 16);
                int base_col = warp_smem_a_k + (lane_id / 16) * vec_size;

                int swizzled_col = swizzle_permuted_A_j(base_row, base_col);
                T *lane_smem_a_ptr = smemA + base_row * BK + swizzled_col + stage_sel * SA_SIZE;

                uint32_t ptr = __cvta_generic_to_shared(lane_smem_a_ptr);
                LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], ptr);
            }

            // 读取B矩阵数据 - 对于16x16x16 MMA，需要读取两组8列的数据
            for (int i = 0; i < WARP_TILE_N; ++i) {
                int warp_smem_b_n = warp_n_id * WMMA_N * WARP_TILE_N + i * WMMA_N;
                int warp_smem_b_k = TILE_K;

                // 第一组8列（0-7列）
                int base_row1 = warp_smem_b_n + (lane_id % 8);
                int base_col1 = warp_smem_b_k + (lane_id / 8) * vec_size;
                int swizzled_col1 = swizzle_permuted_B_j(base_row1, base_col1);
                T *lane_smem_b_ptr1 = smemB + base_row1 * BK + swizzled_col1 + stage_sel * SB_SIZE;
                uint32_t ptr1 = __cvta_generic_to_shared(lane_smem_b_ptr1);
                LDMATRIX_X2(RB[i][0], RB[i][1], ptr1);

                // 第二组8列（8-15列）
                int base_row2 = warp_smem_b_n + 8 + (lane_id % 8);
                int base_col2 = warp_smem_b_k + (lane_id / 8) * vec_size;
                int swizzled_col2 = swizzle_permuted_B_j(base_row2, base_col2);
                T *lane_smem_b_ptr2 = smemB + base_row2 * BK + swizzled_col2 + stage_sel * SB_SIZE;
                uint32_t ptr2 = __cvta_generic_to_shared(lane_smem_b_ptr2);
                LDMATRIX_X2(RB[i][2], RB[i][3], ptr2);
            }

            // 执行16x16x16矩阵乘法
            for (int i = 0; i < WARP_TILE_M; ++i) {
                for (int j = 0; j < WARP_TILE_N; ++j) {
                    MMA161616_BF16(RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3], RC[i][j][4], RC[i][j][5],
                                   RC[i][j][6], RC[i][j][7], RA[i][0], RA[i][1], RA[i][2], RA[i][3], RB[j][0], RB[j][1],
                                   RB[j][2], RB[j][3], RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3], RC[i][j][4],
                                   RC[i][j][5], RC[i][j][6], RC[i][j][7]);
                }
            }
        }
    }

    // 写回结果
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            const int tile_m0 = global_m_base + (warp_m_id * WMMA_M * WARP_TILE_M) + i * WMMA_M;
            const int tile_n0 = global_n_base + (warp_n_id * WMMA_N * WARP_TILE_N) + j * WMMA_N;

            int group = lane_id >> 2;
            int tid4 = lane_id & 3;
            int row0 = group;
            int row1 = group + 8;

            // 对于16x16，列偏移需要覆盖16列，分为两组8列
            int col0 = 2 * tid4;
            int col1 = 2 * tid4 + 1;
            int col2 = 2 * tid4 + 8;
            int col3 = 2 * tid4 + 9;

            // 取出8个累加结果（FP32）
            float v0 = reinterpret_cast<float *>(&RC[i][j][0])[0];
            float v1 = reinterpret_cast<float *>(&RC[i][j][1])[0];
            float v2 = reinterpret_cast<float *>(&RC[i][j][2])[0];
            float v3 = reinterpret_cast<float *>(&RC[i][j][3])[0];
            float v4 = reinterpret_cast<float *>(&RC[i][j][4])[0];
            float v5 = reinterpret_cast<float *>(&RC[i][j][5])[0];
            float v6 = reinterpret_cast<float *>(&RC[i][j][6])[0];
            float v7 = reinterpret_cast<float *>(&RC[i][j][7])[0];

            // 写回全局内存
            if ((tile_m0 + row0) < M && (tile_n0 + col0) < N)
                C[(tile_m0 + row0) * N + (tile_n0 + col0)] = v0;
            if ((tile_m0 + row0) < M && (tile_n0 + col1) < N)
                C[(tile_m0 + row0) * N + (tile_n0 + col1)] = v1;
            if ((tile_m0 + row1) < M && (tile_n0 + col0) < N)
                C[(tile_m0 + row1) * N + (tile_n0 + col0)] = v2;
            if ((tile_m0 + row1) < M && (tile_n0 + col1) < N)
                C[(tile_m0 + row1) * N + (tile_n0 + col1)] = v3;
            if ((tile_m0 + row0) < M && (tile_n0 + col2) < N)
                C[(tile_m0 + row0) * N + (tile_n0 + col2)] = v4;
            if ((tile_m0 + row0) < M && (tile_n0 + col3) < N)
                C[(tile_m0 + row0) * N + (tile_n0 + col3)] = v5;
            if ((tile_m0 + row1) < M && (tile_n0 + col2) < N)
                C[(tile_m0 + row1) * N + (tile_n0 + col2)] = v6;
            if ((tile_m0 + row1) < M && (tile_n0 + col3) < N)
                C[(tile_m0 + row1) * N + (tile_n0 + col3)] = v7;
        }
    }
}

// kernel6 with bias
template <typename T, int BM, int BN, int BK, int WMMA_M, int WMMA_N, int WMMA_K, int WAPR_NUM, int K_STAGE,
          int WARP_TILE_M, int WARP_TILE_N>
__global__ void kernel6_with_bias(const T *A, const T *B, const T *bias, T *C, int M, int N, int K) {
    int warp_id = threadIdx.x / 32;
    constexpr int WARP_N_NUM = BN / (WMMA_N * WARP_TILE_N);
    int warp_n_id = warp_id % WARP_N_NUM;
    int warp_m_id = warp_id / WARP_N_NUM;
    int global_m_base = blockIdx.x * BM;
    int global_n_base = blockIdx.y * BN;
    constexpr int SA_SIZE = BM * BK;
    constexpr int SB_SIZE = BN * BK;
    const int lane_id = threadIdx.x % 32;

    __shared__ T smemA[K_STAGE * BM * BK];
    __shared__ T smemB[K_STAGE * BN * BK];
    uint32_t smem_a_base_ptr = __cvta_generic_to_shared(smemA);
    uint32_t smem_b_base_ptr = __cvta_generic_to_shared(smemB);

    constexpr int vec_size = sizeof(float4) / sizeof(T);

    // 对于16x16x16，我们有8个输出寄存器
    uint32_t RC[WARP_TILE_M][WARP_TILE_N][8];
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            RC[i][j][0] = 0;
            RC[i][j][1] = 0;
            RC[i][j][2] = 0;
            RC[i][j][3] = 0;
            RC[i][j][4] = 0;
            RC[i][j][5] = 0;
            RC[i][j][6] = 0;
            RC[i][j][7] = 0;
        }
    }

    // 加载初始阶段数据
    for (int k_load_stage = 0; k_load_stage < (K_STAGE - 1); ++k_load_stage) {
        // 加载A矩阵数据
        for (int load_idx = threadIdx.x * vec_size; load_idx < BM * BK; load_idx += blockDim.x * vec_size) {
            int smem_row = load_idx / (BK);
            int smem_col = load_idx % (BK);
            int global_row = global_m_base + smem_row;
            int global_col = k_load_stage * BK + smem_col;

            if (global_row < M && (global_col + vec_size - 1) < K) {
                int load_gmem_a_addr = global_row * K + global_col;
                int swizzled_col = swizzle_permuted_A_j(smem_row, smem_col);
                int swizzled_idx = smem_row * BK + swizzled_col;
                uint32_t load_smem_a_ptr = smem_a_base_ptr + (swizzled_idx + k_load_stage * SA_SIZE) * sizeof(T);
                CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
            }
        }

        // 加载B矩阵数据
        for (int load_idx = threadIdx.x * vec_size; load_idx < BN * BK; load_idx += blockDim.x * vec_size) {
            int smem_row = load_idx / (BK);
            int smem_col = load_idx % (BK);
            int global_row = global_n_base + smem_row;
            int global_col = k_load_stage * BK + smem_col;

            if (global_col + vec_size - 1 < K && global_row < N) {
                int load_gmem_b_addr = global_row * K + global_col;
                int swizzled_col = swizzle_permuted_B_j(smem_row, smem_col);
                int swizzled_idx = smem_row * BK + swizzled_col;
                uint32_t load_smem_b_ptr = smem_b_base_ptr + (swizzled_idx + k_load_stage * SB_SIZE) * sizeof(T);
                CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
            }
        }
        CP_ASYNC_COMMIT_GROUP();
    }
    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();

    // 主循环
    uint32_t RA[WARP_TILE_M][4];  // A矩阵保持4个寄存器
    uint32_t RB[WARP_TILE_N][4];  // B矩阵需要4个寄存器来支持16x16
    for (int k_load_base = (K_STAGE - 1) * BK; k_load_base < K; k_load_base += BK) {
        const int k_load_stage = k_load_base / BK;
        int smem_sel = (k_load_stage + 1) % K_STAGE;
        int smem_sel_next = k_load_stage % K_STAGE;

        // 加载A矩阵数据
        for (int load_idx = threadIdx.x * vec_size; load_idx < BM * BK; load_idx += blockDim.x * vec_size) {
            int smem_row = load_idx / (BK);
            int smem_col = load_idx % (BK);
            int global_row = global_m_base + smem_row;
            int global_col = k_load_base + smem_col;

            if (global_row < M && (global_col + vec_size - 1) < K) {
                int load_gmem_a_addr = global_row * K + global_col;
                int swizzled_col = swizzle_permuted_A_j(smem_row, smem_col);
                int swizzled_idx = smem_row * BK + swizzled_col;
                uint32_t load_smem_a_ptr = smem_a_base_ptr + (swizzled_idx + smem_sel_next * SA_SIZE) * sizeof(T);
                CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
            }
        }

        // 加载B矩阵数据
        for (int load_idx = threadIdx.x * vec_size; load_idx < BN * BK; load_idx += blockDim.x * vec_size) {
            int smem_row = load_idx / (BK);
            int smem_col = load_idx % (BK);
            int global_row = global_n_base + smem_row;
            int global_col = k_load_base + smem_col;

            if (global_col + vec_size - 1 < K && global_row < N) {
                int load_gmem_b_addr = global_row * K + global_col;
                int swizzled_col = swizzle_permuted_B_j(smem_row, smem_col);
                int swizzled_idx = smem_row * BK + swizzled_col;
                uint32_t load_smem_b_ptr = smem_b_base_ptr + (swizzled_idx + smem_sel_next * SB_SIZE) * sizeof(T);
                CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
            }
        }
        CP_ASYNC_COMMIT_GROUP();

        // 计算部分
        for (int TILE_K = 0; TILE_K < BK; TILE_K += WMMA_K) {
            // 读取A矩阵数据
            for (int i = 0; i < WARP_TILE_M; ++i) {
                int warp_smem_a_m = warp_m_id * WMMA_M * WARP_TILE_M + i * WMMA_M;
                int warp_smem_a_k = TILE_K;

                int base_row = warp_smem_a_m + (lane_id % 16);
                int base_col = warp_smem_a_k + (lane_id / 16) * vec_size;

                int swizzled_col = swizzle_permuted_A_j(base_row, base_col);
                T *lane_smem_a_ptr = smemA + base_row * BK + swizzled_col + smem_sel * SA_SIZE;

                uint32_t ptr = __cvta_generic_to_shared(lane_smem_a_ptr);
                LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], ptr);
            }

            // 读取B矩阵数据 - 对于16x16x16 MMA，需要读取两组8列的数据
            for (int i = 0; i < WARP_TILE_N; ++i) {
                int warp_smem_b_n = warp_n_id * WMMA_N * WARP_TILE_N + i * WMMA_N;
                int warp_smem_b_k = TILE_K;

                // 第一组8列（0-7列）
                int base_row1 = warp_smem_b_n + (lane_id % 8);
                int base_col1 = warp_smem_b_k + (lane_id / 8) * vec_size;
                int swizzled_col1 = swizzle_permuted_B_j(base_row1, base_col1);
                T *lane_smem_b_ptr1 = smemB + base_row1 * BK + swizzled_col1 + smem_sel * SB_SIZE;
                uint32_t ptr1 = __cvta_generic_to_shared(lane_smem_b_ptr1);
                LDMATRIX_X2(RB[i][0], RB[i][1], ptr1);

                // 第二组8列（8-15列）
                int base_row2 = warp_smem_b_n + 8 + (lane_id % 8);
                int base_col2 = warp_smem_b_k + (lane_id / 8) * vec_size;
                int swizzled_col2 = swizzle_permuted_B_j(base_row2, base_col2);
                T *lane_smem_b_ptr2 = smemB + base_row2 * BK + swizzled_col2 + smem_sel * SB_SIZE;
                uint32_t ptr2 = __cvta_generic_to_shared(lane_smem_b_ptr2);
                LDMATRIX_X2(RB[i][2], RB[i][3], ptr2);
            }

            // 执行16x16x16矩阵乘法
            for (int i = 0; i < WARP_TILE_M; ++i) {
                for (int j = 0; j < WARP_TILE_N; ++j) {
                    MMA161616_BF16(RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3], RC[i][j][4], RC[i][j][5],
                                   RC[i][j][6], RC[i][j][7], RA[i][0], RA[i][1], RA[i][2], RA[i][3], RB[j][0], RB[j][1],
                                   RB[j][2], RB[j][3], RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3], RC[i][j][4],
                                   RC[i][j][5], RC[i][j][6], RC[i][j][7]);
                }
            }
        }

        CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
        __syncthreads();
    }

    // 主循环结束
    if ((K_STAGE - 2) > 0) {
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }

    // 计算剩余阶段
    for (int k_load = 0; k_load < K_STAGE - 1; ++k_load) {
        const int stage_sel = ((K / BK - (K_STAGE - 1) + k_load) % K_STAGE);
        for (int TILE_K = 0; TILE_K < BK; TILE_K += WMMA_K) {
            // 读取A矩阵数据
            for (int i = 0; i < WARP_TILE_M; ++i) {
                int warp_smem_a_m = warp_m_id * WMMA_M * WARP_TILE_M + i * WMMA_M;
                int warp_smem_a_k = TILE_K;

                int base_row = warp_smem_a_m + (lane_id % 16);
                int base_col = warp_smem_a_k + (lane_id / 16) * vec_size;

                int swizzled_col = swizzle_permuted_A_j(base_row, base_col);
                T *lane_smem_a_ptr = smemA + base_row * BK + swizzled_col + stage_sel * SA_SIZE;

                uint32_t ptr = __cvta_generic_to_shared(lane_smem_a_ptr);
                LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], ptr);
            }

            // 读取B矩阵数据 - 对于16x16x16 MMA，需要读取两组8列的数据
            for (int i = 0; i < WARP_TILE_N; ++i) {
                int warp_smem_b_n = warp_n_id * WMMA_N * WARP_TILE_N + i * WMMA_N;
                int warp_smem_b_k = TILE_K;

                // 第一组8列（0-7列）
                int base_row1 = warp_smem_b_n + (lane_id % 8);
                int base_col1 = warp_smem_b_k + (lane_id / 8) * vec_size;
                int swizzled_col1 = swizzle_permuted_B_j(base_row1, base_col1);
                T *lane_smem_b_ptr1 = smemB + base_row1 * BK + swizzled_col1 + stage_sel * SB_SIZE;
                uint32_t ptr1 = __cvta_generic_to_shared(lane_smem_b_ptr1);
                LDMATRIX_X2(RB[i][0], RB[i][1], ptr1);

                // 第二组8列（8-15列）
                int base_row2 = warp_smem_b_n + 8 + (lane_id % 8);
                int base_col2 = warp_smem_b_k + (lane_id / 8) * vec_size;
                int swizzled_col2 = swizzle_permuted_B_j(base_row2, base_col2);
                T *lane_smem_b_ptr2 = smemB + base_row2 * BK + swizzled_col2 + stage_sel * SB_SIZE;
                uint32_t ptr2 = __cvta_generic_to_shared(lane_smem_b_ptr2);
                LDMATRIX_X2(RB[i][2], RB[i][3], ptr2);
            }

            // 执行16x16x16矩阵乘法
            for (int i = 0; i < WARP_TILE_M; ++i) {
                for (int j = 0; j < WARP_TILE_N; ++j) {
                    MMA161616_BF16(RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3], RC[i][j][4], RC[i][j][5],
                                   RC[i][j][6], RC[i][j][7], RA[i][0], RA[i][1], RA[i][2], RA[i][3], RB[j][0], RB[j][1],
                                   RB[j][2], RB[j][3], RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3], RC[i][j][4],
                                   RC[i][j][5], RC[i][j][6], RC[i][j][7]);
                }
            }
        }
    }

    // 写回结果并加上bias
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            const int tile_m0 = global_m_base + (warp_m_id * WMMA_M * WARP_TILE_M) + i * WMMA_M;
            const int tile_n0 = global_n_base + (warp_n_id * WMMA_N * WARP_TILE_N) + j * WMMA_N;

            int group = lane_id >> 2;
            int tid4 = lane_id & 3;
            int row0 = group;
            int row1 = group + 8;

            // 对于16x16，列偏移需要覆盖16列，分为两组8列
            int col0 = 2 * tid4;
            int col1 = 2 * tid4 + 1;
            int col2 = 2 * tid4 + 8;
            int col3 = 2 * tid4 + 9;

            // 取出8个累加结果（FP32）
            float v0 = reinterpret_cast<float *>(&RC[i][j][0])[0];
            float v1 = reinterpret_cast<float *>(&RC[i][j][1])[0];
            float v2 = reinterpret_cast<float *>(&RC[i][j][2])[0];
            float v3 = reinterpret_cast<float *>(&RC[i][j][3])[0];
            float v4 = reinterpret_cast<float *>(&RC[i][j][4])[0];
            float v5 = reinterpret_cast<float *>(&RC[i][j][5])[0];
            float v6 = reinterpret_cast<float *>(&RC[i][j][6])[0];
            float v7 = reinterpret_cast<float *>(&RC[i][j][7])[0];

            // 写回全局内存并加上bias
            if ((tile_m0 + row0) < M && (tile_n0 + col0) < N)
                C[(tile_m0 + row0) * N + (tile_n0 + col0)] = static_cast<T>(v0) + bias[tile_n0 + col0];
            if ((tile_m0 + row0) < M && (tile_n0 + col1) < N)
                C[(tile_m0 + row0) * N + (tile_n0 + col1)] = static_cast<T>(v1) + bias[tile_n0 + col1];
            if ((tile_m0 + row1) < M && (tile_n0 + col0) < N)
                C[(tile_m0 + row1) * N + (tile_n0 + col0)] = static_cast<T>(v2) + bias[tile_n0 + col0];
            if ((tile_m0 + row1) < M && (tile_n0 + col1) < N)
                C[(tile_m0 + row1) * N + (tile_n0 + col1)] = static_cast<T>(v3) + bias[tile_n0 + col1];
            if ((tile_m0 + row0) < M && (tile_n0 + col2) < N)
                C[(tile_m0 + row0) * N + (tile_n0 + col2)] = static_cast<T>(v4) + bias[tile_n0 + col2];
            if ((tile_m0 + row0) < M && (tile_n0 + col3) < N)
                C[(tile_m0 + row0) * N + (tile_n0 + col3)] = static_cast<T>(v5) + bias[tile_n0 + col3];
            if ((tile_m0 + row1) < M && (tile_n0 + col2) < N)
                C[(tile_m0 + row1) * N + (tile_n0 + col2)] = static_cast<T>(v6) + bias[tile_n0 + col2];
            if ((tile_m0 + row1) < M && (tile_n0 + col3) < N)
                C[(tile_m0 + row1) * N + (tile_n0 + col3)] = static_cast<T>(v7) + bias[tile_n0 + col3];
        }
    }
}

// ====== CuTe2 高性能内核实现 ======
// 直接基于SICore的工作实现

// Helper function for address conversion
__device__ __forceinline__ uint32_t smem_u32addr_cute2(const void *smem_ptr) {
    uint32_t addr;
    asm volatile("{.reg .u64 u64addr; cvta.to.shared.u64 u64addr, %1; cvt.u32.u64 %0, u64addr;}\n"
                 : "=r"(addr)
                 : "l"(smem_ptr));
    return addr;
}

#define __cvta_generic_to_shared_cute2(ptr) smem_u32addr_cute2(ptr)

template <typename T, int BM, int BN, int BK, int kStage, typename TiledMMA, typename G2SCopyA, typename G2SCopyB,
          typename SmemLayoutA, typename SmemLayoutB, typename SmemLayoutC, typename S2RCopyAtomA,
          typename S2RCopyAtomB, typename R2SCopyAtomC, typename S2GCopyAtomC, typename S2GCopyC,
          const bool BlockSwizzle>
__global__ void gemm_mma_stages_block_swizzle_tn_cute2_kernel(T *Aptr, T *Bptr, T *Dptr, int m, int n, int k) {
    using namespace cute;
    // 初始化共享内存
    extern __shared__ T shm_data[];

    T *Ashm = shm_data;
    T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

    // 初始化线程块索引
    int idx = threadIdx.x;
    int ix = ((int)BlockSwizzle) * blockIdx.z * gridDim.x + blockIdx.x;
    int iy = blockIdx.y;

    // 边界检查
    if (iy * BM >= m || ix * BN >= n)
        return;

    // 关键修正：使用正确的矩阵布局定义
    // 这里的关键是匹配我们验证过的简单版本的tensor定义
    Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));  // A(m,k) row-major
    Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));  // B(n,k) row-major
    Tensor D = make_tensor(make_gmem_ptr(Dptr), make_shape(m, n), make_stride(n, Int<1>{}));  // D(m,n) row-major

    // 将全局Tensor切片
    Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}), make_coord(iy, _));
    Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), make_coord(ix, _));
    Tensor gD = local_tile(D, make_tile(Int<BM>{}, Int<BN>{}), make_coord(iy, ix));

    // 定义共享内存Tensor
    auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

    // MMA分割
    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
    auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));
    auto tCrD = thr_mma.partition_fragment_C(gD);
    clear(tCrD);

    // 数据拷贝设置
    G2SCopyA g2s_tiled_copy_a;
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
    auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);
    auto tAsA_copy = g2s_thr_copy_a.partition_D(sA);

    G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
    auto tBgB_copy = g2s_thr_copy_b.partition_S(gB);
    auto tBsB_copy = g2s_thr_copy_b.partition_D(sB);

    auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
    auto tAsA = s2r_thr_copy_a.partition_S(sA);
    auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);

    auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
    auto tBsB = s2r_thr_copy_b.partition_S(sB);
    auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB);

    // 主循环前的预取
    int itile_to_read = 0;
    int ismem_read = 0;
    int ismem_write = 0;

#pragma unroll
    for (int istage = 0; istage < kStage - 1; ++istage) {
        cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage), tAsA_copy(_, _, _, istage));
        cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage), tBsB_copy(_, _, _, istage));
        cp_async_fence();

        ++itile_to_read;
        ++ismem_write;
    }

    cp_async_wait<kStage - 2>();
    __syncthreads();

    // 加载第一块数据
    int ik = 0;
    cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik, ismem_read), tCrA_view(_, _, ik));
    cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik, ismem_read), tCrB_view(_, _, ik));

    // 主循环
    int ntile = k / BK;
#pragma unroll 1
    for (int itile = 0; itile < ntile; ++itile) {
        int nk = size<2>(tCrA);

#pragma unroll
        for (int ik = 0; ik < nk; ++ik) {
            int ik_next = (ik + 1) % nk;

            if (ik == nk - 1) {
                cp_async_wait<kStage - 2>();
                __syncthreads();
                ismem_read = (ismem_read + 1) % kStage;
            }

            cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read), tCrA_view(_, _, ik_next));
            cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read), tCrB_view(_, _, ik_next));

            if (ik == 0) {
                if (itile_to_read < ntile) {
                    cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read), tAsA_copy(_, _, _, ismem_write));
                    cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read), tBsB_copy(_, _, _, ismem_write));
                    ++itile_to_read;
                    ismem_write = (ismem_write + 1) % kStage;
                }
                cp_async_fence();
            }

            // 执行MMA操作
            cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
        }
    }

    auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC{});

    auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
    auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(idx);
    auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD);
    auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC);

    S2GCopyC s2g_tiled_copy_c;
    auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
    auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC);
    auto tCgC_s2g = s2g_thr_copy_c.partition_D(gD);

    auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g);
    auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);

    int step = size<3>(tCsC_r2s);
#pragma unroll
    for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
#pragma unroll
        for (int j = 0; j < step; ++j) {
            // 改进的类型转换处理，使用正确的类型名称
            if constexpr (std::is_same_v<T, nv_bfloat16> || std::is_same_v<T, half>) {
                // 对于半精度类型，创建临时tensor进行适当的类型转换
                auto t = make_tensor_like<T>(tCrC_r2sx(_, i + j));

                // 手动进行element-wise拷贝和类型转换
                CUTE_UNROLL
                for (int elem = 0; elem < size(t); ++elem) {
                    if constexpr (std::is_same_v<T, nv_bfloat16>) {
                        t(elem) = __float2bfloat16(__bfloat162float(tCrC_r2sx(_, i + j)(elem)));
                    } else {
                        t(elem) = __float2half(__half2float(tCrC_r2sx(_, i + j)(elem)));
                    }
                }

                cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j));
            } else {
                // 对于其他类型，直接拷贝
                cute::copy(r2s_tiled_copy_c, tCrC_r2sx(_, i + j), tCsC_r2s(_, 0, 0, j));
            }
        }
        __syncthreads();

#pragma unroll
        for (int j = 0; j < step; ++j) {
            cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
        }
        __syncthreads();
    }
}

/**
 * @brief 启动修正版高速CUTE GEMM核函数，完全基于SICore的工作实现
 */
template <typename T, const int Stages = 2, const bool BlockSwizzle = false>
void launch_gemm_mma_stages_block_swizzle_tn_cute2(T *a, T *b, T *c, int M, int N, int K, int swizzle_stride) {
    using namespace cute;

    // 使用与SICore相同的配置
    auto BM = Int<128>{};
    auto BN = Int<128>{};
    auto BK = Int<32>{};
    auto KStage = Int<Stages>{};
    auto kSmemLayoutCBatch = Int<4>{};

    // 定义布局 - 与SICore相同
    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{}, make_layout(make_shape(Int<8>{}, Int<BK>{}), make_stride(Int<BK>{}, Int<1>{}))));
    using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtom{}, make_shape(Int<BM>{}, Int<BK>{}, Int<KStage>{})));
    using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtom{}, make_shape(Int<BN>{}, Int<BK>{}, Int<KStage>{})));

    // 使用专门的bfloat16混合精度MMA：输入bfloat16，累加float精度
    using mma_op = SM80_16x8x16_F32BF16BF16F32_TN;  // bfloat16输入，float32累加
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    static constexpr int kMmaEURepeatM = 2;
    static constexpr int kMmaEURepeatN = 2;
    static constexpr int kMmaEURepeatK = 1;

    using mma_atom_shape = mma_traits::Shape_MNK;
    static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
    static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});
    static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});

    using MMA_EU_RepeatT =
        decltype(make_layout(make_shape(Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
    using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
    using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

    // Copy操作定义 - 与SICore相同
    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;
    using G2SCopyA = decltype(make_tiled_copy(
        g2s_copy_atom{}, make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))));
    using G2SCopyB = G2SCopyA;

    using s2r_copy_op = SM75_U32x4_LDSM_N;
    using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
    using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;
    using S2RCopyAtomA = s2r_copy_atom;
    using S2RCopyAtomB = s2r_copy_atom;

    using SmemLayoutAtomC =
        decltype(composition(Swizzle<3, 3, 3>{}, make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}),
                                                             make_stride(Int<kMmaPN>{}, Int<1>{}))));
    using SmemLayoutC =
        decltype(tile_to_shape(SmemLayoutAtomC{}, make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<kSmemLayoutCBatch>{})));

    static_assert(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) >= size(SmemLayoutC{}),
                  "C shared memory request is larger than A's one pipe");

    using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;
    using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
    using S2GCopyC = decltype(make_tiled_copy(
        S2GCopyAtomC{}, make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))));

    // Grid计算
    int BX = (N + BN - 1) / BN;
    int BY = (M + BM - 1) / BM;
    int BZ = BlockSwizzle ? (N + (swizzle_stride)-1) / (swizzle_stride) : 1;
    BX = BlockSwizzle ? (BX + BZ - 1) / BZ : BX;

    dim3 block(size(MMA{}));
    dim3 grid(BX, BY, BZ);

    static constexpr int shm_size_AB = cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
    static constexpr int shm_size_C = cute::cosize(SmemLayoutC{});
    static constexpr int kShmSize = cute::max(shm_size_AB, shm_size_C) * sizeof(T);

    int shm_size = kShmSize;

    cudaFuncSetAttribute(
        gemm_mma_stages_block_swizzle_tn_cute2_kernel<T, BM, BN, BK, KStage, MMA, G2SCopyA, G2SCopyB, SmemLayoutA,
                                                      SmemLayoutB, SmemLayoutC, S2RCopyAtomA, S2RCopyAtomB,
                                                      R2SCopyAtomC, S2GCopyAtomC, S2GCopyC, BlockSwizzle>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

    gemm_mma_stages_block_swizzle_tn_cute2_kernel<T, BM, BN, BK, KStage, MMA, G2SCopyA, G2SCopyB, SmemLayoutA,
                                                  SmemLayoutB, SmemLayoutC, S2RCopyAtomA, S2RCopyAtomB, R2SCopyAtomC,
                                                  S2GCopyAtomC, S2GCopyC, BlockSwizzle>
        <<<grid, block, shm_size>>>(a, b, c, M, N, K);
}

// cute2的包装器函数
template <typename T>
void cute2_matmul_kernel(const T *a, const T *b, T *c, int M, int N, int K, cudaStream_t stream) {
    // 检查类型支持
    if constexpr (!std::is_same_v<T, nv_bfloat16>) {
        throw std::runtime_error("cute2 kernel currently only supports nv_bfloat16 type");
    }

    cudaStreamSynchronize(stream);

    // 使用2阶段流水线，无Block Swizzle
    launch_gemm_mma_stages_block_swizzle_tn_cute2<T, 2, false>(const_cast<T *>(a), const_cast<T *>(b), c, M, N, K, 1);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA cute2 kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
}

}  // namespace advanced_kernels