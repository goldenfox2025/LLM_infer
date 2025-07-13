#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ptx_common.h"  // 使用正确的路径

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

}  // namespace advanced_kernels