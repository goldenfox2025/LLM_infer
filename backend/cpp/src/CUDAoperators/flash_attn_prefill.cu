#include <float.h>  // 用于 FLT_MAX

#include <stdexcept>  // 用于 std::runtime_error
#include <string>

#include "cudaOP.cuh"

__device__ __forceinline__ float warpReduceSum(float val) {
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return __shfl_sync(0xFFFFFFFF, val, 0);
}

__device__ __forceinline__ float warpReduceMax(float val) {
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return __shfl_sync(0xFFFFFFFF, val, 0);
}

namespace cuda_OP {
// T_r 是一个线程块负责的Q段长度
// current_kv_cache_total_len 是当前线程块负责的token所需要处理的总kv长度
// B_c 是单次加载的kv长度
// B_r 是单次加载的Q段长度
// WARP_NUM 是线程块内的warp数量
// DQKV 是每个头的维度
// 启动参数：[total_seq_len / T_r, n_heads]

template <typename T, int B_c, int B_r, int T_r, int WARP_NUM = 4, int DQKV = 128>
__global__ void flash_attn_prefill_kernel(const T* __restrict__ q_global, const T* __restrict__ k_global,
                                          const T* __restrict__ v_global, T* __restrict__ out_global,
                                          int num_q_heads_total, int num_kv_heads_total, int GQA_n_group,
                                          int current_prefill_q_length, int current_kv_cache_total_len,
                                          int q_offset_in_kv_timeline) {
    // 找到当前线程块负责的Q段和Q头
    const int q_segment_idx = blockIdx.x;
    const int q_head_idx_global = blockIdx.y;
    const int kv_head_idx_global = q_head_idx_global / GQA_n_group;

    // 找到当前线程块内的线程和warp索引
    const int tid = threadIdx.x;
    const int warp_id = tid / warpSize;
    const int lane_id = tid % warpSize;

    // 找到当前线程块负责的Q段起始索引
    const int q_segment_start_idx = q_segment_idx * T_r;

    // 声明共享内存
    __shared__ T q_smem[B_r][DQKV];
    __shared__ T k_smem[B_c][DQKV];
    __shared__ T v_smem[B_c][DQKV];
    __shared__ float scores_smem[B_r][B_c];
    __shared__ T o_smem[B_r][DQKV];
    __shared__ float m_stats[B_r];
    __shared__ float l_stats[B_r];

    // 外循环：遍历当前线程块负责的Q段
    for (int q_block_offset = 0; q_block_offset < T_r; q_block_offset += B_r) {
        // 初始化输出累加器
        for (int row_idx = tid; row_idx < B_r; row_idx += blockDim.x) {
            m_stats[row_idx] = -FLT_MAX;
            l_stats[row_idx] = 0.0f;
            for (int dim_idx = 0; dim_idx < DQKV; ++dim_idx) {
                o_smem[row_idx][dim_idx] = static_cast<T>(0.0f);
            }
        }
        __syncthreads();

        // 从全局内存加载Q块到共享内存
        for (int q_smem_row = 0; q_smem_row < B_r; ++q_smem_row) {
            int q_token_idx = q_segment_start_idx + q_block_offset + q_smem_row;

            bool is_valid_q = (q_token_idx < current_prefill_q_length);
            const T* q_global_ptr = q_global + (q_token_idx * num_q_heads_total + q_head_idx_global) * DQKV;
            for (int dim_idx = tid; dim_idx < DQKV; dim_idx += blockDim.x) {
                if (is_valid_q) {
                    q_smem[q_smem_row][dim_idx] = q_global_ptr[dim_idx];
                } else {
                    q_smem[q_smem_row][dim_idx] = static_cast<T>(0.0f);
                }
            }
        }
        __syncthreads();

        // 当前负责的、大小为B_c的kv块
        for (int kv_block_offset = 0; kv_block_offset < current_kv_cache_total_len; kv_block_offset += B_c) {
            // 加载K, V块到共享内存
            for (int smem_row = 0; smem_row < B_c; ++smem_row) {
                int k_token_idx = kv_block_offset + smem_row;
                bool is_valid_k = (k_token_idx < current_kv_cache_total_len);
                const T* k_global_ptr = k_global + (k_token_idx * num_kv_heads_total + kv_head_idx_global) * DQKV;
                for (int dim_idx = tid; dim_idx < DQKV; dim_idx += blockDim.x) {
                    if (is_valid_k) {
                        k_smem[smem_row][dim_idx] = k_global_ptr[dim_idx];
                    } else {
                        k_smem[smem_row][dim_idx] = static_cast<T>(0.0f);
                    }
                }
            }
            for (int smem_row = 0; smem_row < B_c; ++smem_row) {
                int v_token_idx = kv_block_offset + smem_row;
                bool is_valid_v = (v_token_idx < current_kv_cache_total_len);
                const T* v_global_ptr = v_global + (v_token_idx * num_kv_heads_total + kv_head_idx_global) * DQKV;
                for (int dim_idx = tid; dim_idx < DQKV; dim_idx += blockDim.x) {
                    if (is_valid_v) {
                        v_smem[smem_row][dim_idx] = v_global_ptr[dim_idx];
                    } else {
                        v_smem[smem_row][dim_idx] = static_cast<T>(0.0f);
                    }
                }
            }
            __syncthreads();

            // 计算注意力分数矩阵 S = Q * K^T
            // 一个warp负责一段Q
            constexpr int Q_ROWS_PER_WARP = (B_r + WARP_NUM - 1) / WARP_NUM;
            for (int q_row_in_warp = 0; q_row_in_warp < Q_ROWS_PER_WARP; ++q_row_in_warp) {
                // 找到当前warp负责的Q段
                int q_smem_row = warp_id * Q_ROWS_PER_WARP + q_row_in_warp;
                // 本次循环所负责B_r个Q范围
                if (q_smem_row < B_r) {
                    int q_token_idx = q_segment_start_idx + q_block_offset + q_smem_row;
                    int q_abs_pos = q_offset_in_kv_timeline + q_token_idx;
                    for (int k_smem_row = 0; k_smem_row < B_c; ++k_smem_row) {
                        float dot_sum = 0.0f;
                        for (int d_idx = lane_id; d_idx < DQKV; d_idx += warpSize) {
                            dot_sum += static_cast<float>(q_smem[q_smem_row][d_idx]) *
                                       static_cast<float>(k_smem[k_smem_row][d_idx]);
                        }
                        float score = warpReduceSum(dot_sum);
                        score *= (1.0f / sqrtf(static_cast<float>(DQKV)));

                        if (lane_id == 0) {
                            int k_token_idx = kv_block_offset + k_smem_row;
                            bool q_is_padding = (q_token_idx >= current_prefill_q_length);
                            bool k_is_padding = (k_token_idx >= current_kv_cache_total_len);
                            bool masked_by_causal = (k_token_idx > q_abs_pos);

                            if (q_is_padding || k_is_padding || masked_by_causal) {
                                scores_smem[q_smem_row][k_smem_row] = -FLT_MAX;
                            } else {
                                scores_smem[q_smem_row][k_smem_row] = score;
                            }
                        }
                    }
                }
            }
            __syncthreads();

            // 在线Softmax和累积输出 O = P * V
            // 一个warp负责一个token
            for (int q_row_in_warp = 0; q_row_in_warp < Q_ROWS_PER_WARP; ++q_row_in_warp) {
                int q_smem_row = warp_id * Q_ROWS_PER_WARP + q_row_in_warp;
                if (q_smem_row >= B_r)
                    continue;

                int q_token_idx = q_segment_start_idx + q_block_offset + q_smem_row;
                if (q_token_idx >= current_prefill_q_length)
                    continue;
                // 本次迭代，取出上次存放的m和l
                float m_prev = m_stats[q_smem_row];
                float l_prev = l_stats[q_smem_row];
                // 计算当前warp正在处理的token的m
                float m_block = -FLT_MAX;
                for (int col_idx = lane_id; col_idx < B_c; col_idx += warpSize) {
                    m_block = max(m_block, scores_smem[q_smem_row][col_idx]);
                }
                m_block = warpReduceMax(m_block);
                float m_new = max(m_prev, m_block);

                float scale = expf(m_prev - m_new);
                if (lane_id == 0) {
                    l_stats[q_smem_row] = l_prev * scale;
                }
                // 写入本次迭代的结果
                // online softmax
                // m_new = max(m_old, m_block)
                // scale = exp(m_old - m_new)
                // l_new = new_scale * l_old + l_block_sum

                // 计算本次迭代中，当前warp负责的token的l_block_sum
                float l_block_sum = 0.0f;
                for (int col_idx = lane_id; col_idx < B_c; col_idx += warpSize) {
                    float s = scores_smem[q_smem_row][col_idx];
                    float p = (s == -FLT_MAX) ? 0.0f : expf(s - m_new);
                    scores_smem[q_smem_row][col_idx] = p;
                    l_block_sum += p;
                }
                l_block_sum = warpReduceSum(l_block_sum);

                if (lane_id == 0) {
                    l_stats[q_smem_row] += l_block_sum;
                    m_stats[q_smem_row] = m_new;
                }
                __syncthreads();

                // 计算本次迭代中，当前warp负责的token的输出
                for (int d_idx = lane_id; d_idx < DQKV; d_idx += warpSize) {
                    float pv_sum = 0.0f;
                    for (int k_smem_row = 0; k_smem_row < B_c; ++k_smem_row) {
                        pv_sum += scores_smem[q_smem_row][k_smem_row] * static_cast<float>(v_smem[k_smem_row][d_idx]);
                    }
                    o_smem[q_smem_row][d_idx] =
                        static_cast<T>(static_cast<float>(o_smem[q_smem_row][d_idx]) * scale + pv_sum);
                }
            }
            __syncthreads();
        }

        // 写回最终结果到全局内存
        // 我们还是要确认当前线程块负责的token
        constexpr int Q_ROWS_PER_WARP_WRITE = (B_r + WARP_NUM - 1) / WARP_NUM;
        for (int q_row_in_warp = 0; q_row_in_warp < Q_ROWS_PER_WARP_WRITE; ++q_row_in_warp) {
            int q_smem_row = warp_id * Q_ROWS_PER_WARP_WRITE + q_row_in_warp;

            if (q_smem_row >= B_r)
                continue;

            int q_token_idx = q_segment_start_idx + q_block_offset + q_smem_row;

            if (q_token_idx < current_prefill_q_length) {
                float l_final = l_stats[q_smem_row];
                float inv_l_final = (l_final == 0.0f) ? 0.0f : (1.0f / l_final);
                T* out_global_ptr = out_global + (q_token_idx * num_q_heads_total + q_head_idx_global) * DQKV;
                for (int d_idx = lane_id; d_idx < DQKV; d_idx += warpSize) {
                    out_global_ptr[d_idx] = static_cast<T>(static_cast<float>(o_smem[q_smem_row][d_idx]) * inv_l_final);
                }
            }
        }
        __syncthreads();
    }
}

template <typename T>
void flash_attention_prefill(const Tensor<T>& Q, const Tensor<T>& K, const Tensor<T>& V, Tensor<T>& output, int n_heads,
                             int n_kv_heads, int head_dim, int seq_len, int total_seq_len, int offset,
                             cudaStream_t stream) {
    if (head_dim != 128) {
        throw std::runtime_error("Flash attention prefill currently only supports head_dim=128");
    }

    int n_groups = n_heads / n_kv_heads;

    // 启动配置
    constexpr int B_c = 32;
    constexpr int B_r = 4;
    constexpr int WARP_NUM = 2;
    constexpr int DQKV_val = 128;
    constexpr int T_r = 16;

    // 计算Grid维度
    int num_q_segments = (seq_len + T_r - 1) / T_r;
    dim3 grid(num_q_segments, n_heads);
    dim3 block(WARP_NUM * 32);

    flash_attn_prefill_kernel<T, B_c, B_r, T_r, WARP_NUM, DQKV_val>
        <<<grid, block, 0, stream>>>(Q.data_ptr(), K.data_ptr(), V.data_ptr(), output.data_ptr(), n_heads, n_kv_heads,
                                     n_groups, seq_len, total_seq_len, offset);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error in flash_attention_prefill: " + std::string(cudaGetErrorString(err)));
    }
}

// 显式模板实例化
template void flash_attention_prefill<float>(const Tensor<float>& Q, const Tensor<float>& K, const Tensor<float>& V,
                                             Tensor<float>& output, int n_heads, int n_kv_heads, int head_dim,
                                             int seq_len, int total_seq_len, int offset, cudaStream_t stream);

template void flash_attention_prefill<__nv_bfloat16>(const Tensor<__nv_bfloat16>& Q, const Tensor<__nv_bfloat16>& K,
                                                     const Tensor<__nv_bfloat16>& V, Tensor<__nv_bfloat16>& output,
                                                     int n_heads, int n_kv_heads, int head_dim, int seq_len,
                                                     int total_seq_len, int offset, cudaStream_t stream);

}  // namespace cuda_OP