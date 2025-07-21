#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "cudaOP.cuh"
#include "ptx_common.h"
#define DQKV_VALUE 128
#define B_C_VALUE 16

namespace cuda_OP {

template <typename T>
__device__ void compute_attention_block(int j, int compute_stage_idx, int cache_length, int start_idx,
                                        int total_seq_len, int dqkv, int B_c, T softmax_scale, const float* s_qi,
                                        const T* s_kj, const T* s_vj, float* s_score_buf, float* s_lm, float* s_s_score,
                                        float* s_o) {
    const int d_tid = threadIdx.x;
    const int token_tid = threadIdx.y;
    constexpr int vec_unit = 16 / sizeof(T);
    const int vecCount = dqkv / vec_unit;

    float local_score = 0.0f;
    for (int i = d_tid; i < vecCount; i += blockDim.x) {
#pragma unroll
        for (int l = 0; l < vec_unit; l++) {
            float k_val =
                static_cast<float>(s_kj[compute_stage_idx * B_c * dqkv + token_tid * dqkv + i * vec_unit + l]);
            local_score += s_qi[i * vec_unit + l] * k_val;
        }
    }

    unsigned int mask = 0xFFFFFFFF;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_score += __shfl_down_sync(mask, local_score, offset);
    }

    int token_index = j * B_c + token_tid;
    bool valid = (token_index < cache_length);
    int absolute_token_idx = start_idx + token_index;
    bool absolutely_valid = valid && (absolute_token_idx < total_seq_len);

    if (d_tid == 0) {
        if (absolutely_valid) {
            s_score_buf[token_tid] = local_score * static_cast<float>(softmax_scale);
        } else {
            s_score_buf[token_tid] = -FLT_MAX;
        }
    }
    __syncthreads();

    __shared__ float cur_m_s;
    float warp_val = (d_tid < B_c && threadIdx.y == 0) ? s_score_buf[d_tid] : -FLT_MAX;
    unsigned int mask_max = 0xFFFFFFFF;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        warp_val = fmaxf(warp_val, __shfl_down_sync(mask_max, warp_val, offset));
    }
    if (d_tid == 0 && threadIdx.y == 0) {
        cur_m_s = warp_val;
    }
    __syncthreads();
    float cur_m = cur_m_s;

    __shared__ float cur_l_s;
    float warp_val_l = 0.0f;
    if (d_tid < B_c && threadIdx.y == 0) {
        float score_val = s_score_buf[d_tid];
        float exp_val = expf(score_val - cur_m);
        s_s_score[d_tid] = exp_val;
        warp_val_l = exp_val;
    } else {
        warp_val_l = 0.0f;
    }
    __syncthreads();

    unsigned int mask_sum = 0xFFFFFFFF;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        warp_val_l += __shfl_down_sync(mask_sum, warp_val_l, offset);
    }
    if (d_tid == 0 && threadIdx.y == 0) {
        cur_l_s = warp_val_l;
    }
    __syncthreads();
    float cur_l = cur_l_s;

    float& global_m = s_lm[0];
    float& global_l = s_lm[1];

    if (j == 0) {
        if (token_tid == 0) {
            for (int k_dim = d_tid; k_dim < dqkv; k_dim += blockDim.x) {
                float current_dim_partial_out = 0.0f;
                for (int i_tok = 0; i_tok < B_c; ++i_tok) {
                    float exp_score = s_s_score[i_tok];
                    float v_val = static_cast<float>(s_vj[compute_stage_idx * B_c * dqkv + i_tok * dqkv + k_dim]);
                    current_dim_partial_out = fmaf(exp_score, v_val, current_dim_partial_out);
                }
                s_o[k_dim] = current_dim_partial_out;
            }
        }
        if (token_tid == 0 && d_tid == 0) {
            global_m = cur_m;
            global_l = cur_l;
        }
    } else {
        float old_global_m = global_m;
        float old_global_l = global_l;
        float new_global_m = fmaxf(old_global_m, cur_m);
        float exp_old = __expf(old_global_m - new_global_m);
        float exp_cur = __expf(cur_m - new_global_m);

        if (token_tid == 0) {
            for (int k_dim = d_tid; k_dim < dqkv; k_dim += blockDim.x) {
                float current_dim_partial_out = 0.0f;
                for (int i_tok = 0; i_tok < B_c; ++i_tok) {
                    float exp_score = s_s_score[i_tok];
                    float v_val = static_cast<float>(s_vj[compute_stage_idx * B_c * dqkv + i_tok * dqkv + k_dim]);
                    current_dim_partial_out = fmaf(exp_score, v_val, current_dim_partial_out);
                }
                float old_out_val = s_o[k_dim];
                float new_out_val = old_out_val * exp_old + current_dim_partial_out * exp_cur;
                s_o[k_dim] = new_out_val;
            }
        }

        if (token_tid == 0 && d_tid == 0) {
            float new_global_l = old_global_l * exp_old + cur_l * exp_cur;
            global_m = new_global_m;
            global_l = new_global_l;
        }
    }
    __syncthreads();
}

template <typename T, int STAGE = 2>
__global__ void flash_attention_kernel_graph_fixed(T* q, const T* total_k, const T* total_v, T** output_ptrs,
                                                   int* segment_info, int n_q_h, int n_kv_h, int dqkv, int B_c, int B_r,
                                                   int n_groups, int T_r, T softmax_scale, int* pingpong_index,
                                                   const T* zero_vec) {
    int total_seq_len = segment_info[*pingpong_index];
    const int FIXED_BRANCHES = 3;

    if (blockIdx.y >= FIXED_BRANCHES)
        return;

    int branches_needed = FIXED_BRANCHES;
    int tokens_per_branch = (total_seq_len + branches_needed - 1) / branches_needed;

    int start_idx, end_idx;
    if (blockIdx.y == 0) {
        start_idx = 0;
        end_idx = min(tokens_per_branch, total_seq_len);
    } else if (blockIdx.y == 1) {
        start_idx = tokens_per_branch;
        end_idx = min(2 * tokens_per_branch, total_seq_len);
    } else {  // blockIdx.y == 2
        start_idx = 2 * tokens_per_branch;
        end_idx = total_seq_len;
    }

    int cache_length = end_idx - start_idx;
    if (cache_length <= 0)
        return;

    int T_c = (cache_length + B_c - 1) / B_c;
    T* att_output = output_ptrs[blockIdx.y];

    if (dqkv != DQKV_VALUE || B_c != B_C_VALUE)
        return;

    extern __shared__ char s_mem[];
    float* s_qi = (float*)s_mem;
    T* s_kj = (T*)(s_qi + DQKV_VALUE);
    T* s_vj = (T*)(s_kj + STAGE * B_C_VALUE * DQKV_VALUE);
    float* s_score_buf = (float*)((char*)s_vj + STAGE * B_C_VALUE * DQKV_VALUE * sizeof(T));
    float* s_lm = (float*)(s_score_buf + B_C_VALUE);
    float* s_s_score = (float*)(s_lm + 2);
    float* s_o = (float*)(s_s_score + B_C_VALUE);

    const int d_tid = threadIdx.x;
    const int tid = threadIdx.y * blockDim.x + d_tid;
    const int num_threads = blockDim.x * blockDim.y;
    const int head_id = blockIdx.x;
    const int q_offset = head_id * dqkv;
    const int kv_head = head_id / n_groups;
    constexpr int vec_unit = 16 / sizeof(T);

    if (threadIdx.y == 0) {
        for (int i = d_tid; i < dqkv; i += blockDim.x) {
            s_qi[i] = static_cast<float>(q[q_offset + i]);
        }
    }
    __syncthreads();

    for (int i = 0; i < STAGE - 1; ++i) {
        int token_index_base = i * B_c;
        if (token_index_base >= cache_length)
            continue;
        for (int load_idx = tid * vec_unit; load_idx < B_c * dqkv; load_idx += num_threads * vec_unit) {
            int smem_row = load_idx / dqkv;
            int dim_idx = load_idx % dqkv;
            int absolute_token_idx = start_idx + token_index_base + smem_row;
            bool valid = (token_index_base + smem_row < cache_length) && (absolute_token_idx < total_seq_len);

            const T* k_ptr = total_k + (absolute_token_idx * n_kv_h + kv_head) * dqkv + dim_idx;
            const T* v_ptr = total_v + (absolute_token_idx * n_kv_h + kv_head) * dqkv + dim_idx;

            uint32_t k_smem_cp = __cvta_generic_to_shared(&s_kj[i * B_c * dqkv + smem_row * dqkv + dim_idx]);
            uint32_t v_smem_cp = __cvta_generic_to_shared(&s_vj[i * B_c * dqkv + smem_row * dqkv + dim_idx]);

            if (valid) {
                CP_ASYNC_CG(k_smem_cp, k_ptr, 16);
                CP_ASYNC_CG(v_smem_cp, v_ptr, 16);
            } else {
                CP_ASYNC_CG(k_smem_cp, zero_vec, 16);
                CP_ASYNC_CG(v_smem_cp, zero_vec, 16);
            }
        }
        CP_ASYNC_COMMIT_GROUP();
    }

    int main_loop_iters = (T_c < STAGE) ? 0 : (T_c - (STAGE - 1));
    for (int j = 0; j < main_loop_iters; ++j) {
        int j_load = j + STAGE - 1;
        int compute_stage_idx = j % STAGE;
        int load_stage_idx = j_load % STAGE;

        CP_ASYNC_WAIT_GROUP(STAGE - 2);
        __syncthreads();

        int token_index_base_load = j_load * B_c;
        if (token_index_base_load < cache_length) {
            for (int load_idx = tid * vec_unit; load_idx < B_c * dqkv; load_idx += num_threads * vec_unit) {
                int smem_row = load_idx / dqkv;
                int dim_idx = load_idx % dqkv;
                int absolute_token_idx = start_idx + token_index_base_load + smem_row;
                bool valid = (token_index_base_load + smem_row < cache_length) && (absolute_token_idx < total_seq_len);

                const T* k_ptr = total_k + (absolute_token_idx * n_kv_h + kv_head) * dqkv + dim_idx;
                const T* v_ptr = total_v + (absolute_token_idx * n_kv_h + kv_head) * dqkv + dim_idx;

                uint32_t k_smem_cp =
                    __cvta_generic_to_shared(&s_kj[load_stage_idx * B_c * dqkv + smem_row * dqkv + dim_idx]);
                uint32_t v_smem_cp =
                    __cvta_generic_to_shared(&s_vj[load_stage_idx * B_c * dqkv + smem_row * dqkv + dim_idx]);

                if (valid) {
                    CP_ASYNC_CG(k_smem_cp, k_ptr, 16);
                    CP_ASYNC_CG(v_smem_cp, v_ptr, 16);
                } else {
                    CP_ASYNC_CG(k_smem_cp, zero_vec, 16);
                    CP_ASYNC_CG(v_smem_cp, zero_vec, 16);
                }
            }
            CP_ASYNC_COMMIT_GROUP();
        }

        compute_attention_block(j, compute_stage_idx, cache_length, start_idx, total_seq_len, dqkv, B_c, softmax_scale,
                                s_qi, s_kj, s_vj, s_score_buf, s_lm, s_s_score, s_o);
    }
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();

    for (int i = 0; i < STAGE - 1; ++i) {
        int j_compute = main_loop_iters + i;
        if (j_compute >= T_c)
            break;
        int compute_stage_idx = j_compute % STAGE;
        compute_attention_block(j_compute, compute_stage_idx, cache_length, start_idx, total_seq_len, dqkv, B_c,
                                softmax_scale, s_qi, s_kj, s_vj, s_score_buf, s_lm, s_s_score, s_o);
    }

    if (threadIdx.y == 0) {
        int out_offset = head_id * (dqkv + 2);
        float& global_m = s_lm[0];
        float& global_l = s_lm[1];
        for (int i = d_tid; i < DQKV_VALUE; i += blockDim.x) {
            att_output[out_offset + i] = static_cast<T>(s_o[i]);
        }
        if (d_tid == 0) {
            att_output[out_offset + dqkv] = static_cast<T>(global_m);
            att_output[out_offset + dqkv + 1] = static_cast<T>(global_l);
        }
    }
}

template <typename T>
void flash_attention_graph_fixed(Tensor<T>& Q, const Tensor<T>& total_K, const Tensor<T>& total_V, T** d_output_ptrs,
                                 int* d_segment_info, int n_kv_heads, cudaStream_t stream, int* pingpong_index) {
    int dqkv = Q.sizes()[2];
    if (dqkv != DQKV_VALUE) {
        throw std::runtime_error("dqkv does not match the predefined value");
    }

    static T* d_zero_vec = nullptr;
    static std::once_flag zero_vec_flag;
    std::call_once(zero_vec_flag, [stream]() {
        constexpr int VEC_SIZE = 16 / sizeof(T);
        cudaMalloc(&d_zero_vec, VEC_SIZE * sizeof(T));
        std::vector<T> h_zero_vec(VEC_SIZE, static_cast<T>(0.0f));
        cudaMemcpyAsync(d_zero_vec, h_zero_vec.data(), VEC_SIZE * sizeof(T), cudaMemcpyHostToDevice, stream);
    });

    float softmax_scale = 1.0f / sqrtf(static_cast<float>(dqkv));
    int n_q_h = Q.sizes()[1];
    int n_groups = n_q_h / n_kv_heads;
    int B_r = 1;
    int T_r = 1;
    int B_c = B_C_VALUE;

    const int FIXED_BRANCHES = 3;
    dim3 grid(n_q_h, FIXED_BRANCHES);
    dim3 block(32, B_c);

    constexpr int STAGE = 2;
    size_t smem_size = (DQKV_VALUE * sizeof(float)) +                      // s_qi
                       (STAGE * B_C_VALUE * DQKV_VALUE * sizeof(T)) * 2 +  // s_kj, s_vj
                       (B_C_VALUE * sizeof(float)) +                       // s_score_buf
                       (2 * sizeof(float)) +                               // s_lm
                       (B_C_VALUE * sizeof(float)) +                       // s_s_score
                       (DQKV_VALUE * sizeof(float));                       // s_o

    flash_attention_kernel_graph_fixed<T, STAGE><<<grid, block, smem_size, stream>>>(
        Q.data_ptr(), total_K.data_ptr(), total_V.data_ptr(), d_output_ptrs, d_segment_info, n_q_h, n_kv_heads, dqkv,
        B_c, B_r, n_groups, T_r, static_cast<T>(softmax_scale), pingpong_index, d_zero_vec);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error in flash_attention_graph_fixed: " + std::string(cudaGetErrorString(err)));
    }
}

template void flash_attention_graph_fixed<float>(Tensor<float>& Q, const Tensor<float>& total_K,
                                                 const Tensor<float>& total_V, float** d_output_ptrs,
                                                 int* d_segment_info, int n_kv_heads, cudaStream_t stream,
                                                 int* pingpong_index);

template void flash_attention_graph_fixed<__nv_bfloat16>(Tensor<__nv_bfloat16>& Q, const Tensor<__nv_bfloat16>& total_K,
                                                         const Tensor<__nv_bfloat16>& total_V,
                                                         __nv_bfloat16** d_output_ptrs, int* d_segment_info,
                                                         int n_kv_heads, cudaStream_t stream, int* pingpong_index);

}  // namespace cuda_OP
