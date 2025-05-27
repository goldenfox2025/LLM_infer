#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cudaOP.cuh"

#define DQKV_VALUE 128
#define B_C_VALUE 8
#define MAX_BRANCHES 5

constexpr int WARP_SIZE = 32;

namespace cuda_OP {

// CUDA图优化版本的flash attention kernel
// 直接从连续的KV缓存中读取数据，仿照flash_attention_variable的模式
template <typename T>
__global__ void flash_attention_kernel_graph_fixed(
    T *q,
    const T *total_k,    // 连续的K缓存 [total_seq_len, n_kv_h, dqkv]
    const T *total_v,    // 连续的V缓存 [total_seq_len, n_kv_h, dqkv]
    T **output_ptrs,     // 固定的输出指针数组
    int *segment_info,   // 分段信息：[total_seq_len, branch_count, branch_lengths...]
    int n_q_h, int n_kv_h, int dqkv, int B_c, int B_r, int n_groups, int T_r,
    T softmax_scale) {

  // 从设备内存读取分段信息
  int total_seq_len = segment_info[0];
  // segment_info[1] (active_branches) 已经无用，始终使用固定3分支

  // 固定使用3分支模式，blockIdx.y就是分支索引(0,1,2)
  const int FIXED_BRANCHES = 3;

  // 检查分支ID是否有效
  if (blockIdx.y >= FIXED_BRANCHES) return;

  // 关键修复：使用与普通推理完全相同的分段逻辑
  // 普通推理使用：tokens_per_branch = (total_seq_len + branches_needed - 1) / branches_needed
  // 这是向上取整的除法，确保所有token都被覆盖
  int branches_needed = FIXED_BRANCHES;
  int tokens_per_branch = (total_seq_len + branches_needed - 1) / branches_needed;

  int start_idx, end_idx;
  if (blockIdx.y == 0) {
    start_idx = 0;
    end_idx = min(tokens_per_branch, total_seq_len);
  } else if (blockIdx.y == 1) {
    start_idx = tokens_per_branch;
    end_idx = min(2 * tokens_per_branch, total_seq_len);
  } else { // blockIdx.y == 2
    start_idx = 2 * tokens_per_branch;
    end_idx = total_seq_len;
  }

  int cache_length = end_idx - start_idx;

  // 如果分支长度为0，直接退出
  if (cache_length <= 0) return;

  int T_c = (cache_length + B_c - 1) / B_c;
  T *att_output = output_ptrs[blockIdx.y];

  // 验证参数
  if (dqkv != DQKV_VALUE || B_c != B_C_VALUE) return;

  __shared__ float s_qi[DQKV_VALUE];
  __shared__ float s_vj[B_C_VALUE * DQKV_VALUE];
  __shared__ float s_score_buf[B_C_VALUE];
  __shared__ float s_lm[2];
  __shared__ float s_s_score[B_C_VALUE];
  __shared__ float s_o[DQKV_VALUE];

  const int d_tid = threadIdx.x;
  const int token_tid = threadIdx.y;
  const int head_id = blockIdx.x;
  const int q_offset = head_id * dqkv;
  const int kv_head = head_id / n_groups;

  constexpr int vec_unit = 16 / sizeof(T);
  Vec<T, vec_unit> vq, vk, vv;

  const int vecCount = dqkv / vec_unit;
  for (int i = d_tid; i < vecCount; i += blockDim.x) {
    vq.f4 = *reinterpret_cast<const float4 *>(&q[q_offset + i * vec_unit]);
    if (token_tid < vec_unit)
      s_qi[i * vec_unit + token_tid] = static_cast<float>(vq.t[token_tid]);
  }
  __syncthreads();

  float &global_m = s_lm[0];
  float &global_l = s_lm[1];

  // 遍历 KV 分块
  for (int j = 0; j < T_c; ++j) {
    int token_index = j * B_c + token_tid;
    bool valid = (token_index < cache_length);
    float local_score = 0.0f;

    for (int i = d_tid; i < vecCount; i += blockDim.x) {
      // 关键修复：正确计算在整个KV cache中的绝对索引
      // token_index是分支内相对索引，需要加上start_idx得到绝对位置
      int absolute_token_idx = start_idx + token_index;

      // 关键修复：确保不会越界访问
      if (valid && absolute_token_idx < total_seq_len) {
        int index = (absolute_token_idx * n_kv_h + kv_head) * dqkv + i * vec_unit;
        vk.f4 = *reinterpret_cast<const float4 *>(&total_k[index]);
        vv.f4 = *reinterpret_cast<const float4 *>(&total_v[index]);
#pragma unroll
        for (int l = 0; l < vec_unit; l++) {
          float k_val = static_cast<float>(vk.t[l]);
          float v_val = static_cast<float>(vv.t[l]);
          local_score += s_qi[i * vec_unit + l] * k_val;
          s_vj[token_tid * DQKV_VALUE + i * vec_unit + l] = v_val;
        }
      } else {
#pragma unroll
        for (int l = 0; l < vec_unit; l++) {
          s_vj[token_tid * DQKV_VALUE + i * vec_unit + l] = 0.0f;
        }
      }
    }

    __syncthreads();

    // Warp 内归约 QK Score
    // 关键修复：需要同时检查分支内有效性和绝对索引有效性
    int absolute_token_idx = start_idx + token_index;
    bool absolutely_valid = valid && (absolute_token_idx < total_seq_len);

    if (absolutely_valid) {
      unsigned int mask = 0xFFFFFFFF;
      for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_score += __shfl_down_sync(mask, local_score, offset);
      }
      if (d_tid == 0) {
        s_score_buf[token_tid] = local_score * static_cast<float>(softmax_scale);
      }
    } else {
      if (d_tid == 0) {
        s_score_buf[token_tid] = -FLT_MAX;
      }
    }
    __syncthreads();

    // Local Softmax
    __shared__ float cur_m_s;
    float warp_val = (d_tid < B_c && threadIdx.y == 0) ? s_score_buf[d_tid] : -FLT_MAX;
    unsigned int mask_max = 0xFFFFFFFF;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
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

    // 求和归约
    unsigned int mask_sum = 0xFFFFFFFF;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      warp_val_l += __shfl_down_sync(mask_sum, warp_val_l, offset);
    }
    if (d_tid == 0 && threadIdx.y == 0) {
      cur_l_s = warp_val_l;
    }
    __syncthreads();
    float cur_l = cur_l_s;

    // 计算部分输出
    if (j == 0) {
      // 第一个块: 计算并直接写入 s_o
      if (token_tid == 0) {
        for (int k_dim = d_tid; k_dim < DQKV_VALUE; k_dim += blockDim.x) {
          float current_dim_partial_out = 0.0f;
          for (int i_tok = 0; i_tok < B_c; ++i_tok) {
            float exp_score = s_s_score[i_tok];
            float v_val = s_vj[i_tok * DQKV_VALUE + k_dim];
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
      // 后续块: Online update
      float old_global_m = global_m;
      float old_global_l = global_l;
      float new_global_m = fmaxf(old_global_m, cur_m);
      float exp_old = __expf(old_global_m - new_global_m);
      float exp_cur = __expf(cur_m - new_global_m);

      if (token_tid == 0) {
        for (int k_dim = d_tid; k_dim < DQKV_VALUE; k_dim += blockDim.x) {
          float current_dim_partial_out = 0.0f;
          for (int i_tok = 0; i_tok < B_c; ++i_tok) {
            float exp_score = s_s_score[i_tok];
            float v_val = s_vj[i_tok * DQKV_VALUE + k_dim];
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

  // 写回 att_output
  if (threadIdx.y == 0) {
    int out_offset = head_id * (dqkv + 2);
    for (int i = d_tid; i < DQKV_VALUE; i += blockDim.x) {
      att_output[out_offset + i] = static_cast<T>(s_o[i]);
    }
    if (d_tid == 0) {
      att_output[out_offset + dqkv] = static_cast<T>(global_m);
      att_output[out_offset + dqkv + 1] = static_cast<T>(global_l);
    }
  }
}

// CUDA图优化版本：使用固定内存地址和分段信息的flash attention
// 仿照dynamic_flash_attention_wrapper的模式，直接接受连续的KV缓存
template <typename T>
void flash_attention_graph_fixed(Tensor<T> &Q,
                                 const Tensor<T> &total_K,
                                 const Tensor<T> &total_V,
                                 T **d_output_ptrs,
                                 int *d_segment_info,
                                 int n_kv_heads,
                                 cudaStream_t stream) {

  int dqkv = Q.sizes()[2];
  if (dqkv != DQKV_VALUE) {
    throw std::runtime_error("dqkv 不匹配预定义的值");
  }

  float softmax_scale = 1.0f / sqrtf(static_cast<float>(dqkv));
  int n_q_h = Q.sizes()[1];
  int n_groups = n_q_h / n_kv_heads;
  int B_r = 1;
  int T_r = 1;
  int B_c = B_C_VALUE;

  // 设置kernel参数 - 强制使用3分支，类似flash_attention.cu的稳定模式
  const int FIXED_BRANCHES = 3;
  dim3 grid(n_q_h, FIXED_BRANCHES);
  dim3 block(32, B_c);

  // 启动kernel
  flash_attention_kernel_graph_fixed<T><<<grid, block, 0, stream>>>(
      Q.data_ptr(),
      total_K.data_ptr(),
      total_V.data_ptr(),
      d_output_ptrs,
      d_segment_info,
      n_q_h, n_kv_heads, dqkv, B_c, B_r, n_groups, T_r,
      static_cast<T>(softmax_scale));

  // 检查错误
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA error in flash_attention_graph_fixed: " +
                            std::string(cudaGetErrorString(err)));
  }
}

// 显式模板实例化
template void flash_attention_graph_fixed<float>(
    Tensor<float> &Q,
    const Tensor<float> &total_K,
    const Tensor<float> &total_V,
    float **d_output_ptrs,
    int *d_segment_info,
    int n_kv_heads,
    cudaStream_t stream);

template void flash_attention_graph_fixed<__nv_bfloat16>(
    Tensor<__nv_bfloat16> &Q,
    const Tensor<__nv_bfloat16> &total_K,
    const Tensor<__nv_bfloat16> &total_V,
    __nv_bfloat16 **d_output_ptrs,
    int *d_segment_info,
    int n_kv_heads,
    cudaStream_t stream);

}  // namespace cuda_OP
