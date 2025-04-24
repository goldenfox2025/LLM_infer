#include <cublas_v2.h>
#include <cuda_bf16.h>  // 提供 __nv_bfloat16 定义
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#include <algorithm>  // min
#include <cstdio>     // printf
#include <cstring>    // memcpy
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cudaOP.cuh"

#define DQKV_VALUE 128
#define B_C_VALUE 8

constexpr int WARP_SIZE = 32;
constexpr int MAX_BRANCHES = 5;  // 支持的最大分支数

namespace cuda_OP {

// 最多支持5个分支的flash_attention_kernel
template <typename T>
__global__ void flash_attention_kernel_variable(
    T *q,
    const T *k1, const T *k2, const T *k3, const T *k4, const T *k5,
    const T *v1, const T *v2, const T *v3, const T *v4, const T *v5,
    T *att_output1, T *att_output2, T *att_output3, T *att_output4, T *att_output5,
    int n_q_h, int n_kv_h, int dqkv, int B_c, int B_r, int n_groups, int T_r,
    int cache_length1, int cache_length2, int cache_length3, int cache_length4, int cache_length5,
    int T_c1, int T_c2, int T_c3, int T_c4, int T_c5,
    int branch_count, T softmax_scale) {

  // 检查分支ID是否有效
  if (blockIdx.y >= branch_count) return;

  // 选择当前分支的数据
  const T *k;
  const T *v;
  T *att_output;
  int cache_length;
  int T_c;

  // 根据blockIdx.y选择对应的分支数据
  switch (blockIdx.y) {
    case 0:
      k = k1;
      v = v1;
      att_output = att_output1;
      cache_length = cache_length1;
      T_c = T_c1;
      break;
    case 1:
      k = k2;
      v = v2;
      att_output = att_output2;
      cache_length = cache_length2;
      T_c = T_c2;
      break;
    case 2:
      k = k3;
      v = v3;
      att_output = att_output3;
      cache_length = cache_length3;
      T_c = T_c3;
      break;
    case 3:
      k = k4;
      v = v4;
      att_output = att_output4;
      cache_length = cache_length4;
      T_c = T_c4;
      break;
    case 4:
      k = k5;
      v = v5;
      att_output = att_output5;
      cache_length = cache_length5;
      T_c = T_c5;
      break;
  }

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
    // #pragma unroll
    if (token_tid < vec_unit)
      s_qi[i * vec_unit + token_tid] = static_cast<float>(vq.t[token_tid]);
  }
  __syncthreads();

  float &global_m = s_lm[0];
  float &global_l = s_lm[1];

  // --------------------------
  // 遍历 KV 分块
  // --------------------------

  for (int j = 0; j < T_c; ++j) {
    int token_index = j * B_c + token_tid;
    bool valid = (token_index < cache_length);
    float local_score = 0.0f;

    for (int i = d_tid; i < vecCount; i += blockDim.x) {
      int index = (token_index * n_kv_h + kv_head) * dqkv + i * vec_unit;
      if (valid) {
        vk.f4 = *reinterpret_cast<const float4 *>(&k[index]);
        vv.f4 = *reinterpret_cast<const float4 *>(&v[index]);
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
    if (valid) {
      unsigned int mask = 0xFFFFFFFF;
      for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_score += __shfl_down_sync(mask, local_score, offset);
      }
      if (d_tid == 0) {
        s_score_buf[token_tid] =
            local_score * static_cast<float>(softmax_scale);
      }
    } else {
      if (d_tid == 0) {
        s_score_buf[token_tid] = -FLT_MAX;
      }
    }
    __syncthreads();

    // --------------------------
    // Local Softmax
    // --------------------------
    __shared__ float cur_m_s;  // 使用临时 shared 变量传递归约结果

    float warp_val =
        (d_tid < B_c && threadIdx.y == 0) ? s_score_buf[d_tid] : -FLT_MAX;
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
      s_s_score[d_tid] = exp_val;  // 写入一项
      warp_val_l = exp_val;        // warp_val_l 等于这一项的值
    }
    // 后续 Warp Reduce 会正确地将 8 个 warp_val_l 加起来
    else {
      // 其他线程不参与计算，但 warp_val_l 需初始化为 0 用于归约
      warp_val_l = 0.0f;
    }

    __syncthreads();  // 必须确保 s_s_score 完全写入

    // 求和归约
    unsigned int mask_sum = 0xFFFFFFFF;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      warp_val_l += __shfl_down_sync(mask_sum, warp_val_l, offset);
    }
    if (d_tid == 0 && threadIdx.y == 0) {
      cur_l_s = warp_val_l;
    }
    __syncthreads();
    float cur_l = cur_l_s;  // 所有线程读取 cur_l

    // --------------------------
    // 计算部分输出
    // --------------------------
    if (j == 0) {
      // 第一个块: 计算并直接写入 s_o
      if (token_tid == 0) {  // 只有 y=0 线程工作
        // 外层循环：遍历当前线程负责的维度 k
        for (int k_dim = d_tid; k_dim < DQKV_VALUE; k_dim += blockDim.x) {
          float current_dim_partial_out = 0.0f;  // 初始化该维度的累加器

          // 内层循环：遍历 B_c 个 token，计算 sum(score * V[k])
          for (int i_tok = 0; i_tok < B_c; ++i_tok) {
            float exp_score = s_s_score[i_tok];
            float v_val = s_vj[i_tok * DQKV_VALUE +
                             k_dim];  // 读取 token i 在维度 k 的 V 值
            current_dim_partial_out =
                fmaf(exp_score, v_val, current_dim_partial_out);
          }
          // 将第一个块计算出的结果直接写入 s_o
          s_o[k_dim] = current_dim_partial_out;
        }
      }
      // 初始化全局 m, l (由线程 0,0 完成)
      if (token_tid == 0 && d_tid == 0) {
        global_m = cur_m;
        global_l = cur_l;
      }
    } else {
      // 后续块: Online update
      // 读取旧的全局 m, l (所有线程需要)
      float old_global_m = global_m;
      float old_global_l = global_l;
      // 计算新的全局 m 和缩放因子 (所有线程需要)
      float new_global_m = fmaxf(old_global_m, cur_m);
      float exp_old = __expf(old_global_m - new_global_m);
      float exp_cur = __expf(cur_m - new_global_m);

      if (token_tid == 0) {  // 只有 y=0 线程更新 s_o
        // 外层循环：遍历当前线程负责的维度 k
        for (int k_dim = d_tid; k_dim < DQKV_VALUE; k_dim += blockDim.x) {
          float current_dim_partial_out = 0.0f;  // 初始化该维度的累加器

          // 内层循环：遍历 B_c 个 token，计算当前块对维度 k 的贡献
          for (int i_tok = 0; i_tok < B_c; ++i_tok) {
            float exp_score = s_s_score[i_tok];
            float v_val = s_vj[i_tok * DQKV_VALUE + k_dim];
            current_dim_partial_out =
                fmaf(exp_score, v_val, current_dim_partial_out);
          }

          // 读取旧的 s_o 值
          float old_out_val = s_o[k_dim];
          // 执行 Online Update
          float new_out_val =
              old_out_val * exp_old + current_dim_partial_out * exp_cur;
          // 写回新的 s_o 值
          s_o[k_dim] = new_out_val;
        }
      }

      // 更新全局 m, l (由线程 0,0 完成)
      if (token_tid == 0 && d_tid == 0) {
        float new_global_l = old_global_l * exp_old + cur_l * exp_cur;
        global_m = new_global_m;
        global_l = new_global_l;
      }
    }
    __syncthreads();  // 确保 s_o, m, l 更新对下一轮或写回可见

  }  // end for each chunk (T_c)

  // --------------------------
  // 写回 att_output
  // --------------------------

  // 写回 global memory
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

// 可变分支数量的flash_attention实现
template <typename T>
void flash_attention_variable(Tensor<T>& Q,
                             const std::vector<Tensor<T>>& K_slices,
                             const std::vector<Tensor<T>>& V_slices,
                             std::vector<Tensor<T>>& outputs,
                             cudaStream_t stream) {
  // 验证输入
  int branch_count = K_slices.size();
  if (branch_count == 0 || branch_count > MAX_BRANCHES) {
    throw std::runtime_error("Invalid branch count: " + std::to_string(branch_count));
  }

  if (V_slices.size() != branch_count || outputs.size() != branch_count) {
    throw std::runtime_error("Mismatch in number of K, V, or output tensors");
  }

  int dqkv = Q.sizes()[2];  // 每个 head 内维度
  if (dqkv != DQKV_VALUE) {
    throw std::runtime_error("dqkv 不匹配预定义的值");
  }

  float softmax_scale = 1.0f / sqrtf(static_cast<float>(dqkv));
  int n_q_h = Q.sizes()[1];           // query head 数
  int n_kv_h = K_slices[0].sizes()[1]; // 假设所有K切片具有相同的kv head数
  int n_groups = n_q_h / n_kv_h;
  int B_r = 1;
  int T_r = 1;

  // 每个 chunk 读取的 kv token 数（预设为偶数 B_C_VALUE）
  int B_c = B_C_VALUE;

  // 准备各分支的参数
  const T* k1 = branch_count > 0 ? K_slices[0].data_ptr() : nullptr;
  const T* k2 = branch_count > 1 ? K_slices[1].data_ptr() : nullptr;
  const T* k3 = branch_count > 2 ? K_slices[2].data_ptr() : nullptr;
  const T* k4 = branch_count > 3 ? K_slices[3].data_ptr() : nullptr;
  const T* k5 = branch_count > 4 ? K_slices[4].data_ptr() : nullptr;

  const T* v1 = branch_count > 0 ? V_slices[0].data_ptr() : nullptr;
  const T* v2 = branch_count > 1 ? V_slices[1].data_ptr() : nullptr;
  const T* v3 = branch_count > 2 ? V_slices[2].data_ptr() : nullptr;
  const T* v4 = branch_count > 3 ? V_slices[3].data_ptr() : nullptr;
  const T* v5 = branch_count > 4 ? V_slices[4].data_ptr() : nullptr;

  T* att_output1 = branch_count > 0 ? outputs[0].data_ptr() : nullptr;
  T* att_output2 = branch_count > 1 ? outputs[1].data_ptr() : nullptr;
  T* att_output3 = branch_count > 2 ? outputs[2].data_ptr() : nullptr;
  T* att_output4 = branch_count > 3 ? outputs[3].data_ptr() : nullptr;
  T* att_output5 = branch_count > 4 ? outputs[4].data_ptr() : nullptr;

  int cache_length1 = branch_count > 0 ? K_slices[0].sizes()[0] : 0;
  int cache_length2 = branch_count > 1 ? K_slices[1].sizes()[0] : 0;
  int cache_length3 = branch_count > 2 ? K_slices[2].sizes()[0] : 0;
  int cache_length4 = branch_count > 3 ? K_slices[3].sizes()[0] : 0;
  int cache_length5 = branch_count > 4 ? K_slices[4].sizes()[0] : 0;

  int T_c1 = branch_count > 0 ? (cache_length1 + B_c - 1) / B_c : 0;
  int T_c2 = branch_count > 1 ? (cache_length2 + B_c - 1) / B_c : 0;
  int T_c3 = branch_count > 2 ? (cache_length3 + B_c - 1) / B_c : 0;
  int T_c4 = branch_count > 3 ? (cache_length4 + B_c - 1) / B_c : 0;
  int T_c5 = branch_count > 4 ? (cache_length5 + B_c - 1) / B_c : 0;

  // 设置kernel参数
  dim3 grid(n_q_h, branch_count);
  dim3 block(32, B_c);

  // 启动kernel - 直接传递指针，不需要额外的内存复制
  flash_attention_kernel_variable<T><<<grid, block, 0, stream>>>(
      Q.data_ptr(),
      k1, k2, k3, k4, k5,
      v1, v2, v3, v4, v5,
      att_output1, att_output2, att_output3, att_output4, att_output5,
      n_q_h, n_kv_h, dqkv, B_c, B_r, n_groups, T_r,
      cache_length1, cache_length2, cache_length3, cache_length4, cache_length5,
      T_c1, T_c2, T_c3, T_c4, T_c5,
      branch_count, static_cast<T>(softmax_scale));

  // 检查错误
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA error in flash_attention_variable: " +
                            std::string(cudaGetErrorString(err)));
  }
}

// 特化版本的flash_attention实现 - 1分支
template <typename T>
void flash_attention_specialized_1branch(Tensor<T>& Q,
                                       const std::vector<Tensor<T>>& K_slices,
                                       const std::vector<Tensor<T>>& V_slices,
                                       std::vector<Tensor<T>>& outputs,
                                       cudaStream_t stream) {
  // 验证输入
  if (K_slices.size() != 1 || V_slices.size() != 1 || outputs.size() != 1) {
    throw std::runtime_error("Mismatch in number of K, V, or output tensors for 1-branch specialization");
  }

  int dqkv = Q.sizes()[2];  // 每个 head 内维度
  if (dqkv != DQKV_VALUE) {
    throw std::runtime_error("dqkv 不匹配预定义的值");
  }

  float softmax_scale = 1.0f / sqrtf(static_cast<float>(dqkv));
  int n_q_h = Q.sizes()[1];           // query head 数
  int n_kv_h = K_slices[0].sizes()[1]; // 假设所有K切片具有相同的kv head数
  int n_groups = n_q_h / n_kv_h;
  int B_r = 1;
  int T_r = 1;

  // 每个 chunk 读取的 kv token 数（预设为偶数 B_C_VALUE）
  int B_c = B_C_VALUE;

  // 准备各分支的参数
  const T* k1 = K_slices[0].data_ptr();
  const T* v1 = V_slices[0].data_ptr();
  T* att_output1 = outputs[0].data_ptr();

  int cache_length1 = K_slices[0].sizes()[0];
  int T_c1 = (cache_length1 + B_c - 1) / B_c;

  // 设置kernel参数
  dim3 grid(n_q_h, 1);
  dim3 block(32, B_c);

  // 启动kernel - 直接传递指针，不需要额外的内存复制
  flash_attention_kernel_variable<T><<<grid, block, 0, stream>>>(
      Q.data_ptr(),
      k1, nullptr, nullptr, nullptr, nullptr,
      v1, nullptr, nullptr, nullptr, nullptr,
      att_output1, nullptr, nullptr, nullptr, nullptr,
      n_q_h, n_kv_h, dqkv, B_c, B_r, n_groups, T_r,
      cache_length1, 0, 0, 0, 0,
      T_c1, 0, 0, 0, 0,
      1, static_cast<T>(softmax_scale));
}

// 特化版本的flash_attention实现 - 2分支
template <typename T>
void flash_attention_specialized_2branch(Tensor<T>& Q,
                                       const std::vector<Tensor<T>>& K_slices,
                                       const std::vector<Tensor<T>>& V_slices,
                                       std::vector<Tensor<T>>& outputs,
                                       cudaStream_t stream) {
  // 验证输入
  if (K_slices.size() != 2 || V_slices.size() != 2 || outputs.size() != 2) {
    throw std::runtime_error("Mismatch in number of K, V, or output tensors for 2-branch specialization");
  }

  int dqkv = Q.sizes()[2];  // 每个 head 内维度
  if (dqkv != DQKV_VALUE) {
    throw std::runtime_error("dqkv 不匹配预定义的值");
  }

  float softmax_scale = 1.0f / sqrtf(static_cast<float>(dqkv));
  int n_q_h = Q.sizes()[1];           // query head 数
  int n_kv_h = K_slices[0].sizes()[1]; // 假设所有K切片具有相同的kv head数
  int n_groups = n_q_h / n_kv_h;
  int B_r = 1;
  int T_r = 1;

  // 每个 chunk 读取的 kv token 数（预设为偶数 B_C_VALUE）
  int B_c = B_C_VALUE;

  // 准备各分支的参数
  const T* k1 = K_slices[0].data_ptr();
  const T* k2 = K_slices[1].data_ptr();
  const T* v1 = V_slices[0].data_ptr();
  const T* v2 = V_slices[1].data_ptr();
  T* att_output1 = outputs[0].data_ptr();
  T* att_output2 = outputs[1].data_ptr();

  int cache_length1 = K_slices[0].sizes()[0];
  int cache_length2 = K_slices[1].sizes()[0];
  int T_c1 = (cache_length1 + B_c - 1) / B_c;
  int T_c2 = (cache_length2 + B_c - 1) / B_c;

  // 设置kernel参数
  dim3 grid(n_q_h, 2);
  dim3 block(32, B_c);

  // 启动kernel - 直接传递指针，不需要额外的内存复制
  flash_attention_kernel_variable<T><<<grid, block, 0, stream>>>(
      Q.data_ptr(),
      k1, k2, nullptr, nullptr, nullptr,
      v1, v2, nullptr, nullptr, nullptr,
      att_output1, att_output2, nullptr, nullptr, nullptr,
      n_q_h, n_kv_h, dqkv, B_c, B_r, n_groups, T_r,
      cache_length1, cache_length2, 0, 0, 0,
      T_c1, T_c2, 0, 0, 0,
      2, static_cast<T>(softmax_scale));
}

// 特化版本的flash_attention实现 - 3分支
template <typename T>
void flash_attention_specialized_3branch(Tensor<T>& Q,
                                       const std::vector<Tensor<T>>& K_slices,
                                       const std::vector<Tensor<T>>& V_slices,
                                       std::vector<Tensor<T>>& outputs,
                                       cudaStream_t stream) {
  // 验证输入
  if (K_slices.size() != 3 || V_slices.size() != 3 || outputs.size() != 3) {
    throw std::runtime_error("Mismatch in number of K, V, or output tensors for 3-branch specialization");
  }

  int dqkv = Q.sizes()[2];  // 每个 head 内维度
  if (dqkv != DQKV_VALUE) {
    throw std::runtime_error("dqkv 不匹配预定义的值");
  }

  float softmax_scale = 1.0f / sqrtf(static_cast<float>(dqkv));
  int n_q_h = Q.sizes()[1];           // query head 数
  int n_kv_h = K_slices[0].sizes()[1]; // 假设所有K切片具有相同的kv head数
  int n_groups = n_q_h / n_kv_h;
  int B_r = 1;
  int T_r = 1;

  // 每个 chunk 读取的 kv token 数（预设为偶数 B_C_VALUE）
  int B_c = B_C_VALUE;

  // 准备各分支的参数
  const T* k1 = K_slices[0].data_ptr();
  const T* k2 = K_slices[1].data_ptr();
  const T* k3 = K_slices[2].data_ptr();
  const T* v1 = V_slices[0].data_ptr();
  const T* v2 = V_slices[1].data_ptr();
  const T* v3 = V_slices[2].data_ptr();
  T* att_output1 = outputs[0].data_ptr();
  T* att_output2 = outputs[1].data_ptr();
  T* att_output3 = outputs[2].data_ptr();

  int cache_length1 = K_slices[0].sizes()[0];
  int cache_length2 = K_slices[1].sizes()[0];
  int cache_length3 = K_slices[2].sizes()[0];
  int T_c1 = (cache_length1 + B_c - 1) / B_c;
  int T_c2 = (cache_length2 + B_c - 1) / B_c;
  int T_c3 = (cache_length3 + B_c - 1) / B_c;

  // 设置kernel参数
  dim3 grid(n_q_h, 3);
  dim3 block(32, B_c);

  // 启动kernel - 直接传递指针，不需要额外的内存复制
  flash_attention_kernel_variable<T><<<grid, block, 0, stream>>>(
      Q.data_ptr(),
      k1, k2, k3, nullptr, nullptr,
      v1, v2, v3, nullptr, nullptr,
      att_output1, att_output2, att_output3, nullptr, nullptr,
      n_q_h, n_kv_h, dqkv, B_c, B_r, n_groups, T_r,
      cache_length1, cache_length2, cache_length3, 0, 0,
      T_c1, T_c2, T_c3, 0, 0,
      3, static_cast<T>(softmax_scale));
}

// 特化版本的flash_attention实现 - 4分支
template <typename T>
void flash_attention_specialized_4branch(Tensor<T>& Q,
                                       const std::vector<Tensor<T>>& K_slices,
                                       const std::vector<Tensor<T>>& V_slices,
                                       std::vector<Tensor<T>>& outputs,
                                       cudaStream_t stream) {
  // 验证输入
  if (K_slices.size() != 4 || V_slices.size() != 4 || outputs.size() != 4) {
    throw std::runtime_error("Mismatch in number of K, V, or output tensors for 4-branch specialization");
  }

  int dqkv = Q.sizes()[2];  // 每个 head 内维度
  if (dqkv != DQKV_VALUE) {
    throw std::runtime_error("dqkv 不匹配预定义的值");
  }

  float softmax_scale = 1.0f / sqrtf(static_cast<float>(dqkv));
  int n_q_h = Q.sizes()[1];           // query head 数
  int n_kv_h = K_slices[0].sizes()[1]; // 假设所有K切片具有相同的kv head数
  int n_groups = n_q_h / n_kv_h;
  int B_r = 1;
  int T_r = 1;

  // 每个 chunk 读取的 kv token 数（预设为偶数 B_C_VALUE）
  int B_c = B_C_VALUE;

  // 准备各分支的参数
  const T* k1 = K_slices[0].data_ptr();
  const T* k2 = K_slices[1].data_ptr();
  const T* k3 = K_slices[2].data_ptr();
  const T* k4 = K_slices[3].data_ptr();
  const T* v1 = V_slices[0].data_ptr();
  const T* v2 = V_slices[1].data_ptr();
  const T* v3 = V_slices[2].data_ptr();
  const T* v4 = V_slices[3].data_ptr();
  T* att_output1 = outputs[0].data_ptr();
  T* att_output2 = outputs[1].data_ptr();
  T* att_output3 = outputs[2].data_ptr();
  T* att_output4 = outputs[3].data_ptr();

  int cache_length1 = K_slices[0].sizes()[0];
  int cache_length2 = K_slices[1].sizes()[0];
  int cache_length3 = K_slices[2].sizes()[0];
  int cache_length4 = K_slices[3].sizes()[0];
  int T_c1 = (cache_length1 + B_c - 1) / B_c;
  int T_c2 = (cache_length2 + B_c - 1) / B_c;
  int T_c3 = (cache_length3 + B_c - 1) / B_c;
  int T_c4 = (cache_length4 + B_c - 1) / B_c;

  // 设置kernel参数
  dim3 grid(n_q_h, 4);
  dim3 block(32, B_c);

  // 启动kernel - 直接传递指针，不需要额外的内存复制
  flash_attention_kernel_variable<T><<<grid, block, 0, stream>>>(
      Q.data_ptr(),
      k1, k2, k3, k4, nullptr,
      v1, v2, v3, v4, nullptr,
      att_output1, att_output2, att_output3, att_output4, nullptr,
      n_q_h, n_kv_h, dqkv, B_c, B_r, n_groups, T_r,
      cache_length1, cache_length2, cache_length3, cache_length4, 0,
      T_c1, T_c2, T_c3, T_c4, 0,
      4, static_cast<T>(softmax_scale));
}

// 特化版本的flash_attention实现 - 5分支
template <typename T>
void flash_attention_specialized_5branch(Tensor<T>& Q,
                                       const std::vector<Tensor<T>>& K_slices,
                                       const std::vector<Tensor<T>>& V_slices,
                                       std::vector<Tensor<T>>& outputs,
                                       cudaStream_t stream) {
  // 验证输入
  if (K_slices.size() != 5 || V_slices.size() != 5 || outputs.size() != 5) {
    throw std::runtime_error("Mismatch in number of K, V, or output tensors for 5-branch specialization");
  }

  int dqkv = Q.sizes()[2];  // 每个 head 内维度
  if (dqkv != DQKV_VALUE) {
    throw std::runtime_error("dqkv 不匹配预定义的值");
  }

  float softmax_scale = 1.0f / sqrtf(static_cast<float>(dqkv));
  int n_q_h = Q.sizes()[1];           // query head 数
  int n_kv_h = K_slices[0].sizes()[1]; // 假设所有K切片具有相同的kv head数
  int n_groups = n_q_h / n_kv_h;
  int B_r = 1;
  int T_r = 1;

  // 每个 chunk 读取的 kv token 数（预设为偶数 B_C_VALUE）
  int B_c = B_C_VALUE;

  // 准备各分支的参数
  const T* k1 = K_slices[0].data_ptr();
  const T* k2 = K_slices[1].data_ptr();
  const T* k3 = K_slices[2].data_ptr();
  const T* k4 = K_slices[3].data_ptr();
  const T* k5 = K_slices[4].data_ptr();
  const T* v1 = V_slices[0].data_ptr();
  const T* v2 = V_slices[1].data_ptr();
  const T* v3 = V_slices[2].data_ptr();
  const T* v4 = V_slices[3].data_ptr();
  const T* v5 = V_slices[4].data_ptr();
  T* att_output1 = outputs[0].data_ptr();
  T* att_output2 = outputs[1].data_ptr();
  T* att_output3 = outputs[2].data_ptr();
  T* att_output4 = outputs[3].data_ptr();
  T* att_output5 = outputs[4].data_ptr();

  int cache_length1 = K_slices[0].sizes()[0];
  int cache_length2 = K_slices[1].sizes()[0];
  int cache_length3 = K_slices[2].sizes()[0];
  int cache_length4 = K_slices[3].sizes()[0];
  int cache_length5 = K_slices[4].sizes()[0];
  int T_c1 = (cache_length1 + B_c - 1) / B_c;
  int T_c2 = (cache_length2 + B_c - 1) / B_c;
  int T_c3 = (cache_length3 + B_c - 1) / B_c;
  int T_c4 = (cache_length4 + B_c - 1) / B_c;
  int T_c5 = (cache_length5 + B_c - 1) / B_c;

  // 设置kernel参数
  dim3 grid(n_q_h, 5);
  dim3 block(32, B_c);

  // 启动kernel - 直接传递指针，不需要额外的内存复制
  flash_attention_kernel_variable<T><<<grid, block, 0, stream>>>(
      Q.data_ptr(),
      k1, k2, k3, k4, k5,
      v1, v2, v3, v4, v5,
      att_output1, att_output2, att_output3, att_output4, att_output5,
      n_q_h, n_kv_h, dqkv, B_c, B_r, n_groups, T_r,
      cache_length1, cache_length2, cache_length3, cache_length4, cache_length5,
      T_c1, T_c2, T_c3, T_c4, T_c5,
      5, static_cast<T>(softmax_scale));
}

// 包装函数：根据KV缓存长度动态选择分支数量
template <typename T>
void dynamic_flash_attention_wrapper(Tensor<T>& Q,
                                   const Tensor<T>& total_K,
                                   const Tensor<T>& total_V,
                                   Tensor<T>& att_output,
                                   int n_kv_heads,
                                   cudaStream_t stream) {
  // 获取输入尺寸
  size_t n_q_h = Q.sizes()[1];        // query head 数
  size_t dqkv = Q.sizes()[2];         // 每个 head 内维度
  size_t total_seq_len = total_K.sizes()[0];  // 总的 kv token 数

  // 确定需要的分支数量 - 即使是短序列也使用至少1个分支
  int branches_needed = (total_seq_len + B_C_VALUE - 1) / B_C_VALUE;
  if (branches_needed == 0) branches_needed = 1;

  // 限制最大分支数为5
  branches_needed = std::min(branches_needed, MAX_BRANCHES);

  // 计算每个分支的长度
  size_t tokens_per_branch = (total_seq_len + branches_needed - 1) / branches_needed;

  // 准备K和V的切片 - 直接使用slice方法，不复制数据
  std::vector<Tensor<T>> K_slices;
  std::vector<Tensor<T>> V_slices;
  std::vector<Tensor<T>> branch_outputs;

  for (int i = 0; i < branches_needed; i++) {
    size_t start_idx = i * tokens_per_branch;
    size_t end_idx = std::min(start_idx + tokens_per_branch, total_seq_len);

    if (start_idx >= total_seq_len) break;

    // 使用slice方法创建共享底层数据的视图
    K_slices.push_back(total_K.slice({start_idx, 0, 0},
                                    {end_idx, static_cast<size_t>(n_kv_heads), dqkv}));
    V_slices.push_back(total_V.slice({start_idx, 0, 0},
                                    {end_idx, static_cast<size_t>(n_kv_heads), dqkv}));

    // 创建输出张量
    branch_outputs.push_back(Tensor<T>({n_q_h * (dqkv + 2)}, Device::CUDA));
  }

  // 根据分支数量选择特化版本
  switch (branches_needed) {
    case 1:
      flash_attention_specialized_1branch(Q, K_slices, V_slices, branch_outputs, stream);
      break;
    case 2:
      flash_attention_specialized_2branch(Q, K_slices, V_slices, branch_outputs, stream);
      break;
    case 3:
      flash_attention_specialized_3branch(Q, K_slices, V_slices, branch_outputs, stream);
      break;
    case 4:
      flash_attention_specialized_4branch(Q, K_slices, V_slices, branch_outputs, stream);
      break;
    case 5:
      flash_attention_specialized_5branch(Q, K_slices, V_slices, branch_outputs, stream);
      break;
    default:
      // 不应该到达这里，因为我们已经限制了分支数量
      throw std::runtime_error("Unexpected branch count: " + std::to_string(branches_needed));
  }

  // 使用gather_fa_variable合并结果
  gather_fa_variable(branch_outputs, att_output, stream);
}

// 显式实例化 - 原始函数
template void flash_attention_variable<float>(
    Tensor<float>& Q,
    const std::vector<Tensor<float>>& K_slices,
    const std::vector<Tensor<float>>& V_slices,
    std::vector<Tensor<float>>& outputs,
    cudaStream_t stream);

template void flash_attention_variable<nvbf16>(
    Tensor<nvbf16>& Q,
    const std::vector<Tensor<nvbf16>>& K_slices,
    const std::vector<Tensor<nvbf16>>& V_slices,
    std::vector<Tensor<nvbf16>>& outputs,
    cudaStream_t stream);

// 显式实例化 - 特化版本 (1分支)
template void flash_attention_specialized_1branch<float>(
    Tensor<float>& Q,
    const std::vector<Tensor<float>>& K_slices,
    const std::vector<Tensor<float>>& V_slices,
    std::vector<Tensor<float>>& outputs,
    cudaStream_t stream);

template void flash_attention_specialized_1branch<nvbf16>(
    Tensor<nvbf16>& Q,
    const std::vector<Tensor<nvbf16>>& K_slices,
    const std::vector<Tensor<nvbf16>>& V_slices,
    std::vector<Tensor<nvbf16>>& outputs,
    cudaStream_t stream);

// 显式实例化 - 特化版本 (2分支)
template void flash_attention_specialized_2branch<float>(
    Tensor<float>& Q,
    const std::vector<Tensor<float>>& K_slices,
    const std::vector<Tensor<float>>& V_slices,
    std::vector<Tensor<float>>& outputs,
    cudaStream_t stream);

template void flash_attention_specialized_2branch<nvbf16>(
    Tensor<nvbf16>& Q,
    const std::vector<Tensor<nvbf16>>& K_slices,
    const std::vector<Tensor<nvbf16>>& V_slices,
    std::vector<Tensor<nvbf16>>& outputs,
    cudaStream_t stream);

// 显式实例化 - 特化版本 (3分支)
template void flash_attention_specialized_3branch<float>(
    Tensor<float>& Q,
    const std::vector<Tensor<float>>& K_slices,
    const std::vector<Tensor<float>>& V_slices,
    std::vector<Tensor<float>>& outputs,
    cudaStream_t stream);

template void flash_attention_specialized_3branch<nvbf16>(
    Tensor<nvbf16>& Q,
    const std::vector<Tensor<nvbf16>>& K_slices,
    const std::vector<Tensor<nvbf16>>& V_slices,
    std::vector<Tensor<nvbf16>>& outputs,
    cudaStream_t stream);

// 显式实例化 - 特化版本 (4分支)
template void flash_attention_specialized_4branch<float>(
    Tensor<float>& Q,
    const std::vector<Tensor<float>>& K_slices,
    const std::vector<Tensor<float>>& V_slices,
    std::vector<Tensor<float>>& outputs,
    cudaStream_t stream);

template void flash_attention_specialized_4branch<nvbf16>(
    Tensor<nvbf16>& Q,
    const std::vector<Tensor<nvbf16>>& K_slices,
    const std::vector<Tensor<nvbf16>>& V_slices,
    std::vector<Tensor<nvbf16>>& outputs,
    cudaStream_t stream);

// 显式实例化 - 特化版本 (5分支)
template void flash_attention_specialized_5branch<float>(
    Tensor<float>& Q,
    const std::vector<Tensor<float>>& K_slices,
    const std::vector<Tensor<float>>& V_slices,
    std::vector<Tensor<float>>& outputs,
    cudaStream_t stream);

template void flash_attention_specialized_5branch<nvbf16>(
    Tensor<nvbf16>& Q,
    const std::vector<Tensor<nvbf16>>& K_slices,
    const std::vector<Tensor<nvbf16>>& V_slices,
    std::vector<Tensor<nvbf16>>& outputs,
    cudaStream_t stream);

// 显式实例化 - 包装函数
template void dynamic_flash_attention_wrapper<float>(
    Tensor<float>& Q,
    const Tensor<float>& total_K,
    const Tensor<float>& total_V,
    Tensor<float>& att_output,
    int n_kv_heads,
    cudaStream_t stream);

template void dynamic_flash_attention_wrapper<nvbf16>(
    Tensor<nvbf16>& Q,
    const Tensor<nvbf16>& total_K,
    const Tensor<nvbf16>& total_V,
    Tensor<nvbf16>& att_output,
    int n_kv_heads,
    cudaStream_t stream);

}  // namespace cuda_OP
