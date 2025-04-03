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

#include "cudaOP.cuh"  // 假设此头文件定义了 Tensor<T> 等接口

// 预先确定的参数（根据实际情况调整）
#define DQKV_VALUE 128
#define B_C_VALUE 8

// ----------------------------------------------
// bf16 与 float 转换函数
// ----------------------------------------------
__device__ inline float bf16_to_float(__nv_bfloat16 x) {
  unsigned short raw;
  memcpy(&raw, &x, sizeof(raw));
  unsigned int bits = (static_cast<unsigned int>(raw) << 16);
  float f;
  memcpy(&f, &bits, sizeof(f));
  return f;
}

__device__ inline __nv_bfloat16 float_to_bf16(float f) {
  unsigned int bits;
  memcpy(&bits, &f, sizeof(bits));
  unsigned short raw = static_cast<unsigned short>(bits >> 16);
  __nv_bfloat16 h;
  memcpy(&h, &raw, sizeof(h));
  return h;
}

// ----------------------------------------------
// 自定义指数函数 my_exp
// ----------------------------------------------
template <typename T>
__device__ inline T my_exp(T x);

template <>
__device__ inline float my_exp<float>(float x) {
  return expf(x);
}

template <>
__device__ inline double my_exp<double>(double x) {
  return exp(x);
}

template <>
__device__ inline __nv_bfloat16 my_exp<__nv_bfloat16>(__nv_bfloat16 x) {
  float fx = bf16_to_float(x);
  float ef = expf(fx);
  return float_to_bf16(ef);
}

// ----------------------------------------------
// 自定义最大值函数 my_fmax
// ----------------------------------------------
template <typename T>
__device__ inline T my_fmax(T a, T b);

template <>
__device__ inline float my_fmax<float>(float a, float b) {
  return fmaxf(a, b);
}

template <>
__device__ inline __nv_bfloat16 my_fmax<__nv_bfloat16>(__nv_bfloat16 a,
                                                       __nv_bfloat16 b) {
  float fa = bf16_to_float(a);
  float fb = bf16_to_float(b);
  float fm = fmaxf(fa, fb);
  return float_to_bf16(fm);
}

namespace cuda_OP {

// --------------------------------------------------
// 针对dqkv为128的特别优化
// --------------------------------------------------

template <typename T>
__global__ void flash_attention_kernel_v4(T* q, const T* k, const T* v,
                                          T* att_output, int n_q_h,
                                          int cache_length, int n_kv_h,
                                          int dqkv, int B_c, int B_r,
                                          int n_groups, int T_r, int T_c,
                                          T softmax_scale) {
  // 1. 检查预设参数是否一致
  if (dqkv != DQKV_VALUE || B_c != B_C_VALUE) return;

  // 2. 共享内存声明
  //    - s_qi 用于存放当前 head 的 Q
  //    - s_vj 依旧需要，因为后面要在 softmax 得到权重后，对 V 做加权求和
  //    - s_score_buf 用于存放本 chunk 内每个 token 的 (Q·K) 缩放得分
  //    - s_tmp 用于 block 内归约 (可以和 warp-level reduce 互换)
  //    - s_lm[0] = global_m，s_lm[1] = global_l
  __shared__ float s_qi[DQKV_VALUE];              // 当前 head 的 Query
  __shared__ float s_vj[B_C_VALUE * DQKV_VALUE];  // 当前 chunk 的 Value
  __shared__ float s_score_buf[B_C_VALUE];  // 每个 token 的 (Q·K) * scale
  // __shared__ float s_tmp[DQKV_VALUE * B_C_VALUE];  // 用于 block 归约
  __shared__ float s_lm[2];  // {global_m, global_l}
  // __shared__ float s_max_local[B_C_VALUE];  // 局部最大值归约缓存
  // __shared__ float s_exp_local[B_C_VALUE];  // 局部指数和归约缓存

  __shared__ float s_o[12 * DQKV_VALUE];

  // 线程内变量
  const int d_tid = threadIdx.x;  // 特征维度内线程索引，[0, dqkv)
  const int token_tid = threadIdx.y;  // 当前 chunk 内 token 线程索引，[0, B_c)
  const int head_id = blockIdx.x;  // block 索引 -> head 索引
  const int q_offset = head_id * dqkv;
  const int kv_head = head_id / n_groups;  // KV head 索引

  // 设定一次加载的元素个数 (vec_unit)
  constexpr int vec_unit = 16 / sizeof(T);
  const int vecCount = dqkv / vec_unit;
  Vec<T, vec_unit> vq, vk, vv;  // 用于加载 key, value
  // --------------------------
  // (A) 先将 Q 从 global 加载到共享内存 s_qi
  // --------------------------
  if (token_tid == 0) {
    // 这里只是演示：一个 warp/线程块负责 dqkv 维度的加载
    for (int i = d_tid; i < vecCount; i += blockDim.x) {
      // 每次载入 8 字节数据

      vq.f4 = *reinterpret_cast<const float4*>(&q[q_offset + i * vec_unit]);
#pragma unroll
      for (int j = 0; j < vec_unit; j++) {
        s_qi[i * vec_unit + j] = static_cast<float>(vq.t[j]);
      }
    }
  }
  __syncthreads();
  // 引用全局 softmax 递归变量
  float& global_m = s_lm[0];
  float& global_l = s_lm[1];
  // --------------------------
  // (B) 遍历每个 KV 分块（chunk）
  //     每个分块大小：B_c
  // --------------------------

  for (int j = 0; j < T_c; ++j) {
    int token_index = j * B_c + token_tid;
    bool valid = (token_index < cache_length);
    // (B1) 边读取 Key/Value，边做 Q·K 的局部累加
    float local_score = 0.0f;  // 寄存器内存放 Q·K 的累加结果
    for (int i = d_tid; i < vecCount; i += blockDim.x) {
      if (valid) {
        int index = (token_index * n_kv_h + kv_head) * dqkv + i * vec_unit;
        // 加载 key
        vk.f4 = *reinterpret_cast<const float4*>(&k[index]);
        // 加载 value
        vv.f4 = *reinterpret_cast<const float4*>(&v[index]);
#pragma unroll
        for (int l = 0; l < vec_unit; l++) {
          float k_val = static_cast<float>(vk.t[l]);
          float v_val = static_cast<float>(vv.t[l]);
          // 与 Q 做点积累加
          local_score += s_qi[i * vec_unit + l] * k_val;
          // V 需要存入共享内存，后续做加权求和
          s_vj[token_tid * dqkv + i * vec_unit + l] = v_val;
        }
      } else {
        // 无效位置填零
#pragma unroll
        for (int l = 0; l < vec_unit; l++) {
          s_vj[token_tid * dqkv + i * vec_unit + l] = 0.0f;
        }
      }
    }
    __syncthreads();

    // (B2) 将 local_score 写入 s_tmp 做 block 内归约
    // if (valid) {
    //   int index = token_tid * dqkv + d_tid;
    //   s_tmp[index] = local_score;
    //   __syncthreads();
    //   for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    //     if (d_tid < stride) {
    //       s_tmp[index] += s_tmp[index + stride];
    //     }
    //     __syncthreads();
    //   }
    //   // 该 token 归约完成后，d_tid=0 的线程把 (Q·K)*scale 存到 s_score_buf
    //   if (d_tid == 0) {
    //     s_score_buf[token_tid] =
    //         s_tmp[token_tid * dqkv] * static_cast<float>(softmax_scale);
    //   }
    // } else {
    //   // 无效 token
    //   if (d_tid == 0) {
    //     s_score_buf[token_tid] = -FLT_MAX;
    //   }
    // }
    // __syncthreads();
    if (valid) {
      // 进行 warp 内归约，假设 blockDim.x 为 warpSize (32)
      unsigned int mask = __activemask();
      for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_score += __shfl_down_sync(mask, local_score, offset);
      }
      // 归约完成后，lane 0 存有该 token 的总和
      if (d_tid == 0) {
        s_score_buf[token_tid] =
            local_score * static_cast<float>(softmax_scale);
      }
    } else {
      // 对于无效 token，直接设为 -FLT_MAX
      if (d_tid == 0) {
        s_score_buf[token_tid] = -FLT_MAX;
      }
    }
    __syncthreads();

    // --------------------------
    // (C) 做本分块内的 softmax(局部)，得到局部最大值与指数和
    // --------------------------
    // (C1) 并行归约求最大值 cur_m
    __shared__ float cur_m;
    float warp_val =
        (d_tid < B_c && threadIdx.y == 0) ? s_score_buf[d_tid] : -FLT_MAX;
    // 所有线程都参与 warp 内归约
    unsigned int mask = __activemask();
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      warp_val = fmaxf(warp_val, __shfl_down_sync(mask, warp_val, offset));
    }

    // 最终归约结果由 lane 0 获得
    if (d_tid == 0 && threadIdx.y == 0) {
      cur_m = warp_val;
    }
    __syncthreads();

    // (C2) 并行归约求指数和 cur_l
    // (C2) 并行归约求指数和 cur_l (改为 warp reduce)
    __shared__ float cur_l_s;
    float warp_val_l = 0.0f;
    __shared__ float s_s_score[B_C_VALUE];
    // 只有 (d_tid < B_c && y==0) 的线程去计算 exp(...)，其余线程赋值0
    if (d_tid < B_c && threadIdx.y == 0) {
      warp_val_l = expf(s_score_buf[d_tid] - cur_m);
      s_s_score[d_tid] = warp_val_l;
    }

    // 仍然所有线程都参与同一段归约指令
    unsigned int mask2 = __activemask();
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      warp_val_l += __shfl_down_sync(mask2, warp_val_l, offset);
    }
    // 最终由 lane 0 保存该块内所有 token 的指数和
    if (d_tid == 0 && threadIdx.y == 0) {
      cur_l_s = warp_val_l;
    }
    __syncthreads();

    float cur_l = cur_l_s;
    __syncthreads();

    // --------------------------
    // (D) 在本分块内计算对 V 的加权求和 (部分输出)
    // --------------------------
    float partial_out = 0.0f;  // 当前线程所计算的输出元素的累加器

    // 遍历当前块中的所有 token（长度为 B_c）
    for (int i = 0; i < B_c; ++i) {
      // 获取第 i 个 token 的预计算过的 softmax 分值（已做指数运算）
      // 从共享内存 s_s_score 中读取，共有 B_c 个元素
      float exp_score = s_s_score[i];
      float weight = exp_score / cur_l;  // 归一化 softmax 权重
      float v_val =
          s_vj[i * dqkv + d_tid];  // 获取该 token 的第 d_tid 维 Value 值

      // 将当前 token 的权重值累加到输出结果中。使用 fmaf 可能更高效（FMA 指令）
      partial_out = fmaf(weight, v_val, partial_out);
      // 或者：partial_out += weight * v_val;
    }

    // 将当前线程计算的结果（对应某个 feature 维度 d_tid）
    // 存入共享内存输出缓冲 s_o 中
    // 确保 s_o 的声明正确：__shared__ float s_o[DQKV_VALUE];（或等效大小）

    // 同步 block 中所有线程，确保 s_o 完全写入
    // 因为下一步 online update（步骤 E）会从 s_o 中读取
    __syncthreads();

    // --------------------------
    // (E) 将本分块输出累加到 att_output，并做全局 softmax 递归更新
    // --------------------------
    if (j == 0) {
      // 第一个分块，直接写输出
      if (token_tid == 0) {
        s_o[q_offset + d_tid] = static_cast<T>(partial_out);
      }
      // 全局 softmax 的初始 m, l
      if (token_tid == 0 && d_tid == 0) {
        global_m = cur_m;
        global_l = cur_l;
      }
      __syncthreads();
    } else {
      // 后续分块，做“递归归一化”更新
      float new_global_m = fmaxf(global_m, cur_m);
      float exp_old = expf(global_m - new_global_m);
      float exp_cur = expf(cur_m - new_global_m);
      float new_global_l = global_l * exp_old + cur_l * exp_cur;

      float old_out = static_cast<float>(s_o[q_offset + d_tid]);
      float new_out =
          (global_l * exp_old * old_out + cur_l * exp_cur * partial_out) /
          new_global_l;

      if (token_tid == 0) {
        s_o[q_offset + d_tid] = static_cast<T>(new_out);
      }
      if (token_tid == 0 && d_tid == 0) {
        global_m = new_global_m;
        global_l = new_global_l;
      }
      __syncthreads();
    }
  }  // end for each chunk (T_c)
  // 写回att_output

  if (threadIdx.y == 0) att_output[q_offset + d_tid] = s_o[q_offset + d_tid];
  __syncthreads();
}

// 暂时不考虑动态dqkv

template <typename T>
__global__ void flash_attention_kernel_v3(T* q, const T* k, const T* v,
                                          T* att_output, int n_q_h,
                                          int cache_length, int n_kv_h,
                                          int dqkv, int B_c, int B_r,
                                          int n_groups, int T_r, int T_c,
                                          T softmax_scale) {
  // 检查预设参数是否一致
  if (dqkv != DQKV_VALUE || B_c != B_C_VALUE) return;
  // 共享内存均采用 float 类型
  __shared__ float s_qi[DQKV_VALUE];  // 当前 head 的 Query（float）
  __shared__ float s_kj[B_C_VALUE * DQKV_VALUE];  // 当前 chunk 的 Key（float）
  __shared__ float
      s_vj[B_C_VALUE * DQKV_VALUE];         // 当前 chunk 的 Value（float）
  __shared__ float s_score_buf[B_C_VALUE];  // 每个 token 的得分
  __shared__ float s_tmp[DQKV_VALUE * B_C_VALUE];  // 临时归约数组
  // s_lm[0 ~ B_C_VALUE-1] 用于归约；s_lm[B_C_VALUE]
  // 存全局最大值；s_lm[B_C_VALUE+1] 存全局归一化因子
  __shared__ float s_lm[2];
  __shared__ float s_max_local[B_C_VALUE];  // 局部最大值归约缓存
  __shared__ float s_exp_local[B_C_VALUE];  // 局部指数和归约缓存

  // 线程内变量
  const int d_tid = threadIdx.x;  // 特征维度内线程索引，[0, dqkv)
  const int token_tid = threadIdx.y;  // 当前 chunk 内 token 线程索引，[0, B_c)
  const int head_id = blockIdx.x;
  const int q_offset = head_id * dqkv;
  const int kv_head = head_id / n_groups;  // KV head 索引
  constexpr int vec_unit = 16 / sizeof(T);
  // --------------------------
  // 1. 加载当前 head 的 Query 到共享内存（float4）
  // --------------------------
  Vec<__nv_bfloat16, vec_unit> vec, v1, v2;
  int vecCount = dqkv / vec_unit;
  if (token_tid == 0) {
    // 每次加载 vec_unit 个 T 类型数据，共 dqkv / vec_unit 次迭代
    for (int i = d_tid; i < vecCount; i += blockDim.x) {
      // 每次载入 16 字节数据
      vec.f4 = *reinterpret_cast<const float4*>(&q[q_offset + i * vec_unit]);
#pragma unroll
      for (int j = 0; j < vec_unit; j++) {
        s_qi[i * vec_unit + j] = static_cast<float>(vec.t[j]);
      }
    }
  }
  __syncthreads();

  float& global_m = s_lm[0];
  float& global_l = s_lm[1];
  // --------------------------
  // 2. 遍历每个 KV 分块（chunk）
  // --------------------------
  for (int j = 0; j < T_c; ++j) {
    int token_index = j * B_c + token_tid;
    bool valid = (token_index < cache_length);

    for (int i = d_tid; i < vecCount; i += blockDim.x) {
      if (valid) {
        int index = (token_index * n_kv_h + kv_head) * dqkv + i * vec_unit;
        // 加载 key
        v1.f4 = *reinterpret_cast<const float4*>(&k[index]);
#pragma unroll
        for (int l = 0; l < vec_unit; l++) {
          s_kj[token_tid * dqkv + i * vec_unit + l] =
              static_cast<float>(v1.t[l]);
        }
        // 加载 value
        v2.f4 = *reinterpret_cast<const float4*>(&v[index]);
#pragma unroll
        for (int l = 0; l < vec_unit; l++) {
          s_vj[token_tid * dqkv + i * vec_unit + l] =
              static_cast<float>(v2.t[l]);
        }
      } else {
        // 无效位置填零
#pragma unroll
        for (int l = 0; l < vec_unit; l++) {
          s_kj[token_tid * dqkv + i * vec_unit + l] = 0.0f;
          s_vj[token_tid * dqkv + i * vec_unit + l] = 0.0f;
        }
      }
    }
    __syncthreads();

    // 2.2 计算 Query 与每个 Key 的点积得分（float）
    float score = 0.0f;
    if (valid) {
      for (int i = d_tid; i < dqkv; i += blockDim.x) {
        score += s_qi[i] * s_kj[token_tid * dqkv + i];
      }
      // 每个线程计算的局部得分归约到同一 token 上
      int index = token_tid * dqkv + d_tid;
      s_tmp[index] = score;
      __syncthreads();
      for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (d_tid < stride) {
          s_tmp[index] += s_tmp[index + stride];
        }
        __syncthreads();
      }
      if (d_tid == 0) {
        // 应用 softmax 缩放因子
        s_score_buf[token_tid] =
            s_tmp[token_tid * dqkv] * static_cast<float>(softmax_scale);
      }
    } else {
      if (d_tid == 0) {
        s_score_buf[token_tid] = -FLT_MAX;
      }
    }
    __syncthreads();

    // 3.3 计算当前分块 softmax 归一化参数
    // 3.3.1 并行归约求最大值 cur_m
    if (d_tid == 0) {
      s_max_local[token_tid] = s_score_buf[token_tid];
    }
    __syncthreads();
    for (int stride = B_c / 2; stride > 0; stride >>= 1) {
      if (d_tid == 0 && token_tid < stride) {
        s_max_local[token_tid] =
            fmaxf(s_max_local[token_tid], s_max_local[token_tid + stride]);
      }
      __syncthreads();
    }
    float cur_m = s_max_local[0];
    __syncthreads();

    // 3.3.2 并行归约求指数和 cur_l
    if (d_tid == 0) {
      s_exp_local[token_tid] = expf(s_score_buf[token_tid] - cur_m);
    }
    __syncthreads();
    for (int stride = B_c / 2; stride > 0; stride >>= 1) {
      if (d_tid == 0 && token_tid < stride) {
        s_exp_local[token_tid] += s_exp_local[token_tid + stride];
      }
      __syncthreads();
    }
    float cur_l = s_exp_local[0];
    __syncthreads();

    // 3.4 计算当前分块部分输出 partial_out（float 累积）
    float partial_out = 0.0f;
    for (int i = 0; i < B_c; ++i) {
      float weight = expf(s_score_buf[i] - cur_m) / cur_l;
      partial_out += weight * s_vj[i * dqkv + d_tid];
    }

    // 3.5 更新全局 softmax 参数与输出（递归归一化）
    if (j == 0) {
      if (token_tid == 0) {
        att_output[q_offset + d_tid] = static_cast<T>(partial_out);
      }
      if (token_tid == 0 && d_tid == 0) {
        global_m = cur_m;
        global_l = cur_l;
      }
      __syncthreads();
    } else {
      float new_global_m = fmaxf(global_m, cur_m);
      float exp_old = expf(global_m - new_global_m);
      float exp_cur = expf(cur_m - new_global_m);
      float new_global_l = global_l * exp_old + cur_l * exp_cur;
      float old_out = static_cast<float>(att_output[q_offset + d_tid]);
      float new_out =
          (global_l * exp_old * old_out + cur_l * exp_cur * partial_out) /
          new_global_l;
      if (token_tid == 0) {
        att_output[q_offset + d_tid] = static_cast<T>(new_out);
      }
      if (token_tid == 0 && d_tid == 0) {
        global_m = new_global_m;
        global_l = new_global_l;
      }
      __syncthreads();
    }
  }  // end for each KV chunk
}

// 仅保留float计算版本。BF16计算精度不足。
template <typename T>
__global__ void flash_attention_kernel_v2(T* q, const T* k, const T* v,
                                          T* att_output, int n_q_h,
                                          int cache_length, int n_kv_h,
                                          int dqkv, int B_c, int B_r,
                                          int n_groups, int T_r, int T_c,
                                          T softmax_scale) {
  // 检查预设参数是否一致
  if (dqkv != DQKV_VALUE || B_c != B_C_VALUE) return;

  // 共享内存均采用 float 类型
  __shared__ float s_qi[DQKV_VALUE];  // 当前 head 的 Query（float）
  __shared__ float s_kj[B_C_VALUE * DQKV_VALUE];  // 当前 chunk 的 Key（float）
  __shared__ float
      s_vj[B_C_VALUE * DQKV_VALUE];         // 当前 chunk 的 Value（float）
  __shared__ float s_score_buf[B_C_VALUE];  // 每个 token 的得分
  __shared__ float s_tmp[DQKV_VALUE * B_C_VALUE];  // 临时归约数组
  // s_lm[0 ~ B_C_VALUE-1] 用于归约；s_lm[B_C_VALUE]
  // 存全局最大值；s_lm[B_C_VALUE+1] 存全局归一化因子
  __shared__ float s_lm[2];
  __shared__ float s_max_local[B_C_VALUE];  // 局部最大值归约缓存
  __shared__ float s_exp_local[B_C_VALUE];  // 局部指数和归约缓存

  // 线程内变量
  const int d_tid = threadIdx.x;  // 特征维度内线程索引，[0, dqkv)
  const int token_tid = threadIdx.y;  // 当前 chunk 内 token 线程索引，[0, B_c)
  const int head_id = blockIdx.x;
  const int q_offset = head_id * dqkv;
  const int kv_head = head_id / n_groups;  // KV head 索引

  // --------------------------
  // 1. 初始化输出：转换 0.0f 到类型 T
  // --------------------------
  for (int i = d_tid; i < dqkv; i += blockDim.x) {
    att_output[q_offset + i] = static_cast<T>(0.0f);
  }
  __syncthreads();

  // --------------------------
  // 2. 加载当前 head 的 Query 到共享内存（float）
  // --------------------------
  if (token_tid == 0) {
    for (int i = d_tid; i < dqkv; i += blockDim.x) {
      s_qi[i] = static_cast<float>(q[q_offset + i]);
    }
  }
  __syncthreads();

  // s_lm[B_C_VALUE] 存储全局最大值，s_lm[B_C_VALUE+1] 存储全局归一化因子
  float& global_m = s_lm[0];
  float& global_l = s_lm[1];

  // --------------------------
  // 3. 遍历每个 KV 分块（chunk）
  // --------------------------
  for (int j = 0; j < T_c; ++j) {
    int token_index = j * B_c + token_tid;
    bool valid = (token_index < cache_length);

    // 3.1 加载当前分块的 Key 与 Value（无效位置填 0）
    for (int i = d_tid; i < dqkv; i += blockDim.x) {
      if (valid) {
        int index = (token_index * n_kv_h + kv_head) * dqkv + i;
        s_kj[token_tid * dqkv + i] = static_cast<float>(k[index]);
        s_vj[token_tid * dqkv + i] = static_cast<float>(v[index]);
      } else {
        s_kj[token_tid * dqkv + i] = 0.0f;
        s_vj[token_tid * dqkv + i] = 0.0f;
      }
    }
    __syncthreads();

    // 3.2 计算 Query 与每个 Key 的点积得分（float）
    float score = 0.0f;
    if (valid) {
      for (int i = d_tid; i < dqkv; i += blockDim.x) {
        score += s_qi[i] * s_kj[token_tid * dqkv + i];
      }
      // 每个线程计算的局部得分归约到同一 token 上
      int index = token_tid * dqkv + d_tid;
      s_tmp[index] = score;
      __syncthreads();
      for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (d_tid < stride) {
          s_tmp[index] += s_tmp[index + stride];
        }
        __syncthreads();
      }
      if (d_tid == 0) {
        // 应用 softmax 缩放因子
        s_score_buf[token_tid] =
            s_tmp[token_tid * dqkv] * static_cast<float>(softmax_scale);
      }
    } else {
      if (d_tid == 0) {
        s_score_buf[token_tid] = -FLT_MAX;
      }
    }
    __syncthreads();

    // 3.3 计算当前分块 softmax 归一化参数
    // 3.3.1 并行归约求最大值 cur_m
    if (d_tid == 0) {
      s_max_local[token_tid] = s_score_buf[token_tid];
    }
    __syncthreads();
    for (int stride = B_c / 2; stride > 0; stride >>= 1) {
      if (d_tid == 0 && token_tid < stride) {
        s_max_local[token_tid] =
            fmaxf(s_max_local[token_tid], s_max_local[token_tid + stride]);
      }
      __syncthreads();
    }
    float cur_m = s_max_local[0];
    __syncthreads();

    // 3.3.2 并行归约求指数和 cur_l
    if (d_tid == 0) {
      s_exp_local[token_tid] = expf(s_score_buf[token_tid] - cur_m);
    }
    __syncthreads();
    for (int stride = B_c / 2; stride > 0; stride >>= 1) {
      if (d_tid == 0 && token_tid < stride) {
        s_exp_local[token_tid] += s_exp_local[token_tid + stride];
      }
      __syncthreads();
    }
    float cur_l = s_exp_local[0];
    __syncthreads();

    // 3.4 计算当前分块部分输出 partial_out（float 累积）
    float partial_out = 0.0f;
    for (int i = 0; i < B_c; ++i) {
      float weight = expf(s_score_buf[i] - cur_m) / cur_l;
      partial_out += weight * s_vj[i * dqkv + d_tid];
    }

    // 3.5 更新全局 softmax 参数与输出（递归归一化）
    if (j == 0) {
      if (token_tid == 0) {
        att_output[q_offset + d_tid] = static_cast<T>(partial_out);
      }
      if (token_tid == 0 && d_tid == 0) {
        global_m = cur_m;
        global_l = cur_l;
      }
      __syncthreads();
    } else {
      float new_global_m = fmaxf(global_m, cur_m);
      float exp_old = expf(global_m - new_global_m);
      float exp_cur = expf(cur_m - new_global_m);
      float new_global_l = global_l * exp_old + cur_l * exp_cur;
      float old_out = static_cast<float>(att_output[q_offset + d_tid]);
      float new_out =
          (global_l * exp_old * old_out + cur_l * exp_cur * partial_out) /
          new_global_l;
      if (token_tid == 0) {
        att_output[q_offset + d_tid] = static_cast<T>(new_out);
      }
      if (token_tid == 0 && d_tid == 0) {
        global_m = new_global_m;
        global_l = new_global_l;
      }
      __syncthreads();
    }
  }  // end for each KV chunk
}

// -------------------------------
// host 端调用：设置 grid/block、使用静态共享内存（因此 shmem_bytes
// 设为0），并发起 kernel 调用
template <typename T>
void flash_attention(Tensor<T>& Q, const Tensor<T>& K, const Tensor<T>& V,
                     Tensor<T>& att_output) {
  int dqkv = K.sizes()[2];  // 每个 head 内维度
  if (dqkv != DQKV_VALUE) {
    throw std::runtime_error("dqkv 不匹配预定义的值");
  }
  float softmax_scale = 1.0f / sqrtf(static_cast<float>(dqkv));
  int n_q_h = Q.sizes()[1];         // query head 数
  int cache_length = K.sizes()[0];  // 总的 kv token 数
  int n_kv_h = K.sizes()[1];
  int n_groups = n_q_h / n_kv_h;  // GQA 中的分组数

  // decode 模式下 query 长度为 1
  int B_r = 1;
  int T_r = 1;

  // 每个 chunk 读取的 kv token 数（预设为偶数 B_C_VALUE）
  int B_c = B_C_VALUE;
  int T_c = (cache_length + B_c - 1) / B_c;

  // 每个 block 处理一个 query head
  dim3 grid(n_q_h);
  int threads_x = dqkv;  // dqkv = DQKV_VALUE
  int threads_y = B_c;   // B_c = B_C_VALUE
  dim3 block(threads_x, threads_y);

  // 使用静态共享内存，故 shmem_bytes = 0
  flash_attention_kernel_v4<T><<<grid, block, 0>>>(
      Q.data_ptr(), K.data_ptr(), V.data_ptr(), att_output.data_ptr(), n_q_h,
      cache_length, n_kv_h, dqkv, B_c, B_r, n_groups, T_r, T_c,
      static_cast<T>(softmax_scale));
}

// 显式实例化
template void flash_attention<float>(Tensor<float>&, const Tensor<float>&,
                                     const Tensor<float>&, Tensor<float>&);
template void flash_attention<nvbf16>(Tensor<nvbf16>&, const Tensor<nvbf16>&,
                                      const Tensor<nvbf16>&, Tensor<nvbf16>&);

}  // namespace cuda_OP
