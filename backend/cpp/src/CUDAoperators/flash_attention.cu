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
  __shared__ float s_lm[B_C_VALUE + 2];
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
  float& global_m = s_lm[B_C_VALUE];
  float& global_l = s_lm[B_C_VALUE + 1];

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

// --------------------------------------------------
// flash_attention_kernel_v1 内核（静态共享内存版本）
// --------------------------------------------------
template <typename T>
__global__ void flash_attention_kernel_v1(T* q, const T* k, const T* v,
                                          T* att_output, int n_q_h,
                                          int cache_length, int n_kv_h,
                                          int dqkv, int B_c, int B_r,
                                          int n_groups, int T_r, int T_c,
                                          T softmax_scale) {
  // 检查预设参数是否一致
  if (dqkv != DQKV_VALUE || B_c != B_C_VALUE) return;

  // --------------------------
  // 静态共享内存分区
  // --------------------------
  // 每个 head 内的 query 向量：dqkv
  __shared__ T s_qi[DQKV_VALUE];
  // 当前 chunk 内的 key：B_c * dqkv
  __shared__ T s_kj[B_C_VALUE * DQKV_VALUE];
  // 当前 chunk 内的 value：B_c * dqkv
  __shared__ T s_vj[B_C_VALUE * DQKV_VALUE];
  // 当前 chunk 内每个 token 对应的得分：B_c
  __shared__ T s_score_buf[B_C_VALUE];
  // 临时归约缓存：固定大小 dqkv * B_c
  __shared__ T s_tmp[DQKV_VALUE * B_C_VALUE];
  // 用于存储全局 softmax 参数（后续用于累积全局归一化因子）
  // s_lm[B_c]   —— 存储全局最大值 m_global
  // s_lm[B_c+1] —— 存储全局归一化因子 global_l
  __shared__ T s_lm[B_C_VALUE + 2];

  // 为避免共享内存重用带来的数值干扰，新开辟两个独立区域用于归约计算：
  // s_max_local: 用于并行归约计算当前 chunk 的最大得分 cur_m
  // s_exp_local: 用于并行归约计算指数和 cur_l
  __shared__ T s_max_local[B_C_VALUE];
  __shared__ T s_exp_local[B_C_VALUE];

  // --------------------------
  // 线程和 Block 内变量
  // --------------------------
  const int d_tid = threadIdx.x;  // 特征维度内线程ID, 范围[0, dqkv)
  const int B_c_tid = threadIdx.y;  // 当前 chunk 内 token 线程ID, 范围[0, B_c)
  const int head_id = blockIdx.x;
  const int q_offset = head_id * dqkv;
  const int kv_head = head_id / n_groups;  // KV head 索引

  // --------------------------
  // 初始化输出，所有线程协作（避免写竞争）
  // --------------------------
  for (int x = d_tid; x < dqkv; x += blockDim.x) {
    att_output[q_offset + x] = T(0);
  }
  __syncthreads();

  // --------------------------
  // 加载当前 head 的 Query 向量（仅 B_c_tid==0 的线程负责加载）
  // --------------------------
  if (B_c_tid == 0) {
    for (int x = d_tid; x < dqkv; x += blockDim.x) {
      s_qi[x] = q[q_offset + x];
    }
  }
  __syncthreads();

  // s_lm 用于存储全局 softmax 参数：
  // s_lm[B_c]   —— 全局最大值
  // s_lm[B_c+1] —— 全局归一化因子（累积指数和）
  T& m_global = s_lm[B_c];
  T& global_l = s_lm[B_c + 1];

  // --------------------------
  // 遍历每个 KV 块（chunk）
  // --------------------------
  for (int j = 0; j < T_c; ++j) {
    int token_index = j * B_c + B_c_tid;
    bool valid = token_index < cache_length;

    // 1. 加载当前分块的 Key 和 Value：无效位置填 0
    for (int x = d_tid; x < dqkv; x += blockDim.x) {
      if (valid) {
        int k_index = (token_index * n_kv_h + kv_head) * dqkv + x;
        s_kj[B_c_tid * dqkv + x] = k[k_index];
        s_vj[B_c_tid * dqkv + x] = v[k_index];
      } else {
        s_kj[B_c_tid * dqkv + x] = T(0);
        s_vj[B_c_tid * dqkv + x] = T(0);
      }
    }
    __syncthreads();

    // 2. 计算 Query 与当前分块所有 Key 的点积得分
    T score = T(0);
    if (valid) {
      for (int x = d_tid; x < dqkv; x += blockDim.x) {
        score += s_qi[x] * s_kj[B_c_tid * dqkv + x];
      }
      // 将每个线程计算的局部得分归约到同一 Token 上
      int index = B_c_tid * dqkv + d_tid;
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
        s_score_buf[B_c_tid] = s_tmp[B_c_tid * dqkv] * softmax_scale;
        // 同时将 s_score_buf 的值复制到 s_lm 前 B_c 个位置（原来做归约用）
        s_lm[B_c_tid] = s_tmp[B_c_tid * dqkv] * softmax_scale;
      }
    } else if (d_tid == 0) {
      s_score_buf[B_c_tid] = T(-FLT_MAX + 1);  // 无效 token 得分设为很小值
      s_lm[B_c_tid] = T(-FLT_MAX + 1);
    }
    __syncthreads();

    // 3. 计算当前分块的 Softmax 归一化参数
    // 当前块有效 token 数量
    const int chunk_start = j * B_c;
    const int current_chunk_size = min(B_c, cache_length - chunk_start);

    // ---------------------
    // 3.1 并行归约计算最大值 cur_m
    // ---------------------
    if (d_tid == 0) {
      // 将 s_score_buf 中的值复制到 s_max_local
      s_max_local[B_c_tid] = s_score_buf[B_c_tid];
    }
    __syncthreads();
    for (int stride = B_c / 2; stride > 0; stride >>= 1) {
      if (d_tid == 0 && B_c_tid < stride) {
        s_max_local[B_c_tid] =
            my_fmax(s_max_local[B_c_tid], s_max_local[B_c_tid + stride]);
      }
      __syncthreads();
    }
    T cur_m = s_max_local[0];
    __syncthreads();

    // ---------------------
    // 3.2 并行归约计算归一化因子 cur_l（指数和）
    if (d_tid == 0) {
      // 将每个 token 的指数值计算出来，存入 s_exp_local
      s_exp_local[B_c_tid] = my_exp(s_score_buf[B_c_tid] - cur_m);
    }
    __syncthreads();
    for (int stride = B_c / 2; stride > 0; stride >>= 1) {
      if (d_tid == 0 && B_c_tid < stride) {
        s_exp_local[B_c_tid] += s_exp_local[B_c_tid + stride];
      }
      __syncthreads();
    }
    T cur_l = s_exp_local[0];
    __syncthreads();

    // 如果不希望使用并行归约求和，也可以采用顺序遍历的方式（效果相同，但并行性较差）
    /*
    T cur_l = T(0);
    for (int i = 0; i < B_c; ++i) {
      cur_l += my_exp(s_score_buf[i] - cur_m);
    }
    __syncthreads();
    */

    // 4. 计算当前分块的部分输出 partial_out
    T partial_out = T(0);
    for (int i = 0; i < B_c; ++i) {
      T k_val = s_score_buf[i];
      T weight = my_exp(k_val - cur_m) / cur_l;
      partial_out += weight * s_vj[i * dqkv + d_tid];
    }

    // 5. 更新全局 softmax 参数与输出（采用递归归一化方法）
    if (j == 0) {
      // 第一个分块：直接写入输出
      if (B_c_tid == 0) {
        att_output[q_offset + d_tid] = partial_out;
      }
      if (B_c_tid == 0 && d_tid == 0) {
        m_global = cur_m;
        global_l = cur_l;
      }
      __syncthreads();
    } else {
      // 多个分块累积
      T new_m_global = my_fmax(m_global, cur_m);
      T exp_m_old = my_exp(m_global - new_m_global);
      T exp_m_cur = my_exp(cur_m - new_m_global);
      T new_l_global = global_l * exp_m_old + cur_l * exp_m_cur;

      // 融合之前的输出与当前分块部分输出
      T old_out = att_output[q_offset + d_tid];
      T new_out =
          (global_l * exp_m_old * old_out + cur_l * exp_m_cur * partial_out) /
          new_l_global;
      if (B_c_tid == 0) {
        att_output[q_offset + d_tid] = new_out;
      }
      if (B_c_tid == 0 && d_tid == 0) {
        m_global = new_m_global;
        global_l = new_l_global;
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
  flash_attention_kernel_v2<T><<<grid, block, 0>>>(
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
