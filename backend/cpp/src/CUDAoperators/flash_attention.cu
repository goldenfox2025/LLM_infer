#include <cublas_v2.h>
#include <cuda_bf16.h>  // 提供 __nv_bfloat16 定义
#include <cuda_runtime.h>
#include <math.h>

#include <algorithm>  // min
#include <cstdio>     // printf
#include <cstring>    // memcpy
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cudaOP.cuh"  // 假设此头文件定义了 Tensor<T> 等接口

// ----------------------------------------------
// bf16 与 float 转换函数
// ----------------------------------------------
__device__ inline float bf16_to_float(__nv_bfloat16 x) {
  unsigned short raw;
  memcpy(&raw, &x, sizeof(raw));  // 将 x 的 16 bits 拷贝到 raw
  unsigned int bits = (static_cast<unsigned int>(raw) << 16);
  float f;
  memcpy(&f, &bits, sizeof(f));  // 转换成 float
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
  float fx = bf16_to_float(x);  // 转为 float 计算
  float ef = expf(fx);
  return float_to_bf16(ef);  // 转回 bf16
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
// flash_attention_kernel_v1 内核
// --------------------------------------------------
#include <float.h>
#include <math.h>

template <typename T>
__global__ void flash_attention_kernel_v1(T* q, const T* k, const T* v,
                                          T* att_output, int n_q_h,
                                          int cache_length, int n_kv_h,
                                          int dqkv, int B_c, int B_r,
                                          int n_groups, int T_r, int T_c,
                                          T softmax_scale) {
  // --------------------------
  // 共享内存布局说明
  // --------------------------
  extern __shared__ unsigned char s_raw[];
  T* s_shared = reinterpret_cast<T*>(s_raw);

  T* qi = s_shared;                // [dqkv] —— Query向量
  T* kj = qi + dqkv;               // [B_c * dqkv] —— 分块内Key
  T* vj = kj + B_c * dqkv;         // [B_c * dqkv] —— 分块内Value
  T* score_buf = vj + B_c * dqkv;  // [B_c] —— 存放每个Token的分数
  T* s_tmp = score_buf + B_c;  // [blockDim.x * blockDim.y] —— 临时归约缓存
  T* lm = s_tmp +
          blockDim.x *
              blockDim.y;  // 用于存放全局 softmax 参数：局部最大值及归一化因子

  // --------------------------
  // 线程和Block内变量
  // --------------------------
  const int d_tid = threadIdx.x;    // 特征维度内线程ID
  const int B_c_tid = threadIdx.y;  // 分块内Token线程ID
  const int head_id = blockIdx.x;
  const int q_offset = head_id * dqkv;
  const int kv_head = head_id / n_groups;  // KV的head索引

  // --------------------------
  // 初始化输出，所有线程协作（避免写竞争）
  // --------------------------
  for (int x = d_tid; x < dqkv; x += blockDim.x) {
    att_output[q_offset + x] = T(0);
  }
  __syncthreads();

  // --------------------------
  // 加载当前head的Query向量（仅B_c_tid==0的线程负责加载）
  // --------------------------
  if (B_c_tid == 0) {
    for (int x = d_tid; x < dqkv; x += blockDim.x) {
      qi[x] = q[q_offset + x];
    }
  }
  __syncthreads();

  // lm中的两个位置用于存储全局softmax参数：
  // lm[B_c]   —— 全局最大值
  // lm[B_c+1] —— 全局归一化因子（累积指数和）
  T& m_global = lm[B_c];
  T& global_l = lm[B_c + 1];

  // --------------------------
  // 遍历每个KV块（chunk）
  // --------------------------
  for (int j = 0; j < T_c; ++j) {
    int token_index = j * B_c + B_c_tid;
    bool valid = token_index < cache_length;

    // 1. 加载当前分块的Key和Value：无效位置填0
    for (int x = d_tid; x < dqkv; x += blockDim.x) {
      if (valid) {
        int k_index = (token_index * n_kv_h + kv_head) * dqkv + x;
        kj[B_c_tid * dqkv + x] = k[k_index];
        vj[B_c_tid * dqkv + x] = v[k_index];
      } else {
        kj[B_c_tid * dqkv + x] = T(0);
        vj[B_c_tid * dqkv + x] = T(0);
      }
    }
    __syncthreads();

    // 2. 计算 Query 与当前分块所有 Key 的点积得分
    T score = T(0);
    if (valid) {
      for (int x = d_tid; x < dqkv; x += blockDim.x) {
        score += qi[x] * kj[B_c_tid * dqkv + x];
      }
      // 将每个线程计算的局部得分归约到同一Token
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
        // 应用softmax缩放因子
        score_buf[B_c_tid] = s_tmp[B_c_tid * blockDim.x] * softmax_scale;
      }
    } else if (d_tid == 0) {
      score_buf[B_c_tid] = -FLT_MAX;  // 无效Token分数设为最小值
    }
    __syncthreads();

    // 3. 计算当前分块的Softmax归一化参数
    T cur_m = score_buf[0];
    for (int i = 1; i < B_c; ++i) {
      cur_m = my_fmax(cur_m, score_buf[i]);
    }
    T cur_l = T(0);
    for (int i = 0; i < B_c; ++i) {
      cur_l += my_exp(score_buf[i] - cur_m);
    }

    // 4. 计算当前分块的部分输出 partial_out
    T partial_out = T(0);
    for (int i = 0; i < B_c; ++i) {
      T weight = my_exp(score_buf[i] - cur_m) / cur_l;
      partial_out += weight * vj[i * dqkv + d_tid];
    }

    // 5. 更新全局softmax归一化参数与输出（采用递归归一化方法）
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
// host 端调用：设置 grid/block、共享内存大小，并发起 kernel 调用
template <typename T>
void flash_attention(Tensor<T>& Q, const Tensor<T>& K, const Tensor<T>& V,
                     Tensor<T>& att_output) {
  int dqkv = K.sizes()[2];  // 每个 head 内维度
  float softmax_scale = 1.0f / sqrtf(static_cast<float>(dqkv));
  int n_q_h = Q.sizes()[1];         // query head 数
  int cache_length = K.sizes()[0];  // 总的 kv token 数
  int n_kv_h = K.sizes()[1];
  int n_groups = n_q_h / n_kv_h;  // GQA 中的分组数

  // decode 模式下 query 长度为 1
  int B_r = 1;
  int T_r = 1;

  // 每个 chunk 读取的 kv token 数，例如 B_c = 4
  int B_c = 8;
  int T_c = (cache_length + B_c - 1) / B_c;

  // 每个 block 处理一个 query head
  dim3 grid(n_q_h);
  int threads_x = dqkv;
  int threads_y = B_c;
  dim3 block(threads_x, threads_y);

  // 共享内存大小：
  // qi:         dqkv
  // kj:         B_c * dqkv
  // vj:         B_c * dqkv
  // score_buf:  cache_length
  // s_tmp:      threads_x * threads_y
  // 共享内存大小修正
  size_t shmem_bytes = (dqkv +                     // qi
                        B_c * dqkv +               // kj
                        B_c * dqkv +               // vj
                        B_r * B_c +                // score_buf
                        (threads_x * threads_y) +  // s_tmp
                        B_c + 2  // lm: 全局最大值 + 累积和
                        ) *
                       sizeof(T);

  flash_attention_kernel_v1<T><<<grid, block, shmem_bytes>>>(
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
