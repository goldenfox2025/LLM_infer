#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>  // 用于设备端随机数生成

#include <stdexcept>
#include <vector>

#include "inference.hpp"
#include "tensor.hpp"
#define CUTLASS_CHECK(status)                                             \
  {                                                                       \
    cutlass::Status error = status;                                       \
    if (error != cutlass::Status::kSuccess) {                             \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) \
                << " at: " << __LINE__ << std::endl;                      \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  }

namespace cuda_OP {

template <typename T, int N>
union Vec {
  float4 f4;  // 实际载入 16 字节数据
  T t[N];     // 重解释为 N 个 T 类型元素
};

template <typename T, int N>
union Vec_2 {
  float2 f4;  // 实际载入 8 字节数据
  T t[N];     // 重解释为 N 个 T 类型元素
};
// 定义支持的数据类型别名

using nvbf16 = __nv_bfloat16;
void init_curand(curandState* d_states, unsigned long long seed, int offset,
                 cudaStream_t stream = nullptr);
// 工具函数声明
void checkCudaError(cudaError_t err);
void print_cuda_memory_usage(const char* location);

// 模板化算子函数声明

// 从 embedding_table 中根据 input 索引取值写入 output
template <typename T>
void gather(Tensor<T>* output, const Tensor<uint32_t>* input,
            const Tensor<T>* embedding_table, cudaStream_t stream = nullptr);

// 均方根归一化算子：output = rms_norm(input, weight, eps)
template <typename T>
void rms_norm(Tensor<T>* output, const Tensor<T>* input,
              const Tensor<T>* weight, float eps,
              cudaStream_t stream = nullptr);

template <typename T, typename AccT = float>
void launch_gemmv_scores(const T* x, const T* y, AccT* dst, const int channel_size,
                  const int channel_ratio, const int row_size,
                  const int stride_channel_x, const int stride_channel_y,
                  const int stride_channel_dst, cudaStream_t stream = nullptr);

template <typename T>
void matmul(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>* C,
            cudaStream_t stream = nullptr, const Tensor<T>* bias = nullptr,
            int use_ = 1);

// rope 算子，用于位置编码
template <typename T>
void rope(Tensor<T>* tensor, size_t current_pos, float theta,
          cudaStream_t stream = nullptr);

// softmax 算子，dim 指定操作维度，mask 与 offset 为可选参数
template <typename T>
void softmax(Tensor<T>* output, const Tensor<T>* input, int dim,
             bool mask = true, int offset = 0, cudaStream_t stream = nullptr);

// silu 激活函数算子
template <typename T>
void silu(Tensor<T>* output, const Tensor<T>* input,
          cudaStream_t stream = nullptr);

// 逐元素乘法算子
template <typename T>
void multiply(Tensor<T>* output, const Tensor<T>* A, const Tensor<T>* B,
              cudaStream_t stream = nullptr);

// 逐元素加法算子
template <typename T>
void add(Tensor<T>* output, Tensor<T>* A, Tensor<T>* B,
         cudaStream_t stream = 0);

template <typename T>
uint32_t* sample(Tensor<T>&& input, float temperature, float top_p,
                 size_t top_k, curandState* d_states,
                 cudaStream_t stream = nullptr);

// 计算注意力分数（多头注意力机制相关算子）
template <typename T>
void compute_attention_scores(const Tensor<T>& Q, const Tensor<T>& K,
                              size_t n_q_h, size_t dqkv, Tensor<T>& att_scores,
                              size_t n_kv_h, cudaStream_t stream = nullptr);

// 计算注意力输出
template <typename T>
void compute_att_output(const Tensor<T>& att_probs, const Tensor<T>& V,
                        size_t n_q_h, size_t dqkv, Tensor<T>& att_output,
                        size_t n_kv_h, cudaStream_t stream = nullptr);

// flahattention实现
template <typename T>
void flash_attention(Tensor<T>& Q, const Tensor<T>&& K1,const Tensor<T>&& K2,const Tensor<T>&& K3, const Tensor<T>&& V1 ,const Tensor<T>&& V2 ,const Tensor<T>&& V3 ,
                     Tensor<T>& att_output1, Tensor<T>& att_output2, Tensor<T>& att_output3, cudaStream_t stream = nullptr);



// prefill 版本：计算注意力分数
template <typename T>
void compute_attention_scores_prefill(const Tensor<T>& Q, const Tensor<T>& K,
                                      Tensor<T>& att_scores, size_t dqkv,
                                      cudaStream_t stream = nullptr);

// prefill 版本：计算注意力输出
template <typename T>
void compute_att_output_prefill(const Tensor<T>& att_probs, const Tensor<T>& V,
                                Tensor<T>& att_output, size_t n_q_h,
                                size_t dqkv, size_t total_seq_len,
                                size_t n_kv_h, cudaStream_t stream = nullptr);

template <typename T>
void gather_fa(const Tensor<T>& T1, const Tensor<T>& T2, const Tensor<T>& T3,
               Tensor<T>& T5, cudaStream_t stream = nullptr);

}  // namespace cuda_OP
