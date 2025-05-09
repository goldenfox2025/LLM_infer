#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>  // 提供 __nv_bfloat16 定义
#include <cuda_runtime.h>
#include <curand_kernel.h>  // 用于设备端随机数生成
#include <math.h>

#include <cstdio>  // printf
#include <iostream>
#include <stdexcept>
#include <vector>

#include "inference.hpp"
#include "tensor.hpp"
#define CUTLASS_CHECK(status)                                                                                        \
    {                                                                                                                \
        cutlass::Status error = status;                                                                              \
        if (error != cutlass::Status::kSuccess) {                                                                    \
            std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                                                                      \
        }                                                                                                            \
    }

// CUDA Error Check Macro
#define CUDA_CHECK(call)                                                                                \
    do {                                                                                                \
        cudaError_t err = call;                                                                         \
        if (err != cudaSuccess) {                                                                       \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            throw std::runtime_error(cudaGetErrorString(err));                                          \
        }                                                                                               \
    } while (0)
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

namespace cuda_OP {

// 定义支持的数据类型别名
template <typename T, typename ScaleType = float>
void matmul_quantized_gemv(           // Renamed wrapper
    const Tensor<T> &input,           // [M, K]
    const Tensor<int32_t> &qweight,   // [N, K/8] <- Expected GEMV layout
    const Tensor<ScaleType> &scales,  // [N, G_padded] (G_padded >= G)
    const Tensor<int32_t> &zeros,     // [N, G/8]
    int group_size,
    Tensor<T> *output,  // [M, N] - DIMS NOT CHECKED HERE
    cudaStream_t stream = nullptr, const Tensor<T> *bias = nullptr);
using nvbf16 = __nv_bfloat16;
void init_curand(curandState *d_states, unsigned long long seed, int offset, cudaStream_t stream = nullptr);
// 工具函数声明
void checkCudaError(cudaError_t err);
void print_cuda_memory_usage(const char *location);

template <typename T>
void add_rms(Tensor<T> *output, Tensor<T> *input, const Tensor<T> *add_, const Tensor<T> *weight, float eps,
             cudaStream_t stream = nullptr);

// 从 embedding_table 中根据 input 索引取值写入 output
template <typename T>
void gather(Tensor<T> *output, const Tensor<uint32_t> *input, const Tensor<T> *embedding_table,
            cudaStream_t stream = nullptr);

// 均方根归一化算子：output = rms_norm(input, weight, eps)
template <typename T>
void rms_norm(Tensor<T> *output, const Tensor<T> *input, const Tensor<T> *weight, float eps,
              cudaStream_t stream = nullptr);

template <typename T, typename AccT = float>
void launch_gemv_scores(const T *x, const T *y, AccT *dst, const int channel_size, const int channel_ratio,
                        const int row_size, const int stride_channel_x, const int stride_channel_y,
                        const int stride_channel_dst, cudaStream_t stream = nullptr);

template <typename T>
void matmul(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> *C, cudaStream_t stream = nullptr,
            const Tensor<T> *bias = nullptr, int use_ = 1);

// rope 算子，用于位置编码
template <typename T>
void rope(Tensor<T> *tensor, size_t current_pos, float theta, cudaStream_t stream = nullptr);

// softmax 算子，dim 指定操作维度，mask 与 offset 为可选参数
template <typename T>
void softmax(Tensor<T> *output, const Tensor<T> *input, int dim, bool mask = true, int offset = 0,
             cudaStream_t stream = nullptr);

// silu 激活函数算子
template <typename T>
void silu(Tensor<T> *output, const Tensor<T> *input, cudaStream_t stream = nullptr);

// 逐元素乘法算子
template <typename T>
void multiply(Tensor<T> *output, const Tensor<T> *A, const Tensor<T> *B, cudaStream_t stream = nullptr);

// 逐元素加法算子
template <typename T>
void add(Tensor<T> *output, Tensor<T> *A, Tensor<T> *B, cudaStream_t stream = 0);

template <typename T>
uint32_t *sample(Tensor<T> &&input, float temperature, float top_p, size_t top_k, curandState *d_states,
                 cudaStream_t stream = nullptr);

// 计算注意力分数（多头注意力机制相关算子）
template <typename T>
void compute_attention_scores(const Tensor<T> &Q, const Tensor<T> &K, size_t n_q_h, size_t dqkv, Tensor<T> &att_scores,
                              size_t n_kv_h, cudaStream_t stream = nullptr);

// 计算注意力输出
template <typename T>
void compute_att_output(const Tensor<T> &att_probs, const Tensor<T> &V, size_t n_q_h, size_t dqkv,
                        Tensor<T> &att_output, size_t n_kv_h, cudaStream_t stream = nullptr);

// flahattention实现
template <typename T>
void flash_attention(Tensor<T> &Q, const Tensor<T> &&K1, const Tensor<T> &&K2, const Tensor<T> &&K3,
                     const Tensor<T> &&V1, const Tensor<T> &&V2, const Tensor<T> &&V3, Tensor<T> &att_output1,
                     Tensor<T> &att_output2, Tensor<T> &att_output3, cudaStream_t stream = nullptr);

// prefill 版本：计算注意力分数
template <typename T>
void compute_attention_scores_prefill(const Tensor<T> &Q, const Tensor<T> &K, Tensor<T> &att_scores, size_t dqkv,
                                      cudaStream_t stream = nullptr);

// prefill 版本：计算注意力输出
template <typename T>
void compute_att_output_prefill(const Tensor<T> &att_probs, const Tensor<T> &V, Tensor<T> &att_output, size_t n_q_h,
                                size_t dqkv, size_t total_seq_len, size_t n_kv_h, cudaStream_t stream = nullptr);

// 支持GQA的GEMM计算，用于prefill阶段的注意力分数计算
template <typename T>
void launch_gqa_gemm(const Tensor<T> &Q,  // 查询张量 [batch_size, n_q_heads, seq_len, head_dim]
                     const Tensor<T> &K,  // 键张量 [batch_size, n_kv_heads, seq_len, head_dim]
                     Tensor<T> &scores,   // 输出分数张量 [batch_size, n_q_heads, seq_len, seq_len]

                     cudaStream_t stream = nullptr);  // CUDA流

template <typename T>
void gather_fa(const Tensor<T> &T1, const Tensor<T> &T2, const Tensor<T> &T3, Tensor<T> &T5,
               cudaStream_t stream = nullptr);

// 可变分支数量的gather_fa实现
template <typename T>
void gather_fa_variable(const std::vector<Tensor<T>> &inputs, Tensor<T> &output, cudaStream_t stream = nullptr);

// 特化版本的gather_fa实现 - 1分支
template <typename T>
void gather_fa_specialized_1branch(const std::vector<Tensor<T>> &inputs, Tensor<T> &output,
                                   cudaStream_t stream = nullptr);

// 特化版本的gather_fa实现 - 2分支
template <typename T>
void gather_fa_specialized_2branch(const std::vector<Tensor<T>> &inputs, Tensor<T> &output,
                                   cudaStream_t stream = nullptr);

// 特化版本的gather_fa实现 - 3分支
template <typename T>
void gather_fa_specialized_3branch(const std::vector<Tensor<T>> &inputs, Tensor<T> &output,
                                   cudaStream_t stream = nullptr);

// 特化版本的gather_fa实现 - 4分支
template <typename T>
void gather_fa_specialized_4branch(const std::vector<Tensor<T>> &inputs, Tensor<T> &output,
                                   cudaStream_t stream = nullptr);

// 特化版本的gather_fa实现 - 5分支
template <typename T>
void gather_fa_specialized_5branch(const std::vector<Tensor<T>> &inputs, Tensor<T> &output,
                                   cudaStream_t stream = nullptr);

// 可变分支数量的flash_attention实现
template <typename T>
void flash_attention_variable(Tensor<T> &Q, const std::vector<Tensor<T>> &K_slices,
                              const std::vector<Tensor<T>> &V_slices, std::vector<Tensor<T>> &outputs,
                              cudaStream_t stream = nullptr);

// 特化版本的flash_attention实现 - 1分支
template <typename T>
void flash_attention_specialized_1branch(Tensor<T> &Q, const std::vector<Tensor<T>> &K_slices,
                                         const std::vector<Tensor<T>> &V_slices, std::vector<Tensor<T>> &outputs,
                                         cudaStream_t stream = nullptr);

// 特化版本的flash_attention实现 - 2分支
template <typename T>
void flash_attention_specialized_2branch(Tensor<T> &Q, const std::vector<Tensor<T>> &K_slices,
                                         const std::vector<Tensor<T>> &V_slices, std::vector<Tensor<T>> &outputs,
                                         cudaStream_t stream = nullptr);

// 特化版本的flash_attention实现 - 3分支
template <typename T>
void flash_attention_specialized_3branch(Tensor<T> &Q, const std::vector<Tensor<T>> &K_slices,
                                         const std::vector<Tensor<T>> &V_slices, std::vector<Tensor<T>> &outputs,
                                         cudaStream_t stream = nullptr);

// 特化版本的flash_attention实现 - 4分支
template <typename T>
void flash_attention_specialized_4branch(Tensor<T> &Q, const std::vector<Tensor<T>> &K_slices,
                                         const std::vector<Tensor<T>> &V_slices, std::vector<Tensor<T>> &outputs,
                                         cudaStream_t stream = nullptr);

// 特化版本的flash_attention实现 - 5分支
template <typename T>
void flash_attention_specialized_5branch(Tensor<T> &Q, const std::vector<Tensor<T>> &K_slices,
                                         const std::vector<Tensor<T>> &V_slices, std::vector<Tensor<T>> &outputs,
                                         cudaStream_t stream = nullptr);

// 包装函数：根据KV缓存长度动态选择分支数量
template <typename T>
void dynamic_flash_attention_wrapper(Tensor<T> &Q, const Tensor<T> &total_K, const Tensor<T> &total_V,
                                     Tensor<T> &att_output, int n_kv_heads, cudaStream_t stream = nullptr);

// AWQ量化矩阵乘法
template <typename T, typename ScaleType = float>
void matmul_quantized(const Tensor<T> &input, const Tensor<int32_t> &qweight, const Tensor<ScaleType> &scales,
                      const Tensor<int32_t> &zeros, int group_size, Tensor<T> *output, cudaStream_t stream = nullptr,
                      const Tensor<T> *bias = nullptr);

}  // namespace cuda_OP
