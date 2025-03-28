#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>

#include "inference.hpp"
#include "tensor.hpp"

namespace cuda_OP {

// 定义支持的数据类型别名

using nvbf16 = __nv_bfloat16;

// 工具函数声明
void checkCudaError(cudaError_t err);
void print_cuda_memory_usage(const char* location);

// 模板化算子函数声明

// 从 embedding_table 中根据 input 索引取值写入 output
template <typename T>
void gather(Tensor<T>* output, const Tensor<uint32_t>* input,
            const Tensor<T>* embedding_table);

// 均方根归一化算子：output = rms_norm(input, weight, eps)
template <typename T>
void rms_norm(Tensor<T>* output, const Tensor<T>* input,
              const Tensor<T>* weight, float eps);

/**
 * @brief 矩阵乘法 C = AB
 * @param A 输入矩阵A [M, K]
 * @param B 输入矩阵B [K, N]（转置后）
 * @param stream CUDA流
 * @param bias 可选偏置 [N]，默认为nullptr
 * @return 输出矩阵C [M, N]
 */
template <typename T>
void matmul(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>* C,
            cudaStream_t stream = nullptr, const Tensor<T>* bias = nullptr);

// rope 算子，用于位置编码
template <typename T>
void rope(Tensor<T>* tensor, size_t current_pos, float theta);

// softmax 算子，dim 指定操作维度，mask 与 offset 为可选参数
template <typename T>
void softmax(Tensor<T>* output, const Tensor<T>* input, int dim,
             bool mask = true, int offset = 0);

// silu 激活函数算子
template <typename T>
void silu(Tensor<T>* output, const Tensor<T>* input);

// 逐元素乘法算子
template <typename T>
void multiply(Tensor<T>* output, const Tensor<T>* A, const Tensor<T>* B);

// 逐元素加法算子
template <typename T>
void add(Tensor<T>* output, const Tensor<T>* A, const Tensor<T>* B);

// Layer normalization算子 - 沿最后一个维度进行归一化

// 计算注意力分数（多头注意力机制相关算子）
template <typename T>
void compute_attention_scores(const Tensor<T>& Q, const Tensor<T>& K,
                              size_t n_q_h, size_t dqkv, Tensor<T>& att_scores,
                              size_t n_kv_h);

// 计算注意力输出
template <typename T>
void compute_att_output(const Tensor<T>& att_probs, const Tensor<T>& V,
                        size_t n_q_h, size_t dqkv, Tensor<T>& att_output,
                        size_t n_kv_h);

// flahattention实现
template <typename T>
void flash_attention(Tensor<T>& Q, const Tensor<T>& K, const Tensor<T>& V,
                   Tensor<T>& att_output);

// prefill 版本：计算注意力分数
template <typename T>
void compute_attention_scores_prefill(const Tensor<T>& Q, const Tensor<T>& K,
                                      Tensor<T>& att_scores, size_t dqkv);

// prefill 版本：计算注意力输出
template <typename T>
void compute_att_output_prefill(const Tensor<T>& att_probs, const Tensor<T>& V,
                                Tensor<T>& att_output, size_t n_q_h,
                                size_t dqkv, size_t total_seq_len,
                                size_t n_kv_h);

}  // namespace cuda_OP
