#pragma once

#include "inference.hpp" // 假定 KVCache 定义在此文件中
#include "tensor.hpp"    // 假定 Tensor 模板类定义在此文件中
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

namespace cuda_OP {

void checkCudaError(cudaError_t err);
void print_cuda_memory_usage(const char *location);
void gather(Tensor<float> *output, const Tensor<uint32_t> *input,
            const Tensor<float> *embedding_table);
void rms_norm(Tensor<float> *output, const Tensor<float> *input,
              const Tensor<float> *weight, float eps);
Tensor<float> matmul(const Tensor<float> &A, const Tensor<float> &B,
                     cudaStream_t stream = 0);
void rope(Tensor<float> *tensor, size_t current_pos, float theta);
void softmax(Tensor<float> *output, const Tensor<float> *input, int dim,
             bool mask = true);
void silu(Tensor<float> *output, Tensor<float> *input);
void multiply(Tensor<float> *output, const Tensor<float> *A,
              const Tensor<float> *B);
void compute_attention_scores(const Tensor<float> &Q, const Tensor<float> &K,
                              size_t n_q_h, size_t dqkv,
                              Tensor<float> &att_scores, size_t n_kv_h);
void compute_att_output(const Tensor<float> &att_probs, const Tensor<float> &V,
                        size_t n_q_h, size_t dqkv, Tensor<float> &att_output,
                        size_t n_kv_h);
void compute_attention_scores_prefill(const Tensor<float> &Q,
                                      const Tensor<float> &K,
                                      Tensor<float> &att_scores, size_t dqkv);
void compute_att_output_prefill(const Tensor<float> &att_probs,
                                const Tensor<float> &V,
                                Tensor<float> &att_output, size_t n_q_h,
                                size_t dqkv, size_t total_seq_len,
                                size_t n_kv_h);

} // namespace cuda_OP