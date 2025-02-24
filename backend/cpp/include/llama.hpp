#pragma once
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "inference.hpp"
#include "tensor.hpp"
#include "thread_pool.hpp"
class LlamaModel {
 public:
  LlamaModel(const std::unordered_map<std::string, Tensor<float>>& params,
             const std::unordered_map<std::string, int>& config);
  bool verify_params() const;
  void print_model_info() const;

  // 前向计算
  Tensor<float> forward(const Tensor<uint32_t>* input,
                        ThreadPool& thread_pool,KVCache* kv_cache = nullptr);
  Tensor<float> prefill(const Tensor<uint32_t>* input, KVCache* kv_cache,ThreadPool& thread_pool);

  std::vector<uint32_t> generate(const std::vector<uint32_t>& input_ids,
                                 size_t max_length, float temperature = 1.0f,
                                 float top_p = 0.9f, size_t top_k = 50);

  // Getter方法
  size_t get_n_layers() const { return n_layers_; }
  size_t get_max_seq_len() const { return max_seq_len_; }
  size_t get_head_dim() const { return dqkv_; }
  size_t get_n_kv_heads() const { return n_kv_h_; }
  uint32_t get_eos_token_id() const { return eos_token_id_; }

 private:
  // 基础参数
  size_t vocab_size_;
  size_t n_layers_;
  size_t n_q_h_;
  size_t n_kv_h_;
  size_t d_;     // hidden_size
  size_t dqkv_;  // hidden_size / num_attention_heads
  size_t di_;    // intermediate_size
  float eps_;
  float rope_theta_;
  size_t max_seq_len_;
  uint32_t bos_token_id_;
  uint32_t eos_token_id_;

  // 模型参数
  std::unordered_map<std::string, Tensor<float>> params_;
};