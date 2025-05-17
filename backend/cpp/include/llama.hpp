#pragma once
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "base_model.hpp"
#include "inference.hpp"
#include "tensor.hpp"
#include "thread_pool.hpp"

class LlamaModel : public BaseModel {
 public:
  LlamaModel(const std::unordered_map<std::string, Tensor<float>>& params,
             const std::unordered_map<std::string, int>& config);
  bool verify_params() const override;
  void print_model_info() const override;

  // 前向计算
  uint32_t* forward(const Tensor<uint32_t>* input, ThreadPool& thread_pool,
                    KVCacheBase* kv_cache, size_t top_k, float temperature,
                    float top_p, curandState* d_states = nullptr) override;
  uint32_t* prefill(const Tensor<uint32_t>* input, ThreadPool& thread_pool,
                    KVCacheBase* kv_cache, size_t top_k, float temperature,
                    float top_p, curandState* d_states = nullptr) override;
  Tensor<float> prefill_cpu(const Tensor<uint32_t>* input,
                            KVCache<float>* kv_cache, ThreadPool& thread_pool);
  Tensor<float> prefill_cuda(const Tensor<uint32_t>* input,
                             KVCache<float>* kv_cache);
  Tensor<float> forward_cpu(const Tensor<uint32_t>* input,
                            ThreadPool& thread_pool, KVCache<float>* kv_cache);
  Tensor<float> forward_cuda(const Tensor<uint32_t>* input,
                             KVCache<float>* kv_cache);

  // Getter方法
  size_t get_n_layers() const override { return n_layers_; }
  size_t get_max_seq_len() const override { return max_seq_len_; }
  size_t get_head_dim() const override { return dqkv_; }
  size_t get_n_kv_heads() const override { return n_kv_h_; }
  uint32_t get_eos_token_id() const override { return eos_token_id_; }
  size_t get_hidden_size() const override { return d_; }

  // CUDA support methods
  LlamaModel& cuda() override;
  LlamaModel& cpu() override;
  Device device() const override { return device_; }

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
  Device device_;
};