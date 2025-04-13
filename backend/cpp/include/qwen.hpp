#pragma once
#include <cuda_bf16.h>  // For __nv_bfloat16 support
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "base_model.hpp"
#include "cudaOP.cuh"
#include "inference.hpp"
#include "tensor.hpp"
#include "thread_pool.hpp"

constexpr int kNumStreams = 5;

template <typename T>
class QwenModel : public BaseModel {
 public:
  QwenModel(const std::unordered_map<std::string, Tensor<T>>& params,
            const std::unordered_map<std::string, int>& config);
  ~QwenModel() override;

  bool verify_params() const override;
  void print_model_info() const override;

  // BaseModel 接口实现
  uint32_t* forward(const Tensor<uint32_t>* input, ThreadPool& thread_pool,
                    KVCacheBase* kv_cache, size_t top_k, float temperature,
                    float top_p, curandState* d_states = nullptr) override;
  uint32_t* prefill(const Tensor<uint32_t>* input, ThreadPool& thread_pool,
                    KVCacheBase* kv_cache, size_t top_k, float temperature,
                    float top_p, curandState* d_states = nullptr) override {
    KVCache<T>* typed_cache = dynamic_cast<KVCache<T>*>(kv_cache);
    return cuda_OP::sample(prefill_cuda(input, typed_cache), temperature, top_p,
                           top_k, d_states);
  }

  // Token 生成接口（暂未实现）
  std::vector<uint32_t> generate(const std::vector<uint32_t>& input_ids,
                                 size_t max_length, float temperature = 1.0f,
                                 float top_p = 0.9f, size_t top_k = 50);

  // Getter 方法
  size_t get_n_layers() const override { return n_layers_; }
  size_t get_max_seq_len() const override { return max_position_embeddings_; }
  size_t get_head_dim() const override { return head_dim_; }
  size_t get_n_kv_heads() const override { return n_kv_heads_; }
  uint32_t get_eos_token_id() const override { return eos_token_id_; }
  size_t get_n_heads() const { return n_heads_; }
  size_t get_hidden_size() const { return hidden_size_; }
  size_t get_intermediate_size() const { return intermediate_size_; }
  float get_rms_norm_eps() const { return rms_norm_eps_; }
  float get_rope_theta() const { return rope_theta_; }
  size_t get_vocab_size() const { return vocab_size_; }
  const std::unordered_map<std::string, Tensor<T>>& get_params() const {
    return params_;
  }

  // CUDA 前向接口，支持传入流参数。
  // 当 used_stream 非空时要求使用预先分配的 workspace 避免动态内存分配
  Tensor<T> forward_cuda(const Tensor<uint32_t>* input, KVCache<T>* kv_cache,
                         Tensor<T>* p_output,
                         cudaStream_t used_stream = nullptr);
  Tensor<T> prefill_cuda(const Tensor<uint32_t>* input, KVCache<T>* kv_cache);

  // 设备管理接口
  QwenModel& cuda() override;
  QwenModel& cpu() override;
  Device device() const override { return device_; }

  // 控制 CUDA 图优化开关
  void setGraphEnabled(bool enabled) { graph_enabled_ = enabled; }
  bool getGraphEnabled() const { return graph_enabled_; }

 private:
  // 模型参数
  size_t vocab_size_;
  size_t n_layers_;
  size_t n_heads_;
  size_t n_kv_heads_;
  size_t hidden_size_;
  size_t head_dim_;
  size_t intermediate_size_;
  size_t max_position_embeddings_;
  uint32_t bos_token_id_;
  uint32_t eos_token_id_;
  float rms_norm_eps_;
  float rope_theta_;

  std::unordered_map<std::string, Tensor<T>> params_;
  Device device_;

  std::array<cudaStream_t, kNumStreams> compute_streams_;

  // 预分配工作区（用于 forward_cuda capture 阶段复用，避免动态内存分配）
  Tensor<T> workspace_residual_;  // shape: [1, hidden_size_]
  Tensor<T> workspace_hidden_;    // shape: [1, hidden_size_]

  // 专用捕获/执行流
  cudaStream_t graph_stream_;

  // 预分配的 forward 输出（固定 shape: [1, vocab_size_]）
  Tensor<T> forward_logits_;

  // 捕获期间使用的 CUDA 图对象
  cudaGraph_t forward_graph_;

  // 单个图执行实例（我们每次 forward 重新捕获，只维护一个实例）
  cudaGraphExec_t forward_graph_exec_;

  bool graph_enabled_ = true;
};

extern template class QwenModel<float>;
extern template class QwenModel<__nv_bfloat16>;

std::unordered_map<std::string, Tensor<__nv_bfloat16>> convert_weights_to_bf16(
    const std::unordered_map<std::string, Tensor<float>>& float_weights);
