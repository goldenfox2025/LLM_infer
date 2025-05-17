#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "base_model.hpp"
#include "inference.hpp"
#include "tensor.hpp"
#include "thread_pool.hpp"

// 定义一个结构体来存储 GPU 指针和数量
struct GPUTokens {
  std::vector<uint32_t*> tokens;     // GPU 指针数组
  std::vector<uint32_t> cpu_tokens;  // CPU 上的 token 副本，用于调试

  // 构造函数
  GPUTokens() {}

  // 添加一个 GPU 指针
  void add_token(uint32_t* token_ptr, uint32_t token_value) {
    tokens.push_back(token_ptr);
    cpu_tokens.push_back(token_value);
  }

  // 获取 token 数量
  size_t size() const { return tokens.size(); }

  // 判断是否为空
  bool empty() const { return tokens.empty(); }
};

// 投机解码器类，用于实现投机解码功能
template <typename T>
class SpeculativeDecoder {
 public:
  // 构造函数，接收目标模型和草稿模型
  SpeculativeDecoder(std::shared_ptr<BaseModel> target_model,
                     std::shared_ptr<BaseModel> draft_model,
                     size_t spec_length = 4);  // 默认投机长度为4

  // 生成文本，通过回调函数返回每个token
  void generate_with_callback(const std::vector<uint32_t>& input_ids,
                              size_t max_length, float temperature, float top_p,
                              size_t top_k,
                              std::function<void(uint32_t)> callback);

 private:
  // 目标模型（大模型）
  std::shared_ptr<BaseModel> target_model_;
  // 草稿模型（小模型）
  std::shared_ptr<BaseModel> draft_model_;
  // 目标模型KV缓存
  KVCache<T> target_kv_cache_;
  // 草稿模型KV缓存
  KVCache<T> draft_kv_cache_;
  // 线程池
  ThreadPool thread_pool_;
  // CUDA随机状态
  curandState* d_states;
  // 设备类型
  Device device_;
  // 投机长度（一次生成多少个token）
  size_t spec_length_;

  // 初始化CUDA资源
  void init_cuda_resources();
  // 释放CUDA资源
  void free_cuda_resources();

  // 批量验证草稿模型生成的token
  // 返回最长匹配的长度
  size_t verify_draft_tokens(const std::vector<uint32_t>& prefix_tokens,
                             const std::vector<uint32_t>& draft_tokens,
                             float temperature, float top_p, size_t top_k,
                             std::vector<uint32_t>& verified_tokens);

  // 使用草稿模型生成多个token
  GPUTokens generate_draft_tokens(const uint32_t* input_token,
                                  size_t num_tokens, float temperature,
                                  float top_p, size_t top_k);
};

// 显式声明模板类
extern template class SpeculativeDecoder<__nv_bfloat16>;
