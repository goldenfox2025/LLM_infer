#include "inference.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "llama.hpp"
#include "operators.hpp"

// ------------------------
// KVCache 实现
// ------------------------
KVCache::KVCache(size_t n_layers, size_t max_seq_len, size_t head_dim,
                 size_t initial_size)
    : n_layers_(n_layers),
      max_seq_len_(max_seq_len),
      head_dim_(head_dim),
      current_len_(0) {
  // 分配每一层每个位置的缓存（总共 n_layers * max_seq_len 个 Tensor）
  k_cache_.resize(n_layers_ * max_seq_len_);
  v_cache_.resize(n_layers_ * max_seq_len_);
  for (size_t i = 0; i < n_layers_ * max_seq_len_; i++) {
    // 初始化一个大小为 head_dim 的数据向量，全部置 0
    std::vector<float> k_data(head_dim_, 0.0f);
    std::vector<float> v_data(head_dim_, 0.0f);
    // 使用数据向量构造 Tensor，形状为 [1, head_dim]
    k_cache_[i] = Tensor<float>(std::move(k_data), {1, head_dim_});
    v_cache_[i] = Tensor<float>(std::move(v_data), {1, head_dim_});
  }
  // 如果指定了初始大小，则设置当前长度
  if (initial_size > 0) {
    if (initial_size > max_seq_len_) {
      throw std::runtime_error("Initial size cannot exceed max_seq_len");
    }
    current_len_ = initial_size;
  }
}

void KVCache::resize(size_t new_size) {
  if (new_size > max_seq_len_) {
    throw std::runtime_error("KVCache: Attempted to resize beyond max_seq_len");
  }
  if (new_size <= current_len_) {
    return;
    // throw std::runtime_error(
    //     "KVCache: New size must be larger than current size");
  }
  current_len_ = new_size;
}

void KVCache::clear() { current_len_ = 0; }

Tensor<float>& KVCache::k_cache(size_t layer, size_t pos) {
  size_t index = layer * max_seq_len_ + pos;
  // static int a = 0;
  // a++;
  // std::cout << "[KVCache::k_cache] layer: " << layer << ", pos: " << pos
  //           << ", index: " << index << "a: " << a << std::endl;
  if (index >= k_cache_.size()) {
    throw std::runtime_error("K cache index out of bounds: " +
                             std::to_string(index));
  }
  return k_cache_[index];
}

Tensor<float>& KVCache::v_cache(size_t layer, size_t pos) {
  size_t index = layer * max_seq_len_ + pos;
  // static int a = 0;
  // a++;
  // std::cout << "[KVCache::v_cache] layer: " << layer << ", pos: " << pos
  //           << ", index: " << index << "a: " << a << std::endl;
  if (index >= v_cache_.size()) {
    throw std::runtime_error("V cache index out of bounds: " +
                             std::to_string(index));
  }
  return v_cache_[index];
}

// ------------------------
// InferenceEngine 实现
// ------------------------
InferenceEngine::InferenceEngine(std::shared_ptr<LlamaModel> model)
    : model_(model),
      // 使用模型参数初始化 KVCache
      kv_cache_(model_->get_n_layers(), model_->get_max_seq_len(),
                model_->get_head_dim() * model_->get_n_kv_heads()),
      thread_pool_(4) {}

uint32_t InferenceEngine::generate_next_token(
    ThreadPool& thread_pool, const std::vector<uint32_t>& input_ids,
    float temperature, float top_p, size_t top_k) {
  // 构造输入张量，取 input_ids 中最后一个 token
  Tensor<uint32_t> input({input_ids.back()}, {1});

  // 更新 KV 缓存长度（为新 token 分配缓存空间）
  try {
    kv_cache_.resize(kv_cache_.size() + 1);
  } catch (const std::runtime_error& e) {
    std::cerr << "Error resizing KV cache: " << e.what() << std::endl;
    throw;
  }

  // 前向计算，传入 KVCache 的地址
  Tensor<float> logits = model_->forward(&input, thread_pool, &kv_cache_);

  // 根据 logits 采样下一个 token
  uint32_t next_token = OP::sample(&logits, temperature, top_p, top_k);
  // 获取 token 映射，并打印当前 token（如果映射存在）
  // static const auto token_map = create_token_map();
  // auto it = token_map.find(next_token);
  // if (it != token_map.end()) {
  //   std::cout << it->second;
  // }
  return next_token;
}
#include <functional>  // 添加以使用 std::function

// 在 InferenceEngine 类的实现文件（例如 inference.cpp）中增加：
void InferenceEngine::generate_with_callback(
    const std::vector<uint32_t>& input_ids, size_t max_length,
    float temperature, float top_p, size_t top_k,
    std::function<void(uint32_t)> callback) {
  // 如果需要从头开始，可清空缓存
  // kv_cache_.clear();

  // 让 KVCache 扩容到容纳 input_ids.size() 个位置
  try {
    kv_cache_.resize(kv_cache_.size() + input_ids.size());
  } catch (const std::runtime_error& e) {
    std::cerr << "Error resizing KV cache (prefill): " << e.what() << std::endl;
    throw;
  }

  Tensor<uint32_t> input_tensor(std::vector<uint32_t>(input_ids),
                                {input_ids.size()});

  // 调用 prefill，一次性处理全部 input_ids
  Tensor<float> prefill_logits =
      model_->prefill(&input_tensor, &kv_cache_, thread_pool_);
  // prefill_logits.shape = [seq_len, vocab_size]
  size_t seq_len = input_ids.size();
  if (seq_len == 0) {
    return;
  }

  // 从 prefill 的最后一行 logits 中采样第一个生成 token
  const size_t vocab_size = prefill_logits.sizes()[1];
  std::vector<float> last_row_data(vocab_size, 0.f);
  const float* prefill_ptr =
      prefill_logits.data_ptr() + (seq_len - 1) * vocab_size;
  std::copy(prefill_ptr, prefill_ptr + vocab_size, last_row_data.begin());
  Tensor<float> last_logits(std::move(last_row_data), {1, vocab_size});
  uint32_t next_token = OP::sample(&last_logits, temperature, top_p, top_k);

  // 通过回调函数将 token 传回给 Python
  callback(next_token);
  if (next_token == model_->get_eos_token_id()) {
    return;
  }

  std::vector<uint32_t> output = input_ids;
  output.push_back(next_token);

  // 继续生成直到达到 max_length 或遇到 eos
  while (output.size() < max_length) {
    // if (kv_cache_.size() >= kv_cache_.max_seq_len_) {
    //   callback(-1);  // 传递 -1 表示超出
    //   break;
    // }
    next_token =
        generate_next_token(thread_pool_, output, temperature, top_p, top_k);
    output.push_back(next_token);
    callback(next_token);
    if (next_token == model_->get_eos_token_id()) {
      break;
    }
  }
}

void InferenceEngine::reset() { kv_cache_.clear(); }
