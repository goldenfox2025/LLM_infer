#include <cmath>
#include <fstream>
#include <future>
#include <iostream>
#include <string>

#include "avx_operators.hpp"
#include "inference.hpp"
#include "llama.hpp"
#include "operators.hpp"

LlamaModel::LlamaModel(
    const std::unordered_map<std::string, Tensor<float>>& params,
    const std::unordered_map<std::string, int>& config)
    : params_(params) {
  vocab_size_ = config.at("vocab_size");
  n_layers_ = config.at("num_hidden_layers");
  n_q_h_ = config.at("num_attention_heads");
  n_kv_h_ = config.at("num_key_value_heads");
  d_ = config.at("hidden_size");
  dqkv_ = d_ / n_q_h_;  // 使用 n_q_h_ 计算 dqkv_
  di_ = config.at("intermediate_size");
  eps_ = config.at("rms_norm_eps");
  rope_theta_ = config.at("rope_theta");
  max_seq_len_ = config.at("max_position_embeddings");
  bos_token_id_ = config.at("bos_token_id");
  eos_token_id_ = config.at("eos_token_id");
}

bool LlamaModel::verify_params() const {
  try {
    // 验证权重存在性
    const std::vector<std::string> required_weights = {"embedding_table",
                                                       "lm_head", "rms_out_w"};
    for (const auto& weight : required_weights) {
      if (params_.find(weight) == params_.end()) {
        std::cerr << "Missing required weight: " << weight << std::endl;
        return false;
      }
    }

    // 验证每一层的权重存在和维度
    for (size_t i = 0; i < n_layers_; i++) {
      const std::vector<std::string> layer_weights = {
          "rms_att_w" + std::to_string(i), "wq" + std::to_string(i),
          "wk" + std::to_string(i),        "wv" + std::to_string(i),
          "wo" + std::to_string(i),        "rms_ffn_w" + std::to_string(i),
          "w_up" + std::to_string(i),      "w_down" + std::to_string(i),
          "w_gate" + std::to_string(i)};
      for (const auto& weight : layer_weights) {
        if (params_.find(weight) == params_.end()) {
          std::cerr << "Missing layer weight: " << weight << std::endl;
          return false;
        }
      }

      // 验证权重矩阵维度
      const auto& wq = params_.at("wq" + std::to_string(i));
      const auto& wk = params_.at("wk" + std::to_string(i));
      const auto& wv = params_.at("wv" + std::to_string(i));
      const auto& wo = params_.at("wo" + std::to_string(i));

      if (wq.sizes()[1] != n_q_h_ * dqkv_ || wq.sizes()[0] != d_) {
        std::cerr << "Q weight matrix dimension mismatch at layer " << i
                  << std::endl;
        return false;
      }
      if (wk.sizes()[1] != n_kv_h_ * dqkv_ || wk.sizes()[0] != d_) {
        std::cerr << "K weight matrix dimension mismatch at layer " << i
                  << std::endl;
        return false;
      }
      if (wv.sizes()[1] != n_kv_h_ * dqkv_ || wv.sizes()[0] != d_) {
        std::cerr << "V weight matrix dimension mismatch at layer " << i
                  << std::endl;
        return false;
      }
      if (wo.sizes()[1] != d_ || wo.sizes()[0] != n_q_h_ * dqkv_) {
        std::cerr << "Output projection matrix dimension mismatch at layer "
                  << i << std::endl;
        return false;
      }
    }
    return true;
  } catch (const std::out_of_range& e) {
    std::cerr << "Missing weight: " << e.what() << std::endl;
    return false;
  }
}

void LlamaModel::print_model_info() const {
  std::cout << "Model Info:\n"
            << "vocab_size: " << vocab_size_ << "\n"
            << "n_layers: " << n_layers_ << "\n"
            << "n_q_h: " << n_q_h_ << "\n"
            << "n_kv_h: " << n_kv_h_ << "\n"
            << "hidden_size: " << d_ << "\n"
            << "head_dim: " << dqkv_ << "\n"
            << "intermediate_size: " << di_ << std::endl;
}

// template <typename T>
// void print_tensor_shape(const std::string& filename, const Tensor<T>& tensor)
// { std::ofstream fout(filename, std::ios::binary); if (!fout.is_open()) {
//   std::cerr << "Failed to open file " << filename << " for writing.\n";
//   return;
// }

// // 写出 shape 大小
// size_t ndims = tensor.sizes().size();
// fout.write(reinterpret_cast<const char*>(&ndims), sizeof(size_t));

// // 写出各维度
// for (auto dim : tensor.sizes()) {
//   fout.write(reinterpret_cast<const char*>(&dim), sizeof(size_t));
// }

// // 写出数据
// fout.write(reinterpret_cast<const char*>(tensor.data_ptr()),
//            tensor.numel() * sizeof(T));
// fout.close();
// return;
// -------------------
//   const auto& sizes = tensor.sizes();
//   std::cout << name << " shape: [";
//   for (size_t i = 0; i < sizes.size(); i++) {
//     std::cout << sizes[i] << (i < sizes.size() - 1 ? ", " : "");
//   }
//   std::cout << "]" << std::endl;
// }

Tensor<float> LlamaModel::forward(const Tensor<uint32_t>* input,
                                  ThreadPool& thread_pool, KVCache* kv_cache) {
  //   std::cout << "=== Forward Pass Start ===" << std::endl;
  //   std::cout << "Input token IDs: ";
  // for (size_t i = 0; i < input->numel(); i++) {
  //   std::cout << input->data_ptr()[i] << " ";
  // }
  // std::cout << std::endl;
  const size_t seq_len = input->numel();
  //   std::cout << "Sequence length: " << seq_len << std::endl;
  const size_t n_groups = n_q_h_ / n_kv_h_;
  // std::cout << kv_cache->size() << std::endl;

  const size_t current_pos = kv_cache ? kv_cache->size() - seq_len : 0;
  // 当前 token 在 KVCache 中的索引（新 token 的位置，由 InferenceEngine
  // 扩出来）

  // std::cout << "Current KVCache token position: " << current_pos <<
  // std::endl;
  std::vector<float> residual_data(seq_len * d_);
  std::vector<float> hidden_states_data(seq_len * d_);
  Tensor<float> residual(std::move(residual_data), {seq_len, d_});
  Tensor<float> hidden_states(std::move(hidden_states_data), {seq_len, d_});
  // print_tensor_shape("Initial residual", residual);
  // Embedding查表：将 token ID 映射为嵌入向量，存入 residual
  OP::gather(&residual, input, &params_.at("embedding_table"));

  // print_tensor_shape("Residual after embedding", residual);

  for (size_t layer = 0; layer < n_layers_; layer++) {
    // std::cout << "\n--- Layer " << layer << " Start ---" << std::endl;
    // Attention 层前 RMSNorm
    OP::rms_norm(&hidden_states, &residual,
                 &params_.at("rms_att_w" + std::to_string(layer)), eps_);

    // print_tensor_shape(
    // "Hidden states after RMSNorm (att) - layer " + std::to_string(layer),
    // hidden_states);

    // 计算查询向量

    Tensor<float> wq = params_.at("wq" + std::to_string(layer));
    Tensor<float> wk = params_.at("wk" + std::to_string(layer));
    Tensor<float> wv = params_.at("wv" + std::to_string(layer));

    // 使用线程池并行计算 q_buf, k_buf, v_buf
    Tensor<float> q_buf, k_buf, v_buf;

    auto compute_qkv = [&](std::function<Tensor<float>()> matmul_func,
                           Tensor<float>& result) {
      return [&, matmul_func]() {  // 返回一个 lambda 作为 Task
                                   // 的执行体，捕获 result 的引用
        result = matmul_func();
      };
    };

    // 提交任务到线程池
    thread_pool.enqueueTask(std::make_shared<OpTask>(compute_qkv(
        [&]() { return avx_OP::matmul(hidden_states, wq); }, q_buf)));
    thread_pool.enqueueTask(std::make_shared<OpTask>(compute_qkv(
        [&]() { return avx_OP::matmul(hidden_states, wk); }, k_buf)));
    thread_pool.enqueueTask(std::make_shared<OpTask>(compute_qkv(
        [&]() { return avx_OP::matmul(hidden_states, wv); }, v_buf)));

    thread_pool.waitForAllTasks();  // 等待所有任务完成

    // print_tensor_shape("q_buf - layer " + std::to_string(layer), q_buf);

    Tensor<float> q_buf_view = q_buf.view({seq_len, n_q_h_, dqkv_});
    Tensor<float> k_buf_view = k_buf.view({seq_len, n_kv_h_, dqkv_});
    Tensor<float> v_buf_view = v_buf.view({seq_len, n_kv_h_, dqkv_});
    OP::rope(&q_buf_view, current_pos, rope_theta_);
    OP::rope(&k_buf_view, current_pos, rope_theta_);

    if (kv_cache) {
      //   std::cout << "[Layer " << layer
      //             << "] Writing current token K/V into
      //             KVCache at pos "
      // << current_pos << std::endl;
      kv_cache->k_cache(layer, current_pos) = k_buf_view.clone();
      kv_cache->v_cache(layer, current_pos) = v_buf_view.clone();
    }
    // print_tensor_shape("k_buf_view - layer " +
    // std::to_string(layer),
    //  k_buf_view);
    // print_tensor_shape("v_buf_view - layer " +
    // std::to_string(layer),
    //  v_buf_view);

    // Attention计算
    // 从 KVCache 中读取当前层所有已缓存 token 的 K/V 用于计算注意力
    size_t cache_length = (kv_cache ? kv_cache->size() : 0);

    // std::cout << "[Layer " << layer << "] Cache length: " << cache_length
    //           << std::endl;
    std::vector<float> att_scores(n_q_h_ * cache_length, 0.0f);

    // 对每个查询头计算得分（注意：seq_len 为1，这里是decode）
    for (size_t q_head = 0; q_head < n_q_h_; q_head++) {
      size_t kv_head = q_head / n_groups;  // 映射到键/值的头
      float* q_ptr = q_buf_view.data_ptr() + q_head * dqkv_;
      for (size_t pos = 0; pos < cache_length; pos++) {
        // 从 KVCache 中读取缓存的 key，重塑为 [n_kv_h_,
        // dqkv_]
        Tensor<float>& cached_key = kv_cache->k_cache(layer, pos);
        Tensor<float> cached_key_view = cached_key.view({n_kv_h_, dqkv_});
        float* key_ptr = cached_key_view.data_ptr() + kv_head * dqkv_;
        float dot = 0.0f;
        for (size_t d = 0; d < dqkv_; d++) {
          dot += q_ptr[d] * key_ptr[d];
        }
        float scale = 1.0f / sqrtf((float)dqkv_);
        att_scores[q_head * cache_length + pos] = dot * scale;
      }
    }
    // std::cout << "[Layer " << layer << "] Attention scores
    // computed."
    //           << std::endl;
    Tensor<float> att_scores_tensor(std::move(att_scores),
                                    {n_q_h_, cache_length});
    // 分配一个输出 Tensor，形状同 att_scores_tensor
    Tensor<float> att_probs = att_scores_tensor.clone();
    OP::softmax(&att_probs, &att_scores_tensor, /*dim=*/1, false, n_q_h_);

    // std::cout << "[Layer " << layer << "] Attention
    // probabilities computed."
    //           << std::endl;
    // 根据注意力概率对所有缓存中的 V 进行加权求和，得到注意力输出

    std::vector<float> att_out(n_q_h_ * dqkv_, 0.0f);
    for (size_t q_head = 0; q_head < n_q_h_; q_head++) {
      size_t kv_head = q_head / n_groups;
      for (size_t d = 0; d < dqkv_; d++) {
        float weighted_sum = 0.0f;
        for (size_t pos = 0; pos < cache_length; pos++) {
          Tensor<float>& cached_val = kv_cache->v_cache(layer, pos);
          Tensor<float> cached_val_view = cached_val.view({n_kv_h_, dqkv_});
          float* val_ptr = cached_val_view.data_ptr() + kv_head * dqkv_;
          weighted_sum +=
              att_probs.data_ptr()[q_head * cache_length + pos] * val_ptr[d];
        }
        att_out[q_head * dqkv_ + d] = weighted_sum;
      }
    }
    // std::cout << "[Layer " << layer << "] Attention output
    // computed."
    //           << std::endl;
    Tensor<float> att_heads(std::move(att_out), {1, n_q_h_ * dqkv_});
    // 将注意力头合并后映射回 d_ 维空间
    Tensor<float> wo = params_.at("wo" + std::to_string(layer));
    Tensor<float> att_proj = avx_OP::matmul(att_heads, wo);
    // print_tensor_shape("Attention projection - layer " +
    // std::to_string(layer),
    //                    att_proj);

    // 将注意力输出加到 residual 上
    residual = residual + att_proj;
    // print_tensor_shape(
    //     "Residual after attention - layer " +
    //     std::to_string(layer), residual);

    // ------------------ FFN 模块 ------------------
    // FFN前 RMSNorm
    OP::rms_norm(&hidden_states, &residual,
                 &params_.at("rms_ffn_w" + std::to_string(layer)), eps_);
    // print_tensor_shape(
    //     "Hidden states after RMSNorm (FFN) - layer " +
    //     std::to_string(layer), hidden_states);

    // FFN计算：分别计算 gate 和 up 分支
    Tensor<float> w_gate = params_.at("w_gate" + std::to_string(layer));
    Tensor<float> w_up = params_.at("w_up" + std::to_string(layer));
    Tensor<float> gate_buf = avx_OP::matmul(hidden_states, w_gate);
    Tensor<float> up_buf = avx_OP::matmul(hidden_states, w_up);

    // print_tensor_shape(
    //     "Gate buffer before SiLU - layer " +
    //     std::to_string(layer), gate_buf);
    OP::silu(&gate_buf, &gate_buf);
    // print_tensor_shape(
    //     "Gate buffer after SiLU - layer " +
    //     std::to_string(layer), gate_buf);
    OP::multiply(&gate_buf, &gate_buf, &up_buf);
    // print_tensor_shape(
    //     "Gate buffer after multiplication - layer " +
    //     std::to_string(layer), gate_buf);

    // FFN下投影及残差连接
    Tensor<float> w_down = params_.at("w_down" + std::to_string(layer));
    Tensor<float> ffn_out = avx_OP::matmul(gate_buf, w_down);
    residual = residual + ffn_out;
    // print_tensor_shape("Residual after FFN - layer " +
    // std::to_string(layer),
    //                    residual);
    // std::cout << "--- Layer " << layer << " End ---" <<
    // std::endl;
  }

  // 最后一步：整体 RMSNorm 和 LM Head 计算 logits
  OP::rms_norm(&hidden_states, &residual, &params_.at("rms_out_w"), eps_);
  // print_tensor_shape("Hidden states after final RMSNorm", hidden_states);
  // 提取最后一个 token 的 hidden state
  std::vector<float> last_hidden_data(d_, 0.0f);
  Tensor<float> last_hidden(std::move(last_hidden_data), {1, d_});
  size_t start = (seq_len - 1) * d_;
  std::copy(hidden_states.data_ptr() + start,
            hidden_states.data_ptr() + start + d_, last_hidden.data_ptr());
  // print_tensor_shape("Last hidden state", last_hidden);

  // 将 hidden state 投影为 logits
  Tensor<float> lm_head = params_.at("lm_head");
  Tensor<float> logits = avx_OP::matmul(last_hidden, lm_head);
  // print_tensor_shape("Logits", logits);

  //   std::cout << "=== Forward Pass End ===" << std::endl;
  return logits;
}
