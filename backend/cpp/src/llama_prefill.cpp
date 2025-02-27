#include <cmath>
#include <cstring>
#include <future>
#include <iostream>
#include <string>

#include "avx_operators.hpp"
#include "inference.hpp"
#include "llama.hpp"
#include "operators.hpp"
#include "thread_pool.hpp"
Tensor<float> LlamaModel::prefill(const Tensor<uint32_t>* input,
                                  KVCache* kv_cache, ThreadPool& thread_pool) {
  const size_t seq_len = input->numel();
  size_t offset = 0;
  if (kv_cache) {
    offset = kv_cache->size() - seq_len;
  }

  std::vector<float> residual_data(seq_len * d_);
  Tensor<float> residual(std::move(residual_data), {seq_len, d_});

  OP::gather(&residual, input, &params_.at("embedding_table"));

  std::vector<float> hidden_states_data(seq_len * d_);
  Tensor<float> hidden_states(std::move(hidden_states_data), {seq_len, d_});

  const size_t n_groups = n_q_h_ / n_kv_h_;

  // 遍历每一层
  for (size_t layer = 0; layer < n_layers_; layer++) {
    // RMSNorm（注意力前）
    OP::rms_norm(&hidden_states, &residual,
                 &params_.at("rms_att_w" + std::to_string(layer)), eps_);

    // 并行计算 Q、K 和 V
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
    // 在这里，q_buf, k_buf, v_buf 都已经计算完成，可以继续后续操作
    Tensor<float> q_buf_view = q_buf.view({seq_len, n_q_h_, dqkv_});
    Tensor<float> k_buf_view = k_buf.view({seq_len, n_kv_h_, dqkv_});
    OP::rope(&q_buf_view, offset, rope_theta_);
    OP::rope(&k_buf_view, offset, rope_theta_);

    if (kv_cache) {
      Tensor<float> k_buf_contiguous = k_buf.clone();
      Tensor<float> v_buf_contiguous = v_buf.clone();
      size_t row_size = n_kv_h_ * dqkv_;
      for (size_t i = 0; i < seq_len; i++) {
        const float* k_ptr = k_buf_contiguous.data_ptr() + i * row_size;
        const float* v_ptr = v_buf_contiguous.data_ptr() + i * row_size;
        std::vector<float> k_i(k_ptr, k_ptr + row_size);
        std::vector<float> v_i(v_ptr, v_ptr + row_size);
        kv_cache->k_cache(layer, offset + i) =
            Tensor<float>(std::move(k_i), {1, row_size});
        kv_cache->v_cache(layer, offset + i) =
            Tensor<float>(std::move(v_i), {1, row_size});
      }
    }

    Tensor<float> Q_3d = q_buf.view({seq_len, n_q_h_, dqkv_});
    size_t total_seq_len = seq_len;
    Tensor<float> total_K, total_V;
    if (offset != 0) {
      size_t cached_len = offset;
      total_seq_len = cached_len + seq_len;
      size_t row_size = n_kv_h_ * dqkv_;

      std::vector<float> total_K_data(total_seq_len * row_size);
      std::vector<float> total_V_data(total_seq_len * row_size);

      for (size_t pos = 0; pos < cached_len; pos++) {
        Tensor<float>& cached_k = kv_cache->k_cache(layer, pos);
        Tensor<float>& cached_v = kv_cache->v_cache(layer, pos);
        memcpy(&total_K_data[pos * row_size], cached_k.data_ptr(),
               row_size * sizeof(float));
        memcpy(&total_V_data[pos * row_size], cached_v.data_ptr(),
               row_size * sizeof(float));
      }

      memcpy(&total_K_data[cached_len * row_size], k_buf_view.data_ptr(),
             seq_len * row_size * sizeof(float));
      Tensor<float> v_buf_view = v_buf.view({seq_len, n_kv_h_, dqkv_});
      memcpy(&total_V_data[cached_len * row_size], v_buf_view.data_ptr(),
             seq_len * row_size * sizeof(float));

      total_K = Tensor<float>(std::move(total_K_data),
                              {total_seq_len, n_kv_h_, dqkv_});
      total_V = Tensor<float>(std::move(total_V_data),
                              {total_seq_len, n_kv_h_, dqkv_});
    } else {
      total_K = k_buf.view({seq_len, n_kv_h_, dqkv_}).clone();
      total_V = v_buf.view({seq_len, n_kv_h_, dqkv_}).clone();
    }

    std::vector<float> att_scores(seq_len * n_q_h_ * total_seq_len, 0.0f);
    for (size_t i = 0; i < seq_len; i++) {
      for (size_t qh = 0; qh < n_q_h_; qh++) {
        size_t kv_head = qh / n_groups;
        const float* q_ptr = Q_3d.data_ptr() + (i * n_q_h_ + qh) * dqkv_;
        for (size_t j = 0; j < total_seq_len; j++) {
          const float* k_ptr =
              total_K.data_ptr() + (j * n_kv_h_ + kv_head) * dqkv_;
          float dot = 0.0f;
          for (size_t d = 0; d < dqkv_; d++) {
            dot += q_ptr[d] * k_ptr[d];
          }
          dot /= std::sqrt(float(dqkv_));
          att_scores[i * (n_q_h_ * total_seq_len) + qh * total_seq_len + j] =
              dot;
        }
      }
    }

    // // 应用因果 mask
    // // 对于新 token，其绝对位置为 (kv_cache ? offset+i : i)
    // for (size_t i = 0; i < seq_len; i++) {
    //   size_t abs_pos = (kv_cache ? offset + i : i);
    //   for (size_t j = abs_pos + 1; j < total_seq_len; j++) {
    //     for (size_t qh = 0; qh < n_q_h_; qh++) {
    //       att_scores[i * (n_q_h_ * total_seq_len) + qh * total_seq_len + j] =
    //           -1e9f;
    //     }
    //   }
    // }

    Tensor<float> att_scores_tensor(std::move(att_scores),
                                    {seq_len, n_q_h_, total_seq_len});
    OP::softmax(&att_scores_tensor, &att_scores_tensor, /*dim=*/2, true, n_q_h_,
                offset);

    std::vector<float> att_out(seq_len * n_q_h_ * dqkv_, 0.0f);
    float* att_ptr = att_scores_tensor.data_ptr();
    for (size_t i = 0; i < seq_len; i++) {
      for (size_t qh = 0; qh < n_q_h_; qh++) {
        size_t kv_head = qh / n_groups;
        for (size_t d = 0; d < dqkv_; d++) {
          float sum_val = 0.f;
          for (size_t j = 0; j < total_seq_len; j++) {
            float att_w =
                att_ptr[i * (n_q_h_ * total_seq_len) + qh * total_seq_len + j];
            const float* v_ptr =
                total_V.data_ptr() + (j * n_kv_h_ + kv_head) * dqkv_;
            sum_val += att_w * v_ptr[d];
          }
          att_out[(i * n_q_h_ + qh) * dqkv_ + d] = sum_val;
        }
      }
    }

    Tensor<float> att_heads(std::move(att_out), {seq_len, n_q_h_ * dqkv_});

    Tensor<float> wo = params_.at("wo" + std::to_string(layer));
    Tensor<float> att_proj = avx_OP::matmul(att_heads, wo);
    residual = residual + att_proj;
    // FFN 前 RMSNorm
    OP::rms_norm(&hidden_states, &residual,
                 &params_.at("rms_ffn_w" + std::to_string(layer)), eps_);

    // FFN：Gate, Up, Down 分支
    Tensor<float> w_gate = params_.at("w_gate" + std::to_string(layer));
    Tensor<float> w_up = params_.at("w_up" + std::to_string(layer));
    Tensor<float> w_down = params_.at("w_down" + std::to_string(layer));

    Tensor<float> gate_buf = avx_OP::matmul(hidden_states, w_gate);
    Tensor<float> up_buf = avx_OP::matmul(hidden_states, w_up);

    OP::silu(&gate_buf, &gate_buf);
    OP::multiply(&gate_buf, &gate_buf, &up_buf);

    Tensor<float> ffn_out = avx_OP::matmul(gate_buf, w_down);
    residual = residual + ffn_out;
  }

  std::vector<float> final_h_data(seq_len * d_);
  Tensor<float> final_h(std::move(final_h_data), {seq_len, d_});
  OP::rms_norm(&final_h, &residual, &params_.at("rms_out_w"), eps_);

  Tensor<float> lm_head = params_.at("lm_head");
  Tensor<float> logits = avx_OP::matmul(final_h, lm_head);

  return logits;
}
