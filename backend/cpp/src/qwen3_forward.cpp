#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "cudaOP.cuh"
#include "operators/unified_operators.hpp"
#include "qwen3.hpp"
#include "tensor.hpp"

// -------------------------------
// forward: 前向传播接口
// -------------------------------
template <typename T>
uint32_t *Qwen3Model<T>::forward(const Tensor<uint32_t> *input,
                                 ThreadPool &thread_pool, KVCacheBase *kv_cache,
                                 size_t top_k, float temperature, float top_p,
                                 curandState *d_states) {
  KVCache<T> *typed_cache = dynamic_cast<KVCache<T> *>(kv_cache);

  return cuda_OP::sample(forward_cuda(input, typed_cache), temperature, top_p,
                         top_k, d_states);
}

// -------------------------------
// forward_cuda: CUDA 版本的前向传播实现
// -------------------------------
template <typename T>
Tensor<T> Qwen3Model<T>::forward_cuda(const Tensor<uint32_t> *input,
                                      KVCache<T> *kv_cache) {
  // 确保输入在 CUDA 上

  if (input->device() != Device::CUDA) {
    throw std::runtime_error("Input tensor must be on CUDA device");
  }

  // 获取输入信息，前向传播时序列长度固定为1
  const size_t seq_len = 1;

  // 计算起始KV缓存位置
  size_t offset = 0;
  if (kv_cache) {
    if (kv_cache->device() != Device::CUDA) {
      throw std::runtime_error("KVCache must be on CUDA device");
    }

    offset = kv_cache->size() - seq_len;
  }

  // 创建residual和hidden_states张量
  Tensor<T> residual({seq_len, hidden_size_}, Device::CUDA);
  Tensor<T> hidden_states({seq_len, hidden_size_}, Device::CUDA);

  // Token嵌入 (从embedding_table中获取token嵌入)
  cuda_OP::gather(&residual, input, &params_.at("token_embeddings.weight"));

  // 主循环：遍历所有Transformer层
  for (size_t i = 0; i < n_layers_; i++) {
    std::string layer_prefix = "layers." + std::to_string(i) + ".";

    // Attention 输入层归一化 (RMSNorm)
    auto &attention_norm_weight = params_.at("rms_att_w" + std::to_string(i));
    operators_->rms_norm(&hidden_states, &residual, &attention_norm_weight,
                         rms_norm_eps_);

    // 获取偏置项（如果存在）
    const Tensor<T> *q_bias = nullptr;
    const Tensor<T> *k_bias = nullptr;
    const Tensor<T> *v_bias = nullptr;
    const Tensor<T> *o_bias = nullptr;

    try {
      q_bias = &params_.at(layer_prefix + "self_attn.q_proj.bias");
    } catch (const std::out_of_range &) {
    }

    try {
      k_bias = &params_.at(layer_prefix + "self_attn.k_proj.bias");
    } catch (const std::out_of_range &) {
    }

    try {
      v_bias = &params_.at(layer_prefix + "self_attn.v_proj.bias");
    } catch (const std::out_of_range &) {
    }

    try {
      o_bias = &params_.at(layer_prefix + "self_attn.o_proj.bias");
    } catch (const std::out_of_range &) {
    }

    // 创建q/k/v向量的缓冲区
    Tensor<T> q_buf({seq_len, n_heads_ * head_dim_}, Device::CUDA);
    Tensor<T> k_buf({seq_len, n_kv_heads_ * head_dim_}, Device::CUDA);
    Tensor<T> v_buf({seq_len, n_kv_heads_ * head_dim_}, Device::CUDA);

    // 获取权重（自动处理量化与非量化情况）
    auto q_weight = get_weight("wq" + std::to_string(i));
    auto k_weight = get_weight("wk" + std::to_string(i));
    auto v_weight = get_weight("wv" + std::to_string(i));

    // 使用operators_->matmul
    operators_->matmul(&q_buf, &hidden_states, q_weight, q_bias);
    operators_->matmul(&k_buf, &hidden_states, k_weight, k_bias);
    operators_->matmul(&v_buf, &hidden_states, v_weight, v_bias);

    // Qwen3特有: Q/K归一化
    auto &q_norm_weight = params_.at("q_norm" + std::to_string(i));
    auto &k_norm_weight = params_.at("k_norm" + std::to_string(i));

    // 将q_buf和k_buf重塑为3D张量视图
    Tensor<T> q_buf_view = q_buf.view({seq_len, n_heads_, head_dim_});
    Tensor<T> k_buf_view = k_buf.view({seq_len, n_kv_heads_, head_dim_});

    // 对整个q_buf_view和k_buf_view执行RMS归一化
    // RMS归一化实现会自动处理最后一维（head_dim_）
    operators_->rms_norm(&q_buf_view, &q_buf_view, &q_norm_weight,
                         rms_norm_eps_);
    operators_->rms_norm(&k_buf_view, &k_buf_view, &k_norm_weight,
                         rms_norm_eps_);

    // 应用旋转位置编码 (RoPE)，计算注意力，并更新KV缓存
    // 使用 n_heads_ * head_dim_ 而不是 hidden_size_ 来创建注意力输出张量
    // 这样可以同时支持 0.6B 和 1.7B 模型
    Tensor<T> attn_output({seq_len, n_heads_ * head_dim_}, Device::CUDA);

    // 应用RoPE（旋转位置编码）
    operators_->rope(&q_buf_view, offset, rope_theta_);
    operators_->rope(&k_buf_view, offset, rope_theta_);

    // 存储K, V到缓存
    // 假设k_buf_view和v_buf已经准备好
    Tensor<T> v_buf_view = v_buf.view({seq_len, n_kv_heads_, head_dim_});
    for (size_t j = 0; j < seq_len; j++) {
      // 获取缓存中的位置
      Tensor<T> &k_cache_slice = kv_cache->k_cache(i, offset + j);
      Tensor<T> &v_cache_slice = kv_cache->v_cache(i, offset + j);

      // 从当前k_buf_view和v_buf_view中拷贝到缓存
      size_t head_size = n_kv_heads_ * head_dim_;
      cudaMemcpy(k_cache_slice.data_ptr(),
                 k_buf_view.data_ptr() + j * head_size, head_size * sizeof(T),
                 cudaMemcpyDeviceToDevice);
      cudaMemcpy(v_cache_slice.data_ptr(),
                 v_buf_view.data_ptr() + j * head_size, head_size * sizeof(T),
                 cudaMemcpyDeviceToDevice);
    }

    // 从缓存获取完整的K和V序列
    auto [k_cache_tensor, v_cache_tensor] = kv_cache->get_contiguous_tensor(i);
    size_t total_seq_len = offset + seq_len;

    Tensor<T> k_cache_view =
        k_cache_tensor.view({total_seq_len, n_kv_heads_, head_dim_});
    Tensor<T> v_cache_view =
        v_cache_tensor.view({total_seq_len, n_kv_heads_, head_dim_});

    // 使用动态Flash-Attention包装函数计算自注意力
    // 确保att_out_view是3D张量 [seq_len, n_heads_, head_dim_]
    Tensor<T> att_out_view = attn_output.view({n_heads_, head_dim_});
    cuda_OP::dynamic_flash_attention_wrapper(
        q_buf_view, k_cache_view, v_cache_view, att_out_view, n_kv_heads_);

    // 确保attn_output是2D张量 [seq_len, n_heads_ *
    // head_dim_]，用于后续的matmul操作
    attn_output = attn_output.view({seq_len, n_heads_ * head_dim_});
    // 注意力输出投影
    Tensor<T> attn_proj({seq_len, hidden_size_}, Device::CUDA);

    // 获取权重（自动处理量化与非量化情况）
    auto o_weight = get_weight("wo" + std::to_string(i));

    // 执行矩阵乘法
    operators_->matmul(&attn_proj, &attn_output, o_weight, o_bias);

    // 第一个残差连接
    operators_->add(&residual, &residual, &attn_proj);

    // FFN 输入层归一化 (RMSNorm)
    auto &ffn_norm_weight = params_.at("rms_ffn_w" + std::to_string(i));
    operators_->rms_norm(&hidden_states, &residual, &ffn_norm_weight,
                         rms_norm_eps_);

    // MLP (Feed-Forward Network)
    const Tensor<T> *gate_bias = nullptr;
    const Tensor<T> *up_bias = nullptr;
    const Tensor<T> *down_bias = nullptr;

    try {
      gate_bias = &params_.at(layer_prefix + "mlp.gate_proj.bias");
    } catch (const std::out_of_range &) {
    }

    try {
      up_bias = &params_.at(layer_prefix + "mlp.up_proj.bias");
    } catch (const std::out_of_range &) {
    }

    try {
      down_bias = &params_.at(layer_prefix + "mlp.down_proj.bias");
    } catch (const std::out_of_range &) {
    }

    // 计算Gate和Up投影
    Tensor<T> gate_buf({seq_len, intermediate_size_}, Device::CUDA);
    Tensor<T> up_buf({seq_len, intermediate_size_}, Device::CUDA);

    // 获取权重（自动处理量化与非量化情况）
    auto gate_weight = get_weight("w_gate" + std::to_string(i));
    auto up_weight = get_weight("w_up" + std::to_string(i));

    // 执行矩阵乘法
    operators_->matmul(&gate_buf, &hidden_states, gate_weight, gate_bias);
    operators_->matmul(&up_buf, &hidden_states, up_weight, up_bias);

    // 应用SiLU激活函数到gate_buf并与up_buf相乘
    operators_->silu(&gate_buf, &gate_buf);               // SiLU激活
    operators_->multiply(&gate_buf, &gate_buf, &up_buf);  // 逐元素相乘

    // Down投影
    Tensor<T> ffn_out({seq_len, hidden_size_}, Device::CUDA);

    // 获取权重（自动处理量化与非量化情况）
    auto down_weight = get_weight("w_down" + std::to_string(i));

    // 执行矩阵乘法
    operators_->matmul(&ffn_out, &gate_buf, down_weight, down_bias);

    // 残差连接
    operators_->add(&residual, &residual, &ffn_out);
  }

  // 最终的LayerNorm (RMSNorm)
  auto &norm_weight = params_.at("rms_out_w");
  Tensor<T> final_h({seq_len, hidden_size_}, Device::CUDA);
  operators_->rms_norm(&final_h, &residual, &norm_weight, rms_norm_eps_);

  // LM head投影到词汇表大小
  auto lm_head_weight = get_weight("lm_head");
  const Tensor<T> *lm_head_bias = nullptr;

  Tensor<T> logits({seq_len, vocab_size_}, Device::CUDA);
  // 使用operators_->matmul
  operators_->matmul(&logits, &final_h, lm_head_weight, lm_head_bias);

  // 返回logits
  return logits;
}

// 显式实例化模板函数
template uint32_t *Qwen3Model<__nv_bfloat16>::forward(
    const Tensor<uint32_t> *input, ThreadPool &thread_pool,
    KVCacheBase *kv_cache, size_t top_k, float temperature, float top_p,
    curandState *d_states);

template Tensor<__nv_bfloat16> Qwen3Model<__nv_bfloat16>::forward_cuda(
    const Tensor<uint32_t> *input, KVCache<__nv_bfloat16> *kv_cache);
