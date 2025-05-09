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
uint32_t *Qwen3Model<T>::forward(const Tensor<uint32_t> *input, ThreadPool &thread_pool, KVCacheBase *kv_cache,
                                 size_t top_k, float temperature, float top_p, curandState *d_states) {
    KVCache<T> *typed_cache = dynamic_cast<KVCache<T> *>(kv_cache);

    return cuda_OP::sample(forward_cuda(input, typed_cache), temperature, top_p, top_k, d_states);
}

// -------------------------------
// forward_cuda: CUDA 版本的前向传播实现
// -------------------------------
template <typename T>
Tensor<T> Qwen3Model<T>::forward_cuda(const Tensor<uint32_t> *input, KVCache<T> *kv_cache) {
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
        operators_->rms_norm(&hidden_states, &residual, &attention_norm_weight, rms_norm_eps_);

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

        // Q, K, V 投影
        if (quant_type_ == 1) {
            // AWQ量化版本
            std::string q_weight_key = "wq" + std::to_string(i);
            std::string k_weight_key = "wk" + std::to_string(i);
            std::string v_weight_key = "wv" + std::to_string(i);

            std::string q_qweight_key = q_weight_key + ".qweight";
            std::string q_scales_key = q_weight_key + ".scales";
            std::string q_qzeros_key = q_weight_key + ".qzeros";

            std::string k_qweight_key = k_weight_key + ".qweight";
            std::string k_scales_key = k_weight_key + ".scales";
            std::string k_qzeros_key = k_weight_key + ".qzeros";

            std::string v_qweight_key = v_weight_key + ".qweight";
            std::string v_scales_key = v_weight_key + ".scales";
            std::string v_qzeros_key = v_weight_key + ".qzeros";

            // 检查是否存在量化权重
            auto q_qweight_it = qweight_params_.find(q_qweight_key);
            auto k_qweight_it = qweight_params_.find(k_qweight_key);
            auto v_qweight_it = qweight_params_.find(v_qweight_key);

            if (q_qweight_it != qweight_params_.end() && scales_params_.find(q_scales_key) != scales_params_.end() &&
                qzeros_params_.find(q_qzeros_key) != qzeros_params_.end()) {
                // 使用量化矩阵乘法
                cuda_OP::matmul_quantized_gemv(hidden_states, qweight_params_.at(q_qweight_key),
                                               scales_params_.at(q_scales_key), qzeros_params_.at(q_qzeros_key),
                                               group_size_, &q_buf, nullptr, q_bias);
            } else {
                throw std::runtime_error("Missing quantized weights for Q projection");
            }

            if (k_qweight_it != qweight_params_.end() && scales_params_.find(k_scales_key) != scales_params_.end() &&
                qzeros_params_.find(k_qzeros_key) != qzeros_params_.end()) {
                // 使用量化矩阵乘法
                cuda_OP::matmul_quantized_gemv(hidden_states, qweight_params_.at(k_qweight_key),
                                               scales_params_.at(k_scales_key), qzeros_params_.at(k_qzeros_key),
                                               group_size_, &k_buf, nullptr, k_bias);
            } else {
                throw std::runtime_error("Missing quantized weights for K projection");
            }

            if (v_qweight_it != qweight_params_.end() && scales_params_.find(v_scales_key) != scales_params_.end() &&
                qzeros_params_.find(v_qzeros_key) != qzeros_params_.end()) {
                // 使用量化矩阵乘法
                cuda_OP::matmul_quantized_gemv(hidden_states, qweight_params_.at(v_qweight_key),
                                               scales_params_.at(v_scales_key), qzeros_params_.at(v_qzeros_key),
                                               group_size_, &v_buf, nullptr, v_bias);
            } else {
                throw std::runtime_error("Missing quantized weights for V projection");
            }
        } else {
            // 非量化版本
            auto &q_weight = params_.at("wq" + std::to_string(i));
            auto &k_weight = params_.at("wk" + std::to_string(i));
            auto &v_weight = params_.at("wv" + std::to_string(i));

            // 使用WeightTensor包装权重
            const op::WeightTensor<T> q_weight_tensor(&q_weight);
            const op::WeightTensor<T> k_weight_tensor(&k_weight);
            const op::WeightTensor<T> v_weight_tensor(&v_weight);

            // 使用operators_->matmul
            operators_->matmul(&q_buf, &hidden_states, q_weight_tensor, q_bias);
            operators_->matmul(&k_buf, &hidden_states, k_weight_tensor, k_bias);
            operators_->matmul(&v_buf, &hidden_states, v_weight_tensor, v_bias);
        }

        // Qwen3特有: Q/K归一化
        auto &q_norm_weight = params_.at("q_norm" + std::to_string(i));
        auto &k_norm_weight = params_.at("k_norm" + std::to_string(i));

        // 将q_buf和k_buf重塑为3D张量视图
        Tensor<T> q_buf_view = q_buf.view({seq_len, n_heads_, head_dim_});
        Tensor<T> k_buf_view = k_buf.view({seq_len, n_kv_heads_, head_dim_});

        // 对整个q_buf_view和k_buf_view执行RMS归一化
        // RMS归一化实现会自动处理最后一维（head_dim_）
        operators_->rms_norm(&q_buf_view, &q_buf_view, &q_norm_weight, rms_norm_eps_);
        operators_->rms_norm(&k_buf_view, &k_buf_view, &k_norm_weight, rms_norm_eps_);

        // 应用旋转位置编码 (RoPE)，计算注意力，并更新KV缓存
        Tensor<T> attn_output({seq_len, hidden_size_}, Device::CUDA);

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
            cudaMemcpy(k_cache_slice.data_ptr(), k_buf_view.data_ptr() + j * head_size, head_size * sizeof(T),
                       cudaMemcpyDeviceToDevice);
            cudaMemcpy(v_cache_slice.data_ptr(), v_buf_view.data_ptr() + j * head_size, head_size * sizeof(T),
                       cudaMemcpyDeviceToDevice);
        }

        // 从缓存获取完整的K和V序列
        auto [k_cache_tensor, v_cache_tensor] = kv_cache->get_contiguous_tensor(i);
        size_t total_seq_len = offset + seq_len;

        Tensor<T> k_cache_view = k_cache_tensor.view({total_seq_len, n_kv_heads_, head_dim_});
        Tensor<T> v_cache_view = v_cache_tensor.view({total_seq_len, n_kv_heads_, head_dim_});

        // 使用动态Flash-Attention包装函数计算自注意力
        Tensor<T> att_out_view = attn_output.view({n_heads_, head_dim_});
        cuda_OP::dynamic_flash_attention_wrapper(q_buf_view, k_cache_view, v_cache_view, att_out_view, n_kv_heads_);

        // 注意力输出投影
        Tensor<T> attn_proj({seq_len, hidden_size_}, Device::CUDA);
        if (quant_type_ == 1) {
            // 量化版本
            std::string o_weight_key = "wo" + std::to_string(i);
            std::string o_qweight_key = o_weight_key + ".qweight";
            std::string o_scales_key = o_weight_key + ".scales";
            std::string o_qzeros_key = o_weight_key + ".qzeros";

            // 创建量化权重
            if (qweight_params_.find(o_qweight_key) != qweight_params_.end() &&
                scales_params_.find(o_scales_key) != scales_params_.end() &&
                qzeros_params_.find(o_qzeros_key) != qzeros_params_.end()) {
                // 构建WeightTensor - 使用指针传递
                const Tensor<int32_t> *qweight = &qweight_params_.at(o_qweight_key);
                const Tensor<T> *scales = &scales_params_.at(o_scales_key);
                const Tensor<int32_t> *qzeros = &qzeros_params_.at(o_qzeros_key);
                op::WeightTensor<T> o_weight(qweight, scales, qzeros, group_size_);
                operators_->matmul(&attn_proj, &attn_output, o_weight, o_bias);
            } else {
                throw std::runtime_error("Missing quantized weights for O projection");
            }
        } else {
            // 非量化版本
            auto &o_weight = params_.at("wo" + std::to_string(i));
            // 使用WeightTensor包装权重，执行matmul
            const op::WeightTensor<T> weight_tensor(&o_weight);
            attn_output = attn_output.view({seq_len, n_heads_ * head_dim_});
            operators_->matmul(&attn_proj, &attn_output, weight_tensor, o_bias);
        }

        // 第一个残差连接
        operators_->add(&residual, &residual, &attn_proj);

        // FFN 输入层归一化 (RMSNorm)
        auto &ffn_norm_weight = params_.at("rms_ffn_w" + std::to_string(i));
        operators_->rms_norm(&hidden_states, &residual, &ffn_norm_weight, rms_norm_eps_);

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

        if (quant_type_ == 1) {
            // 量化版本
            std::string gate_weight_key = "w_gate" + std::to_string(i);
            std::string up_weight_key = "w_up" + std::to_string(i);

            std::string gate_qweight_key = gate_weight_key + ".qweight";
            std::string gate_scales_key = gate_weight_key + ".scales";
            std::string gate_qzeros_key = gate_weight_key + ".qzeros";

            std::string up_qweight_key = up_weight_key + ".qweight";
            std::string up_scales_key = up_weight_key + ".scales";
            std::string up_qzeros_key = up_weight_key + ".qzeros";

            // 检查量化权重是否存在
            if (qweight_params_.find(gate_qweight_key) != qweight_params_.end() &&
                scales_params_.find(gate_scales_key) != scales_params_.end() &&
                qzeros_params_.find(gate_qzeros_key) != qzeros_params_.end()) {
                // 构建量化权重张量 - 使用指针传递
                const Tensor<int32_t> *gate_qweight = &qweight_params_.at(gate_qweight_key);
                const Tensor<T> *gate_scales = &scales_params_.at(gate_scales_key);
                const Tensor<int32_t> *gate_qzeros = &qzeros_params_.at(gate_qzeros_key);
                op::WeightTensor<T> gate_weight(gate_qweight, gate_scales, gate_qzeros, group_size_);
                operators_->matmul(&gate_buf, &hidden_states, gate_weight, gate_bias);
            } else {
                throw std::runtime_error("Missing quantized weights for Gate projection");
            }

            if (qweight_params_.find(up_qweight_key) != qweight_params_.end() &&
                scales_params_.find(up_scales_key) != scales_params_.end() &&
                qzeros_params_.find(up_qzeros_key) != qzeros_params_.end()) {
                // 构建量化权重张量 - 使用指针传递
                const Tensor<int32_t> *up_qweight = &qweight_params_.at(up_qweight_key);
                const Tensor<T> *up_scales = &scales_params_.at(up_scales_key);
                const Tensor<int32_t> *up_qzeros = &qzeros_params_.at(up_qzeros_key);
                op::WeightTensor<T> up_weight(up_qweight, up_scales, up_qzeros, group_size_);
                operators_->matmul(&up_buf, &hidden_states, up_weight, up_bias);
            } else {
                throw std::runtime_error("Missing quantized weights for Up projection");
            }
        } else {
            // 非量化版本
            auto &gate_weight = params_.at("w_gate" + std::to_string(i));
            auto &up_weight = params_.at("w_up" + std::to_string(i));

            // 使用WeightTensor包装权重
            const op::WeightTensor<T> gate_weight_tensor(&gate_weight);
            const op::WeightTensor<T> up_weight_tensor(&up_weight);

            // 使用operators_->matmul
            operators_->matmul(&gate_buf, &hidden_states, gate_weight_tensor, gate_bias);
            operators_->matmul(&up_buf, &hidden_states, up_weight_tensor, up_bias);
        }

        // 应用SiLU激活函数到gate_buf并与up_buf相乘
        operators_->silu(&gate_buf, &gate_buf);               // SiLU激活
        operators_->multiply(&gate_buf, &gate_buf, &up_buf);  // 逐元素相乘

        // Down投影
        Tensor<T> ffn_out({seq_len, hidden_size_}, Device::CUDA);

        if (quant_type_ == 1) {
            // 量化版本
            std::string down_weight_key = "w_down" + std::to_string(i);
            std::string down_qweight_key = down_weight_key + ".qweight";
            std::string down_scales_key = down_weight_key + ".scales";
            std::string down_qzeros_key = down_weight_key + ".qzeros";

            if (qweight_params_.find(down_qweight_key) != qweight_params_.end() &&
                scales_params_.find(down_scales_key) != scales_params_.end() &&
                qzeros_params_.find(down_qzeros_key) != qzeros_params_.end()) {
                // 构建量化权重张量 - 使用指针传递
                const Tensor<int32_t> *down_qweight = &qweight_params_.at(down_qweight_key);
                const Tensor<T> *down_scales = &scales_params_.at(down_scales_key);
                const Tensor<int32_t> *down_qzeros = &qzeros_params_.at(down_qzeros_key);
                op::WeightTensor<T> down_weight(down_qweight, down_scales, down_qzeros, group_size_);
                operators_->matmul(&ffn_out, &gate_buf, down_weight, down_bias);
            } else {
                throw std::runtime_error("Missing quantized weights for Down projection");
            }
        } else {
            // 非量化版本
            auto &down_weight = params_.at("w_down" + std::to_string(i));

            // 使用WeightTensor包装权重
            const op::WeightTensor<T> down_weight_tensor(&down_weight);

            // 使用operators_->matmul
            operators_->matmul(&ffn_out, &gate_buf, down_weight_tensor, down_bias);
        }

        // 残差连接
        operators_->add(&residual, &residual, &ffn_out);
    }

    // 最终的LayerNorm (RMSNorm)
    auto &norm_weight = params_.at("rms_out_w");
    Tensor<T> final_h({seq_len, hidden_size_}, Device::CUDA);
    operators_->rms_norm(&final_h, &residual, &norm_weight, rms_norm_eps_);

    // LM head投影到词汇表大小
    auto &lm_head_weight = params_.at("lm_head");
    const Tensor<T> *lm_head_bias = nullptr;
    // 使用WeightTensor包装lm_head权重
    const op::WeightTensor<T> lm_head_tensor(&lm_head_weight);

    Tensor<T> logits({seq_len, vocab_size_}, Device::CUDA);
    // 使用operators_->matmul
    operators_->matmul(&logits, &final_h, lm_head_tensor, lm_head_bias);

    // 返回logits
    return logits;
}

// 显式实例化模板函数
template uint32_t *Qwen3Model<__nv_bfloat16>::forward(const Tensor<uint32_t> *input, ThreadPool &thread_pool,
                                                      KVCacheBase *kv_cache, size_t top_k, float temperature,
                                                      float top_p, curandState *d_states);

template Tensor<__nv_bfloat16> Qwen3Model<__nv_bfloat16>::forward_cuda(const Tensor<uint32_t> *input,
                                                                       KVCache<__nv_bfloat16> *kv_cache);
