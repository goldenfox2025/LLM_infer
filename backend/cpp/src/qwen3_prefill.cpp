#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "qwen3.hpp"

// -------------------------------
// prefill: 预填充接口
// -------------------------------
template <typename T>
uint32_t *Qwen3Model<T>::prefill(const Tensor<uint32_t> *input, ThreadPool &thread_pool, KVCacheBase *kv_cache,
                                 size_t top_k, float temperature, float top_p, curandState *d_states) {
    KVCache<T> *typed_cache = dynamic_cast<KVCache<T> *>(kv_cache);

    return operators_->sample(prefill_cuda(input, typed_cache), temperature, top_p, top_k, d_states);
}

// -------------------------------
// prefill_cuda: CUDA 版本的预填充实现
// -------------------------------
template <typename T>
Tensor<T> Qwen3Model<T>::prefill_cuda(const Tensor<uint32_t> *input, KVCache<T> *kv_cache) {
    // 确保输入在 CUDA 上
    if (input->device() != Device::CUDA) {
        throw std::runtime_error("Input tensor must be on CUDA device");
    }

    // 获取输入信息
    const size_t seq_len = input->sizes()[0];

    // 计算起始KV缓存位置
    size_t offset = 0;
    if (kv_cache) {
        if (kv_cache->device() != Device::CUDA) {
            throw std::runtime_error("KVCache must be on CUDA device");
        }

        offset = kv_cache->size() - seq_len;
    }

    // 创建residual和hidden_states张量，在prefill阶段自动使用prefill buffer
    Tensor<T> residual({seq_len, hidden_size_}, Device::CUDA);
    Tensor<T> hidden_states({seq_len, hidden_size_}, Device::CUDA);

    // Token嵌入 (从embedding_table中获取token嵌入)
    operators_->gather(&residual, input, &params_.at("token_embeddings.weight"));

    // 初始化计算流同步
    cudaStreamSynchronize(compute_streams_[3]);
    cudaStreamSynchronize(compute_streams_[4]);

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

        // 获取权重（自动处理量化与非量化情况）
        auto q_weight = get_weight("wq" + std::to_string(i));
        auto k_weight = get_weight("wk" + std::to_string(i));
        auto v_weight = get_weight("wv" + std::to_string(i));

        // debugPrintTensor(hidden_states, "hidden_states");
        // debugPrintTensor(*q_weight.tensor(), "q_weight");
        // 使用不同的计算流并行处理Q/K/V矩阵乘法
        operators_->matmul(&q_buf, &hidden_states, q_weight, q_bias, compute_streams_[0]);
        operators_->matmul(&k_buf, &hidden_states, k_weight, k_bias, compute_streams_[1]);
        operators_->matmul(&v_buf, &hidden_states, v_weight, v_bias, compute_streams_[2]);

        // 等待Q/K/V计算完成
        cudaStreamSynchronize(compute_streams_[0]);
        cudaStreamSynchronize(compute_streams_[1]);
        cudaStreamSynchronize(compute_streams_[2]);

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
        // 使用 n_heads_ * head_dim_ 而不是 hidden_size_ 来创建注意力输出张量
        // 这样可以同时支持 0.6B 和 1.7B 模型
        Tensor<T> attn_output({seq_len, n_heads_ * head_dim_}, Device::CUDA);

        // 应用RoPE（旋转位置编码）
        operators_->rope(&q_buf_view, offset, rope_theta_);
        operators_->rope(&k_buf_view, offset, rope_theta_);

        // 存储K, V到缓存
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

        // // ========== 对比两种注意力计算方法 ==========

        // // 创建两个独立的输出张量
        Tensor<T> attn_output_flash({seq_len, n_heads_ * head_dim_}, Device::CUDA);
        // Tensor<T> attn_output_separate({seq_len, n_heads_ * head_dim_}, Device::CUDA);

        // std::cout << "Layer " << i << ": 开始对比两种注意力计算方法..." << std::endl;

        // // 方法1: Flash Attention (标记为错误的)
        // {
        Tensor<T> att_out_view_flash = attn_output_flash.view({seq_len, n_heads_, head_dim_});
        operators_->flash_attention_prefill(q_buf_view, k_cache_view, v_cache_view, att_out_view_flash, n_heads_,
                                            n_kv_heads_, head_dim_, seq_len, total_seq_len, offset);
        // }

        // // 方法2: 分开的计算路径 (标记为正确的)
        // {
        //     Tensor<T> att_scores({seq_len, n_heads_, total_seq_len}, Device::CUDA);
        //     operators_->compute_attention_scores_prefill(q_buf_view, k_cache_view, att_scores, head_dim_);

        //     // Softmax处理注意力分数（prefill版本需要设置mask=true）
        //     operators_->softmax(&att_scores, &att_scores, /*dim=*/2, true, offset);

        //     // 计算注意力输出（prefill版本）
        //     Tensor<T> att_out_view_separate = attn_output_separate.view({seq_len, n_heads_, head_dim_});
        //     operators_->compute_attention_output_prefill(att_scores, v_cache_view, att_out_view_separate, n_heads_,
        //                                                  head_dim_, total_seq_len, n_kv_heads_);
        // }

        // // 同步计算流确保两种计算都完成
        // cudaDeviceSynchronize();

        // // 对比两个结果
        // {
        //     const size_t total_elements = seq_len * n_heads_ * head_dim_;
        //     const float tolerance = 1e-1f;  // 误差容忍度

        //     // 将数据复制到CPU进行比较
        //     std::vector<float> flash_data(total_elements);
        //     std::vector<float> separate_data(total_elements);

        //     // 复制数据到CPU并转换为float进行比较
        //     if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        //         // 为bfloat16数据创建CPU缓冲区
        //         std::vector<__nv_bfloat16> flash_bf16(total_elements);
        //         std::vector<__nv_bfloat16> separate_bf16(total_elements);

        //         // 复制数据到CPU
        //         cudaMemcpy(flash_bf16.data(), attn_output_flash.data_ptr(), total_elements * sizeof(__nv_bfloat16),
        //                    cudaMemcpyDeviceToHost);
        //         cudaMemcpy(separate_bf16.data(), attn_output_separate.data_ptr(),
        //                    total_elements * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

        //         // 转换为float
        //         for (size_t idx = 0; idx < total_elements; ++idx) {
        //             flash_data[idx] = static_cast<float>(flash_bf16[idx]);
        //             separate_data[idx] = static_cast<float>(separate_bf16[idx]);
        //         }
        //     } else if constexpr (std::is_same_v<T, float>) {
        //         // 直接复制float数据
        //         cudaMemcpy(flash_data.data(), attn_output_flash.data_ptr(), total_elements * sizeof(float),
        //                    cudaMemcpyDeviceToHost);
        //         cudaMemcpy(separate_data.data(), attn_output_separate.data_ptr(), total_elements * sizeof(float),
        //                    cudaMemcpyDeviceToHost);
        //     } else {
        //         // 对于其他数据类型，先复制然后转换
        //         std::vector<T> flash_raw(total_elements);
        //         std::vector<T> separate_raw(total_elements);

        //         cudaMemcpy(flash_raw.data(), attn_output_flash.data_ptr(), total_elements * sizeof(T),
        //                    cudaMemcpyDeviceToHost);
        //         cudaMemcpy(separate_raw.data(), attn_output_separate.data_ptr(), total_elements * sizeof(T),
        //                    cudaMemcpyDeviceToHost);

        //         for (size_t idx = 0; idx < total_elements; ++idx) {
        //             flash_data[idx] = static_cast<float>(flash_raw[idx]);
        //             separate_data[idx] = static_cast<float>(separate_raw[idx]);
        //         }
        //     }

        //     // 比较结果
        //     float max_diff = 0.01f;
        //     size_t max_diff_idx = 0;
        //     size_t first_error_idx = total_elements;  // 初始化为最大值，表示没有错误

        //     for (size_t idx = 0; idx < total_elements; ++idx) {
        //         float diff = std::abs(flash_data[idx] - separate_data[idx]);

        //         if (diff > max_diff) {
        //             max_diff = diff;
        //             max_diff_idx = idx;
        //         }

        //         if (diff > tolerance && first_error_idx == total_elements) {
        //             first_error_idx = idx;
        //         }
        //     }

        //     std::cout << "Layer " << i << " 结果对比:" << std::endl;
        //     std::cout << "  最大差距: " << std::scientific << std::setprecision(6) << max_diff
        //               << " (位置: " << max_diff_idx << ")" << std::endl;

        //     if (max_diff > tolerance) {
        //         std::cout << "  最大差距详情 - Flash: " << std::fixed << std::setprecision(6)
        //                   << flash_data[max_diff_idx] << ", Separate: " << separate_data[max_diff_idx] << std::endl;
        //     }

        //     if (first_error_idx < total_elements) {
        //         std::cout << "  ❌ 发现误差超出容忍度! (容忍度: " << std::scientific << tolerance << ")" <<
        //         std::endl; std::cout << "  第一个错误位置: " << first_error_idx << " (差距: " << std::scientific
        //                   << std::setprecision(6)
        //                   << std::abs(flash_data[first_error_idx] - separate_data[first_error_idx]) << ")" <<
        //                   std::endl;
        //         std::cout << "  第一个错误及其后5个值:" << std::endl;

        //         for (size_t j = 0; j < 6 && (first_error_idx + j) < total_elements; ++j) {
        //             size_t idx = first_error_idx + j;
        //             float diff = std::abs(flash_data[idx] - separate_data[idx]);
        //             std::cout << "    [" << std::setw(6) << idx << "] Flash: " << std::fixed << std::setprecision(8)
        //                       << flash_data[idx] << ", Separate: " << std::setprecision(8) << separate_data[idx]
        //                       << ", 差距: " << std::scientific << std::setprecision(6) << diff;
        //             if (diff > tolerance) {
        //                 std::cout << " ⚠️ 超出容忍度";
        //             }
        //             std::cout << std::endl;
        //         }

        //         // 额外统计信息
        //         size_t error_count = 0;
        //         for (size_t idx = 0; idx < total_elements; ++idx) {
        //             if (std::abs(flash_data[idx] - separate_data[idx]) > tolerance) {
        //                 error_count++;
        //             }
        //         }
        //         std::cout << "  📊 总计: " << error_count << "/" << total_elements << " 个元素超出容忍度 ("
        //                   << std::fixed << std::setprecision(2) << (100.0 * error_count / total_elements) << "%)"
        //                   << std::endl;
        //     } else {
        //         std::cout << "  ✅ 所有值都在容忍度范围内 (容忍度: " << std::scientific << tolerance << ")"
        //                   << std::endl;
        //     }
        //     std::cout << std::endl;
        // }

        // 使用正确的方法的结果继续后续计算
        attn_output = attn_output_flash;
        attn_output = attn_output.view({seq_len, n_heads_ * head_dim_});
        // 注意力输出投影
        Tensor<T> attn_proj({seq_len, hidden_size_}, Device::CUDA);

        // 获取权重（自动处理量化与非量化情况）
        auto o_weight = get_weight("wo" + std::to_string(i));

        // 执行矩阵乘法
        operators_->matmul(&attn_proj, &attn_output, o_weight, o_bias);

        // 残差连接
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
template uint32_t *Qwen3Model<__nv_bfloat16>::prefill(const Tensor<uint32_t> *input, ThreadPool &thread_pool,
                                                      KVCacheBase *kv_cache, size_t top_k, float temperature,
                                                      float top_p, curandState *d_states);

template Tensor<__nv_bfloat16> Qwen3Model<__nv_bfloat16>::prefill_cuda(const Tensor<uint32_t> *input,
                                                                       KVCache<__nv_bfloat16> *kv_cache);
