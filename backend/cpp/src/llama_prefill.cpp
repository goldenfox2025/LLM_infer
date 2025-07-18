#include <cuda_runtime.h>

#include <algorithm>  // std::min
#include <cmath>
#include <cstring>
#include <future>
#include <iostream>
#include <string>
#include <vector>

#include "avx_operators.hpp"
#include "cudaOP.cuh"
#include "inference.hpp"
#include "llama.hpp"
#include "operators.hpp"
#include "thread_pool.hpp"

template <typename T>
void debugPrintTensor(const Tensor<T>& tensor, const std::string& tensor_name, size_t num_to_print = 10) {
    std::cout << "[Debug] " << tensor_name << ":\n";

    // 1) 打印 shape
    std::cout << "  shape: [";
    for (auto s : tensor.sizes()) {
        std::cout << s << " ";
    }
    std::cout << "]\n";

    // 2) 打印 strides
    std::cout << "  strides: [";
    for (auto st : tensor.strides()) {
        std::cout << st << " ";
    }
    std::cout << "]\n";

    // 3) 打印 device
    std::cout << "  device: ";
    if (tensor.device() == Device::CPU) {
        std::cout << "CPU";
    } else if (tensor.device() == Device::CUDA) {
        std::cout << "CUDA";
    } else {
        std::cout << "UNKNOWN";
    }
    std::cout << "\n";

    // 4) 打印
    size_t offset = 1536;
    size_t total_elements = tensor.numel();
    size_t n_print = (total_elements > offset) ? std::min(num_to_print, total_elements - offset) : 0;

    std::cout << "  elements from offset " << offset << " (" << n_print << " element(s)): ";
    if (tensor.device() == Device::CPU) {
        const T* ptr = tensor.data_ptr();
        for (size_t i = 0; i < n_print; i++) {
            std::cout << ptr[offset + i] << " ";
        }
        std::cout << "\n";
    } else {
        // 从 GPU 拷贝到 CPU，再打印
        std::vector<T> host_buffer(n_print);
        cudaError_t err =
            cudaMemcpy(host_buffer.data(), tensor.data_ptr() + offset, n_print * sizeof(T), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cout << "  [Error] cudaMemcpy failed\n";
            return;
        }
        for (size_t i = 0; i < n_print; i++) {
            std::cout << host_buffer[i] << " ";
        }
        std::cout << "\n";
    }
}

// =========== 2. CPU版本 prefill_cpu
// ===========
Tensor<float> LlamaModel::prefill_cpu(const Tensor<uint32_t>* input, KVCache<float>* typed_kv_cache,
                                      ThreadPool& thread_pool) {
    // 首先转换为正确的KVCache类型

    const size_t seq_len = input->numel();
    size_t offset = 0;
    if (typed_kv_cache) {
        offset = typed_kv_cache->size() - seq_len;
    }

    std::vector<float> residual_data(seq_len * d_);
    Tensor<float> residual(std::move(residual_data), {seq_len, d_});

    // (1) gather
    OP::gather(&residual, input, &params_.at("embedding_table"));
    // debugPrintTensor(residual, "CPU residual after gather");

    std::vector<float> hidden_states_data(seq_len * d_);
    Tensor<float> hidden_states(std::move(hidden_states_data), {seq_len, d_});

    const size_t n_groups = n_q_h_ / n_kv_h_;

    // 遍历每一层
    for (size_t layer = 0; layer < n_layers_; layer++) {
        // (2) RMSNorm（注意力前）
        OP::rms_norm(&hidden_states, &residual, &params_.at("rms_att_w" + std::to_string(layer)), eps_);
        // debugPrintTensor(hidden_states, "CPU hidden_states after RMSNorm(att)");

        // 并行计算 Q、K 和 V
        Tensor<float> wq = params_.at("wq" + std::to_string(layer));
        Tensor<float> wk = params_.at("wk" + std::to_string(layer));
        Tensor<float> wv = params_.at("wv" + std::to_string(layer));

        Tensor<float> q_buf, k_buf, v_buf;

        // 提交任务到线程池
        thread_pool.enqueueTask(std::make_shared<OpTask>([&]() { q_buf = avx_OP::matmul(hidden_states, wq); }));
        thread_pool.enqueueTask(std::make_shared<OpTask>([&]() { k_buf = avx_OP::matmul(hidden_states, wk); }));
        thread_pool.enqueueTask(std::make_shared<OpTask>([&]() { v_buf = avx_OP::matmul(hidden_states, wv); }));

        thread_pool.waitForAllTasks();  // 等待所有任务完成

        // 打印 Q/K/V matmul结果
        // debugPrintTensor(q_buf, "CPU q_buf after matmul");
        // debugPrintTensor(k_buf, "CPU k_buf after matmul");
        // debugPrintTensor(v_buf, "CPU v_buf after matmul");
        // rope
        Tensor<float> q_buf_view = q_buf.view({seq_len, n_q_h_, dqkv_});
        Tensor<float> k_buf_view = k_buf.view({seq_len, n_kv_h_, dqkv_});
        OP::rope(&q_buf_view, offset, rope_theta_);
        OP::rope(&k_buf_view, offset, rope_theta_);
        // debugPrintTensor(q_buf_view, "CPU q_buf_view after rope");
        // debugPrintTensor(k_buf_view, "CPU k_buf_view after rope");

        if (typed_kv_cache) {
            Tensor<float> k_buf_contiguous = k_buf;
            Tensor<float> v_buf_contiguous = v_buf;
            size_t row_size = n_kv_h_ * dqkv_;
            for (size_t i = 0; i < seq_len; i++) {
                const float* k_ptr = k_buf_contiguous.data_ptr() + i * row_size;
                const float* v_ptr = v_buf_contiguous.data_ptr() + i * row_size;
                std::vector<float> k_i(k_ptr, k_ptr + row_size);
                std::vector<float> v_i(v_ptr, v_ptr + row_size);
                typed_kv_cache->k_cache(layer, offset + i) = Tensor<float>(std::move(k_i), {1, row_size});
                typed_kv_cache->v_cache(layer, offset + i) = Tensor<float>(std::move(v_i), {1, row_size});
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
                Tensor<float>& cached_k = typed_kv_cache->k_cache(layer, pos);
                Tensor<float>& cached_v = typed_kv_cache->v_cache(layer, pos);
                memcpy(&total_K_data[pos * row_size], cached_k.data_ptr(), row_size * sizeof(float));
                memcpy(&total_V_data[pos * row_size], cached_v.data_ptr(), row_size * sizeof(float));
            }

            memcpy(&total_K_data[cached_len * row_size], k_buf_view.data_ptr(), seq_len * row_size * sizeof(float));

            Tensor<float> v_buf_view = v_buf.view({seq_len, n_kv_h_, dqkv_});
            memcpy(&total_V_data[cached_len * row_size], v_buf_view.data_ptr(), seq_len * row_size * sizeof(float));

            total_K = Tensor<float>(std::move(total_K_data), {total_seq_len, n_kv_h_, dqkv_});
            total_V = Tensor<float>(std::move(total_V_data), {total_seq_len, n_kv_h_, dqkv_});
        } else {
            total_K = k_buf.view({seq_len, n_kv_h_, dqkv_});
            total_V = v_buf.view({seq_len, n_kv_h_, dqkv_});
        }

        // 计算注意力分数
        std::vector<float> att_scores(seq_len * n_q_h_ * total_seq_len, 0.0f);
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t qh = 0; qh < n_q_h_; qh++) {
                size_t kv_head = qh / n_groups;
                const float* q_ptr = Q_3d.data_ptr() + (i * n_q_h_ + qh) * dqkv_;

                // 调试：仅在第一个查询（s==0 且 qh==0）打印前几个特征值
                for (size_t j = 0; j < total_seq_len; j++) {
                    const float* k_ptr = total_K.data_ptr() + (j * n_kv_h_ + kv_head) * dqkv_;
                    float dot = 0.0f;

                    // 打印每个点积计算的中间值
                    for (size_t d = 0; d < dqkv_; d++) {
                        dot += q_ptr[d] * k_ptr[d];

                        // 只打印前两个特征的 q_ptr 和 k_ptr
                        if (i == 1 && qh == 1 && j == 1 && d < 5) {
                            std::cout << "[CPU Debug] i=" << i << ", qh=" << qh << ", j=" << j << ", d=" << d
                                      << ": q_ptr[d]=" << q_ptr[d] << ", k_ptr[d]=" << k_ptr[d] << std::endl;
                        }
                    }

                    dot /= std::sqrt(float(dqkv_));

                    // 输出部分结果以检查
                    if (i == 1 && qh == 1 && j == 1) {
                        std::cout << "[CPU Debug] i=" << i << ", qh=" << qh << ", j=" << j << ": dot=" << dot
                                  << std::endl;
                    }

                    // 计算att_scores
                    att_scores[i * (n_q_h_ * total_seq_len) + qh * total_seq_len + j] = dot;

                    // 再次输出存储到att_scores的值
                    if (i == 1 && qh == 1 && j == 1) {
                        std::cout << "[CPU Debug] i=" << i << ", qh=" << qh << ", j=" << j << ": att_scores["
                                  << i * (n_q_h_ * total_seq_len) + qh * total_seq_len + j
                                  << "] = " << att_scores[i * (n_q_h_ * total_seq_len) + qh * total_seq_len + j]
                                  << std::endl;
                    }
                }
            }
        }

        Tensor<float> att_scores_tensor(std::move(att_scores), {seq_len, n_q_h_, total_seq_len});
        // debugPrintTensor(att_scores_tensor, "compute_attention_scores_prefill");
        OP::softmax(&att_scores_tensor, &att_scores_tensor, /*dim=*/2, true, n_q_h_, offset);

        // debugPrintTensor(att_scores_tensor, "softmax");
        //  计算注意力输出
        std::vector<float> att_out(seq_len * n_q_h_ * dqkv_, 0.0f);
        float* att_ptr = att_scores_tensor.data_ptr();
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t qh = 0; qh < n_q_h_; qh++) {
                size_t kv_head = qh / n_groups;
                for (size_t d = 0; d < dqkv_; d++) {
                    float sum_val = 0.f;
                    for (size_t j = 0; j < total_seq_len; j++) {
                        float att_w = att_ptr[i * (n_q_h_ * total_seq_len) + qh * total_seq_len + j];
                        const float* v_ptr = total_V.data_ptr() + (j * n_kv_h_ + kv_head) * dqkv_;
                        sum_val += att_w * v_ptr[d];
                    }
                    att_out[(i * n_q_h_ + qh) * dqkv_ + d] = sum_val;
                }
            }
        }

        Tensor<float> att_heads(std::move(att_out), {seq_len, n_q_h_ * dqkv_});
        // debugPrintTensor(att_heads, "compute_att_output_prefill");
        Tensor<float> wo = params_.at("wo" + std::to_string(layer));
        Tensor<float> att_proj = avx_OP::matmul(att_heads, wo);

        residual = residual + att_proj;

        // (3) FFN 前 RMSNorm
        OP::rms_norm(&hidden_states, &residual, &params_.at("rms_ffn_w" + std::to_string(layer)), eps_);
        // debugPrintTensor(hidden_states, "CPU hidden_states after RMSNorm(ffn)");

        // FFN
        Tensor<float> w_gate = params_.at("w_gate" + std::to_string(layer));
        Tensor<float> w_up = params_.at("w_up" + std::to_string(layer));
        Tensor<float> w_down = params_.at("w_down" + std::to_string(layer));

        Tensor<float> gate_buf = avx_OP::matmul(hidden_states, w_gate);
        Tensor<float> up_buf = avx_OP::matmul(hidden_states, w_up);
        OP::silu(&gate_buf, &gate_buf);
        OP::multiply(&gate_buf, &gate_buf, &up_buf);

        Tensor<float> ffn_out = avx_OP::matmul(gate_buf, w_down);
        residual = residual + ffn_out;
        // debugPrintTensor(residual, "CPU residual after FFN");
    }

    // 最终 RMSNorm
    std::vector<float> final_h_data(seq_len * d_);
    Tensor<float> final_h(std::move(final_h_data), {seq_len, d_});
    OP::rms_norm(&final_h, &residual, &params_.at("rms_out_w"), eps_);
    // debugPrintTensor(final_h, "CPU final_h after final RMSNorm");

    Tensor<float> lm_head = params_.at("lm_head");
    Tensor<float> logits = avx_OP::matmul(final_h, lm_head);
    // debugPrintTensor(logits, "CPU final logits");

    return logits;
}

// =========== 3. CUDA版本 prefill_cuda
// ===========
Tensor<float> LlamaModel::prefill_cuda(const Tensor<uint32_t>* input, KVCache<float>* typed_kv_cache) {
    // 1. 输入参数验证和初始化
    if (!input || !input->data_ptr()) {
        throw std::runtime_error("Input tensor is null or has null data pointer");
    }
    if (input->device() != Device::CUDA) {
        throw std::runtime_error("Input tensor must be on CUDA device");
    }
    const size_t seq_len = input->numel();
    size_t offset = 0;
    if (typed_kv_cache) {
        if (typed_kv_cache->device() != Device::CUDA) {
            throw std::runtime_error("KVCache must be on CUDA device");
        }
        offset = typed_kv_cache->size() - seq_len;
    }

    // 2. 分配和初始化CUDA张量
    Tensor<float> residual({seq_len, d_}, Device::CUDA);
    Tensor<float> hidden_states({seq_len, d_}, Device::CUDA);

    // 3. 执行gather操作
    cuda_OP::gather(&residual, input, &params_.at("embedding_table"));

    const size_t n_groups = n_q_h_ / n_kv_h_;

    // 4. 遍历每一层
    for (size_t layer = 0; layer < n_layers_; layer++) {
        // (2) RMSNorm（注意力前）
        cuda_OP::rms_norm(&hidden_states, &residual, &params_.at("rms_att_w" + std::to_string(layer)), eps_);

        // 并行计算 Q、K、V
        Tensor<float>& wq = params_.at("wq" + std::to_string(layer));
        Tensor<float>& wk = params_.at("wk" + std::to_string(layer));
        Tensor<float>& wv = params_.at("wv" + std::to_string(layer));

        cudaStream_t streams[3];
        for (int i = 0; i < 3; i++) {
            cudaError_t err = cudaStreamCreate(&streams[i]);
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to create CUDA stream");
            }
        }

        // 预先分配输出张量，避免内部重复分配
        // Q: [seq_len, n_q_h_ * dqkv_]
        Tensor<float> q_buf({seq_len, n_q_h_ * dqkv_}, Device::CUDA);
        cuda_OP::matmul(hidden_states, wq, &q_buf, streams[0]);

        // K,V: [seq_len, n_kv_h_ * dqkv_]
        Tensor<float> k_buf({seq_len, n_kv_h_ * dqkv_}, Device::CUDA);
        cuda_OP::matmul(hidden_states, wk, &k_buf, streams[1]);

        Tensor<float> v_buf({seq_len, n_kv_h_ * dqkv_}, Device::CUDA);
        cuda_OP::matmul(hidden_states, wv, &v_buf, streams[2]);

        for (int i = 0; i < 3; i++) {
            cudaStreamSynchronize(streams[i]);
            cudaStreamDestroy(streams[i]);
        }

        // 4.4 调整形状并应用RoPE
        Tensor<float> q_buf_view = q_buf.view({seq_len, n_q_h_, dqkv_});
        Tensor<float> k_buf_view = k_buf.view({seq_len, n_kv_h_, dqkv_});
        cuda_OP::rope(&q_buf_view, offset, rope_theta_);
        cuda_OP::rope(&k_buf_view, offset, rope_theta_);

        // 4.5 保存KV Cache
        if (typed_kv_cache) {
            size_t row_size = n_kv_h_ * dqkv_;
            for (size_t i = 0; i < seq_len; i++) {
                Tensor<float> k_i({1, row_size}, Device::CUDA);
                Tensor<float> v_i({1, row_size}, Device::CUDA);
                cudaMemcpy(k_i.data_ptr(), k_buf.data_ptr() + i * row_size, row_size * sizeof(float),
                           cudaMemcpyDeviceToDevice);
                cudaMemcpy(v_i.data_ptr(), v_buf.data_ptr() + i * row_size, row_size * sizeof(float),
                           cudaMemcpyDeviceToDevice);
                typed_kv_cache->k_cache(layer, offset + i) = std::move(k_i);
                typed_kv_cache->v_cache(layer, offset + i) = std::move(v_i);
            }
        }

        // 4.6 准备注意力计算
        Tensor<float> Q_3d = q_buf_view;
        Tensor<float> total_K, total_V;
        size_t total_seq_len = seq_len;
        if (offset != 0) {
            size_t cached_len = offset;
            total_seq_len = cached_len + seq_len;
            size_t row_size = n_kv_h_ * dqkv_;
            total_K = Tensor<float>({total_seq_len, n_kv_h_, dqkv_}, Device::CUDA);
            total_V = Tensor<float>({total_seq_len, n_kv_h_, dqkv_}, Device::CUDA);
            // 拼接缓存 K、V
            for (size_t pos = 0; pos < cached_len; pos++) {
                Tensor<float>& cached_k = typed_kv_cache->k_cache(layer, pos);
                Tensor<float>& cached_v = typed_kv_cache->v_cache(layer, pos);
                cudaMemcpy(total_K.data_ptr() + pos * row_size, cached_k.data_ptr(), row_size * sizeof(float),
                           cudaMemcpyDeviceToDevice);
                cudaMemcpy(total_V.data_ptr() + pos * row_size, cached_v.data_ptr(), row_size * sizeof(float),
                           cudaMemcpyDeviceToDevice);
            }
            // 复制当前 K、V
            cudaMemcpy(total_K.data_ptr() + offset * row_size, k_buf.data_ptr(), seq_len * row_size * sizeof(float),
                       cudaMemcpyDeviceToDevice);
            Tensor<float> v_buf_view = v_buf.view({seq_len, n_kv_h_, dqkv_});
            cudaMemcpy(total_V.data_ptr() + offset * row_size, v_buf_view.data_ptr(),
                       seq_len * row_size * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            total_K = k_buf.view({seq_len, n_kv_h_, dqkv_});
            total_V = v_buf.view({seq_len, n_kv_h_, dqkv_});
        }

        // 4.8 计算注意力分数 -> softmax -> 注意力输出
        Tensor<float> att_scores({seq_len, n_q_h_, total_seq_len}, Device::CUDA);
        cuda_OP::compute_attention_scores_prefill(Q_3d, total_K, att_scores, dqkv_);
        cuda_OP::softmax(&att_scores, &att_scores, /*dim=*/2, true, offset);
        Tensor<float> att_heads({seq_len, n_q_h_, dqkv_}, Device::CUDA);
        cuda_OP::compute_att_output_prefill(att_scores, total_V, att_heads, n_q_h_, dqkv_, total_seq_len, n_kv_h_);

        // 将注意力输出投影回原始维度
        Tensor<float> att_heads_reshaped = att_heads.view({seq_len, n_q_h_ * dqkv_});
        Tensor<float>& wo = params_.at("wo" + std::to_string(layer));
        Tensor<float> att_proj({seq_len, d_}, Device::CUDA);
        cuda_OP::matmul(att_heads_reshaped, wo, &att_proj);
        residual = residual + att_proj;

        // (3) FFN 前的 RMSNorm
        cuda_OP::rms_norm(&hidden_states, &residual, &params_.at("rms_ffn_w" + std::to_string(layer)), eps_);

        // FFN部分
        Tensor<float>& w_gate = params_.at("w_gate" + std::to_string(layer));
        Tensor<float>& w_up = params_.at("w_up" + std::to_string(layer));
        Tensor<float>& w_down = params_.at("w_down" + std::to_string(layer));

        Tensor<float> gate_buf({seq_len, w_gate.sizes()[1]}, Device::CUDA);
        cuda_OP::matmul(hidden_states, w_gate, &gate_buf);
        Tensor<float> up_buf({seq_len, w_up.sizes()[1]}, Device::CUDA);
        cuda_OP::matmul(hidden_states, w_up, &up_buf);

        cuda_OP::silu(&gate_buf, &gate_buf);
        cuda_OP::multiply(&gate_buf, &gate_buf, &up_buf);
        Tensor<float> ffn_out({seq_len, w_down.sizes()[1]}, Device::CUDA);
        cuda_OP::matmul(gate_buf, w_down, &ffn_out);
        residual = residual + ffn_out;
    }

    // 5. 最终 RMSNorm + lm_head
    Tensor<float> final_h({seq_len, d_}, Device::CUDA);
    cuda_OP::rms_norm(&final_h, &residual, &params_.at("rms_out_w"), eps_);
    Tensor<float>& lm_head = params_.at("lm_head");
    size_t vocab_size = lm_head.sizes()[1];
    Tensor<float> logits({seq_len, vocab_size}, Device::CUDA);
    cuda_OP::matmul(final_h, lm_head, &logits);

    return logits;
}

uint32_t* LlamaModel::prefill(const Tensor<uint32_t>* input, ThreadPool& thread_pool, KVCacheBase* kv_cache,
                              size_t top_k, float temperature, float top_p, curandState* d_states) {
    if (!input) {
        throw std::invalid_argument("Input tensor cannot be null");
    }
    auto typed_kv_cache = dynamic_cast<KVCache<float>*>(kv_cache);
    if (!typed_kv_cache) {
        throw std::runtime_error("Invalid KVCache type for LlamaModel::prefill_cuda");
    }
    try {
        if (device_ == Device::CUDA) {
            Tensor<float> prefill_logits = prefill_cuda(input, typed_kv_cache);

            size_t seq_len = input->numel();

            // 从 prefill 的最后一行 logits 中提取最后一个 token 的 logits
            const size_t vocab_size = prefill_logits.sizes()[1];

            // 创建一个新的 CUDA 张量，只包含最后一行的 logits
            Tensor<float> last_logits({1, vocab_size}, Device::CUDA);

            // 复制最后一行的 logits 到新的张量
            cudaMemcpy(last_logits.data_ptr(), prefill_logits.data_ptr() + (seq_len - 1) * vocab_size,
                       vocab_size * sizeof(float), cudaMemcpyDeviceToDevice);

            // 使用 cuda_OP::sample 在 GPU 上进行采样，返回 GPU 内存上的指针
            return cuda_OP::sample(std::move(last_logits), temperature, top_p, top_k, d_states);

        } else {
            Tensor<float> prefill_logits = prefill_cpu(input, typed_kv_cache, thread_pool);

            size_t seq_len = input->numel();

            // 从 prefill 的最后一行 logits 中采样第一个生成 token
            const size_t vocab_size = prefill_logits.sizes()[1];
            std::vector<float> last_row_data(vocab_size, 0.f);
            const float* prefill_ptr = prefill_logits.data_ptr() + (seq_len - 1) * vocab_size;
            std::copy(prefill_ptr, prefill_ptr + vocab_size, last_row_data.begin());
            Tensor<float> last_logits(std::move(last_row_data), {1, vocab_size},
                                      Device::CPU);  // 强制使用CPU设备
            return OP::sample(&last_logits, temperature, top_p, top_k);
        }
    } catch (const std::exception& e) {
        std::cerr << "[LlamaModel::prefill] Exception: " << e.what() << std::endl;
        throw;
    }
}
