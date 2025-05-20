#include "../include/qwen3.hpp"

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
#include "tensor.hpp"

// -------------------------------
// Qwen3Model<T> 构造函数
// -------------------------------
template <typename T>
Qwen3Model<T>::Qwen3Model(const std::unordered_map<std::string, Tensor<T>> &params,
                          const std::unordered_map<std::string, int> &config)
    : params_(params) {
    // 从 config 中提取基本参数
    vocab_size_ = config.at("vocab_size");
    n_layers_ = config.at("n_layers");
    n_heads_ = config.at("n_heads");
    n_kv_heads_ = config.at("n_kv_heads");
    hidden_size_ = config.at("hidden_size");
    intermediate_size_ = config.at("intermediate_size");
    max_position_embeddings_ = config.at("max_position_embeddings");
    bos_token_id_ = static_cast<uint32_t>(config.at("bos_token_id"));
    eos_token_id_ = static_cast<uint32_t>(config.at("eos_token_id"));
    rms_norm_eps_ = static_cast<float>(config.at("rms_norm_eps"));
    rope_theta_ = static_cast<float>(config.at("rope_theta"));

    // 优先使用配置中的head_dim，如果没有则计算
    if (config.find("head_dim") != config.end()) {
        head_dim_ = config.at("head_dim");
        std::cout << "使用配置中的head_dim: " << head_dim_ << std::endl;
    }

    // 检查并修正hidden_size与head_dim*n_heads的不一致
    size_t expected_hidden_size = head_dim_ * n_heads_;

    device_ = Device::CUDA;

    // 初始化算子接口
    operators_ = std::make_unique<op::UnifiedOperators<T>>(device_);

    // 初始化CUDA流和事件
    for (int i = 0; i < kNumStreams; ++i) {
        cudaStreamCreate(&compute_streams_[i]);
    }

    for (int i = 0; i < 3; ++i) {
        // 使用 cudaEventDisableTiming
        // 可以获得微小的性能提升，因为我们只关心完成状态，不测量时间
        cudaEventCreateWithFlags(&fa_done_events_[i], cudaEventDisableTiming);
    }
}

// 带量化参数的构造函数
template <typename T>
Qwen3Model<T>::Qwen3Model(const std::unordered_map<std::string, Tensor<T>> &params,
                          const std::unordered_map<std::string, Tensor<int32_t>> &qweight_params,
                          const std::unordered_map<std::string, Tensor<T>> &scales_params,
                          const std::unordered_map<std::string, Tensor<int32_t>> &qzeros_params,
                          const std::unordered_map<std::string, int> &config)
    : params_(params), qweight_params_(qweight_params), scales_params_(scales_params), qzeros_params_(qzeros_params) {
    // 从 config 中提取基本参数
    vocab_size_ = config.at("vocab_size");
    n_layers_ = config.at("n_layers");
    n_heads_ = config.at("n_heads");
    n_kv_heads_ = config.at("n_kv_heads");
    hidden_size_ = config.at("hidden_size");
    intermediate_size_ = config.at("intermediate_size");
    max_position_embeddings_ = config.at("max_position_embeddings");
    bos_token_id_ = static_cast<uint32_t>(config.at("bos_token_id"));
    eos_token_id_ = static_cast<uint32_t>(config.at("eos_token_id"));
    rms_norm_eps_ = static_cast<float>(config.at("rms_norm_eps"));
    rope_theta_ = static_cast<float>(config.at("rope_theta"));
    // 优先使用配置中的head_dim，如果没有则计算
    if (config.find("head_dim") != config.end()) {
        head_dim_ = config.at("head_dim");
        std::cout << "使用配置中的head_dim: " << head_dim_ << std::endl;
    }

    device_ = Device::CUDA;  // 默认在CUDA上初始化
    quant_type_ = 1;
    // 设置量化类型和分组大小
    if (config.find("group_size") != config.end()) {
        group_size_ = config.at("group_size");
    }

    // 初始化算子接口
    operators_ = std::make_unique<op::UnifiedOperators<T>>(device_);

    // 初始化CUDA流和事件
    for (int i = 0; i < kNumStreams; ++i) {
        cudaStreamCreate(&compute_streams_[i]);
    }

    for (int i = 0; i < 3; ++i) {
        cudaEventCreateWithFlags(&fa_done_events_[i], cudaEventDisableTiming);
    }
}

template <typename T>
Qwen3Model<T>::~Qwen3Model() {
    for (cudaStream_t stream : compute_streams_) {
        if (stream) {
            // 最好在销毁流之前同步它，确保所有工作完成
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
        }
    }

    for (int i = 0; i < 3; ++i) {
        if (fa_done_events_[i]) {
            cudaEventDestroy(fa_done_events_[i]);
        }
    }
}

// -------------------------------
// 参数验证：检查全局与层级关键参数是否存在
// -------------------------------
template <typename T>
bool Qwen3Model<T>::verify_params() const {
    // 检查基本权重是否存在
    std::vector<std::string> base_weights = {"token_embeddings.weight", "rms_out_w", "lm_head"};

    for (const auto &weight_name : base_weights) {
        if (params_.find(weight_name) == params_.end()) {
            std::cerr << "缺少基本权重: " << weight_name << std::endl;
            return false;
        }
    }

    // 调试信息: 打印量化参数中的键名
    if (quant_type_ == 1) {
        std::cout << "\n=== 量化参数键名调试信息 ===" << std::endl;
        std::cout << "qweight_params_ 键名 (" << qweight_params_.size() << " 项):" << std::endl;
        for (const auto &[key, _] : qweight_params_) {
            std::cout << "  " << key << std::endl;
        }

        std::cout << "scales_params_ 键名 (" << scales_params_.size() << " 项):" << std::endl;
        for (const auto &[key, _] : scales_params_) {
            std::cout << "  " << key << std::endl;
        }

        std::cout << "qzeros_params_ 键名 (" << qzeros_params_.size() << " 项):" << std::endl;
        for (const auto &[key, _] : qzeros_params_) {
            std::cout << "  " << key << std::endl;
        }
        std::cout << "================================\n" << std::endl;
    }

    // 检查每层权重是否存在
    for (size_t i = 0; i < n_layers_; i++) {
        std::string layer_id = std::to_string(i);
        std::vector<std::string> layer_weights = {"rms_att_w" + layer_id, "rms_ffn_w" + layer_id, "wq" + layer_id,
                                                  "wk" + layer_id,        "wv" + layer_id,        "wo" + layer_id,
                                                  "q_norm" + layer_id,    "k_norm" + layer_id,    "w_gate" + layer_id,
                                                  "w_up" + layer_id,      "w_down" + layer_id};

        for (const auto &weight_name : layer_weights) {
            // 非量化权重或者非线性层权重直接在params_中查找
            if (weight_name.find("rms_") == 0 || weight_name.find("q_norm") == 0 || weight_name.find("k_norm") == 0) {
                if (params_.find(weight_name) == params_.end()) {
                    std::cerr << "缺少层权重: " << weight_name << std::endl;
                    return false;
                }
                continue;  // 对于非量化的层级权重，直接检查完成，进入下一个循环
            }

            // 对于可能量化的线性层权重，检查是否存在于params_或量化参数中
            if (params_.find(weight_name) == params_.end()) {
                // 对于量化模型，检查是否存在量化版本的权重
                if (quant_type_ == 1) {
                    // 针对线性层权重检查量化版本
                    if (weight_name.find("wq") == 0 || weight_name.find("wk") == 0 || weight_name.find("wv") == 0 ||
                        weight_name.find("wo") == 0 || weight_name.find("w_gate") == 0 ||
                        weight_name.find("w_up") == 0 || weight_name.find("w_down") == 0) {
                        // 尝试多种可能的键名格式
                        std::vector<std::string> possible_qweight_keys = {
                            weight_name,
                            weight_name + ".qweight",
                        };

                        std::vector<std::string> possible_scales_keys = {
                            weight_name,
                            weight_name + ".scales",
                        };

                        std::vector<std::string> possible_qzeros_keys = {
                            weight_name,
                            weight_name + ".qzeros",
                        };

                        bool found_qweight = false;
                        bool found_scales = false;
                        bool found_qzeros = false;

                        // 检查是否存在任何一种可能的键名
                        for (const auto &key : possible_qweight_keys) {
                            if (qweight_params_.find(key) != qweight_params_.end()) {
                                found_qweight = true;
                                break;
                            }
                        }

                        for (const auto &key : possible_scales_keys) {
                            if (scales_params_.find(key) != scales_params_.end()) {
                                found_scales = true;
                                break;
                            }
                        }

                        for (const auto &key : possible_qzeros_keys) {
                            if (qzeros_params_.find(key) != qzeros_params_.end()) {
                                found_qzeros = true;
                                break;
                            }
                        }

                        // 如果找到了所有三种权重，那么认为权重存在
                        if (found_qweight && found_scales && found_qzeros) {
                            continue;  // 权重存在，继续检查下一个权重
                        }

                        // 打印调试信息
                        std::cerr << "缺少层权重: " << weight_name << std::endl;
                        if (!found_qweight) {
                            std::cerr << "  缺少qweight: ";
                            for (const auto &key : possible_qweight_keys) {
                                std::cerr << key << " ";
                            }
                            std::cerr << std::endl;
                        }
                        if (!found_scales) {
                            std::cerr << "  缺少scales: ";
                            for (const auto &key : possible_scales_keys) {
                                std::cerr << key << " ";
                            }
                            std::cerr << std::endl;
                        }
                        if (!found_qzeros) {
                            std::cerr << "  缺少qzeros: ";
                            for (const auto &key : possible_qzeros_keys) {
                                std::cerr << key << " ";
                            }
                            std::cerr << std::endl;
                        }
                        return false;
                    }
                } else {
                    std::cerr << "缺少层权重: " << weight_name << std::endl;
                    return false;
                }
            }
        }
    }

    return true;
}

// -------------------------------
// 打印模型信息
// -------------------------------
template <typename T>
void Qwen3Model<T>::print_model_info() const {
    std::cout << "\n=== Qwen3 Model Information ===" << std::endl;
    std::cout << "Vocab Size: " << vocab_size_ << std::endl;
    std::cout << "Hidden Size: " << hidden_size_ << std::endl;
    std::cout << "Num Layers: " << n_layers_ << std::endl;
    std::cout << "Num Attention Heads: " << n_heads_ << std::endl;
    std::cout << "Num KV Heads: " << n_kv_heads_ << std::endl;
    std::cout << "Head Dimension: " << head_dim_ << std::endl;
    std::cout << "Intermediate Size: " << intermediate_size_ << std::endl;
    std::cout << "Max Position Embeddings: " << max_position_embeddings_ << std::endl;
    std::cout << "RMS Norm Epsilon: " << rms_norm_eps_ << std::endl;
    std::cout << "RoPE Theta: " << rope_theta_ << std::endl;

    if (quant_type_ > 0) {
        std::cout << "Quantization: AWQ (group_size=" << group_size_ << ")" << std::endl;
    } else {
        std::cout << "Quantization: None" << std::endl;
    }

    std::cout << "Device: " << (device_ == Device::CUDA ? "CUDA" : "CPU") << std::endl;
    std::cout << "================================\n" << std::endl;
}

// -------------------------------
// cuda()：将所有参数移到 CUDA，并设置设备
// -------------------------------
template <typename T>
Qwen3Model<T> &Qwen3Model<T>::cuda() {
    for (auto &kv : params_) {
        if (kv.second.device() != Device::CUDA) {
            kv.second.cuda();
        }
    }

    // 移动量化参数到CUDA
    if (quant_type_ == 1) {
        for (auto &kv : qweight_params_) {
            if (kv.second.device() != Device::CUDA) {
                kv.second.cuda();
            }
        }

        for (auto &kv : scales_params_) {
            if (kv.second.device() != Device::CUDA) {
                kv.second.cuda();
            }
        }

        for (auto &kv : qzeros_params_) {
            if (kv.second.device() != Device::CUDA) {
                kv.second.cuda();
            }
        }
    }

    device_ = Device::CUDA;

    // 更新算子接口
    if (operators_) {
        operators_->cuda();
    } else {
        operators_ = std::make_unique<op::UnifiedOperators<T>>(Device::CUDA);
    }

    return *this;
}

// -------------------------------
// cpu()：Qwen3 模型仅支持 CUDA，故调用 cpu() 抛出异常
// -------------------------------
template <typename T>
Qwen3Model<T> &Qwen3Model<T>::cpu() {
    // 更新算子接口（虽然会抛出异常，但保持一致性）
    if (operators_) {
        operators_->cpu();
    }

    throw std::runtime_error("Qwen3Model only supports CUDA execution.");
    return *this;
}

// 显式实例化模板类
template class Qwen3Model<__nv_bfloat16>;
