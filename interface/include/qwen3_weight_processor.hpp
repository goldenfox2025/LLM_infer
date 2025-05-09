#pragma once

#include <iostream>
#include <tuple>

#include "weight_processor_utils.hpp"

// Qwen3 模型权重处理器命名空间
namespace qwen3_weight_processor {

// 前向声明所有的辅助函数
inline void process_global_weights_bf16(const py::dict& weights,
                                        std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_weights);

inline void process_layer_weights_bf16(const py::dict& weights,
                                       std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_weights);

inline void process_global_weights_awq(const py::dict& weights,
                                       std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_weights);

inline void process_quantized_weights_awq(const py::dict& weights,
                                          std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_weights,
                                          std::unordered_map<std::string, Tensor<int32_t>>& cpp_qweight_params,
                                          std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_scales_params,
                                          std::unordered_map<std::string, Tensor<int32_t>>& cpp_qzeros_params);

// 转置AWQ量化权重的辅助函数
// 这个版本仅打印日志，但不实际转置，保持原始的KN格式
inline void transpose_awq_weight(const std::string& key, const std::vector<size_t>& original_shape,
                                 std::unordered_map<std::string, Tensor<int32_t>>& qweights,
                                 std::unordered_map<std::string, Tensor<__nv_bfloat16>>& scales,
                                 std::unordered_map<std::string, Tensor<int32_t>>& qzeros) {
    if (qweights.find(key) == qweights.end() || scales.find(key) == scales.end() || qzeros.find(key) == qzeros.end()) {
        std::cerr << "警告: 缺少量化参数: " << key << std::endl;
        return;
    }

    // 获取原始形状
    size_t K = original_shape[0];  // 输入维度
    size_t N = original_shape[1];  // 输出维度

    std::cout << "  保持量化权重: " << key << " 原始格式 [" << K << ", " << N << "]" << std::endl;
    std::cout << "  不执行转置，保持KN格式以便与矩阵乘法匹配" << std::endl;

    // 不做任何操作，保持原始权重格式
}

// 处理 BF16 全局权重
inline void process_global_weights_bf16(const py::dict& weights,
                                        std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_weights) {
    // 全局权重映射
    const std::unordered_map<std::string, std::string> global_weights_map = {
        {"model.embed_tokens.weight", "token_embeddings.weight"},
        {"model.norm.weight", "rms_out_w"},
        {"lm_head.weight", "lm_head"}};

    // 处理非量化的全局权重
    for (const auto& [src_key, dst_key] : global_weights_map) {
        if (weights.contains(src_key)) {
            weight_processor_utils::print_processing_info(src_key, dst_key);
            py::object tensor = weights[src_key.c_str()];

            // 计算参数数量并累加到总数
            size_t params_count = weight_processor_utils::calculate_params_count(tensor);

            Tensor<__nv_bfloat16> bf16_tensor = weight_processor_utils::convert_bf16_tensor(tensor);
            if (dst_key == "lm_head") {
                // lm_head 权重需要逻辑转置，与矩阵乘法操作匹配
                auto shape_obj = tensor.attr("shape");
                auto shape_tuple = shape_obj.cast<py::tuple>();
                std::vector<size_t> shape;
                for (size_t i = 0; i < shape_tuple.size(); i++) {
                    shape.push_back(shape_tuple[i].cast<size_t>());
                }

                if (shape.size() >= 2) {
                    std::cout << "  lm_head 原始形状: [" << shape[0] << ", " << shape[1] << "]" << std::endl;
                    std::cout << "  对lm_head执行逻辑转置" << std::endl;
                    // 对lm_head进行逻辑转置，保持KN格式但在访问时被视为NK格式
                    cpp_weights.emplace(dst_key, bf16_tensor.transpose(-1, -2));
                } else {
                    // 如果形状不是2D，也进行转置处理
                    std::cout << "  lm_head 不是2D张量，执行默认转置" << std::endl;
                    cpp_weights.emplace(dst_key, bf16_tensor.transpose(-1, -2));
                }
            } else {
                cpp_weights.emplace(dst_key, std::move(bf16_tensor));
            }
        }
    }
}

// 处理 BF16 层级权重
inline void process_layer_weights_bf16(const py::dict& weights,
                                       std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_weights) {
    // 层级权重映射
    const std::vector<std::pair<std::string, std::string>> layer_key_mapping = {
        {"input_layernorm.weight", "rms_att_w"}, {"post_attention_layernorm.weight", "rms_ffn_w"},
        {"self_attn.q_proj.weight", "wq"},       {"self_attn.k_proj.weight", "wk"},
        {"self_attn.v_proj.weight", "wv"},       {"self_attn.o_proj.weight", "wo"},
        {"self_attn.q_norm.weight", "q_norm"},   {"self_attn.k_norm.weight", "k_norm"},
        {"mlp.gate_proj.weight", "w_gate"},      {"mlp.up_proj.weight", "w_up"},
        {"mlp.down_proj.weight", "w_down"}};

    // 需要转置的线性层列表（仅进行逻辑转置，不进行物理转置）
    const std::unordered_set<std::string> transpose_layers = {"wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down"};

    for (auto item : weights) {
        std::string key = py::str(item.first).cast<std::string>();
        if (key.find("model.layers.") == 0) {
            for (const auto& [src_suffix, dst_prefix] : layer_key_mapping) {
                std::string pattern = "." + src_suffix;
                if (key.find(pattern) != std::string::npos) {
                    // 提取层索引，构造目标键名
                    size_t start = std::string("model.layers.").size();
                    size_t end = key.find('.', start);
                    std::string layer_str = key.substr(start, end - start);
                    int layer = std::stoi(layer_str);
                    std::string dst_key = dst_prefix + std::to_string(layer);

                    weight_processor_utils::print_processing_info(key, dst_key);

                    // 计算参数数量并累加到总数
                    py::object tensor = py::reinterpret_borrow<py::object>(item.second);
                    size_t params_count = weight_processor_utils::calculate_params_count(tensor);

                    // 转换为bf16张量
                    Tensor<__nv_bfloat16> bf16_tensor = weight_processor_utils::convert_bf16_tensor(tensor);

                    // 检查是否需要转置（从KN转为NK格式）- 这里只进行逻辑转置，没有物理数据移动
                    if (transpose_layers.find(dst_prefix) != transpose_layers.end()) {
                        std::cout << "  对权重进行逻辑转置: " << dst_key << std::endl;
                        cpp_weights.emplace(dst_key, bf16_tensor.transpose(-1, -2));
                    } else {
                        cpp_weights.emplace(dst_key, std::move(bf16_tensor));
                    }
                    break;
                }
            }
        }
    }
}

// 处理 AWQ 全局权重
inline void process_global_weights_awq(const py::dict& weights,
                                       std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_weights) {
    // 全局权重映射
    const std::unordered_map<std::string, std::string> global_weights_map = {
        {"model.embed_tokens.weight", "token_embeddings.weight"},
        {"model.norm.weight", "rms_out_w"},
        {"lm_head.weight", "lm_head"}};

    // 处理非量化的全局权重
    for (const auto& [src_key, dst_key] : global_weights_map) {
        if (weights.contains(src_key)) {
            weight_processor_utils::print_processing_info(src_key, dst_key);
            py::object tensor = weights[src_key.c_str()];

            // 计算参数数量并累加到总数
            size_t params_count = weight_processor_utils::calculate_params_count(tensor);

            Tensor<__nv_bfloat16> bf16_tensor = weight_processor_utils::convert_bf16_tensor(tensor);
            if (dst_key == "lm_head") {
                // lm_head 权重需要逻辑转置，与矩阵乘法操作匹配
                auto shape_obj = tensor.attr("shape");
                auto shape_tuple = shape_obj.cast<py::tuple>();
                std::vector<size_t> shape;
                for (size_t i = 0; i < shape_tuple.size(); i++) {
                    shape.push_back(shape_tuple[i].cast<size_t>());
                }

                // 检查并显示形状信息
                if (shape.size() >= 2) {
                    std::cout << "  AWQ lm_head 原始形状: [" << shape[0] << ", " << shape[1] << "]" << std::endl;
                    std::cout << "  对AWQ lm_head执行逻辑转置" << std::endl;
                    // 对lm_head进行逻辑转置
                    cpp_weights.emplace(dst_key, bf16_tensor.transpose(-1, -2));
                } else {
                    // 如果形状不是2D，也进行转置处理
                    std::cout << "  AWQ lm_head 不是2D张量，执行默认转置" << std::endl;
                    cpp_weights.emplace(dst_key, bf16_tensor.transpose(-1, -2));
                }
            } else {
                cpp_weights.emplace(dst_key, std::move(bf16_tensor));
            }
        }
    }
}

// 处理 AWQ 量化权重
inline void process_quantized_weights_awq(const py::dict& weights,
                                          std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_weights,
                                          std::unordered_map<std::string, Tensor<int32_t>>& cpp_qweight_params,
                                          std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_scales_params,
                                          std::unordered_map<std::string, Tensor<int32_t>>& cpp_qzeros_params) {
    // 添加处理变量：标记哪些权重需要转置
    // 这里我们先收集所有权重，处理完后再进行转置
    std::unordered_set<std::string> weights_to_transpose;
    std::unordered_map<std::string, std::vector<size_t>> original_shapes;

    for (auto item : weights) {
        std::string key = py::str(item.first).cast<std::string>();

        // 处理层级权重
        if (key.find("model.layers.") == 0) {
            // 处理量化权重
            if (key.find(".qweight") != std::string::npos) {
                // 提取层索引和权重名称
                size_t start = std::string("model.layers.").size();
                size_t end = key.find('.', start);
                std::string layer_str = key.substr(start, end - start);
                int layer = std::stoi(layer_str);

                // 提取权重类型
                std::string weight_type = key.substr(end + 1);
                weight_type = weight_type.substr(0, weight_type.find(".qweight"));

                // 映射权重类型到目标键名前缀
                std::string dst_prefix;
                if (weight_type == "self_attn.q_proj") {
                    dst_prefix = "wq";
                } else if (weight_type == "self_attn.k_proj") {
                    dst_prefix = "wk";
                } else if (weight_type == "self_attn.v_proj") {
                    dst_prefix = "wv";
                } else if (weight_type == "self_attn.o_proj") {
                    dst_prefix = "wo";
                } else if (weight_type == "mlp.gate_proj") {
                    dst_prefix = "w_gate";
                } else if (weight_type == "mlp.up_proj") {
                    dst_prefix = "w_up";
                } else if (weight_type == "mlp.down_proj") {
                    dst_prefix = "w_down";
                } else {
                    continue;  // 跳过未知权重类型
                }

                std::string dst_key = dst_prefix + std::to_string(layer);
                weight_processor_utils::print_processing_info(key, dst_key + ".qweight");

                // 处理量化权重
                py::object tensor = py::reinterpret_borrow<py::object>(item.second);
                size_t params_count = weight_processor_utils::calculate_params_count(tensor);

                // 转换为int32张量
                py::array_t<int32_t> np_array = tensor.cast<py::array_t<int32_t>>();
                std::vector<size_t> shape;
                for (int i = 0; i < np_array.ndim(); i++) {
                    shape.push_back(np_array.shape(i));
                }

                // 记录原始形状，稍后用于转置
                original_shapes[dst_key] = shape;

                // 标记此权重需要转置（所有线性层权重都需要从KN转为NK）
                weights_to_transpose.insert(dst_key);

                std::vector<int32_t> data(np_array.data(), np_array.data() + np_array.size());
                cpp_qweight_params.emplace(dst_key, Tensor<int32_t>(std::move(data), shape));
            }

            // 处理缩放因子
            else if (key.find(".scales") != std::string::npos) {
                // 提取层索引和权重名称
                size_t start = std::string("model.layers.").size();
                size_t end = key.find('.', start);
                std::string layer_str = key.substr(start, end - start);
                int layer = std::stoi(layer_str);

                // 提取权重类型
                std::string weight_type = key.substr(end + 1);
                weight_type = weight_type.substr(0, weight_type.find(".scales"));

                // 映射权重类型到目标键名前缀
                std::string dst_prefix;
                if (weight_type == "self_attn.q_proj") {
                    dst_prefix = "wq";
                } else if (weight_type == "self_attn.k_proj") {
                    dst_prefix = "wk";
                } else if (weight_type == "self_attn.v_proj") {
                    dst_prefix = "wv";
                } else if (weight_type == "self_attn.o_proj") {
                    dst_prefix = "wo";
                } else if (weight_type == "mlp.gate_proj") {
                    dst_prefix = "w_gate";
                } else if (weight_type == "mlp.up_proj") {
                    dst_prefix = "w_up";
                } else if (weight_type == "mlp.down_proj") {
                    dst_prefix = "w_down";
                } else {
                    continue;  // 跳过未知权重类型
                }

                std::string dst_key = dst_prefix + std::to_string(layer);
                weight_processor_utils::print_processing_info(key, dst_key + ".scales");

                // 处理缩放因子
                py::object tensor = py::reinterpret_borrow<py::object>(item.second);
                size_t params_count = weight_processor_utils::calculate_params_count(tensor);

                // 转换为bf16张量
                Tensor<__nv_bfloat16> bf16_tensor = weight_processor_utils::convert_bf16_tensor(tensor);
                cpp_scales_params.emplace(dst_key, std::move(bf16_tensor));
            }

            // 处理量化零点
            else if (key.find(".qzeros") != std::string::npos) {
                // 提取层索引和权重名称
                size_t start = std::string("model.layers.").size();
                size_t end = key.find('.', start);
                std::string layer_str = key.substr(start, end - start);
                int layer = std::stoi(layer_str);

                // 提取权重类型
                std::string weight_type = key.substr(end + 1);
                weight_type = weight_type.substr(0, weight_type.find(".qzeros"));

                // 映射权重类型到目标键名前缀
                std::string dst_prefix;
                if (weight_type == "self_attn.q_proj") {
                    dst_prefix = "wq";
                } else if (weight_type == "self_attn.k_proj") {
                    dst_prefix = "wk";
                } else if (weight_type == "self_attn.v_proj") {
                    dst_prefix = "wv";
                } else if (weight_type == "self_attn.o_proj") {
                    dst_prefix = "wo";
                } else if (weight_type == "mlp.gate_proj") {
                    dst_prefix = "w_gate";
                } else if (weight_type == "mlp.up_proj") {
                    dst_prefix = "w_up";
                } else if (weight_type == "mlp.down_proj") {
                    dst_prefix = "w_down";
                } else {
                    continue;  // 跳过未知权重类型
                }

                std::string dst_key = dst_prefix + std::to_string(layer);
                weight_processor_utils::print_processing_info(key, dst_key + ".qzeros");

                // 处理量化零点
                py::object tensor = py::reinterpret_borrow<py::object>(item.second);
                size_t params_count = weight_processor_utils::calculate_params_count(tensor);

                // 转换为int32张量
                py::array_t<int32_t> np_array = tensor.cast<py::array_t<int32_t>>();
                std::vector<size_t> shape;
                for (int i = 0; i < np_array.ndim(); i++) {
                    shape.push_back(np_array.shape(i));
                }
                std::vector<int32_t> data(np_array.data(), np_array.data() + np_array.size());
                cpp_qzeros_params.emplace(dst_key, Tensor<int32_t>(std::move(data), shape));
            }

            // 处理非量化层级权重
            else if (key.find("input_layernorm.weight") != std::string::npos ||
                     key.find("post_attention_layernorm.weight") != std::string::npos ||
                     key.find("self_attn.q_norm.weight") != std::string::npos ||
                     key.find("self_attn.k_norm.weight") != std::string::npos) {
                // 提取层索引
                size_t start = std::string("model.layers.").size();
                size_t end = key.find('.', start);
                std::string layer_str = key.substr(start, end - start);
                int layer = std::stoi(layer_str);

                // 构造目标键名
                std::string dst_key;
                if (key.find("input_layernorm.weight") != std::string::npos) {
                    dst_key = "rms_att_w" + std::to_string(layer);
                } else if (key.find("post_attention_layernorm.weight") != std::string::npos) {
                    dst_key = "rms_ffn_w" + std::to_string(layer);
                } else if (key.find("self_attn.q_norm.weight") != std::string::npos) {
                    dst_key = "q_norm" + std::to_string(layer);
                } else if (key.find("self_attn.k_norm.weight") != std::string::npos) {
                    dst_key = "k_norm" + std::to_string(layer);
                }

                weight_processor_utils::print_processing_info(key, dst_key);

                // 处理非量化层级权重
                py::object tensor = py::reinterpret_borrow<py::object>(item.second);
                size_t params_count = weight_processor_utils::calculate_params_count(tensor);

                // 转换为bf16张量
                Tensor<__nv_bfloat16> bf16_tensor = weight_processor_utils::convert_bf16_tensor(tensor);
                cpp_weights.emplace(dst_key, std::move(bf16_tensor));
            }
        }
    }

    // 处理需要转置的量化权重（从KN转为NK格式）
    for (const auto& key : weights_to_transpose) {
        if (original_shapes.find(key) != original_shapes.end()) {
            // 调用辅助函数转置量化权重
            transpose_awq_weight(key, original_shapes[key], cpp_qweight_params, cpp_scales_params, cpp_qzeros_params);
        } else {
            std::cerr << "警告: 无法找到权重的原始形状: " << key << std::endl;
        }
    }
}

// 处理 Qwen3 模型权重（BF16）
inline std::unordered_map<std::string, Tensor<__nv_bfloat16>> process_weights_bf16(const py::dict& weights) {
    std::unordered_map<std::string, Tensor<__nv_bfloat16>> cpp_weights;

    // 初始化进度条
    size_t total_weights = weights.size();
    weight_processor_utils::init_progress(total_weights, "Qwen3 BF16");

    // 处理全局权重
    process_global_weights_bf16(weights, cpp_weights);

    // 处理层级权重
    process_layer_weights_bf16(weights, cpp_weights);

    // 完成进度条
    weight_processor_utils::finish_progress();

    if (cpp_weights.find("lm_head") == cpp_weights.end()) {
        std::cout << "\nWarning: lm_head not found in BF16 weights, creating from "
                     "token_embeddings.weight"
                  << std::endl;
        if (cpp_weights.find("token_embeddings.weight") != cpp_weights.end()) {
            Tensor<__nv_bfloat16> lm_head = cpp_weights.at("token_embeddings.weight").transpose(-1, -2);
            cpp_weights.emplace("lm_head", std::move(lm_head));
        }
    }

    return cpp_weights;
}

// 处理 Qwen3 AWQ 量化权重
inline std::tuple<
    std::unordered_map<std::string, Tensor<__nv_bfloat16>>, std::unordered_map<std::string, Tensor<int32_t>>,
    std::unordered_map<std::string, Tensor<__nv_bfloat16>>, std::unordered_map<std::string, Tensor<int32_t>>>
process_weights_awq(const py::dict& weights) {
    std::unordered_map<std::string, Tensor<__nv_bfloat16>> cpp_weights;
    std::unordered_map<std::string, Tensor<int32_t>> cpp_qweight_params;
    std::unordered_map<std::string, Tensor<__nv_bfloat16>> cpp_scales_params;
    std::unordered_map<std::string, Tensor<int32_t>> cpp_qzeros_params;

    // 初始化进度条
    size_t total_weights = weights.size();
    weight_processor_utils::init_progress(total_weights, "Qwen3 AWQ");

    // 处理全局权重
    process_global_weights_awq(weights, cpp_weights);

    // 处理量化权重
    process_quantized_weights_awq(weights, cpp_weights, cpp_qweight_params, cpp_scales_params, cpp_qzeros_params);

    // 完成进度条
    weight_processor_utils::finish_progress();

    // 注意：移除从embed_tokens.weight转置生成lm_head的逻辑
    // 如果 lm_head 缺失，报告错误
    if (cpp_weights.find("lm_head") == cpp_weights.end()) {
        std::cout << "\nWarning: lm_head 未在AWQ权重中找到！" << std::endl;
    }

    return {cpp_weights, cpp_qweight_params, cpp_scales_params, cpp_qzeros_params};
}

}  // namespace qwen3_weight_processor
