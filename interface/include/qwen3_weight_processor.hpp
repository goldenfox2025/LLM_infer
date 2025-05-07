#pragma once

#include <tuple>
#include <iostream>

#include "weight_processor_utils.hpp"

// Qwen3 模型权重处理器命名空间
namespace qwen3_weight_processor {

// 前向声明所有的辅助函数
inline void process_global_weights_bf16(
      const py::dict& weights,
      std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_weights);

inline void process_layer_weights_bf16(
      const py::dict& weights,
      std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_weights);

inline void process_global_weights_awq(
      const py::dict& weights,
      std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_weights);

inline void process_quantized_weights_awq(
      const py::dict& weights,
      std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_weights,
      std::unordered_map<std::string, Tensor<int32_t>>& cpp_qweight_params,
      std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_scales_params,
      std::unordered_map<std::string, Tensor<int32_t>>& cpp_qzeros_params);

// 处理 BF16 全局权重
inline void process_global_weights_bf16(
    const py::dict& weights,
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
        cpp_weights.emplace(dst_key, bf16_tensor.transpose(-1, -2));
      } else {
        cpp_weights.emplace(dst_key, std::move(bf16_tensor));
      }
    }
  }
}

// 处理 BF16 层级权重
inline void process_layer_weights_bf16(
    const py::dict& weights,
    std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_weights) {
  // 层级权重映射
  const std::vector<std::pair<std::string, std::string>> layer_key_mapping = {
      {"input_layernorm.weight", "rms_att_w"},
      {"post_attention_layernorm.weight", "rms_ffn_w"},
      {"self_attn.q_proj.weight", "wq"},
      {"self_attn.k_proj.weight", "wk"},
      {"self_attn.v_proj.weight", "wv"},
      {"self_attn.o_proj.weight", "wo"},
      {"mlp.gate_proj.weight", "w_gate"},
      {"mlp.up_proj.weight", "w_up"},
      {"mlp.down_proj.weight", "w_down"}};

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
          cpp_weights.emplace(dst_key, std::move(bf16_tensor));
          break;
        }
      }
    }
  }
}

// 处理 AWQ 全局权重
inline void process_global_weights_awq(
    const py::dict& weights,
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
        cpp_weights.emplace(dst_key, bf16_tensor.transpose(-1, -2));
      } else {
        cpp_weights.emplace(dst_key, std::move(bf16_tensor));
      }
    }
  }
}

// 处理 AWQ 量化权重
inline void process_quantized_weights_awq(
    const py::dict& weights,
    std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_weights,
    std::unordered_map<std::string, Tensor<int32_t>>& cpp_qweight_params,
    std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_scales_params,
    std::unordered_map<std::string, Tensor<int32_t>>& cpp_qzeros_params) {
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
        std::vector<int32_t> data(np_array.data(),
                                  np_array.data() + np_array.size());
        cpp_qweight_params.emplace(dst_key,
                                  Tensor<int32_t>(std::move(data), shape));
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
        std::vector<int32_t> data(np_array.data(),
                                  np_array.data() + np_array.size());
        cpp_qzeros_params.emplace(dst_key,
                                 Tensor<int32_t>(std::move(data), shape));
      }

      // 处理非量化层级权重
      else if (key.find("input_layernorm.weight") != std::string::npos ||
               key.find("post_attention_layernorm.weight") != std::string::npos) {
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
}

// 处理 Qwen3 模型权重（BF16）
inline std::unordered_map<std::string, Tensor<__nv_bfloat16>>
process_weights_bf16(const py::dict& weights) {
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

  // 如果 lm_head 缺失，则尝试从 embed_tokens.weight 转置生成
  if (cpp_weights.find("lm_head") == cpp_weights.end()) {
    std::cout << "\nWarning: lm_head not found in weights, creating from "
                "embed_tokens.weight"
              << std::endl;
    if (cpp_weights.find("token_embeddings.weight") != cpp_weights.end()) {
      Tensor<__nv_bfloat16> lm_head =
          cpp_weights.at("token_embeddings.weight").transpose(-1, -2);
      cpp_weights.emplace("lm_head", std::move(lm_head));
    } else {
      std::cerr << "Error: embed_tokens.weight not found in weights"
                << std::endl;
    }
  }

  return cpp_weights;
}

// 处理 Qwen3 AWQ 量化权重
inline std::tuple<std::unordered_map<std::string, Tensor<__nv_bfloat16>>,
                  std::unordered_map<std::string, Tensor<int32_t>>,
                  std::unordered_map<std::string, Tensor<__nv_bfloat16>>,
                  std::unordered_map<std::string, Tensor<int32_t>>>
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
  process_quantized_weights_awq(weights, cpp_weights, cpp_qweight_params,
                              cpp_scales_params, cpp_qzeros_params);
  // 完成进度条
  weight_processor_utils::finish_progress();

  // 如果 lm_head 缺失，则尝试从 embed_tokens.weight 转置生成
  if (cpp_weights.find("lm_head") == cpp_weights.end()) {
    std::cout << "\nWarning: lm_head not found in AWQ weights, creating from "
                "embed_tokens.weight"
              << std::endl;
    if (cpp_weights.find("token_embeddings.weight") != cpp_weights.end()) {
      Tensor<__nv_bfloat16> lm_head =
          cpp_weights.at("token_embeddings.weight").transpose(-1, -2);
      cpp_weights.emplace("lm_head", std::move(lm_head));
    } else {
      std::cerr << "Error: embed_tokens.weight not found in AWQ weights"
                << std::endl;
    }
  }

  return {cpp_weights, cpp_qweight_params, cpp_scales_params,
          cpp_qzeros_params};
}

} // namespace qwen3_weight_processor
