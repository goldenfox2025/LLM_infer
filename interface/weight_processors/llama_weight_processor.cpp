#include "llama_weight_processor.hpp"

#include <iostream>

// 处理 Llama 模型权重（FP32）
std::unordered_map<std::string, Tensor<float>>
LlamaWeightProcessor::process_weights(const py::dict& weights) {
  std::unordered_map<std::string, Tensor<float>> cpp_weights;

  // 初始化进度条
  size_t total_weights = weights.size();
  init_progress(total_weights, "Llama");

  // 处理全局权重
  process_global_weights(weights, cpp_weights);

  // 处理层级权重
  process_layer_weights(weights, cpp_weights);

  // 完成进度条
  finish_progress();

  return cpp_weights;
}

// 处理全局权重（embedding, norm, lm_head）
void LlamaWeightProcessor::process_global_weights(
    const py::dict& weights,
    std::unordered_map<std::string, Tensor<float>>& cpp_weights) {
  // 全局权重映射：embedding、归一化、lm_head
  const std::unordered_map<std::string, std::string> key_mapping = {
      {"model.embed_tokens.weight", "embedding_table"},
      {"model.norm.weight", "rms_out_w"},
      {"lm_head.weight", "lm_head"}};

  // 如果没有 embedding_table，则使用 lm_head 的权重作为 embedding
  if (!weights.contains("model.embed_tokens.weight") &&
      weights.contains("lm_head.weight")) {
    py::array_t<float> np_array =
        weights["lm_head.weight"].cast<py::array_t<float>>();
    std::vector<size_t> shape;
    for (int i = 0; i < np_array.ndim(); i++) {
      shape.push_back(np_array.shape(i));
    }
    std::cout << "No embedding_table found, using lm_head as embedding"
              << std::endl;
    std::vector<float> data(np_array.data(), np_array.data() + np_array.size());
    cpp_weights.emplace("embedding_table",
                        Tensor<float>(std::move(data), shape));
  }

  // 处理全局权重
  for (const auto& [src_key, dst_key] : key_mapping) {
    if (weights.contains(src_key)) {
      print_processing_info(src_key, dst_key);
      py::array_t<float> np_array =
          weights[src_key.c_str()].cast<py::array_t<float>>();
      std::vector<size_t> shape;
      for (int i = 0; i < np_array.ndim(); i++) {
        shape.push_back(np_array.shape(i));
      }
      std::vector<float> data(np_array.data(),
                              np_array.data() + np_array.size());
      if (dst_key == "lm_head") {
        cpp_weights.emplace(
            dst_key, Tensor<float>(std::move(data), shape).transpose(-1, -2));
      } else {
        cpp_weights.emplace(dst_key, Tensor<float>(std::move(data), shape));
      }
    }
  }
}

// 处理层级权重
void LlamaWeightProcessor::process_layer_weights(
    const py::dict& weights,
    std::unordered_map<std::string, Tensor<float>>& cpp_weights) {
  // 层级权重映射
  const std::vector<std::pair<std::string, std::string>> layer_key_mapping = {
      {"input_layernorm.weight", "rms_att_w"},
      {"post_attention_layernorm.weight", "rms_ffn_w"},
      {"self_attn.q_proj.weight", "wq"},
      {"self_attn.k_proj.weight", "wk"},
      {"self_attn.v_proj.weight", "wv"},
      {"self_attn.o_proj.weight", "wo"},
      {"mlp.up_proj.weight", "w_up"},
      {"mlp.down_proj.weight", "w_down"},
      {"mlp.gate_proj.weight", "w_gate"}};

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

          print_processing_info(key, dst_key);

          py::array_t<float> np_array = item.second.cast<py::array_t<float>>();
          std::vector<size_t> shape;
          for (int i = 0; i < np_array.ndim(); i++) {
            shape.push_back(np_array.shape(i));
          }
          std::vector<float> data(np_array.data(),
                                  np_array.data() + np_array.size());
          // 对部分矩阵需要转置 但本质上没有转置
          // 这里是为了适配cpu算子（支持stride） 然而cuda没有支持
          if (dst_prefix == "wq" || dst_prefix == "wk" || dst_prefix == "wv" ||
              dst_prefix == "wo" || dst_prefix == "w_up" ||
              dst_prefix == "w_down" || dst_prefix == "w_gate") {
            cpp_weights.emplace(
                dst_key,
                Tensor<float>(std::move(data), shape).transpose(-1, -2));
          } else {
            cpp_weights.emplace(dst_key, Tensor<float>(std::move(data), shape));
          }
        }
      }
    }
  }
}
