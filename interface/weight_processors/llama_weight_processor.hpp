#pragma once

#include "weight_processor_base.hpp"

// Llama 模型权重处理器类
class LlamaWeightProcessor : public WeightProcessorBase {
 public:
  // 处理 Llama 模型权重（FP32）
  static std::unordered_map<std::string, Tensor<float>> process_weights(
      const py::dict& weights);

 private:
  // 处理全局权重（embedding, norm, lm_head）
  static void process_global_weights(
      const py::dict& weights,
      std::unordered_map<std::string, Tensor<float>>& cpp_weights);

  // 处理层级权重
  static void process_layer_weights(
      const py::dict& weights,
      std::unordered_map<std::string, Tensor<float>>& cpp_weights);
};
