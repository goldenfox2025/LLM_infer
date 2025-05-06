#pragma once

#include <tuple>

#include "weight_processor_base.hpp"

// Qwen 模型权重处理器类
class QwenWeightProcessor : public WeightProcessorBase {
 public:
  // 处理 Qwen 模型权重（FP32）
  static std::unordered_map<std::string, Tensor<float>> process_weights_fp32(
      const py::dict& weights);

  // 处理 Qwen 模型权重（BF16）
  static std::unordered_map<std::string, Tensor<__nv_bfloat16>>
  process_weights_bf16(const py::dict& weights);

  // 处理 Qwen AWQ 量化权重
  static std::tuple<std::unordered_map<std::string, Tensor<__nv_bfloat16>>,
                    std::unordered_map<std::string, Tensor<int32_t>>,
                    std::unordered_map<std::string, Tensor<__nv_bfloat16>>,
                    std::unordered_map<std::string, Tensor<int32_t>>>
  process_weights_awq(const py::dict& weights);

 private:
  // 处理 FP32 全局权重
  static void process_global_weights_fp32(
      const py::dict& weights,
      std::unordered_map<std::string, Tensor<float>>& cpp_weights);

  // 处理 FP32 层级权重
  static void process_layer_weights_fp32(
      const py::dict& weights,
      std::unordered_map<std::string, Tensor<float>>& cpp_weights);

  // 处理 BF16 全局权重
  static void process_global_weights_bf16(
      const py::dict& weights,
      std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_weights);

  // 处理 BF16 层级权重
  static void process_layer_weights_bf16(
      const py::dict& weights,
      std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_weights);

  // 处理 AWQ 全局权重
  static void process_global_weights_awq(
      const py::dict& weights,
      std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_weights);

  // 处理 AWQ 量化权重
  static void process_quantized_weights_awq(
      const py::dict& weights,
      std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_weights,
      std::unordered_map<std::string, Tensor<int32_t>>& cpp_qweight_params,
      std::unordered_map<std::string, Tensor<__nv_bfloat16>>& cpp_scales_params,
      std::unordered_map<std::string, Tensor<int32_t>>& cpp_qzeros_params);
};
