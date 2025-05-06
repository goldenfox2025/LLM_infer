#pragma once

#include "include/llama_weight_processor.hpp"
#include "include/qwen_weight_processor.hpp"

// 权重处理器类，提供统一的接口访问各种模型的权重处理器
class WeightProcessor {
 public:
  // 处理 Llama 模型权重（FP32）
  static std::unordered_map<std::string, Tensor<float>> process_llama_weights(
      const py::dict& weights) {
    return LlamaWeightProcessor::process_weights(weights);
  }

  // 处理 Qwen 模型权重（FP32）
  static std::unordered_map<std::string, Tensor<float>>
  process_qwen_weights_fp32(const py::dict& weights) {
    return QwenWeightProcessor::process_weights_fp32(weights);
  }

  // 处理 Qwen 模型权重（BF16）
  static std::unordered_map<std::string, Tensor<__nv_bfloat16>>
  process_qwen_weights_bf16(const py::dict& weights) {
    return QwenWeightProcessor::process_weights_bf16(weights);
  }

  // 处理 Qwen AWQ 量化权重
  static std::tuple<std::unordered_map<std::string, Tensor<__nv_bfloat16>>,
                    std::unordered_map<std::string, Tensor<int32_t>>,
                    std::unordered_map<std::string, Tensor<__nv_bfloat16>>,
                    std::unordered_map<std::string, Tensor<int32_t>>>
  process_qwen_weights_awq(const py::dict& weights) {
    return QwenWeightProcessor::process_weights_awq(weights);
  }

  // 辅助函数：将 PyTorch 张量转换为 __nv_bfloat16 类型的 Tensor
  static Tensor<__nv_bfloat16> convert_bf16_tensor(const py::object& tensor) {
    return WeightProcessorBase::convert_bf16_tensor(tensor);
  }
};
