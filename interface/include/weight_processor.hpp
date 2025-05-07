#pragma once

#include <cuda_bf16.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "include/weight_processor_utils.hpp"
#include "include/llama_weight_processor.hpp"
#include "include/qwen3_weight_processor.hpp"
#include "include/qwen_weight_processor.hpp"

namespace py = pybind11;

// 权重处理器命名空间，提供统一的接口访问各种模型的权重处理器
namespace weight_processor {

// 处理 Llama 模型权重（FP32）
inline std::unordered_map<std::string, Tensor<float>> process_llama_weights(
    const py::dict& weights) {
  return llama_weight_processor::process_weights(weights);
}

// 处理 Qwen 模型权重（FP32）
inline std::unordered_map<std::string, Tensor<float>>
process_qwen_weights_fp32(const py::dict& weights) {
  return qwen_weight_processor::process_weights_fp32(weights);
}

// 处理 Qwen 模型权重（BF16）
inline std::unordered_map<std::string, Tensor<__nv_bfloat16>>
process_qwen_weights_bf16(const py::dict& weights) {
  return qwen_weight_processor::process_weights_bf16(weights);
}

// 处理 Qwen AWQ 量化权重
inline std::tuple<std::unordered_map<std::string, Tensor<__nv_bfloat16>>,
                  std::unordered_map<std::string, Tensor<int32_t>>,
                  std::unordered_map<std::string, Tensor<__nv_bfloat16>>,
                  std::unordered_map<std::string, Tensor<int32_t>>>
process_qwen_weights_awq(const py::dict& weights) {
  return qwen_weight_processor::process_weights_awq(weights);
}

// 处理 Qwen3 模型权重（BF16）
inline std::unordered_map<std::string, Tensor<__nv_bfloat16>>
process_qwen3_weights_bf16(const py::dict& weights) {
  return qwen3_weight_processor::process_weights_bf16(weights);
}

// 处理 Qwen3 AWQ 量化权重
inline std::tuple<std::unordered_map<std::string, Tensor<__nv_bfloat16>>,
                  std::unordered_map<std::string, Tensor<int32_t>>,
                  std::unordered_map<std::string, Tensor<__nv_bfloat16>>,
                  std::unordered_map<std::string, Tensor<int32_t>>>
process_qwen3_weights_awq(const py::dict& weights) {
  return qwen3_weight_processor::process_weights_awq(weights);
}

// 辅助函数：将 PyTorch 张量转换为 __nv_bfloat16 类型的 Tensor
inline Tensor<__nv_bfloat16> convert_bf16_tensor(const py::object& tensor) {
  return weight_processor_utils::convert_bf16_tensor(tensor);
}

} // namespace weight_processor
