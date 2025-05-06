#pragma once

#include <cuda_bf16.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "backend/cpp/include/tensor.hpp"

namespace py = pybind11;

// 基础权重处理器类，提供通用的权重处理功能
class WeightProcessorBase {
 public:
  // 辅助函数：将 PyTorch 张量转换为 __nv_bfloat16 类型的 Tensor
  static Tensor<__nv_bfloat16> convert_bf16_tensor(const py::object& tensor);

  // 辅助函数：将 PyTorch 张量转换为 float 类型的 Tensor
  static Tensor<float> convert_float_tensor(const py::object& tensor);

  // 辅助函数：从 PyTorch 张量提取形状信息
  static std::vector<size_t> get_tensor_shape(const py::object& tensor);

  // 辅助函数：打印权重处理进度
  static void print_processing_info(const std::string& key,
                                    const std::string& dst_key);

  // 辅助函数：计算张量中的参数数量
  static size_t calculate_params_count(const py::object& tensor);

  // 辅助函数：计算张量形状中的参数数量
  static size_t calculate_params_from_shape(const std::vector<size_t>& shape);

  // 进度条相关函数
  static void init_progress(size_t total_weights,
                            const std::string& model_type);
  static void update_progress(const std::string& key,
                              const std::string& dst_key);
  static void finish_progress();

 protected:
  // 进度条状态
  static size_t total_weights_;
  static size_t processed_weights_;
  static std::string current_model_type_;
  static bool progress_initialized_;
  static size_t total_params_count_;
};
