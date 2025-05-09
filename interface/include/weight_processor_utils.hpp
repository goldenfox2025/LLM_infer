#pragma once

#include <cuda_bf16.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "backend/cpp/include/tensor.hpp"

namespace py = pybind11;

/**
 * 权重处理工具命名空间
 * 提供通用的权重处理功能，包括张量转换、权重格式处理和进度显示
 */
namespace weight_processor_utils {

// 静态状态变量 - 使用内联变量防止多重定义
inline size_t total_weights = 0;
inline size_t processed_weights = 0;
inline std::string current_model_type = "";
inline bool progress_initialized = false;
inline size_t total_params_count = 0;

// 前向声明 - 在使用前确保已定义
inline void update_progress(const std::string& key, const std::string& dst_key);

/**
 * 从 PyTorch 张量提取形状信息
 * @param tensor PyTorch 张量对象
 * @return 包含形状信息的向量
 */
inline std::vector<size_t> get_tensor_shape(const py::object& tensor) {
    py::tuple shape_tuple = tensor.attr("shape");
    std::vector<size_t> shape;
    for (size_t i = 0; i < py::len(shape_tuple); ++i) {
        shape.push_back(shape_tuple[i].cast<size_t>());
    }
    return shape;
}

/**
 * 计算张量形状中的参数数量
 * @param shape 形状信息向量
 * @return 参数总数
 */
inline size_t calculate_params_from_shape(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        return 0;
    }
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
}

/**
 * 计算张量中的参数数量
 * @param tensor PyTorch 张量对象
 * @return 参数总数
 */
inline size_t calculate_params_count(const py::object& tensor) {
    std::vector<size_t> shape = get_tensor_shape(tensor);
    return calculate_params_from_shape(shape);
}

/**
 * 将 PyTorch 张量转换为 __nv_bfloat16 类型的 Tensor
 * @param tensor PyTorch 张量对象
 * @return 转换后的 bf16 Tensor
 */
inline Tensor<__nv_bfloat16> convert_bf16_tensor(const py::object& tensor) {
    try {
        py::object torch_module = py::module::import("torch");

        // 确保张量在CPU上，便于数据访问
        py::object cpu_tensor = tensor.attr("detach")().attr("cpu")();

        // 获取形状
        std::vector<size_t> shape = get_tensor_shape(cpu_tensor);

        // 计算元素总数
        size_t numel = 1;
        for (auto dim : shape) {
            numel *= dim;
        }

        // 预分配数据向量
        std::vector<__nv_bfloat16> data;
        data.reserve(numel);

        if (py::hasattr(cpu_tensor, "element_size") && py::hasattr(cpu_tensor, "data_ptr")) {
            // 这里使用pytorch的接口获取字节长度
            size_t element_size = cpu_tensor.attr("element_size")().cast<size_t>();

            // 检查数据类型 - pytorch中的数据类型
            std::string dtype_str = py::str(cpu_tensor.attr("dtype")).cast<std::string>();

            // 确认是否为bfloat16类型或者是其他2字节类型(如fp16)
            if (element_size == 2) {
                // 根据数据类型进行不同处理
                if (dtype_str.find("bfloat16") != std::string::npos) {
                    // 是bfloat16类型，可以直接复制
                    uintptr_t data_ptr = cpu_tensor.attr("data_ptr")().cast<uintptr_t>();
                    const __nv_bfloat16* ptr = reinterpret_cast<const __nv_bfloat16*>(data_ptr);
                    // 每个元素直接拷贝二进制数据
                    for (size_t i = 0; i < numel; ++i) {
                        __nv_bfloat16 bits = ptr[i];
                        data.push_back(bits);
                    }
                } else if (dtype_str.find("float16") != std::string::npos ||
                           dtype_str.find("half") != std::string::npos) {
                    // 是fp16类型，需要先转换

                    // 首先将fp16先转为float32，分析是否存在很小的非零值
                    py::object float_tensor = cpu_tensor.attr("to")(torch_module.attr("float"));

                    // 提取统计信息以便处理
                    py::object max_val = torch_module.attr("max")(float_tensor);
                    py::object min_val = torch_module.attr("min")(float_tensor);
                    float max_value = max_val.cast<float>();
                    float min_value = min_val.cast<float>();

                    // 如果值特别小，强制替换为更合理的值，防止转换为0
                    // 检查最大值 - 如果一个tensor中最大值小于一个阈值，认为这是scales值
                    if (max_value > 0 && max_value < 0.3) {
                        // 这很可能是量化scales，小的值很容易丢失

                        // 获取numpy数组用于处理
                        py::array_t<float> np_array = float_tensor.attr("numpy")().cast<py::array_t<float>>();
                        py::buffer_info buffer = np_array.request();
                        float* float_ptr = static_cast<float*>(buffer.ptr);

                        // 处理每个元素 - 防止非常小的值在bf16中变为0
                        std::vector<float> fixed_data(numel);
                        size_t zeroes_fixed = 0;

                        for (size_t i = 0; i < numel; ++i) {
                            float val = float_ptr[i];
                            // 如果值非常小（可能会被bf16表示为0），但不是精确的0
                            if (val != 0.0f && std::abs(val) < 0.001f) {
                                // 对于真实很小的正值，至少保证它们在bf16中有表示
                                // 小值中的0往往是强制量化为0，不需要处理
                                fixed_data[i] = val < 0 ? -0.001f : 0.001f;
                                zeroes_fixed++;
                            } else {
                                fixed_data[i] = val;
                            }
                        }

                        // 转换为bf16
                        for (size_t i = 0; i < numel; ++i) {
                            data.push_back(__nv_bfloat16(fixed_data[i]));
                        }
                    } else {
                        // 普通情况，使用标准转换流程
                        py::array_t<float> np_array = float_tensor.attr("numpy")().cast<py::array_t<float>>();
                        py::buffer_info buffer = np_array.request();
                        float* float_ptr = static_cast<float*>(buffer.ptr);

                        for (size_t i = 0; i < numel; ++i) {
                            data.push_back(__nv_bfloat16(float_ptr[i]));
                        }
                    }
                } else {
                    // 未知的2字节类型，也转换为float再处理
                    py::object float_tensor = cpu_tensor.attr("to")(torch_module.attr("float"));

                    // 转为numpy再获取数据
                    py::array_t<float> np_array = float_tensor.attr("numpy")().cast<py::array_t<float>>();
                    py::buffer_info buffer = np_array.request();
                    float* float_ptr = static_cast<float*>(buffer.ptr);

                    for (size_t i = 0; i < numel; ++i) {
                        data.push_back(__nv_bfloat16(float_ptr[i]));
                    }
                }
            } else {
                // 如果不是2字节元素，先转换为float再处理
                py::object float_tensor = cpu_tensor.attr("to")(torch_module.attr("float"));

                // 转为numpy再获取数据
                py::array_t<float> np_array = float_tensor.attr("numpy")().cast<py::array_t<float>>();
                py::buffer_info buffer = np_array.request();
                float* float_ptr = static_cast<float*>(buffer.ptr);

                for (size_t i = 0; i < numel; ++i) {
                    data.push_back(__nv_bfloat16(float_ptr[i]));
                }
            }
        } else {
            // 备选方案：使用循环直接访问每个元素
            std::cerr << "Warning: Using fallback element-wise access for conversion" << std::endl;
            // 转为float32
            py::object float_tensor = cpu_tensor.attr("to")(torch_module.attr("float"));
            for (size_t i = 0; i < numel; ++i) {
                // 使用索引操作访问每个元素
                py::object item = float_tensor.attr("flatten")()[py::int_(i)];
                float value = item.cast<float>();
                data.push_back(__nv_bfloat16(value));
            }
        }

        return Tensor<__nv_bfloat16>(std::move(data), shape);
    } catch (const std::exception& e) {
        std::cerr << "Exception in convert_bf16_tensor: " << e.what() << std::endl;
        throw;
    }
}

/**
 * 将 PyTorch 张量转换为 float 类型的 Tensor
 * @param tensor PyTorch 张量对象
 * @return 转换后的 float Tensor
 */
inline Tensor<float> convert_float_tensor(const py::object& tensor) {
    try {
        // 确保张量在CPU上，便于数据访问
        py::object cpu_tensor = tensor.attr("detach")().attr("cpu")();

        // 转换为numpy数组
        py::array_t<float> np_array = cpu_tensor.attr("numpy")().cast<py::array_t<float>>();

        // 获取形状
        std::vector<size_t> shape;
        for (int i = 0; i < np_array.ndim(); i++) {
            shape.push_back(np_array.shape(i));
        }

        // 复制数据
        std::vector<float> data(np_array.data(), np_array.data() + np_array.size());

        return Tensor<float>(std::move(data), shape);
    } catch (const std::exception& e) {
        std::cerr << "Exception in convert_float_tensor: " << e.what() << std::endl;
        throw;
    }
}

/**
 * 更新进度条
 * @param key 源键名
 * @param dst_key 目标键名
 */
inline void update_progress(const std::string& key, const std::string& dst_key) {
    if (!progress_initialized) {
        return;
    }

    // 更新处理的权重数
    processed_weights++;

    // 计算进度百分比
    float percentage = static_cast<float>(processed_weights) / total_weights * 100.0f;
    int bar_width = static_cast<int>(percentage / 2.0f);

    // 清除当前行
    std::cout << "\r";

    // 打印进度条
    std::cout << "进度: [";
    std::cout << std::string(bar_width, '=');
    if (bar_width < 50) {
        std::cout << ">";
        std::cout << std::string(49 - bar_width, ' ');
    } else {
        std::cout << "=";
    }
    std::cout << "] " << std::fixed << std::setprecision(1) << percentage << "%";

    // 可选：打印当前处理的键
    // if (key.length() > 30) {
    //   std::cout << " " << key.substr(0, 27) << "...";
    // } else {
    //   std::cout << " " << key;
    // }

    std::cout << std::flush;
}

/**
 * 打印权重处理进度信息
 * @param key 源键名
 * @param dst_key 目标键名
 */
inline void print_processing_info(const std::string& key, const std::string& dst_key) {
    if (progress_initialized) {
        // 如果进度条已初始化，则更新进度
        update_progress(key, dst_key);
    } else {
        // 否则使用简单的打印
        std::cout << "Processing key: " << key << " -> " << dst_key << std::endl;
    }
}

/**
 * 初始化进度条
 * @param total_weights 权重总数
 * @param model_type 模型类型描述
 */
inline void init_progress(size_t total_weight_count, const std::string& model_type) {
    // 如果上一次进度条没有正确完成，先强制完成它
    if (progress_initialized) {
        std::cout << "\r进度: [" << std::string(50, '=') << "] 100%";
        std::cout << "\n\033[1;32m✓ 上一次权重处理已强制完成!\033[0m\n" << std::endl;
    }

    // 重置所有状态变量
    total_weights = total_weight_count;
    processed_weights = 0;
    total_params_count = 0;  // 重置参数计数
    current_model_type = model_type;
    progress_initialized = true;

    // 打印进度条标题
    std::cout << "\n\033[1;36m处理 " << model_type << " 模型权重\033[0m" << std::endl;
    std::cout << "总权重数: " << total_weights << std::endl;
    std::cout << "进度: [" << std::string(50, ' ') << "] 0%" << std::flush;
}

/**
 * 完成进度条
 */
inline void finish_progress() {
    if (!progress_initialized) {
        return;
    }

    // 打印完成信息
    std::cout << "\r进度: [" << std::string(50, '=') << "] 100%";
    std::cout << "\n\033[1;32m✓ 权重处理完成!\033[0m" << std::endl;

    // 打印总参数量信息
    if (total_params_count > 0) {
        double params_in_millions = static_cast<double>(total_params_count) / 1000000.0;
        std::cout << "总参数量: " << std::fixed << std::setprecision(2) << params_in_millions << " 百万 ("
                  << total_params_count << " 个参数)\n"
                  << std::endl;
    } else {
        std::cout << std::endl;
    }

    // 重置进度条状态
    progress_initialized = false;
    processed_weights = 0;
    total_weights = 0;
    total_params_count = 0;
    current_model_type = "";
}

/**
 * 更新参数计数
 * @param count 需要增加的参数数量
 */
inline void update_params_count(size_t count) {
    total_params_count += count;
}

}  // namespace weight_processor_utils