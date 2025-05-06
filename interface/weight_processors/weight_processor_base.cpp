#include "weight_processor_base.hpp"

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

// 初始化静态成员变量
size_t WeightProcessorBase::total_weights_ = 0;
size_t WeightProcessorBase::processed_weights_ = 0;
std::string WeightProcessorBase::current_model_type_ = "";
bool WeightProcessorBase::progress_initialized_ = false;

// 辅助函数：将 PyTorch 张量转换为 __nv_bfloat16 类型的 Tensor
Tensor<__nv_bfloat16> WeightProcessorBase::convert_bf16_tensor(
    const py::object& tensor) {
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

    if (py::hasattr(cpu_tensor, "element_size") &&
        py::hasattr(cpu_tensor, "data_ptr")) {
      // 这里使用pytorch的接口获取字节长度
      size_t element_size = cpu_tensor.attr("element_size")().cast<size_t>();

      // 确认是否为bfloat16类型 其实也可能是fp16 但先不管
      if (element_size == 2) {
        // 获取数据指针，这会返回一个整数，表示内存地址
        uintptr_t data_ptr = cpu_tensor.attr("data_ptr")().cast<uintptr_t>();
        const __nv_bfloat16* ptr =
            reinterpret_cast<const __nv_bfloat16*>(data_ptr);
        // 每个元素直接拷贝二进制数据
        for (size_t i = 0; i < numel; ++i) {
          __nv_bfloat16 bits = ptr[i];
          data.push_back(bits);
        }
      } else {
        // 如果不是2字节元素，先转换为float再处理 以防万一
        // std::cerr
        //     << "Warning: Input tensor is not bfloat16, converting through
        //     float"
        //     << std::endl;

        py::object float_tensor =
            cpu_tensor.attr("to")(torch_module.attr("float"));

        // 转为numpy再获取数据
        py::array_t<float> np_array =
            float_tensor.attr("numpy")().cast<py::array_t<float>>();
        py::buffer_info buffer = np_array.request();
        float* float_ptr = static_cast<float*>(buffer.ptr);

        for (size_t i = 0; i < numel; ++i) {
          data.push_back(__nv_bfloat16(float_ptr[i]));
        }
      }
    } else {
      // 备选方案：使用循环直接访问每个元素
      std::cerr << "Warning: Using fallback element-wise access for conversion"
                << std::endl;
      // 转为float32
      py::object float_tensor =
          cpu_tensor.attr("to")(torch_module.attr("float"));
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

// 辅助函数：将 PyTorch 张量转换为 float 类型的 Tensor
Tensor<float> WeightProcessorBase::convert_float_tensor(
    const py::object& tensor) {
  try {
    // 确保张量在CPU上，便于数据访问
    py::object cpu_tensor = tensor.attr("detach")().attr("cpu")();

    // 转换为numpy数组
    py::array_t<float> np_array =
        cpu_tensor.attr("numpy")().cast<py::array_t<float>>();

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

// 辅助函数：从 PyTorch 张量提取形状信息
std::vector<size_t> WeightProcessorBase::get_tensor_shape(
    const py::object& tensor) {
  py::tuple shape_tuple = tensor.attr("shape");
  std::vector<size_t> shape;
  for (size_t i = 0; i < py::len(shape_tuple); ++i) {
    shape.push_back(shape_tuple[i].cast<size_t>());
  }
  return shape;
}

// 辅助函数：打印权重处理进度
void WeightProcessorBase::print_processing_info(const std::string& key,
                                                const std::string& dst_key) {
  if (progress_initialized_) {
    // 如果进度条已初始化，则更新进度
    update_progress(key, dst_key);
  } else {
    // 否则使用简单的打印
    std::cout << "Processing key: " << key << " -> " << dst_key << std::endl;
  }
}

// 初始化进度条
void WeightProcessorBase::init_progress(size_t total_weights,
                                        const std::string& model_type) {
  // 如果上一次进度条没有正确完成，先强制完成它
  if (progress_initialized_) {
    std::cout << "\r进度: [" << std::string(50, '=') << "] 100%";
    std::cout << "\n\033[1;32m✓ 上一次权重处理已强制完成!\033[0m\n"
              << std::endl;
  }

  // 重置所有状态变量
  total_weights_ = total_weights;
  processed_weights_ = 0;
  current_model_type_ = model_type;
  progress_initialized_ = true;

  // 打印进度条标题
  std::cout << "\n\033[1;36m处理 " << model_type << " 模型权重\033[0m"
            << std::endl;
  std::cout << "总权重数: " << total_weights << std::endl;
  std::cout << "进度: [" << std::string(50, ' ') << "] 0%" << std::flush;
}

// 更新进度条
void WeightProcessorBase::update_progress(const std::string& key,
                                          const std::string& dst_key) {
  if (!progress_initialized_) {
    return;
  }

  // 更新处理的权重数
  processed_weights_++;

  // 计算进度百分比
  float percentage =
      static_cast<float>(processed_weights_) / total_weights_ * 100.0f;
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

  // 打印当前处理的键
  //   if (key.length() > 30) {
  //     std::cout << " " << key.substr(0, 27) << "...";
  //   } else {
  //     std::cout << " " << key;
  //   }

  std::cout << std::flush;
}

// 完成进度条
void WeightProcessorBase::finish_progress() {
  if (!progress_initialized_) {
    return;
  }

  // 打印完成信息
  std::cout << "\r进度: [" << std::string(50, '=') << "] 100%";
  std::cout << "\n\033[1;32m✓ 权重处理完成!\033[0m\n" << std::endl;

  // 重置进度条状态
  progress_initialized_ = false;
  processed_weights_ = 0;
  total_weights_ = 0;
  current_model_type_ = "";
}
