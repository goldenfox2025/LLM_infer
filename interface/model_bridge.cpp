// model_bridge.cpp
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "CudaMemoryPool.hpp"
#include "base_model.hpp"
#include "device_manager.hpp"
#include "inference.hpp"
#include "model_factory.hpp"
#include "model_initializer.hpp"
#include "weight_processor.hpp"

namespace py = pybind11;
class infer_base;

// 使用 model_factory.hpp 中定义的 ModelType 和 ModelFactory

// 全局模型与推理引擎实例 便于维护生命周期
std::shared_ptr<BaseModel> g_model;
std::unique_ptr<infer_base> g_engine;

bool init_model(py::dict config, py::dict weights,
                const std::string& model_type) {
  try {
    // 打印配置与权重调试信息
    ModelInitializer::print_config_and_weights_info(config, weights);

    // 根据模型类型初始化不同的模型
    bool result = false;

    if (model_type == "llama") {
      result = ModelInitializer::init_llama_model(config, weights, g_model,
                                                  g_engine);
    } else if (model_type == "qwen") {
      result = ModelInitializer::init_qwen_fp32_model(config, weights, g_model,
                                                      g_engine);
    } else if (model_type == "qwen_bf16") {
      result = ModelInitializer::init_qwen_bf16_model(config, weights, g_model,
                                                      g_engine);
    } else if (model_type == "qwen_awq") {
      result = ModelInitializer::init_qwen_awq_model(config, weights, g_model,
                                                     g_engine);
    } else {
      throw std::runtime_error("Unsupported model type: " + model_type);
    }

    if (result) {
      // 打印模型信息
      g_model->print_model_info();

      // 构建基础配置用于初始化CUDA内存池
      std::unordered_map<std::string, int> cpp_config =
          ModelInitializer::build_base_config(config);

      // 添加模型特定的配置
      if (model_type == "llama") {
        cpp_config["num_hidden_layers"] =
            config["num_hidden_layers"].cast<int>();
      } else {
        cpp_config["n_layers"] = config["num_hidden_layers"].cast<int>();
      }

      // 初始化CUDA内存池
      ModelInitializer::init_cuda_memory_pool(cpp_config);
    }

    return result;
  } catch (const std::exception& e) {
    std::cerr << "Error initializing model: " << e.what() << std::endl;
    return false;
  }
}

// 每生成一个 token 时通过回调返回

void generate_text_stream(const std::vector<uint32_t>& input_ids,
                          py::function callback, size_t max_length = 100,
                          float temperature = 1.0f, float top_p = 0.9f,
                          size_t top_k = 50) {
  if (!g_engine) {
    throw std::runtime_error("Model not initialized");
  }

  g_engine->generate_with_callback(input_ids, max_length, temperature, top_p,
                                   top_k, [callback](uint32_t token) {
                                     py::gil_scoped_acquire acquire;
                                     callback(token);
                                   });
}

// 设置默认设备
bool set_default_device(const std::string& device_str) {
  try {
    Device device;
    if (device_str == "cuda" || device_str == "CUDA") {
      device = Device::CUDA;
      // 检查CUDA是否可用
      if (!DeviceManager::instance().isCudaAvailable()) {
        std::cerr << "CUDA requested but not available. Falling back to CPU."
                  << std::endl;
        device = Device::CPU;
      }
    } else if (device_str == "cpu" || device_str == "CPU") {
      device = Device::CPU;
    } else {
      std::cerr << "Invalid device: " << device_str
                << ". Valid options are 'cuda' or 'cpu'." << std::endl;
      return false;
    }

    DeviceManager::instance().setDefaultDevice(device);
    std::cout << "Default device set to: "
              << (device == Device::CUDA ? "CUDA" : "CPU") << std::endl;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Error setting default device: " << e.what() << std::endl;
    return false;
  }
}

// 获取当前默认设备
std::string get_default_device() {
  Device device = DeviceManager::instance().getDefaultDevice();
  return (device == Device::CUDA) ? "cuda" : "cpu";
}

// Pybind11 模块定义
PYBIND11_MODULE(model_bridge, m) {
  m.def("init_model", &init_model, py::arg("config"), py::arg("weights"),
        py::arg("model_type") = "llama", "Initialize and verify the model");
  m.def("generate_text_stream", &generate_text_stream, py::arg("input_ids"),
        py::arg("callback"), py::arg("max_length") = 100,
        py::arg("temperature") = 1.0f, py::arg("top_p") = 0.9f,
        py::arg("top_k") = 50, "Stream generated tokens via callback");
  m.def("set_default_device", &set_default_device, py::arg("device"),
        "Set the default device for model execution (cuda or cpu)");
  m.def("get_default_device", &get_default_device,
        "Get the current default device for model execution");
}
