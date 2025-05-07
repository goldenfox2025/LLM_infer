#include "model_initializer.hpp"

#include <iostream>
#include <stdexcept>

#include "CudaMemoryPool.hpp"
#include "device_manager.hpp"
#include "include/weight_processor.hpp"
#include "model_factory.hpp"

// 初始化 Qwen3 BF16 模型
bool ModelInitializer::init_qwen3_bf16_model(
    py::dict config, py::dict weights, std::shared_ptr<BaseModel>& model,
    std::unique_ptr<infer_base>& engine) {
  try {
    // 构建配置
    std::unordered_map<std::string, int> cpp_config = build_base_config(config);

    // Qwen3 BF16 专有配置
    cpp_config["n_layers"] = config["num_hidden_layers"].cast<int>();
    cpp_config["n_heads"] = config["num_attention_heads"].cast<int>();
    cpp_config["n_kv_heads"] = config["num_key_value_heads"].cast<int>();
    cpp_config["intermediate_size"] = config["intermediate_size"].cast<int>();
    cpp_config["rms_norm_eps"] =
        static_cast<int>(config["rms_norm_eps"].cast<float>() * 1000000.0f);
    cpp_config["rope_theta"] =
        static_cast<int>(config["rope_theta"].cast<float>());

    // 处理 Qwen3 BF16 权重（输入必须为 PyTorch bf16 类型）
    auto cpp_weights_bf16 = WeightProcessor::process_qwen3_weights_bf16(weights);

    // 获取默认设备
    Device default_device = DeviceManager::instance().getDefaultDevice();

    // Qwen3 模型目前仅支持 CUDA，如果默认设备是CPU，发出警告并强制使用CUDA
    if (default_device == Device::CPU) {
      std::cerr
          << "Warning: Qwen3 BF16 model currently only supports CUDA execution. "
          << "Forcing CUDA device despite CPU being requested." << std::endl;
      default_device = Device::CUDA;
    }

    model = ModelFactory::create_model_bf16(ModelType::QWEN3_BF16,
                                           cpp_weights_bf16, cpp_config);
    engine =
        std::make_unique<InferenceEngine<__nv_bfloat16>>(model, default_device);
    model->cuda();
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Error initializing Qwen3 BF16 model: " << e.what()
              << std::endl;
    return false;
  }
}

// 初始化 Qwen3 AWQ 模型
bool ModelInitializer::init_qwen3_awq_model(
    py::dict config, py::dict weights, std::shared_ptr<BaseModel>& model,
    std::unique_ptr<infer_base>& engine) {
  try {
    // 构建配置
    std::unordered_map<std::string, int> cpp_config = build_base_config(config);

    // Qwen3 AWQ 配置
    cpp_config["n_layers"] = config["num_hidden_layers"].cast<int>();
    cpp_config["n_heads"] = config["num_attention_heads"].cast<int>();
    cpp_config["n_kv_heads"] = config["num_key_value_heads"].cast<int>();
    cpp_config["intermediate_size"] = config["intermediate_size"].cast<int>();
    cpp_config["rms_norm_eps"] =
        static_cast<int>(config["rms_norm_eps"].cast<float>() * 1000000.0f);
    cpp_config["rope_theta"] =
        static_cast<int>(config["rope_theta"].cast<float>());

    // 设置量化类型和分组大小
    cpp_config["quant_type"] = 1;  // AWQ量化

    // 如果配置中有分组大小，则使用配置中的值
    if (config.contains("group_size")) {
      cpp_config["group_size"] = config["group_size"].cast<int>();
    } else {
      // 默认分组大小为128
      cpp_config["group_size"] = 128;
    }

    // 处理 Qwen3 AWQ 权重
    auto [bf16_weights, qweight_params, scales_params, qzeros_params] =
        WeightProcessor::process_qwen3_weights_awq(weights);

    // 获取默认设备
    Device default_device = DeviceManager::instance().getDefaultDevice();

    // Qwen3 模型目前仅支持 CUDA，如果默认设备是CPU，发出警告并强制使用CUDA
    if (default_device == Device::CPU) {
      std::cerr
          << "Warning: Qwen3 AWQ model currently only supports CUDA execution. "
          << "Forcing CUDA device despite CPU being requested." << std::endl;
      default_device = Device::CUDA;
    }

    // 创建带量化参数的模型
    model = ModelFactory::create_model_quantized(
        ModelType::QWEN3_AWQ, bf16_weights, qweight_params, scales_params,
        qzeros_params, cpp_config);

    engine =
        std::make_unique<InferenceEngine<__nv_bfloat16>>(model, default_device);
    model->cuda();
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Error initializing Qwen3 AWQ model: " << e.what() << std::endl;
    return false;
  }
}
