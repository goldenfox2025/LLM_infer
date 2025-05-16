#include "model_initializer.hpp"

#include <iostream>
#include <stdexcept>

#include "CudaMemoryPool.hpp"
#include "device_manager.hpp"
#include "include/weight_processor.hpp"
#include "model_factory.hpp"

// 打印配置和权重信息
void ModelInitializer::print_config_and_weights_info(py::dict config,
                                                     py::dict weights) {
  std::cout << "\n===== Configuration Items =====" << std::endl;
  for (const auto& item : config) {
    std::string key = py::str(item.first).cast<std::string>();
    std::string type_str = py::str(item.second.get_type()).cast<std::string>();
    std::cout << "Config key: " << key << ", type: " << type_str << std::endl;
  }
  std::cout << "\n===== Weight Items =====" << std::endl;
  for (const auto& item : weights) {
    std::string key = py::str(item.first).cast<std::string>();
    std::string type_str = py::str(item.second.get_type()).cast<std::string>();
    std::cout << "Weight key: " << key << ", type: " << type_str << std::endl;
  }
}

// 构建基础配置
std::unordered_map<std::string, int> ModelInitializer::build_base_config(
    py::dict config) {
  std::unordered_map<std::string, int> cpp_config;
  cpp_config["vocab_size"] = config["vocab_size"].cast<int>();
  cpp_config["hidden_size"] = config["hidden_size"].cast<int>();
  cpp_config["max_position_embeddings"] =
      config["max_position_embeddings"].cast<int>();
  cpp_config["bos_token_id"] = config["bos_token_id"].cast<int>();
  cpp_config["eos_token_id"] = config["eos_token_id"].cast<int>();
  return cpp_config;
}

// 初始化 Llama 模型
bool ModelInitializer::init_llama_model(py::dict config, py::dict weights,
                                        std::shared_ptr<BaseModel>& model,
                                        std::unique_ptr<infer_base>& engine) {
  try {
    // 构建配置
    std::unordered_map<std::string, int> cpp_config = build_base_config(config);

    // Llama 专有配置
    cpp_config["num_hidden_layers"] = config["num_hidden_layers"].cast<int>();
    cpp_config["num_attention_heads"] =
        config["num_attention_heads"].cast<int>();
    cpp_config["num_key_value_heads"] =
        config["num_key_value_heads"].cast<int>();
    cpp_config["intermediate_size"] = config["intermediate_size"].cast<int>();
    cpp_config["rms_norm_eps"] =
        static_cast<int>(config["rms_norm_eps"].cast<float>());
    cpp_config["rope_theta"] =
        static_cast<int>(config["rope_theta"].cast<float>());

    // 处理 Llama 权重（FP32）
    auto cpp_weights_fp32 = weight_processor::process_llama_weights(weights);

    // 创建 Llama 模型（根据默认设备配置）
    Device default_device = DeviceManager::instance().getDefaultDevice();
    model = ModelFactory::create_model(ModelType::LLAMA, cpp_weights_fp32,
                                       cpp_config);

    // 如果默认设备是CUDA，则移动模型到CUDA
    if (default_device == Device::CUDA) {
      model->cuda();
    }

    engine = std::make_unique<InferenceEngine<float>>(model, default_device);
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Error initializing Llama model: " << e.what() << std::endl;
    return false;
  }
}

// 初始化 Qwen FP32 模型
bool ModelInitializer::init_qwen_fp32_model(
    py::dict config, py::dict weights, std::shared_ptr<BaseModel>& model,
    std::unique_ptr<infer_base>& engine) {
  try {
    // 构建配置
    std::unordered_map<std::string, int> cpp_config = build_base_config(config);

    // Qwen FP32 配置
    cpp_config["n_layers"] = config["num_hidden_layers"].cast<int>();
    cpp_config["n_heads"] = config["num_attention_heads"].cast<int>();
    cpp_config["n_kv_heads"] = config["num_key_value_heads"].cast<int>();
    cpp_config["intermediate_size"] = config["intermediate_size"].cast<int>();
    cpp_config["rms_norm_eps"] =
        static_cast<int>(config["rms_norm_eps"].cast<float>());
    cpp_config["rope_theta"] =
        static_cast<int>(config["rope_theta"].cast<float>());

    // 处理 Qwen FP32 权重
    auto cpp_weights_fp32 =
        weight_processor::process_qwen_weights_fp32(weights);
    model = ModelFactory::create_model(ModelType::QWEN, cpp_weights_fp32,
                                       cpp_config);

    // 获取默认设备
    Device default_device = DeviceManager::instance().getDefaultDevice();

    // Qwen 模型目前仅支持 CUDA，如果默认设备是CPU，发出警告并强制使用CUDA
    if (default_device == Device::CPU) {
      std::cerr
          << "Warning: Qwen model currently only supports CUDA execution. "
          << "Forcing CUDA device despite CPU being requested." << std::endl;
      default_device = Device::CUDA;
    }

    model->cuda();
    engine = std::make_unique<InferenceEngine<float>>(model, default_device);
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Error initializing Qwen FP32 model: " << e.what()
              << std::endl;
    return false;
  }
}

// 初始化 Qwen BF16 模型
bool ModelInitializer::init_qwen_bf16_model(
    py::dict config, py::dict weights, std::shared_ptr<BaseModel>& model,
    std::unique_ptr<infer_base>& engine) {
  try {
    // 构建配置
    std::unordered_map<std::string, int> cpp_config = build_base_config(config);

    // Qwen BF16 专有配置
    cpp_config["n_layers"] = config["num_hidden_layers"].cast<int>();
    cpp_config["n_heads"] = config["num_attention_heads"].cast<int>();
    cpp_config["n_kv_heads"] = config["num_key_value_heads"].cast<int>();
    cpp_config["intermediate_size"] = config["intermediate_size"].cast<int>();
    cpp_config["rms_norm_eps"] =
        static_cast<int>(config["rms_norm_eps"].cast<float>());
    cpp_config["rope_theta"] =
        static_cast<int>(config["rope_theta"].cast<float>());

    // 处理 Qwen BF16 权重（输入必须为 PyTorch bf16 类型）
    auto cpp_weights_bf16 =
        weight_processor::process_qwen_weights_bf16(weights);

    // 获取默认设备
    Device default_device = DeviceManager::instance().getDefaultDevice();

    // Qwen 模型目前仅支持 CUDA，如果默认设备是CPU，发出警告并强制使用CUDA
    if (default_device == Device::CPU) {
      std::cerr
          << "Warning: Qwen BF16 model currently only supports CUDA execution. "
          << "Forcing CUDA device despite CPU being requested." << std::endl;
      default_device = Device::CUDA;
    }

    model = ModelFactory::create_model_bf16(ModelType::QWEN_BF16,
                                            cpp_weights_bf16, cpp_config);
    engine =
        std::make_unique<InferenceEngine<__nv_bfloat16>>(model, default_device);
    model->cuda();
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Error initializing Qwen BF16 model: " << e.what()
              << std::endl;
    return false;
  }
}

// 初始化 Qwen AWQ 模型
bool ModelInitializer::init_qwen_awq_model(
    py::dict config, py::dict weights, std::shared_ptr<BaseModel>& model,
    std::unique_ptr<infer_base>& engine) {
  try {
    // 构建配置
    std::unordered_map<std::string, int> cpp_config = build_base_config(config);

    // Qwen AWQ 配置
    cpp_config["n_layers"] = config["num_hidden_layers"].cast<int>();
    cpp_config["n_heads"] = config["num_attention_heads"].cast<int>();
    cpp_config["n_kv_heads"] = config["num_key_value_heads"].cast<int>();
    cpp_config["intermediate_size"] = config["intermediate_size"].cast<int>();
    cpp_config["rms_norm_eps"] =
        static_cast<int>(config["rms_norm_eps"].cast<float>());
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

    // 处理 Qwen AWQ 权重
    auto [bf16_weights, qweight_params, scales_params, qzeros_params] =
        weight_processor::process_qwen_weights_awq(weights);

    // 获取默认设备
    Device default_device = DeviceManager::instance().getDefaultDevice();

    // Qwen 模型目前仅支持 CUDA，如果默认设备是CPU，发出警告并强制使用CUDA
    if (default_device == Device::CPU) {
      std::cerr
          << "Warning: Qwen AWQ model currently only supports CUDA execution. "
          << "Forcing CUDA device despite CPU being requested." << std::endl;
      default_device = Device::CUDA;
    }

    // 创建带量化参数的模型
    model = ModelFactory::create_model_quantized(
        ModelType::QWEN_AWQ, bf16_weights, qweight_params, scales_params,
        qzeros_params, cpp_config);

    engine =
        std::make_unique<InferenceEngine<__nv_bfloat16>>(model, default_device);
    model->cuda();
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Error initializing Qwen AWQ model: " << e.what() << std::endl;
    return false;
  }
}

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
        static_cast<int>(config["rms_norm_eps"].cast<float>());
    cpp_config["rope_theta"] =
        static_cast<int>(config["rope_theta"].cast<float>());

    // 如果配置中有head_dim，则使用配置中的值
    if (config.contains("head_dim")) {
      cpp_config["head_dim"] = config["head_dim"].cast<int>();
      std::cout << "使用配置中的head_dim: " << cpp_config["head_dim"]
                << std::endl;
    }

    // 处理 Qwen3 BF16 权重（输入必须为 PyTorch bf16 类型）
    auto cpp_weights_bf16 =
        weight_processor::process_qwen3_weights_bf16(weights);

    // 获取默认设备
    Device default_device = DeviceManager::instance().getDefaultDevice();

    // Qwen3 模型目前仅支持 CUDA，如果默认设备是CPU，发出警告并强制使用CUDA
    if (default_device == Device::CPU) {
      std::cerr << "Warning: Qwen3 BF16 model currently only supports CUDA "
                   "execution. "
                << "Forcing CUDA device despite CPU being requested."
                << std::endl;
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
        static_cast<int>(config["rms_norm_eps"].cast<float>());
    cpp_config["rope_theta"] =
        static_cast<int>(config["rope_theta"].cast<float>());

    // 如果配置中有head_dim，则使用配置中的值
    if (config.contains("head_dim")) {
      cpp_config["head_dim"] = config["head_dim"].cast<int>();
      std::cout << "使用配置中的head_dim: " << cpp_config["head_dim"]
                << std::endl;
    }

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
        weight_processor::process_qwen3_weights_awq(weights);

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
    std::cerr << "Error initializing Qwen3 AWQ model: " << e.what()
              << std::endl;
    return false;
  }
}

// 初始化 CUDA 内存池
bool ModelInitializer::init_cuda_memory_pool(
    const std::unordered_map<std::string, int>& config) {
  try {
    // 获取模型配置参数
    size_t hidden_dim = config.at("hidden_size");

    // 获取最大序列长度，如果配置中有的话
    size_t seq_len = 32;  // 默认序列长度
    if (config.find("max_position_embeddings") != config.end()) {
      seq_len = std::min(
          seq_len, static_cast<size_t>(config.at("max_position_embeddings")));
    }

    // 检查GPU内存状态
    size_t free_memory = 0, total_memory = 0;
    cudaError_t err = cudaMemGetInfo(&free_memory, &total_memory);
    if (err == cudaSuccess) {
      std::cout << "Current GPU memory: " << (free_memory / (1024 * 1024))
                << " MB free, " << (total_memory / (1024 * 1024)) << " MB total"
                << std::endl;

      // 只有当可用内存超过1GB时才开启prefill模式
      if (free_memory > 1024 * 1024 * 1024) {
        // 计算prefill内存大小，使用可用内存的10%，但不超过256MB
        size_t prefill_size =
            std::min(free_memory / 10, static_cast<size_t>(256 * 1024 * 1024));

        // 开启prefill模式
        std::cout << "Enabling prefill mode with initial buffer size: "
                  << (prefill_size / (1024 * 1024)) << " MB" << std::endl;
        GlobalCudaMemoryPool::enable_prefill_mode(prefill_size);
      } else {
        std::cout << "Skipping prefill mode due to low available memory"
                  << std::endl;
      }
    }
    return true;
  } catch (const std::exception& e) {
    // 如果prefill模式初始化失败，只打印警告，不影响模型加载
    std::cerr << "Warning: Failed to initialize prefill mode: " << e.what()
              << std::endl;
    return false;
  }
}
