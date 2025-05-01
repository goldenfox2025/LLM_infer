// model_bridge.cpp
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "base_model.hpp"
#include "inference.hpp"
#include "llama.hpp"
#include "qwen.hpp"
#include "CudaMemoryPool.hpp"
#include <iostream>

namespace py = pybind11;
class infer_base;

// 模型类型枚举
enum class ModelType {
  LLAMA,
  QWEN,
  QWEN_BF16,
  QWEN_AWQ,
};

// ModelFactory 用于根据模型类型创建对应实例
class ModelFactory {
 public:
  static std::shared_ptr<BaseModel> create_model(
      ModelType type,
      const std::unordered_map<std::string, Tensor<float>>& weights,
      const std::unordered_map<std::string, int>& config) {
    switch (type) {
      case ModelType::LLAMA: {
        auto model = std::make_shared<LlamaModel>(weights, config);
        model->print_model_info();
        if (!model->verify_params()) {
          throw std::runtime_error("Model parameter verification failed");
        }
        return model;
      }
      case ModelType::QWEN: {
        auto model = std::make_shared<QwenModel<float>>(weights, config);
        model->print_model_info();
        if (!model->verify_params()) {
          throw std::runtime_error("Model parameter verification failed");
        }
        return model;
      }
      default:
        throw std::runtime_error(
            "Unsupported model type for FP32 weights in create_model");
    }
  }

  // 创建带量化参数的模型
  static std::shared_ptr<BaseModel> create_model_quantized(
      ModelType type,
      const std::unordered_map<std::string, Tensor<__nv_bfloat16>>& weights,
      const std::unordered_map<std::string, Tensor<int32_t>>& qweight_params,
      const std::unordered_map<std::string, Tensor<float>>& scales_params,
      const std::unordered_map<std::string, Tensor<int32_t>>& qzeros_params,
      const std::unordered_map<std::string, int>& config) {
    switch (type) {
      case ModelType::QWEN_AWQ: {
        auto model = std::make_shared<QwenModel<__nv_bfloat16>>(
            weights, qweight_params, scales_params, qzeros_params, config);
        model->print_model_info();
        if (!model->verify_params()) {
          throw std::runtime_error("Model parameter verification failed");
        }
        return model;
      }
      default:
        throw std::runtime_error(
            "Unsupported model type for quantized weights in create_model_quantized");
    }
  }
};

// 全局模型与推理引擎实例 便于维护生命周期
std::shared_ptr<BaseModel> g_model;
std::unique_ptr<infer_base> g_engine;

// 辅助函数：将 PyTorch 张量转换为 __nv_bfloat16 类型的 Tensor

Tensor<__nv_bfloat16> convert_bf16_tensor(const py::object& tensor) {
  try {
    py::object torch_module = py::module::import("torch");

    // 确保张量在CPU上，便于数据访问
    // attr是一种运行时反射
    py::object cpu_tensor = tensor.attr("detach")().attr("cpu")();

    // 获取形状
    py::tuple shape_tuple = cpu_tensor.attr("shape");
    std::vector<size_t> shape;
    for (size_t i = 0; i < py::len(shape_tuple); ++i) {
      shape.push_back(shape_tuple[i].cast<size_t>());
    }

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
        std::cerr
            << "Warning: Input tensor is not bfloat16, converting through float"
            << std::endl;

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

//
// 辅助函数：处理 Llama 模型权重（FP32）
//
std::unordered_map<std::string, Tensor<float>> process_llama_weights(
    const py::dict& weights) {
  std::unordered_map<std::string, Tensor<float>> cpp_weights;

  // 全局权重映射：embedding、归一化、lm_head
  const std::unordered_map<std::string, std::string> key_mapping = {
      {"model.embed_tokens.weight", "embedding_table"},
      {"model.norm.weight", "rms_out_w"},
      {"lm_head.weight", "lm_head"}};

  // 如果没有 embedding_table，则使用 lm_head 的权重作为 embedding
  if (!weights.contains("model.embed_tokens.weight") &&
      weights.contains("lm_head.weight")) {
    py::array_t<float> np_array =
        weights["lm_head.weight"].cast<py::array_t<float>>();
    std::vector<size_t> shape;
    for (int i = 0; i < np_array.ndim(); i++) {
      shape.push_back(np_array.shape(i));
    }
    std::cout << "No embedding_table found, using lm_head as embedding"
              << std::endl;
    std::vector<float> data(np_array.data(), np_array.data() + np_array.size());
    cpp_weights.emplace("embedding_table",
                        Tensor<float>(std::move(data), shape));
  }
  // 处理全局权重
  for (const auto& [src_key, dst_key] : key_mapping) {
    if (weights.contains(src_key)) {
      std::cout << "Processing key: " << src_key << " -> " << dst_key
                << std::endl;
      py::array_t<float> np_array =
          weights[src_key.c_str()].cast<py::array_t<float>>();
      std::vector<size_t> shape;
      for (int i = 0; i < np_array.ndim(); i++) {
        shape.push_back(np_array.shape(i));
      }
      std::vector<float> data(np_array.data(),
                              np_array.data() + np_array.size());
      if (dst_key == "lm_head") {
        cpp_weights.emplace(
            dst_key, Tensor<float>(std::move(data), shape).transpose(-1, -2));
      } else {
        cpp_weights.emplace(dst_key, Tensor<float>(std::move(data), shape));
      }
    }
  }
  // 处理层级权重：通过遍历 weights 中所有以 "model.layers." 开头的键进行处理
  const std::vector<std::pair<std::string, std::string>> layer_key_mapping = {
      {"input_layernorm.weight", "rms_att_w"},
      {"post_attention_layernorm.weight", "rms_ffn_w"},
      {"self_attn.q_proj.weight", "wq"},
      {"self_attn.k_proj.weight", "wk"},
      {"self_attn.v_proj.weight", "wv"},
      {"self_attn.o_proj.weight", "wo"},
      {"mlp.up_proj.weight", "w_up"},
      {"mlp.down_proj.weight", "w_down"},
      {"mlp.gate_proj.weight", "w_gate"}};
  for (auto item : weights) {
    std::string key = py::str(item.first).cast<std::string>();
    if (key.find("model.layers.") == 0) {
      for (const auto& [src_suffix, dst_prefix] : layer_key_mapping) {
        std::string pattern = "." + src_suffix;
        if (key.find(pattern) != std::string::npos) {
          // 提取层索引，构造目标键名
          size_t start = std::string("model.layers.").size();
          size_t end = key.find('.', start);
          std::string layer_str = key.substr(start, end - start);
          int layer = std::stoi(layer_str);
          std::string dst_key = dst_prefix + std::to_string(layer);
          std::cout << "Processing layer key: " << key << " -> " << dst_key
                    << std::endl;
          py::array_t<float> np_array = item.second.cast<py::array_t<float>>();
          std::vector<size_t> shape;
          for (int i = 0; i < np_array.ndim(); i++) {
            shape.push_back(np_array.shape(i));
          }
          std::vector<float> data(np_array.data(),
                                  np_array.data() + np_array.size());
          // 对部分矩阵需要转置 但本质上没有转置
          // 这里是为了适配cpu算子（支持stride） 然而cuda没有支持
          if (dst_prefix == "wq" || dst_prefix == "wk" || dst_prefix == "wv" ||
              dst_prefix == "wo" || dst_prefix == "w_up" ||
              dst_prefix == "w_down" || dst_prefix == "w_gate") {
            cpp_weights.emplace(
                dst_key,
                Tensor<float>(std::move(data), shape).transpose(-1, -2));
          } else {
            cpp_weights.emplace(dst_key, Tensor<float>(std::move(data), shape));
          }
        }
      }
    }
  }
  return cpp_weights;
}

// 辅助函数：处理 Qwen 模型权重（FP32）

std::unordered_map<std::string, Tensor<float>> process_qwen_weights_fp32(
    const py::dict& weights) {
  std::unordered_map<std::string, Tensor<float>> cpp_weights;
  // 全局权重映射
  const std::unordered_map<std::string, std::string> qwen_key_mapping = {
      {"model.embed_tokens.weight", "token_embeddings.weight"},
      {"model.norm.weight", "norm.weight"},
      {"lm_head.weight", "lm_head"},
      {"transformer.wte.weight", "token_embeddings.weight"},
      {"transformer.ln_f.weight", "norm.weight"}};
  for (const auto& [src_key, dst_key] : qwen_key_mapping) {
    if (weights.contains(src_key)) {
      std::cout << "Processing Qwen FP32 key: " << src_key << " -> " << dst_key
                << std::endl;
      py::array_t<float> np_array =
          weights[src_key.c_str()].cast<py::array_t<float>>();
      std::vector<size_t> shape;
      for (int i = 0; i < np_array.ndim(); i++) {
        shape.push_back(np_array.shape(i));
      }
      std::vector<float> data(np_array.data(),
                              np_array.data() + np_array.size());
      if (dst_key == "lm_head") {
        cpp_weights.emplace(
            dst_key, Tensor<float>(std::move(data), shape).transpose(-1, -2));
      } else {
        cpp_weights.emplace(dst_key, Tensor<float>(std::move(data), shape));
      }
    }
  }
  // 处理层级权重
  const std::vector<std::pair<std::string, std::string>>
      qwen_layer_key_mapping = {
          {"input_layernorm.weight", "input_layernorm.weight"},
          {"post_attention_layernorm.weight",
           "post_attention_layernorm.weight"},
          {"self_attn.q_proj.weight", "self_attn.q_proj.weight"},
          {"self_attn.k_proj.weight", "self_attn.k_proj.weight"},
          {"self_attn.v_proj.weight", "self_attn.v_proj.weight"},
          {"self_attn.o_proj.weight", "self_attn.o_proj.weight"},
          {"mlp.gate_proj.weight", "mlp.gate_proj.weight"},
          {"mlp.up_proj.weight", "mlp.up_proj.weight"},
          {"mlp.down_proj.weight", "mlp.down_proj.weight"}};
  const std::vector<std::pair<std::string, std::string>>
      qwen_layer_bias_mapping = {
          {"self_attn.q_proj.bias", "self_attn.q_proj.bias"},
          {"self_attn.k_proj.bias", "self_attn.k_proj.bias"},
          {"self_attn.v_proj.bias", "self_attn.v_proj.bias"},
          {"self_attn.o_proj.bias", "self_attn.o_proj.bias"}};

  for (auto item : weights) {
    std::string key = py::str(item.first).cast<std::string>();
    if (key.find("model.layers.") == 0) {
      for (const auto& [src_suffix, dst_suffix] : qwen_layer_key_mapping) {
        std::string pattern = "." + src_suffix;
        if (key.find(pattern) != std::string::npos) {
          size_t start = std::string("model.layers.").size();
          size_t end = key.find('.', start);
          std::string layer_str = key.substr(start, end - start);
          int layer = std::stoi(layer_str);
          std::string dst_key =
              "layers." + std::to_string(layer) + "." + dst_suffix;
          std::cout << "Processing Qwen FP32 layer key: " << key << " -> "
                    << dst_key << std::endl;
          py::array_t<float> np_array = item.second.cast<py::array_t<float>>();
          std::vector<size_t> shape;
          for (int i = 0; i < np_array.ndim(); i++) {
            shape.push_back(np_array.shape(i));
          }
          std::vector<float> data(np_array.data(),
                                  np_array.data() + np_array.size());
          // 部分矩阵需要转置
          if (src_suffix.find("proj.weight") != std::string::npos) {
            cpp_weights.emplace(
                dst_key,
                Tensor<float>(std::move(data), shape).transpose(-1, -2));
          } else {
            cpp_weights.emplace(dst_key, Tensor<float>(std::move(data), shape));
          }
        }
      }
      for (const auto& [src_suffix, dst_suffix] : qwen_layer_bias_mapping) {
        std::string pattern = "." + src_suffix;
        if (key.find(pattern) != std::string::npos) {
          size_t start = std::string("model.layers.").size();
          size_t end = key.find('.', start);
          std::string layer_str = key.substr(start, end - start);
          int layer = std::stoi(layer_str);
          std::string dst_key =
              "layers." + std::to_string(layer) + "." + dst_suffix;
          std::cout << "Processing Qwen FP32 bias key: " << key << " -> "
                    << dst_key << std::endl;
          py::array_t<float> np_array = item.second.cast<py::array_t<float>>();
          std::vector<size_t> shape;
          for (int i = 0; i < np_array.ndim(); i++) {
            shape.push_back(np_array.shape(i));
          }
          std::vector<float> data(np_array.data(),
                                  np_array.data() + np_array.size());
          cpp_weights.emplace(dst_key, Tensor<float>(std::move(data), shape));
        }
      }
    }
  }
  // 如果 lm_head 缺失，则从 token_embeddings.weight 转置生成
  if (cpp_weights.find("lm_head") == cpp_weights.end()) {
    std::cout << "Warning: lm_head not found in weights dict, creating from "
                 "token_embeddings.weight"
              << std::endl;
    if (cpp_weights.find("token_embeddings.weight") != cpp_weights.end()) {
      try {
        Tensor<float> lm_head =
            cpp_weights.at("token_embeddings.weight").transpose(-1, -2);
        cpp_weights.emplace("lm_head", std::move(lm_head));
      } catch (const std::exception& e) {
        std::cerr << "Error creating lm_head: " << e.what() << std::endl;
        throw;
      }
    }
  }
  return cpp_weights;
}

// 处理 Qwen 模型权重（BF16）
std::unordered_map<std::string, Tensor<__nv_bfloat16>>
process_qwen_weights_bf16(const py::dict& weights) {
  std::unordered_map<std::string, Tensor<__nv_bfloat16>> cpp_bf16_weights;
  // 全局权重映射表
  const std::unordered_map<std::string, std::string> global_weights_map = {
      {"model.embed_tokens.weight", "token_embeddings.weight"},
      {"model.norm.weight", "norm.weight"},
      {"lm_head.weight", "lm_head"},
      {"model.lm_head.weight", "lm_head"}};
  for (const auto& [src_key, dst_key] : global_weights_map) {
    if (weights.contains(src_key)) {
      std::cout << "Processing Qwen BF16 global key: " << src_key << " -> "
                << dst_key << std::endl;
      py::object tensor = weights[src_key.c_str()];
      Tensor<__nv_bfloat16> bf16_tensor = convert_bf16_tensor(tensor);
      if (dst_key == "lm_head") {
        cpp_bf16_weights.emplace(dst_key, bf16_tensor.transpose(-1, -2));
      } else {
        cpp_bf16_weights.emplace(dst_key, std::move(bf16_tensor));
      }
    }
  }
  // 处理层级权重（权重与偏置）
  const std::vector<std::pair<std::string, std::string>>
      qwen_layer_key_mapping = {
          {"input_layernorm.weight", "input_layernorm.weight"},
          {"post_attention_layernorm.weight",
           "post_attention_layernorm.weight"},
          {"self_attn.q_proj.weight", "self_attn.q_proj.weight"},
          {"self_attn.k_proj.weight", "self_attn.k_proj.weight"},
          {"self_attn.v_proj.weight", "self_attn.v_proj.weight"},
          {"self_attn.o_proj.weight", "self_attn.o_proj.weight"},
          {"mlp.gate_proj.weight", "mlp.gate_proj.weight"},
          {"mlp.up_proj.weight", "mlp.up_proj.weight"},
          {"mlp.down_proj.weight", "mlp.down_proj.weight"}};
  const std::vector<std::pair<std::string, std::string>>
      qwen_layer_bias_mapping = {
          {"self_attn.q_proj.bias", "self_attn.q_proj.bias"},
          {"self_attn.k_proj.bias", "self_attn.k_proj.bias"},
          {"self_attn.v_proj.bias", "self_attn.v_proj.bias"},
          {"self_attn.o_proj.bias", "self_attn.o_proj.bias"}};
  for (auto item : weights) {
    std::string key = py::str(item.first).cast<std::string>();
    if (key.find("model.layers.") == 0) {
      for (const auto& [src_suffix, dst_suffix] : qwen_layer_key_mapping) {
        std::string pattern = "." + src_suffix;
        if (key.find(pattern) != std::string::npos) {
          size_t start = std::string("model.layers.").size();
          size_t end = key.find('.', start);
          std::string layer_str = key.substr(start, end - start);
          int layer = std::stoi(layer_str);
          std::string dst_key =
              "layers." + std::to_string(layer) + "." + dst_suffix;
          std::cout << "Processing Qwen BF16 layer key: " << key << " -> "
                    << dst_key << std::endl;
          py::object tensor = weights[key.c_str()];
          Tensor<__nv_bfloat16> bf16_tensor = convert_bf16_tensor(tensor);
          if (src_suffix.find("proj.weight") != std::string::npos ||
              src_suffix.find("gate_proj.weight") != std::string::npos ||
              src_suffix.find("up_proj.weight") != std::string::npos ||
              src_suffix.find("down_proj.weight") != std::string::npos) {
            cpp_bf16_weights.emplace(dst_key, bf16_tensor.transpose(-1, -2));
          } else {
            cpp_bf16_weights.emplace(dst_key, std::move(bf16_tensor));
          }
        }
      }
      for (const auto& [src_suffix, dst_suffix] : qwen_layer_bias_mapping) {
        std::string pattern = "." + src_suffix;
        if (key.find(pattern) != std::string::npos) {
          size_t start = std::string("model.layers.").size();
          size_t end = key.find('.', start);
          std::string layer_str = key.substr(start, end - start);
          int layer = std::stoi(layer_str);
          std::string dst_key =
              "layers." + std::to_string(layer) + "." + dst_suffix;
          std::cout << "Processing Qwen BF16 bias key: " << key << " -> "
                    << dst_key << std::endl;
          py::object tensor = weights[key.c_str()];
          Tensor<__nv_bfloat16> bf16_tensor = convert_bf16_tensor(tensor);
          cpp_bf16_weights.emplace(dst_key, std::move(bf16_tensor));
        }
      }
    }
  }
  // 如果 lm_head 缺失，则尝试从 token_embeddings.weight 转置生成
  if (cpp_bf16_weights.find("lm_head") == cpp_bf16_weights.end()) {
    std::cout << "Warning: lm_head not found in BF16 weights, creating from "
                 "token_embeddings.weight"
              << std::endl;
    if (cpp_bf16_weights.find("token_embeddings.weight") !=
        cpp_bf16_weights.end()) {
      Tensor<__nv_bfloat16> lm_head =
          cpp_bf16_weights.at("token_embeddings.weight").transpose(-1, -2);
      cpp_bf16_weights.emplace("lm_head", std::move(lm_head));
    } else {
      std::cerr << "Error: token_embeddings.weight not found in BF16 weights"
                << std::endl;
    }
  }
  return cpp_bf16_weights;
}

// 处理 Qwen AWQ 量化权重
std::tuple<std::unordered_map<std::string, Tensor<__nv_bfloat16>>,
           std::unordered_map<std::string, Tensor<int32_t>>,
           std::unordered_map<std::string, Tensor<float>>,
           std::unordered_map<std::string, Tensor<int32_t>>>
process_qwen_weights_awq(const py::dict& weights) {
  std::unordered_map<std::string, Tensor<__nv_bfloat16>> cpp_bf16_weights;
  std::unordered_map<std::string, Tensor<int32_t>> cpp_qweight_params;
  std::unordered_map<std::string, Tensor<float>> cpp_scales_params;
  std::unordered_map<std::string, Tensor<int32_t>> cpp_qzeros_params;

  // 全局权重映射表（非量化权重）
  const std::unordered_map<std::string, std::string> global_weights_map = {
      {"model.embed_tokens.weight", "token_embeddings.weight"},
      {"model.norm.weight", "norm.weight"},
      {"lm_head.weight", "lm_head"},
      {"model.lm_head.weight", "lm_head"}
  };

  // 处理非量化的全局权重
  for (const auto& [src_key, dst_key] : global_weights_map) {
    if (weights.contains(src_key)) {
      std::cout << "Processing Qwen AWQ global key: " << src_key << " -> "
                << dst_key << std::endl;
      py::object tensor = weights[src_key.c_str()];
      Tensor<__nv_bfloat16> bf16_tensor = convert_bf16_tensor(tensor);
      if (dst_key == "lm_head") {
        cpp_bf16_weights.emplace(dst_key, bf16_tensor.transpose(-1, -2));
      } else {
        cpp_bf16_weights.emplace(dst_key, std::move(bf16_tensor));
      }
    }
  }

  // 处理量化权重
  for (auto item : weights) {
    std::string key = py::str(item.first).cast<std::string>();

    // 处理量化权重 (qweight)
    if (key.find(".qweight") != std::string::npos) {
      std::string base_key = key.substr(0, key.find(".qweight"));
      std::string dst_key = base_key;

      // 转换层级路径格式
      if (base_key.find("model.layers.") == 0) {
        size_t start = std::string("model.layers.").size();
        size_t end = base_key.find('.', start);
        std::string layer_str = base_key.substr(start, end - start);
        int layer = std::stoi(layer_str);

        // 替换路径前缀
        dst_key = "layers." + std::to_string(layer) + base_key.substr(end);
      }

      std::string full_dst_key = dst_key + ".qweight";
      std::cout << "Processing Qwen AWQ qweight: " << key << " -> " << full_dst_key << std::endl;

      // 转换为int32_t张量
      py::array_t<int32_t> np_array = item.second.cast<py::array_t<int32_t>>();
      std::vector<size_t> shape;
      for (int i = 0; i < np_array.ndim(); i++) {
        shape.push_back(np_array.shape(i));
      }
      std::vector<int32_t> data(np_array.data(), np_array.data() + np_array.size());
      cpp_qweight_params.emplace(full_dst_key, Tensor<int32_t>(std::move(data), shape));
    }

    // 处理量化缩放因子 (scales)
    else if (key.find(".scales") != std::string::npos) {
      std::string base_key = key.substr(0, key.find(".scales"));
      std::string dst_key = base_key;

      // 转换层级路径格式
      if (base_key.find("model.layers.") == 0) {
        size_t start = std::string("model.layers.").size();
        size_t end = base_key.find('.', start);
        std::string layer_str = base_key.substr(start, end - start);
        int layer = std::stoi(layer_str);

        // 替换路径前缀
        dst_key = "layers." + std::to_string(layer) + base_key.substr(end);
      }

      std::string full_dst_key = dst_key + ".scales";
      std::cout << "Processing Qwen AWQ scales: " << key << " -> " << full_dst_key << std::endl;

      // 转换为float张量
      py::array_t<float> np_array = item.second.cast<py::array_t<float>>();
      std::vector<size_t> shape;
      for (int i = 0; i < np_array.ndim(); i++) {
        shape.push_back(np_array.shape(i));
      }
      std::vector<float> data(np_array.data(), np_array.data() + np_array.size());
      cpp_scales_params.emplace(full_dst_key, Tensor<float>(std::move(data), shape));
    }

    // 处理量化零点 (qzeros)
    else if (key.find(".qzeros") != std::string::npos) {
      std::string base_key = key.substr(0, key.find(".qzeros"));
      std::string dst_key = base_key;

      // 转换层级路径格式
      if (base_key.find("model.layers.") == 0) {
        size_t start = std::string("model.layers.").size();
        size_t end = base_key.find('.', start);
        std::string layer_str = base_key.substr(start, end - start);
        int layer = std::stoi(layer_str);

        // 替换路径前缀
        dst_key = "layers." + std::to_string(layer) + base_key.substr(end);
      }

      std::string full_dst_key = dst_key + ".qzeros";
      std::cout << "Processing Qwen AWQ qzeros: " << key << " -> " << full_dst_key << std::endl;

      // 转换为int32_t张量
      py::array_t<int32_t> np_array = item.second.cast<py::array_t<int32_t>>();
      std::vector<size_t> shape;
      for (int i = 0; i < np_array.ndim(); i++) {
        shape.push_back(np_array.shape(i));
      }
      std::vector<int32_t> data(np_array.data(), np_array.data() + np_array.size());
      cpp_qzeros_params.emplace(full_dst_key, Tensor<int32_t>(std::move(data), shape));
    }

    // 处理非量化的层级权重
    else if (key.find("model.layers.") == 0 &&
             key.find(".qweight") == std::string::npos &&
             key.find(".scales") == std::string::npos &&
             key.find(".qzeros") == std::string::npos) {
      // 处理层级权重（如 layernorm 和 bias）
      const std::vector<std::pair<std::string, std::string>> qwen_layer_key_mapping = {
          {"input_layernorm.weight", "input_layernorm.weight"},
          {"post_attention_layernorm.weight", "post_attention_layernorm.weight"},
          {"self_attn.q_proj.bias", "self_attn.q_proj.bias"},
          {"self_attn.k_proj.bias", "self_attn.k_proj.bias"},
          {"self_attn.v_proj.bias", "self_attn.v_proj.bias"},
          {"self_attn.o_proj.bias", "self_attn.o_proj.bias"}
      };

      for (const auto& [src_suffix, dst_suffix] : qwen_layer_key_mapping) {
        std::string pattern = "." + src_suffix;
        if (key.find(pattern) != std::string::npos) {
          size_t start = std::string("model.layers.").size();
          size_t end = key.find('.', start);
          std::string layer_str = key.substr(start, end - start);
          int layer = std::stoi(layer_str);
          std::string dst_key = "layers." + std::to_string(layer) + "." + dst_suffix;
          std::cout << "Processing Qwen AWQ layer key: " << key << " -> " << dst_key << std::endl;
          py::object tensor = weights[key.c_str()];
          Tensor<__nv_bfloat16> bf16_tensor = convert_bf16_tensor(tensor);
          cpp_bf16_weights.emplace(dst_key, std::move(bf16_tensor));
          break;
        }
      }
    }
    // 处理lm_head (特殊处理)
    else if (key == "lm_head.weight") {
      std::cout << "Processing Qwen AWQ lm_head" << std::endl;
      py::object tensor = weights[key.c_str()];
      Tensor<__nv_bfloat16> bf16_tensor = convert_bf16_tensor(tensor);
      cpp_bf16_weights.emplace("lm_head", bf16_tensor.transpose(-1, -2));
    }
  }

  // 如果 lm_head 缺失，则尝试从 token_embeddings.weight 转置生成
  if (cpp_bf16_weights.find("lm_head") == cpp_bf16_weights.end()) {
    std::cout << "Warning: lm_head not found in AWQ weights, creating from "
                 "token_embeddings.weight" << std::endl;
    if (cpp_bf16_weights.find("token_embeddings.weight") != cpp_bf16_weights.end()) {
      Tensor<__nv_bfloat16> lm_head =
          cpp_bf16_weights.at("token_embeddings.weight").transpose(-1, -2);
      cpp_bf16_weights.emplace("lm_head", std::move(lm_head));
    } else {
      std::cerr << "Error: token_embeddings.weight not found in AWQ weights"
                << std::endl;
    }
  }

  return {cpp_bf16_weights, cpp_qweight_params, cpp_scales_params, cpp_qzeros_params};
}

bool init_model(py::dict config, py::dict weights,
                const std::string& model_type) {
  try {
    // 打印配置与权重调试信息
    std::cout << "\n===== Configuration Items =====" << std::endl;
    for (const auto& item : config) {
      std::string key = py::str(item.first).cast<std::string>();
      std::string type_str =
          py::str(item.second.get_type()).cast<std::string>();
      std::cout << "Config key: " << key << ", type: " << type_str << std::endl;
    }
    std::cout << "\n===== Weight Items =====" << std::endl;
    for (const auto& item : weights) {
      std::string key = py::str(item.first).cast<std::string>();
      std::string type_str =
          py::str(item.second.get_type()).cast<std::string>();
      std::cout << "Weight key: " << key << ", type: " << type_str << std::endl;
    }

    // 构建配置字典
    std::unordered_map<std::string, int> cpp_config;
    cpp_config["vocab_size"] = config["vocab_size"].cast<int>();
    cpp_config["hidden_size"] = config["hidden_size"].cast<int>();
    cpp_config["max_position_embeddings"] =
        config["max_position_embeddings"].cast<int>();
    cpp_config["bos_token_id"] = config["bos_token_id"].cast<int>();
    cpp_config["eos_token_id"] = config["eos_token_id"].cast<int>();

    ModelType type;
    std::unordered_map<std::string, Tensor<float>> cpp_weights_fp32;
    std::unordered_map<std::string, Tensor<__nv_bfloat16>> cpp_weights_bf16;
    std::unordered_map<std::string, Tensor<int32_t>> cpp_qweight_params;
    std::unordered_map<std::string, Tensor<float>> cpp_scales_params;
    std::unordered_map<std::string, Tensor<int32_t>> cpp_qzeros_params;

    if (model_type == "llama") {
      type = ModelType::LLAMA;
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
      cpp_weights_fp32 = process_llama_weights(weights);
      // 创建 Llama 模型（默认在 CUDA 上运行 支持CPU）
      g_model = ModelFactory::create_model(type, cpp_weights_fp32, cpp_config);
      g_model->cuda();
      g_engine =
          std::make_unique<InferenceEngine<float>>(g_model, Device::CUDA);

    } else if (model_type == "qwen") {
      type = ModelType::QWEN;
      // Qwen FP32 配置
      cpp_config["n_layers"] = config["num_hidden_layers"].cast<int>();
      cpp_config["n_heads"] = config["num_attention_heads"].cast<int>();
      cpp_config["n_kv_heads"] = config["num_key_value_heads"].cast<int>();
      cpp_config["intermediate_size"] = config["intermediate_size"].cast<int>();
      cpp_config["vocab_size"] = config["vocab_size"].cast<int>();
      cpp_config["hidden_size"] = config["hidden_size"].cast<int>();
      cpp_config["max_position_embeddings"] =
          config["max_position_embeddings"].cast<int>();
      cpp_config["bos_token_id"] = config["bos_token_id"].cast<int>();
      cpp_config["eos_token_id"] = config["eos_token_id"].cast<int>();
      cpp_config["rms_norm_eps"] =
          static_cast<int>(config["rms_norm_eps"].cast<float>());
      cpp_config["rope_theta"] =
          static_cast<int>(config["rope_theta"].cast<float>());
      // 处理 Qwen FP32 权重
      cpp_weights_fp32 = process_qwen_weights_fp32(weights);
      g_model = ModelFactory::create_model(type, cpp_weights_fp32, cpp_config);
      // Qwen 模型仅支持 CUDA，故必须移动到 CUDA
      g_model->cuda();
      g_engine =
          std::make_unique<InferenceEngine<float>>(g_model, Device::CUDA);

    } else if (model_type == "qwen_bf16") {
      type = ModelType::QWEN_BF16;
      // Qwen BF16 专有配置：注意部分浮点参数转换（如 rms_norm_eps 按 1e6 缩放）
      cpp_config["n_layers"] = config["num_hidden_layers"].cast<int>();
      cpp_config["n_heads"] = config["num_attention_heads"].cast<int>();
      cpp_config["n_kv_heads"] = config["num_key_value_heads"].cast<int>();
      cpp_config["intermediate_size"] = config["intermediate_size"].cast<int>();
      cpp_config["vocab_size"] = config["vocab_size"].cast<int>();
      cpp_config["hidden_size"] = config["hidden_size"].cast<int>();
      cpp_config["max_position_embeddings"] =
          config["max_position_embeddings"].cast<int>();
      cpp_config["bos_token_id"] = config["bos_token_id"].cast<int>();
      cpp_config["eos_token_id"] = config["eos_token_id"].cast<int>();
      cpp_config["rms_norm_eps"] =
          static_cast<int>(config["rms_norm_eps"].cast<float>());
      cpp_config["rope_theta"] =
          static_cast<int>(config["rope_theta"].cast<float>());
      // 处理 Qwen BF16 权重（输入必须为 PyTorch bf16 类型）
      cpp_weights_bf16 = process_qwen_weights_bf16(weights);

      g_model = std::make_shared<QwenModel<__nv_bfloat16>>(cpp_weights_bf16,
                                                           cpp_config);
      g_engine = std::make_unique<InferenceEngine<__nv_bfloat16>>(g_model,
                                                                  Device::CUDA);
      g_model->cuda();
    } else if (model_type == "qwen_awq") {
      type = ModelType::QWEN_AWQ;
      // Qwen AWQ 配置
      cpp_config["n_layers"] = config["num_hidden_layers"].cast<int>();
      cpp_config["n_heads"] = config["num_attention_heads"].cast<int>();
      cpp_config["n_kv_heads"] = config["num_key_value_heads"].cast<int>();
      cpp_config["intermediate_size"] = config["intermediate_size"].cast<int>();
      cpp_config["vocab_size"] = config["vocab_size"].cast<int>();
      cpp_config["hidden_size"] = config["hidden_size"].cast<int>();
      cpp_config["max_position_embeddings"] =
          config["max_position_embeddings"].cast<int>();
      cpp_config["bos_token_id"] = config["bos_token_id"].cast<int>();
      cpp_config["eos_token_id"] = config["eos_token_id"].cast<int>();
      cpp_config["rms_norm_eps"] =
          static_cast<int>(config["rms_norm_eps"].cast<float>());
      cpp_config["rope_theta"] =
          static_cast<int>(config["rope_theta"].cast<float>());

      // 设置量化类型和分组大小
      cpp_config["quant_type"] = 1; // AWQ量化

      // 如果配置中有分组大小，则使用配置中的值
      if (config.contains("group_size")) {
        cpp_config["group_size"] = config["group_size"].cast<int>();
      } else {
        // 默认分组大小为128
        cpp_config["group_size"] = 128;
      }

      // 处理 Qwen AWQ 权重
      auto [bf16_weights, qweight_params, scales_params, qzeros_params] =
          process_qwen_weights_awq(weights);

      // 创建带量化参数的模型
      g_model = std::make_shared<QwenModel<__nv_bfloat16>>(
          bf16_weights, qweight_params, scales_params, qzeros_params, cpp_config);

      g_engine = std::make_unique<InferenceEngine<__nv_bfloat16>>(g_model,
                                                                  Device::CUDA);
      g_model->cuda();
    } else {
      throw std::runtime_error("Unsupported model type: " + model_type);
    }
    g_model->print_model_info();

    // 初始化CUDA内存池，为prefill阶段预分配内存
    try {
      // 获取模型配置参数
      size_t hidden_dim = cpp_config["hidden_size"];

      // 获取最大序列长度，如果配置中有的话
      size_t seq_len = 32; // 默认序列长度
      if (cpp_config.find("max_position_embeddings") != cpp_config.end()) {
        seq_len = std::min(seq_len, static_cast<size_t>(cpp_config["max_position_embeddings"]));
      }

      // 检查GPU内存状态
      size_t free_memory = 0, total_memory = 0;
      cudaError_t err = cudaMemGetInfo(&free_memory, &total_memory);
      if (err == cudaSuccess) {
        std::cout << "Current GPU memory: " << (free_memory / (1024 * 1024))
                  << " MB free, " << (total_memory / (1024 * 1024))
                  << " MB total" << std::endl;

        // 只有当可用内存超过1GB时才开启prefill模式
        if (free_memory > 1024 * 1024 * 1024) {
          // 计算prefill内存大小，使用可用内存的10%，但不超过256MB
          size_t prefill_size = std::min(free_memory / 10, static_cast<size_t>(256 * 1024 * 1024));

          // 开启prefill模式
          std::cout << "Enabling prefill mode with initial buffer size: "
                    << (prefill_size / (1024 * 1024)) << " MB" << std::endl;
          GlobalCudaMemoryPool::enable_prefill_mode(prefill_size);
        } else {
          std::cout << "Skipping prefill mode due to low available memory" << std::endl;
        }
      }
    } catch (const std::exception& e) {
      // 如果prefill模式初始化失败，只打印警告，不影响模型加载
      std::cerr << "Warning: Failed to initialize prefill mode: " << e.what() << std::endl;
    }

    return true;
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

// Pybind11 模块定义

PYBIND11_MODULE(model_bridge, m) {
  m.def("init_model", &init_model, py::arg("config"), py::arg("weights"),
        py::arg("model_type") = "llama", "Initialize and verify the model");
  m.def("generate_text_stream", &generate_text_stream, py::arg("input_ids"),
        py::arg("callback"), py::arg("max_length") = 100,
        py::arg("temperature") = 1.0f, py::arg("top_p") = 0.9f,
        py::arg("top_k") = 50, "Stream generated tokens via callback");
}
