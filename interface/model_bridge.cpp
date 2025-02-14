

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>

#include "inference.hpp"
#include "llama.hpp"

enum class ModelType {
  LLAMA,
  // 未来可以添加其他模型类型
};

class ModelFactory {
 public:
  static std::shared_ptr<LlamaModel> create_model(
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
      default:
        throw std::runtime_error("Unsupported model type");
    }
  }
};

namespace py = pybind11;

// 全局变量存储模型和推理引擎实例
std::shared_ptr<LlamaModel> g_model;
std::unique_ptr<InferenceEngine> g_engine;

bool init_model(py::dict config, py::dict weights,
                const std::string& model_type) {
  try {
    std::unordered_map<std::string, int> cpp_config;
    // 基础配置
    cpp_config["vocab_size"] = config["vocab_size"].cast<int>();
    cpp_config["num_hidden_layers"] = config["num_hidden_layers"].cast<int>();
    cpp_config["num_attention_heads"] =
        config["num_attention_heads"].cast<int>();
    cpp_config["num_key_value_heads"] =
        config["num_key_value_heads"].cast<int>();
    cpp_config["hidden_size"] = config["hidden_size"].cast<int>();
    cpp_config["intermediate_size"] = config["intermediate_size"].cast<int>();
    cpp_config["max_position_embeddings"] =
        config["max_position_embeddings"].cast<int>();
    cpp_config["bos_token_id"] = config["bos_token_id"].cast<int>();
    cpp_config["eos_token_id"] = config["eos_token_id"].cast<int>();

    // 浮点数配置
    cpp_config["rms_norm_eps"] =
        static_cast<int>(config["rms_norm_eps"].cast<float>());
    cpp_config["rope_theta"] =
        static_cast<int>(config["rope_theta"].cast<float>());

    // 转换权重
    std::unordered_map<std::string, Tensor<float>> cpp_weights;

    // 定义权重键名映射
    const std::unordered_map<std::string, std::string> key_mapping = {
        {"model.embed_tokens.weight", "embedding_table"},
        {"model.norm.weight", "rms_out_w"},
        {"lm_head.weight", "lm_head"}};

    // 如果没有 embedding_table，使用 lm_head 的权重
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
      std::vector<float> data(np_array.data(),
                              np_array.data() + np_array.size());
      cpp_weights.emplace("embedding_table",
                          Tensor<float>(std::move(data), shape));
    }

    // 打印调试信息
    for (const auto& [key, value] : weights) {
      std::cout << "Available weight key: " << py::str(key) << std::endl;
    }

    // 层内权重的键名模板
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

    // 处理非层级权重
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

        } else
          cpp_weights.emplace(dst_key, Tensor<float>(std::move(data), shape));
      }
    }

    // 处理层级权重
    int num_layers = config["num_hidden_layers"].cast<int>();
    for (int layer = 0; layer < num_layers; layer++) {
      for (const auto& [src_suffix, dst_prefix] : layer_key_mapping) {
        std::string src_key =
            "model.layers." + std::to_string(layer) + "." + src_suffix;
        std::string dst_key = dst_prefix + std::to_string(layer);  // 移除方括号

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
          if (dst_prefix == "wq" || dst_prefix == "wk" || dst_prefix == "wv" ||
              dst_prefix == "wo" || dst_prefix == "w_up" ||
              dst_prefix == "w_down" || dst_prefix == "w_gate") {
            cpp_weights.emplace(
                dst_key,
                Tensor<float>(std::move(data), shape).transpose(-1, -2));

          } else
            cpp_weights.emplace(dst_key, Tensor<float>(std::move(data), shape));
        }
      }
    }

    // 确定模型类型
    ModelType type;
    if (model_type == "llama") {
      type = ModelType::LLAMA;
    } else {
      throw std::runtime_error("Unsupported model type: " + model_type);
    }

    // 创建模型并验证
    std::cout << "Model Info:" << std::endl;
    std::cout << "vocab_size: " << cpp_config["vocab_size"] << std::endl;
    std::cout << "n_layers: " << cpp_config["num_hidden_layers"] << std::endl;
    std::cout << "n_q_h: " << cpp_config["num_attention_heads"] << std::endl;
    std::cout << "n_kv_h: " << cpp_config["num_key_value_heads"] << std::endl;
    std::cout << "hidden_size: " << cpp_config["hidden_size"] << std::endl;
    std::cout << "head_dim: "
              << cpp_config["hidden_size"] / cpp_config["num_attention_heads"]
              << std::endl;
    std::cout << "intermediate_size: " << cpp_config["intermediate_size"]
              << std::endl;

    // 创建模型实例
    g_model = ModelFactory::create_model(type, cpp_weights, cpp_config);

    // 创建推理引擎
    g_engine = std::make_unique<InferenceEngine>(g_model);

    return true;
  } catch (const std::exception& e) {
    std::cerr << "Error initializing model: " << e.what() << std::endl;
    return false;
  }
}

void generate_text_stream(const std::vector<uint32_t>& input_ids,
                          py::function callback, size_t max_length = 100,
                          float temperature = 1.0f, float top_p = 0.9f,
                          size_t top_k = 50) {
  if (!g_engine) {
    throw std::runtime_error("Model not initialized");
  }
  // 调用带回调的生成接口，同时在每次回调前确保获取 GIL
  g_engine->generate_with_callback(input_ids, max_length, temperature, top_p,
                                   top_k, [callback](uint32_t token) {
                                     // 确保在调用 Python 回调前获取 GIL
                                     py::gil_scoped_acquire acquire;
                                     callback(token);
                                   });
}

PYBIND11_MODULE(model_bridge, m) {
  m.def("init_model", &init_model, py::arg("config"), py::arg("weights"),
        py::arg("model_type") = "llama", "Initialize and verify the model");

  // 新增：流式生成函数
  m.def("generate_text_stream", &generate_text_stream, py::arg("input_ids"),
        py::arg("callback"), py::arg("max_length") = 100,
        py::arg("temperature") = 1.0f, py::arg("top_p") = 0.9f,
        py::arg("top_k") = 50, "Stream generated tokens via callback");
}