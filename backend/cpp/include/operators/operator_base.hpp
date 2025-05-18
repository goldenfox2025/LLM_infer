#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>  // 添加这个头文件以声明 curandState 类型

#include <memory>
#include <string>
#include <unordered_map>

#include "tensor.hpp"

namespace op {

// 前向声明 WeightTensor
template <typename T>
class WeightTensor;

// MatMul实现类型枚举
enum class MatmulType { DEFAULT, CUBLAS, CUTLASS, AWQ };

// 算子类型枚举
enum class OperatorType {
  ROPE,
  RMS_NORM,
  MATMUL,
  GATHER,
  SOFTMAX,
  ADD,
  MULTIPLY,
  SILU,
  SAMPLE,
  DYNAMIC_FLASH_ATTENTION,
  ATTENTION_SCORES_PREFILL,
  ATTENTION_OUTPUT_PREFILL,
  // 添加更多算子类型...
};

// 算子平台枚举
enum class OperatorPlatform {
  CPU,
  CUDA,
};

// 基础算子接口
class OperatorBase {
 public:
  virtual ~OperatorBase() = default;

  // 获取算子类型
  virtual OperatorType type() const = 0;

  // 获取算子平台
  virtual OperatorPlatform platform() const = 0;

  // 获取算子名称
  virtual std::string name() const = 0;
};

// RoPE算子接口
template <typename T>
class RopeOperator : public OperatorBase {
 public:
  virtual ~RopeOperator() = default;

  // RoPE算子实现 - 使用一重指针
  virtual void operator()(Tensor<T>* tensor, size_t offset, float theta,
                          cudaStream_t stream = nullptr) = 0;

  // 获取算子类型
  OperatorType type() const override { return OperatorType::ROPE; }

  // 获取算子名称
  std::string name() const override { return "rope"; }
};

// RMS Norm算子接口
template <typename T>
class RmsNormOperator : public OperatorBase {
 public:
  virtual ~RmsNormOperator() = default;

  // RMS Norm算子实现 - 使用一重指针
  virtual void operator()(Tensor<T>* output, Tensor<T>* input,
                          Tensor<T>* weight, float eps,
                          cudaStream_t stream = nullptr) = 0;

  // 获取算子类型
  OperatorType type() const override { return OperatorType::RMS_NORM; }

  // 获取算子名称
  std::string name() const override { return "rms_norm"; }
};

// Add算子接口
template <typename T>
class AddOperator : public OperatorBase {
 public:
  virtual ~AddOperator() = default;

  // Add算子实现 - 使用一重指针
  virtual void operator()(Tensor<T>* output, Tensor<T>* input_a,
                          Tensor<T>* input_b,
                          cudaStream_t stream = nullptr) = 0;

  // 获取算子类型
  OperatorType type() const override { return OperatorType::ADD; }

  // 获取算子名称
  std::string name() const override { return "add"; }
};

// Multiply算子接口
template <typename T>
class MultiplyOperator : public OperatorBase {
 public:
  virtual ~MultiplyOperator() = default;

  // Multiply算子实现 - 使用一重指针
  virtual void operator()(Tensor<T>* output, Tensor<T>* input_a,
                          Tensor<T>* input_b,
                          cudaStream_t stream = nullptr) = 0;

  // 获取算子类型
  OperatorType type() const override { return OperatorType::MULTIPLY; }

  // 获取算子名称
  std::string name() const override { return "multiply"; }
};

// SiLU算子接口
template <typename T>
class SiluOperator : public OperatorBase {
 public:
  virtual ~SiluOperator() = default;

  // SiLU算子实现 - 使用一重指针
  virtual void operator()(Tensor<T>* output, Tensor<T>* input,
                          cudaStream_t stream = nullptr) = 0;

  // 获取算子类型
  OperatorType type() const override { return OperatorType::SILU; }

  // 获取算子名称
  std::string name() const override { return "silu"; }
};

// MatMul算子接口
template <typename T>
class MatmulOperator : public OperatorBase {
 public:
  virtual ~MatmulOperator() = default;

  // MatMul算子实现 - 使用WeightTensor作为参数
  virtual void operator()(Tensor<T>* output, Tensor<T>* input,
                          const WeightTensor<T>& weight,
                          const Tensor<T>* bias = nullptr,
                          cudaStream_t stream = nullptr) = 0;

  // 获取算子类型
  OperatorType type() const override { return OperatorType::MATMUL; }

  // 获取算子名称
  std::string name() const override { return "matmul"; }
};

// MatMul算子实现的基类
template <typename T>
class MatmulOperatorImpl : public MatmulOperator<T> {
 public:
  virtual ~MatmulOperatorImpl() = default;

  // 获取MatMul实现类型
  virtual MatmulType impl_type() const = 0;
};

// Gather算子接口
template <typename T>
class GatherOperator : public OperatorBase {
 public:
  virtual ~GatherOperator() = default;

  // Gather算子实现
  virtual void operator()(Tensor<T>* output, const Tensor<uint32_t>* input,
                          const Tensor<T>* embedding_table,
                          cudaStream_t stream = nullptr) = 0;

  // 获取算子类型
  OperatorType type() const override { return OperatorType::GATHER; }

  // 获取算子名称
  std::string name() const override { return "gather"; }
};

// Sample算子接口
template <typename T>
class SampleOperator : public OperatorBase {
 public:
  virtual ~SampleOperator() = default;

  // Sample算子实现
  virtual uint32_t* operator()(Tensor<T>&& logits, float temperature,
                               float top_p, size_t top_k, curandState* d_states,
                               cudaStream_t stream = nullptr) = 0;

  // 获取算子类型
  OperatorType type() const override { return OperatorType::SAMPLE; }

  // 获取算子名称
  std::string name() const override { return "sample"; }
};

// DynamicFlashAttention算子接口
template <typename T>
class DynamicFlashAttentionOperator : public OperatorBase {
 public:
  virtual ~DynamicFlashAttentionOperator() = default;

  // DynamicFlashAttention算子实现
  virtual void operator()(Tensor<T>& Q, const Tensor<T>& K, const Tensor<T>& V,
                          Tensor<T>& output, int n_kv_heads,
                          cudaStream_t stream = nullptr) = 0;

  // 获取算子类型
  OperatorType type() const override {
    return OperatorType::DYNAMIC_FLASH_ATTENTION;
  }

  // 获取算子名称
  std::string name() const override { return "dynamic_flash_attention"; }
};

// AttentionScoresPrefill算子接口
template <typename T>
class AttentionScoresPrefillOperator : public OperatorBase {
 public:
  virtual ~AttentionScoresPrefillOperator() = default;

  // AttentionScoresPrefill算子实现
  virtual void operator()(const Tensor<T>& Q, const Tensor<T>& K,
                          Tensor<T>& att_scores, size_t head_dim,
                          cudaStream_t stream = nullptr) = 0;

  // 获取算子类型
  OperatorType type() const override {
    return OperatorType::ATTENTION_SCORES_PREFILL;
  }

  // 获取算子名称
  std::string name() const override { return "attention_scores_prefill"; }
};

// AttentionOutputPrefill算子接口
template <typename T>
class AttentionOutputPrefillOperator : public OperatorBase {
 public:
  virtual ~AttentionOutputPrefillOperator() = default;

  // AttentionOutputPrefill算子实现
  virtual void operator()(const Tensor<T>& att_scores, const Tensor<T>& V,
                          Tensor<T>& att_output, size_t n_heads,
                          size_t head_dim, size_t total_seq_len,
                          size_t n_kv_heads, cudaStream_t stream = nullptr) = 0;

  // 获取算子类型
  OperatorType type() const override {
    return OperatorType::ATTENTION_OUTPUT_PREFILL;
  }

  // 获取算子名称
  std::string name() const override { return "attention_output_prefill"; }
};

// Softmax算子接口
template <typename T>
class SoftmaxOperator : public OperatorBase {
 public:
  virtual ~SoftmaxOperator() = default;

  // Softmax算子实现
  virtual void operator()(Tensor<T>* output, const Tensor<T>* input, int dim,
                          bool mask = false, int offset = 0,
                          cudaStream_t stream = nullptr) = 0;

  // 获取算子类型
  OperatorType type() const override { return OperatorType::SOFTMAX; }

  // 获取算子名称
  std::string name() const override { return "softmax"; }
};

// 其他算子接口可以在这里添加...

// 算子管理器 - 单例模式
template <typename T>
class OperatorRegistry {
 public:
  static OperatorRegistry<T>& instance() {
    static OperatorRegistry<T> instance;
    return instance;
  }

  // 注册算子
  void registerOperator(OperatorType type, OperatorPlatform platform,
                        std::shared_ptr<OperatorBase> op) {
    std::string key = getKey(type, platform);
    if (operators_.find(key) != operators_.end()) {
      return;
    }
    operators_[key] = op;
  }

  // 获取算子
  template <typename OpType>
  std::shared_ptr<OpType> getOperator(OperatorType type,
                                      OperatorPlatform platform) {
    std::string key = getKey(type, platform);
    if (operators_.find(key) == operators_.end()) {
      return nullptr;
    }
    return std::dynamic_pointer_cast<OpType>(operators_[key]);
  }

  // 检查算子是否已注册
  bool hasOperator(OperatorType type, OperatorPlatform platform) {
    std::string key = getKey(type, platform);
    return operators_.find(key) != operators_.end();
  }

 private:
  OperatorRegistry() = default;
  ~OperatorRegistry() = default;

  // 禁止拷贝和赋值
  OperatorRegistry(const OperatorRegistry&) = delete;
  OperatorRegistry& operator=(const OperatorRegistry&) = delete;

  // 生成算子键
  std::string getKey(OperatorType type, OperatorPlatform platform) {
    return std::to_string(static_cast<int>(type)) + "_" +
           std::to_string(static_cast<int>(platform));
  }

  // 存储算子的映射表
  std::unordered_map<std::string, std::shared_ptr<OperatorBase>> operators_;
};

}  // namespace op
