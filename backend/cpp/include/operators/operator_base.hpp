#pragma once

#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <unordered_map>

#include "tensor.hpp"

namespace op {

// 算子类型枚举
enum class OperatorType {
  ROPE,
  RMS_NORM,
  MATMUL,
  GATHER,
  SOFTMAX,
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

  // RoPE算子实现 - 使用二重指针以支持CUDA图优化
  virtual void operator()(Tensor<T>** tensor, size_t* offset, float theta,
                          cudaStream_t stream = nullptr) = 0;

  // 获取算子类型
  OperatorType type() const override { return OperatorType::ROPE; }

  // 获取算子名称
  std::string name() const override { return "rope"; }
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
