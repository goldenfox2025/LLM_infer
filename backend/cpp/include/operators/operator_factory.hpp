#pragma once

#include <memory>

#include "operators/cpu/rms_norm_cpu.hpp"
#include "operators/cpu/rope_cpu.hpp"
#include "operators/cuda/rms_norm_cuda.cuh"
#include "operators/cuda/rope_cuda.cuh"
#include "operators/operator_base.hpp"

namespace op {

// 算子工厂类
template <typename T>
class OperatorFactory {
 public:
  // 创建并注册所有CPU算子
  static void registerCPUOperators() {
    auto& registry = OperatorRegistry<T>::instance();

    // 注册RoPE CPU算子
    auto rope_cpu = std::make_shared<RopeCPUOperator<T>>();
    registry.registerOperator(OperatorType::ROPE, OperatorPlatform::CPU,
                              rope_cpu);

    // 注册RMS Norm CPU算子
    auto rms_norm_cpu = std::make_shared<RmsNormCPUOperator<T>>();
    registry.registerOperator(OperatorType::RMS_NORM, OperatorPlatform::CPU,
                              rms_norm_cpu);

    // 注册其他CPU算子...
  }

  // 创建并注册所有CUDA算子
  static void registerCUDAOperators() {
    auto& registry = OperatorRegistry<T>::instance();

    // 注册RoPE CUDA算子
    auto rope_cuda = std::make_shared<RopeCUDAOperator<T>>();
    registry.registerOperator(OperatorType::ROPE, OperatorPlatform::CUDA,
                              rope_cuda);

    // 注册RMS Norm CUDA算子
    auto rms_norm_cuda = std::make_shared<RmsNormCUDAOperator<T>>();
    registry.registerOperator(OperatorType::RMS_NORM, OperatorPlatform::CUDA,
                              rms_norm_cuda);

    // 注册其他CUDA算子...
  }

  // 获取算子
  static std::shared_ptr<RopeOperator<T>> getRopeOperator(
      OperatorPlatform platform) {
    auto& registry = OperatorRegistry<T>::instance();
    return registry.template getOperator<RopeOperator<T>>(OperatorType::ROPE,
                                                          platform);
  }

  // 获取RMS Norm算子
  static std::shared_ptr<RmsNormOperator<T>> getRmsNormOperator(
      OperatorPlatform platform) {
    auto& registry = OperatorRegistry<T>::instance();
    return registry.template getOperator<RmsNormOperator<T>>(
        OperatorType::RMS_NORM, platform);
  }

  // 添加其他算子的获取方法...
};

}  // namespace op
