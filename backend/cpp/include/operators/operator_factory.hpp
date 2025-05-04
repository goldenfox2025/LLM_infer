#pragma once

#include <memory>
#include "operators/operator_base.hpp"
#include "operators/cpu/rope_cpu.hpp"
#include "operators/cuda/rope_cuda.cuh"

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
        registry.registerOperator(OperatorType::ROPE, OperatorPlatform::CPU, rope_cpu);
        
        // 注册其他CPU算子...
    }
    
    // 创建并注册所有CUDA算子
    static void registerCUDAOperators() {
        auto& registry = OperatorRegistry<T>::instance();
        
        // 注册RoPE CUDA算子
        auto rope_cuda = std::make_shared<RopeCUDAOperator<T>>();
        registry.registerOperator(OperatorType::ROPE, OperatorPlatform::CUDA, rope_cuda);
        
        // 注册其他CUDA算子...
    }
    
    // 获取算子
    static std::shared_ptr<RopeOperator<T>> getRopeOperator(OperatorPlatform platform) {
        auto& registry = OperatorRegistry<T>::instance();
        return registry.template getOperator<RopeOperator<T>>(OperatorType::ROPE, platform);
    }
    
    // 添加其他算子的获取方法...
};

} // namespace op
