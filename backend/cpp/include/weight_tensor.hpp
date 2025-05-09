#pragma once

#include <cuda_bf16.h>

#include <memory>
#include <stdexcept>
#include <string>

#include "tensor.hpp"

namespace op {

// 权重张量封装类，用于统一处理普通权重和量化权重
template <typename T>
class WeightTensor {
   public:
    // 普通权重构造函数
    WeightTensor(const Tensor<T>* tensor) : tensor_(tensor), is_quantized_(false) {
    }

    // 量化权重构造函数
    WeightTensor(const Tensor<int32_t>* qweight, const Tensor<T>* scales, const Tensor<int32_t>* qzeros, int group_size)
        : qweight_(qweight), scales_(scales), qzeros_(qzeros), group_size_(group_size), is_quantized_(true) {
    }

    // 判断是否为量化权重
    bool is_quantized() const {
        return is_quantized_;
    }

    // 获取普通权重
    const Tensor<T>* tensor() const {
        if (is_quantized_)
            throw std::runtime_error("Not a regular tensor");
        return tensor_;
    }

    // 获取量化权重相关参数
    const Tensor<int32_t>* qweight() const {
        if (!is_quantized_)
            throw std::runtime_error("Not a quantized tensor");
        return qweight_;
    }

    const Tensor<T>* scales() const {
        if (!is_quantized_)
            throw std::runtime_error("Not a quantized tensor");
        return scales_;
    }

    const Tensor<int32_t>* qzeros() const {
        if (!is_quantized_)
            throw std::runtime_error("Not a quantized tensor");
        return qzeros_;
    }

    int group_size() const {
        if (!is_quantized_)
            throw std::runtime_error("Not a quantized tensor");
        return group_size_;
    }

   private:
    // 普通权重
    const Tensor<T>* tensor_ = nullptr;

    // 量化权重
    const Tensor<int32_t>* qweight_ = nullptr;
    const Tensor<T>* scales_ = nullptr;
    const Tensor<int32_t>* qzeros_ = nullptr;
    int group_size_ = 0;

    // 是否为量化权重
    bool is_quantized_ = false;
};

}  // namespace op
