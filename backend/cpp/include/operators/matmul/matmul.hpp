#pragma once

#include "operators/operator_base.hpp"
#include "weight_tensor.hpp"

namespace op {

// 前向声明
template <typename T>
class MatmulSelector;

template <typename T>
class MatmulCPUOperator;

template <typename T>
class MatmulCUDAOperator;

template <typename T>
class CublasMatmulCUDAOperator;

template <typename T>
class CutlassMatmulCUDAOperator;

template <typename T>
class AwqMatmulCUDAOperator;

}  // namespace op