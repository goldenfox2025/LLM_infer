# 统一算子库设计文档

## 1. 设计思路

### 1.1 核心理念

统一算子库的核心设计理念是**分离接口与实现**，同时提供一个统一的注册和管理机制。这样做有几个主要优势：

1. **平台无关性**：通过抽象接口，模型代码不需要关心算子的具体实现是在CPU还是CUDA上
2. **易于扩展**：新增算子或新增平台实现只需遵循统一接口
3. **运行时选择**：根据设备类型和张量类型动态选择最合适的算子实现
4. **简洁明了**：使用一重指针传递算子数据，接口更加简洁

### 1.2 架构设计

算子库采用以下分层架构：

1. **基础抽象层**：
   - `OperatorBase`：所有算子的基类，定义通用接口
   - `XxxOperator<T>`：特定算子的模板接口类（如`RopeOperator<T>`）

2. **平台实现层**：
   - `XxxCPUOperator<T>`：CPU实现（如`RopeCPUOperator<T>`）
   - `XxxCUDAOperator<T>`：CUDA实现（如`RopeCUDAOperator<T>`）

3. **注册与管理层**：
   - `OperatorRegistry<T>`：单例模式的算子注册表
   - `OperatorFactory<T>`：创建和获取算子的工厂类

4. **统一接口层**：
   - `UnifiedOperators<T>`：对外提供统一接口，内部根据平台选择具体实现

### 1.3 关键技术点

1. **一重指针设计**：
   - 使用一重指针传递张量参数，接口更加简洁
   - 对于固定不变的参数，直接传值
   - 通过标签固定内存分配，确保张量内存地址固定，便于CUDA图优化

2. **类型安全**：
   - 使用模板参数`<T>`确保类型安全，支持不同数据类型（float、__nv_bfloat16等）
   - 使用`std::is_same_v`等类型特性处理特殊类型

3. **单例模式**：
   - `OperatorRegistry`采用单例模式，确保全局只有一个算子注册表
   - 算子本身是无状态的，可以安全地在多个推理实例间共享

4. **流参数传递**：
   - 所有算子接口都包含`cudaStream_t`参数
   - 即使在CPU实现中也保留此参数（默认为nullptr），保持接口一致性

## 2. 添加新算子流程

### 2.1 定义算子接口

在`backend/cpp/include/operators/operator_base.hpp`中添加新算子的接口类：

```cpp
// 新算子接口
template <typename T>
class NewOperator : public OperatorBase {
public:
    virtual ~NewOperator() = default;

    // 新算子实现 - 使用一重指针
    virtual void operator()(Tensor<T>* input, size_t param, float fixed_param, cudaStream_t stream = nullptr) = 0;

    // 获取算子类型
    OperatorType type() const override { return OperatorType::NEW_OP; }

    // 获取算子名称
    std::string name() const override { return "new_op"; }
};
```

同时，在`OperatorType`枚举中添加新算子类型：

```cpp
enum class OperatorType {
    ROPE,
    RMS_NORM,
    MATMUL,
    // ...
    NEW_OP,  // 添加新算子类型
};
```

### 2.2 实现CPU版本

1. 创建头文件`backend/cpp/include/operators/cpu/new_op_cpu.hpp`：

```cpp
#pragma once

#include "operators/operator_base.hpp"

namespace op {

template <typename T>
class NewOpCPUOperator : public NewOperator<T> {
public:
    NewOpCPUOperator() = default;
    ~NewOpCPUOperator() override = default;

    // 实现CPU版本
    void operator()(Tensor<T>* input, size_t param, float fixed_param, cudaStream_t stream = nullptr) override {
        // CPU实现逻辑...
    }

    // 获取算子平台
    OperatorPlatform platform() const override { return OperatorPlatform::CPU; }
};

} // namespace op
```

2. 创建源文件`backend/cpp/src/operators/cpu/new_op_cpu.cpp`：

```cpp
#include "operators/cpu/new_op_cpu.hpp"

namespace op {

// 显式模板实例化
template class NewOpCPUOperator<float>;
// 如果需要支持其他类型，添加对应的实例化

} // namespace op
```

### 2.3 实现CUDA版本

1. 创建头文件`backend/cpp/include/operators/cuda/new_op_cuda.cuh`：

```cpp
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "operators/operator_base.hpp"

namespace op {

template <typename T>
class NewOpCUDAOperator : public NewOperator<T> {
public:
    NewOpCUDAOperator() = default;
    ~NewOpCUDAOperator() override = default;

    // 实现CUDA版本
    void operator()(Tensor<T>* input, size_t param, float fixed_param, cudaStream_t stream = nullptr) override;

    // 获取算子平台
    OperatorPlatform platform() const override { return OperatorPlatform::CUDA; }
};

} // namespace op
```

2. 创建源文件`backend/cpp/src/operators/cuda/new_op_cuda.cu`：

```cpp
#include "operators/cuda/new_op_cuda.cuh"
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace op {

// CUDA kernel定义
template <typename T>
__global__ void new_op_kernel(T* input, size_t param, float fixed_param) {
    // CUDA kernel实现...
}

// 实现CUDA版本
template <typename T>
void NewOpCUDAOperator<T>::operator()(Tensor<T>* input, size_t param, float fixed_param, cudaStream_t stream) {

    // 参数检查
    // ...

    // 计算kernel启动参数
    int threads = 256;
    int blocks = (input->numel() + threads - 1) / threads;

    // 启动kernel
    new_op_kernel<T><<<blocks, threads, 0, stream>>>(
        input->data_ptr(), param, fixed_param);

    // 错误检查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after new_op kernel launch: "
                  << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA new_op kernel launch failed");
    }
}

// 显式模板实例化
template class NewOpCUDAOperator<float>;
template class NewOpCUDAOperator<__nv_bfloat16>;

} // namespace op
```

### 2.4 更新工厂类

在`backend/cpp/include/operators/operator_factory.hpp`中添加新算子的注册和获取方法：

```cpp
// 创建并注册所有CPU算子
static void registerCPUOperators() {
    auto& registry = OperatorRegistry<T>::instance();

    // 注册已有算子...

    // 注册新算子
    auto new_op_cpu = std::make_shared<NewOpCPUOperator<T>>();
    registry.registerOperator(OperatorType::NEW_OP, OperatorPlatform::CPU, new_op_cpu);
}

// 创建并注册所有CUDA算子
static void registerCUDAOperators() {
    auto& registry = OperatorRegistry<T>::instance();

    // 注册已有算子...

    // 注册新算子
    auto new_op_cuda = std::make_shared<NewOpCUDAOperator<T>>();
    registry.registerOperator(OperatorType::NEW_OP, OperatorPlatform::CUDA, new_op_cuda);
}

// 获取新算子
static std::shared_ptr<NewOperator<T>> getNewOperator(OperatorPlatform platform) {
    auto& registry = OperatorRegistry<T>::instance();
    return registry.template getOperator<NewOperator<T>>(OperatorType::NEW_OP, platform);
}
```

### 2.5 更新统一接口

在`backend/cpp/include/operators/unified_operators.hpp`中添加新算子的接口方法：

```cpp
// 新算子接口
void new_op(Tensor<T>* input, size_t param, float fixed_param, cudaStream_t stream = nullptr) {
    auto op = OperatorFactory<T>::getNewOperator(platform_);
    if (!op) {
        // 错误处理...
        throw std::runtime_error("New operator not registered for the current platform");
    }

    // 直接调用算子
    (*op)(input, param, fixed_param, stream);
}
```

### 2.6 使用新算子

在模型代码中使用新算子：

```cpp
// 在模型代码中使用新算子
size_t param_value = 42;
operators_->new_op(&input_tensor, param_value, 3.14f, compute_streams_[0]);
```

## 3. 高级优化技巧

### 3.1 CUDA图优化

通过标签固定内存分配，使得算子可以支持CUDA图优化：

1. **图创建**：在第一次调用时创建CUDA图
2. **固定内存**：通过标签确保张量内存地址固定，无需重建图
3. **图执行**：重复执行已创建的图，减少内核启动开销

### 3.2 算子融合

对于常见的算子组合，可以实现融合算子以减少内存访问和内核启动开销：

```cpp
// 融合算子示例：RMS Norm + Add
void add_rms(Tensor<T>* input, Tensor<T>* residual, Tensor<T>* output,
             Tensor<T>* weight, float eps, cudaStream_t stream = nullptr) {
    // 实现逻辑...
}
```

### 3.3 量化支持

为算子添加量化支持，以提高性能和减少内存占用：

```cpp
// 量化矩阵乘法算子
void matmul_quantized(Tensor<T>* input, Tensor<int8_t>* weight,
                      Tensor<float>* scales, Tensor<int8_t>* zeros,
                      int group_size, Tensor<T>* output,
                      cudaStream_t stream = nullptr, const Tensor<T>* bias = nullptr) {
    // 实现逻辑...
}
```

## 4. 最佳实践

1. **参数分析**：仔细分析每个参数，确定哪些需要使用标签固定内存
2. **错误处理**：提供清晰的错误信息，便于调试
3. **性能优化**：针对不同的硬件平台优化算子实现
4. **测试验证**：编写单元测试，确保算子在各种情况下都能正确工作
5. **文档注释**：为每个算子提供详细的文档，说明其功能、参数和使用方法

