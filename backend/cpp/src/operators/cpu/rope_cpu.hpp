#pragma once

#include <cmath>
#include "operators/operator_base.hpp"

namespace op {

template <typename T>
class RopeCPUOperator : public RopeOperator<T> {
public:
    RopeCPUOperator() = default;
    ~RopeCPUOperator() override = default;
    
    // 实现CPU版本的RoPE
    void operator()(Tensor<T>* x, size_t offset, float theta, cudaStream_t stream = nullptr) override {
        const auto& sizes = x->sizes();
        if (sizes.size() < 3) {
            throw std::runtime_error("rope: tensor must be at least 3D");
        }
        const size_t seq_len = sizes[0];
        const size_t n_heads = sizes[1];
        const size_t head_dim = sizes[2];
        const size_t dim_half = head_dim / 2;
        
        for (size_t s = 0; s < seq_len; s++) {
            for (size_t h = 0; h < n_heads; h++) {
                T* head_ptr = x->data_ptr() + s * n_heads * head_dim + h * head_dim;
                for (size_t i = 0; i < dim_half; i++) {
                    float freq = 1.0f / powf(theta, (2.0f * i) / head_dim);
                    float val = (s + offset) * freq;
                    float cos_val = cosf(val);
                    float sin_val = sinf(val);
                    const T x0 = head_ptr[i];
                    const T x1 = head_ptr[i + dim_half];
                    head_ptr[i] = x0 * cos_val - x1 * sin_val;
                    head_ptr[i + dim_half] = x0 * sin_val + x1 * cos_val;
                }
            }
        }
    }
    
    // 获取算子平台
    OperatorPlatform platform() const override { return OperatorPlatform::CPU; }
};

} // namespace op
