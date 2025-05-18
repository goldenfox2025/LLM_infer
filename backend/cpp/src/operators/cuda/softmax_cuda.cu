#include <cmath>
#include <iostream>
#include <stdexcept>

#include "operators/cuda/softmax_cuda.cuh"

// 使用 CUDA_CHECK 宏，但不包含整个 common.hpp
#define CUDA_CHECK(call)                                                \
  do {                                                                  \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
      fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                 \
      throw std::runtime_error(cudaGetErrorString(err));                \
    }                                                                   \
  } while (0)

namespace op {

// CUDA kernel 实现
template <typename T>
__global__ void softmax_kernel(T* output, const T* input, int dim_size,
                               int stride, int batch_size, bool mask,
                               int offset) {
  // 每个线程处理一个批次
  int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch_idx >= batch_size) return;

  // 计算当前批次的起始位置
  int batch_offset = batch_idx * stride * dim_size;
  const T* batch_input = input + batch_offset;
  T* batch_output = output + batch_offset;

  // 找到最大值
  float max_val = -INFINITY;
  for (int i = 0; i < dim_size; i++) {
    // 如果使用掩码，并且当前位置小于offset，则跳过
    if (mask && i < offset) {
      continue;
    }
    float val = static_cast<float>(batch_input[i * stride]);
    max_val = max(max_val, val);
  }

  // 计算指数和
  float sum = 0.0f;
  for (int i = 0; i < dim_size; i++) {
    if (mask && i < offset) {
      batch_output[i * stride] = static_cast<T>(0.0f);
    } else {
      float val = static_cast<float>(batch_input[i * stride]);
      float exp_val = expf(val - max_val);
      batch_output[i * stride] = static_cast<T>(exp_val);
      sum += exp_val;
    }
  }

  // 归一化
  for (int i = 0; i < dim_size; i++) {
    if (!(mask && i < offset)) {
      batch_output[i * stride] =
          static_cast<T>(static_cast<float>(batch_output[i * stride]) / sum);
    }
  }
}

// 算子实现
template <typename T>
void SoftmaxCUDAOperator<T>::operator()(Tensor<T>* output,
                                        const Tensor<T>* input, int dim,
                                        bool mask, int offset,
                                        cudaStream_t stream) {
  // 检查输入
  if (input->device() != Device::CUDA || output->device() != Device::CUDA) {
    throw std::runtime_error(
        "Softmax: Input and output tensors must be on CUDA device");
  }

  // 获取维度信息
  const auto& sizes = input->sizes();
  if (dim < 0) dim += sizes.size();
  if (dim < 0 || dim >= sizes.size()) {
    throw std::runtime_error("Softmax: Invalid dimension");
  }

  // 计算维度大小和步长
  int dim_size = sizes[dim];
  int stride = 1;
  for (int i = dim + 1; i < sizes.size(); i++) {
    stride *= sizes[i];
  }

  // 计算批次大小
  int batch_size = input->numel() / (dim_size * stride);

  // 启动 kernel
  int threads_per_block = 256;
  int blocks = (batch_size + threads_per_block - 1) / threads_per_block;
  softmax_kernel<T><<<blocks, threads_per_block, 0, stream>>>(
      output->data_ptr(), input->data_ptr(), dim_size, stride, batch_size, mask,
      offset);

  // 检查错误
  CUDA_CHECK(cudaGetLastError());
}

// 显式实例化
template class SoftmaxCUDAOperator<float>;
template class SoftmaxCUDAOperator<__nv_bfloat16>;

}  // namespace op
