#include <cuda_runtime.h>
#include <math.h>

#include <cstdio>
#include <iostream>
#include <vector>

#include "cudaOP.cuh"

namespace cuda_OP {

// --------------------------------------------------
// LayerNorm的CUDA内核实现 - 第一阶段：计算均值和方差
// --------------------------------------------------
template <typename T>
__global__ void layer_norm_compute_stats_kernel(const T* input, float* mean, float* var, 
                                              int batch_size, int hidden_size) {
  // 使用warp-level reduction实现更高效的统计量计算
  // 每个线程块处理一个batch元素
  const int bidx = blockIdx.x;  // batch index
  
  // 阻止超出范围的访问
  if (bidx >= batch_size) return;
  
  // 指向当前batch的输入数据
  const T* input_row = input + bidx * hidden_size;
  
  // 线程内部统计
  float local_sum = 0.0f;
  float local_sq_sum = 0.0f;
  
  // 线程合作计算总和和平方和
  for (int tid = threadIdx.x; tid < hidden_size; tid += blockDim.x) {
    float val = static_cast<float>(input_row[tid]);
    local_sum += val;
    local_sq_sum += val * val;
  }
  
  // 使用shared memory进行块内归约
  __shared__ float s_sum[32];  // 假设每个warp使用一个元素
  __shared__ float s_sq_sum[32];
  
  // warp内归约
  unsigned int warp_id = threadIdx.x / 32;
  unsigned int lane_id = threadIdx.x % 32;
  
  // 使用warp级别的规约来计算均值和方差
  // 先在warp内进行规约
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    local_sq_sum += __shfl_down_sync(0xffffffff, local_sq_sum, offset);
  }
  
  // 保存warp的结果到shared memory
  if (lane_id == 0) {
    s_sum[warp_id] = local_sum;
    s_sq_sum[warp_id] = local_sq_sum;
  }
  
  // 确保所有warp都已写入
  __syncthreads();
  
  // 仅使用第一个warp来归约所有warp的结果
  if (warp_id == 0) {
    // 从shared memory加载自己的值
    local_sum = (lane_id < blockDim.x / 32) ? s_sum[lane_id] : 0.0f;
    local_sq_sum = (lane_id < blockDim.x / 32) ? s_sq_sum[lane_id] : 0.0f;
    
    // warp内规约
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
      local_sq_sum += __shfl_down_sync(0xffffffff, local_sq_sum, offset);
    }
    
    // 第一个线程写入结果
    if (lane_id == 0) {
      // 计算均值
      mean[bidx] = local_sum / static_cast<float>(hidden_size);
      // 计算方差：E[X^2] - E[X]^2
      var[bidx] = local_sq_sum / static_cast<float>(hidden_size) - 
                  (mean[bidx] * mean[bidx]);
    }
  }
}

// --------------------------------------------------
// LayerNorm的CUDA内核实现 - 第二阶段：归一化、缩放和偏移
// --------------------------------------------------
template <typename T>
__global__ void layer_norm_apply_kernel(const T* input, T* output, 
                                      const float* mean, const float* var,
                                      const T* weight, const T* bias,
                                      int batch_size, int hidden_size, 
                                      float eps) {
  const int bidx = blockIdx.x;  // batch index
  const int tid = blockIdx.y * blockDim.x + threadIdx.x;  // 线程在特征维度上的索引
  
  // 阻止超出范围的访问
  if (bidx >= batch_size || tid >= hidden_size) return;
  
  // 获取当前元素的索引
  const int idx = bidx * hidden_size + tid;
  
  // 应用LayerNorm公式：(x - E[x]) / sqrt(Var[x] + eps) * weight + bias
  float val = static_cast<float>(input[idx]);
  float normalized = (val - mean[bidx]) / sqrtf(var[bidx] + eps);
  
  if (weight != nullptr && bias != nullptr) {
    // 应用缩放和偏移
    output[idx] = static_cast<T>(normalized * static_cast<float>(weight[tid]) + 
                                static_cast<float>(bias[tid]));
  } else if (weight != nullptr) {
    // 只应用缩放
    output[idx] = static_cast<T>(normalized * static_cast<float>(weight[tid]));
  } else {
    // 不应用缩放和偏移
    output[idx] = static_cast<T>(normalized);
  }
}

// --------------------------------------------------
// Host端LayerNorm实现
// --------------------------------------------------
template <typename T>
void layer_norm(Tensor<T>* output, const Tensor<T>* input,
               const Tensor<T>* weight, const Tensor<T>* bias,
               float eps) {
  // 确保张量在CUDA设备上
  if (input->device() != Device::CUDA || output->device() != Device::CUDA) {
    throw std::runtime_error("Input and output tensors must be on CUDA device for layer_norm");
  }
  
  if (weight != nullptr && weight->device() != Device::CUDA) {
    throw std::runtime_error("Weight tensor must be on CUDA device for layer_norm");
  }
  
  if (bias != nullptr && bias->device() != Device::CUDA) {
    throw std::runtime_error("Bias tensor must be on CUDA device for layer_norm");
  }
  
  // 获取维度信息
  const auto& sizes = input->sizes();
  int ndim = sizes.size();
  
  if (ndim < 2) {
    throw std::runtime_error("Input tensor must have at least 2 dimensions for layer_norm");
  }
  
  // BatchSize是所有除了最后一维的乘积
  int batch_size = 1;
  for (int i = 0; i < ndim - 1; ++i) {
    batch_size *= sizes[i];
  }
  int hidden_size = sizes[ndim - 1];
  
  // 验证权重和偏置的维度
  if (weight != nullptr && weight->numel() != hidden_size) {
    throw std::runtime_error("Weight tensor size mismatch in layer_norm");
  }
  if (bias != nullptr && bias->numel() != hidden_size) {
    throw std::runtime_error("Bias tensor size mismatch in layer_norm");
  }
  
  // 分配临时存储用于均值和方差
  float* mean_device;
  float* var_device;
  cudaMalloc(&mean_device, batch_size * sizeof(float));
  cudaMalloc(&var_device, batch_size * sizeof(float));
  
  // 计算均值和方差
  int block_size = 256;
  int grid_size = batch_size;
  
  layer_norm_compute_stats_kernel<T><<<grid_size, block_size>>>(
    input->data_ptr(), mean_device, var_device, batch_size, hidden_size
  );
  
  // 应用归一化
  dim3 grid(batch_size, (hidden_size + block_size - 1) / block_size);
  layer_norm_apply_kernel<T><<<grid, block_size>>>(
    input->data_ptr(), output->data_ptr(), mean_device, var_device,
    weight != nullptr ? weight->data_ptr() : nullptr,
    bias != nullptr ? bias->data_ptr() : nullptr,
    batch_size, hidden_size, eps
  );
  
  // 检查CUDA错误
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA error in layer_norm: " + 
                            std::string(cudaGetErrorString(err)));
  }
  
  // 清理临时内存
  cudaFree(mean_device);
  cudaFree(var_device);
}

// 显式实例化模板函数
template void layer_norm<float>(Tensor<float>*, const Tensor<float>*,
                              const Tensor<float>*, const Tensor<float>*,
                              float);
template void layer_norm<nvbf16>(Tensor<nvbf16>*, const Tensor<nvbf16>*,
                               const Tensor<nvbf16>*, const Tensor<nvbf16>*,
                               float);

} // namespace cuda_OP
