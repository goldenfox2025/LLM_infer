#include <cmath>      // 包含 cmath 用于数学函数，如 powf, cosf, sinf
#include <iostream>   // 包含 iostream 用于错误输出 (std::cerr)
#include <stdexcept>  // 包含 stdexcept 用于抛出运行时错误 (std::runtime_error)

#include "operators/cuda/rope_cuda.cuh"  // 包含 RoPE CUDA 操作的头文件

namespace op {  // 定义在 op 命名空间内

// RoPE 操作的 CUDA 核函数 (通用模板版本)
template <typename T>
__global__ void rope_kernel(T *tensor, size_t seq_len, size_t n_heads,
                            size_t head_dim, size_t offset, float theta) {
  // 计算全局线程索引
  // Grid 维度设计为覆盖所有需要旋转的维度对
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // 总共需要处理的旋转对数量
  size_t total_rotations = seq_len * n_heads * (head_dim / 2);

  // 检查线程索引是否越界
  if (idx < total_rotations) {
    // 计算当前线程负责处理的维度、头和序列索引
    size_t dim_half = head_dim / 2;  // 头维度的一半
    size_t rot_dim =
        idx % dim_half;  // 当前处理的旋转维度索引 (0 到 dim_half-1)
    size_t tmp = idx / dim_half;
    size_t head_idx = tmp % n_heads;  // 当前处理的头索引
    size_t seq_idx = tmp / n_heads;   // 当前处理的序列位置索引

    // 计算指向当前处理的头的起始位置的指针
    T *head_ptr = tensor + seq_idx * n_heads * head_dim + head_idx * head_dim;

    // 计算 RoPE 旋转所需的频率、角度值、余弦和正弦
    // RoPE 频率计算公式: 1.0 / (theta^(2k / d))
    float freq = 1.0f / powf(theta, (2.0f * rot_dim) / head_dim);
    // 旋转角度 = (序列位置 + 偏移量) * 频率
    float val = (seq_idx + offset) * freq;
    float cos_val = cosf(val);  // 计算余弦值
    float sin_val = sinf(val);  // 计算正弦值

    // 应用旋转变换：
    // x_new = x * cos(m*theta_i) - y * sin(m*theta_i)
    // y_new = x * sin(m*theta_i) + y * cos(m*theta_i)
    // 其中 x 是 head_ptr[rot_dim], y 是 head_ptr[rot_dim + dim_half]
    float x0 = static_cast<float>(head_ptr[rot_dim]);
    float x1 = static_cast<float>(head_ptr[rot_dim + dim_half]);
    head_ptr[rot_dim] = static_cast<T>(x0 * cos_val - x1 * sin_val);
    head_ptr[rot_dim + dim_half] = static_cast<T>(x0 * sin_val + x1 * cos_val);
  }
}

// RoPE 操作的 CUDA 核函数 (__nv_bfloat16 特化版本)
// 这个版本为 bfloat16 做了优化，每个线程处理两个维度对 (共4个元素)
template <>
__global__ void rope_kernel<__nv_bfloat16>(__nv_bfloat16 *tensor,
                                           size_t seq_len, size_t n_heads,
                                           size_t head_dim, size_t offset,
                                           float theta) {
  // 计算全局线程索引
  // Grid 维度设计为覆盖所有需要处理的 bf16x2 对
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t head_dim_half = head_dim / 2;
  // 每个头需要处理的 bf16 元素对的数量 (head_dim/2 个元素 / 每个线程处理 2
  // 个元素 = head_dim/4)
  size_t pairs_per_head_half = head_dim_half / 2;
  // 总共需要处理的 bf16 元素对数量
  size_t total_vec_rotations = seq_len * n_heads * pairs_per_head_half;

  // 检查线程索引是否越界
  if (idx < total_vec_rotations) {
    // 计算当前线程负责处理的维度、头和序列索引
    size_t rot_dim_vec = idx % pairs_per_head_half;  // 当前处理的 bf16 对的索引
    size_t tmp = idx / pairs_per_head_half;
    size_t head_idx = tmp % n_heads;  // 当前处理的头索引
    size_t seq_idx = tmp / n_heads;   // 当前处理的序列位置索引

    // 计算指向当前处理的头的起始位置的指针
    __nv_bfloat16 *head_ptr =
        tensor + seq_idx * n_heads * head_dim + head_idx * head_dim;

    // 每个线程处理两个相邻的维度对
    size_t rot_dim = rot_dim_vec * 2;  // 第一个维度对的起始索引

    // --- 处理第一个维度对 (rot_dim, rot_dim + head_dim_half) ---
    float freq1 = 1.0f / powf(theta, (2.0f * rot_dim) / head_dim);
    float val1 = (seq_idx + offset) * freq1;
    float cos_val1 = cosf(val1);
    float sin_val1 = sinf(val1);

    float x0_1 = static_cast<float>(head_ptr[rot_dim]);
    float x1_1 = static_cast<float>(head_ptr[rot_dim + head_dim_half]);
    head_ptr[rot_dim] =
        static_cast<__nv_bfloat16>(x0_1 * cos_val1 - x1_1 * sin_val1);
    head_ptr[rot_dim + head_dim_half] =
        static_cast<__nv_bfloat16>(x0_1 * sin_val1 + x1_1 * cos_val1);

    // --- 处理第二个维度对 (rot_dim + 1, rot_dim + 1 + head_dim_half) ---
    // 注意：频率需要用 rot_dim + 1 来计算
    float freq2 = 1.0f / powf(theta, (2.0f * (rot_dim + 1)) / head_dim);
    float val2 = (seq_idx + offset) * freq2;
    float cos_val2 = cosf(val2);
    float sin_val2 = sinf(val2);

    float x0_2 = static_cast<float>(head_ptr[rot_dim + 1]);
    float x1_2 = static_cast<float>(head_ptr[rot_dim + 1 + head_dim_half]);
    head_ptr[rot_dim + 1] =
        static_cast<__nv_bfloat16>(x0_2 * cos_val2 - x1_2 * sin_val2);
    head_ptr[rot_dim + 1 + head_dim_half] =
        static_cast<__nv_bfloat16>(x0_2 * sin_val2 + x1_2 * cos_val2);
  }
}

// RoPE CUDA 算子的实现类
template <typename T>
void RopeCUDAOperator<T>::operator()(Tensor<T> *x, size_t offset, float theta,
                                     cudaStream_t stream) {
  // 获取输入张量的维度信息
  const auto &sizes = x->sizes();
  // 输入张量至少需要包含序列长度、头数量和头维度这三维
  if (sizes.size() < 3) {
    throw std::runtime_error(
        "RoPE: 输入张量至少需要是 3D (seq_len, n_heads, head_dim)");
  }

  // 确定序列长度、头数和头维度
  size_t seq_len, n_heads, head_dim;
  size_t batch_size = 1;

  // 根据张量维度确定批次大小和其他参数
  if (sizes.size() == 3) {
    // 3D张量: [seq_len, n_heads, head_dim]
    seq_len = sizes[0];
    n_heads = sizes[1];
    head_dim = sizes[2];
  } else {
    // 4D或更高维张量: [batch_size, seq_len, n_heads, head_dim]
    // 计算批次大小（除了最后三个维度外的所有维度的乘积）
    for (size_t i = 0; i < sizes.size() - 3; ++i) {
      batch_size *= sizes[i];
    }
    seq_len = sizes[sizes.size() - 3];
    n_heads = sizes[sizes.size() - 2];
    head_dim = sizes[sizes.size() - 1];
  }

  // 如果头维度为0，则无需执行任何操作
  if (head_dim == 0) return;
  // RoPE 操作要求头维度必须是偶数，因为它总是成对操作
  if (head_dim % 2 != 0) {
    throw std::runtime_error("RoPE: 头维度 (head_dim) 必须是偶数");
  }

  // 设置 CUDA 核函数启动配置
  int threads = 256;  // 每个块的线程数，这是一个常用的值，可以根据具体 GPU
                      // 架构和问题规模进行调优
  int blocks = 0;     // 需要启动的块数量，稍后计算
  void *kernel_ptr = nullptr;  // 指向要启动的核函数的指针

  // 根据数据类型 T 选择合适的核函数并计算 Grid 大小 (blocks)
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    // 对于 BF16 类型，我们使用特化的核函数
    // 特化核函数要求 head_dim 是 4 的倍数，因为它一次处理 4 个 bf16
    // (两个维度对)
    if (head_dim % 4 != 0) {
      // 如果不满足要求，可以选择回退到通用核函数或报错。这里选择报错。
      throw std::runtime_error(
          "RoPE (BF16): 优化的 BF16 内核要求 head_dim 是 4 的倍数");
    }
    // 计算 BF16 版本核函数需要的总工作项数（每个工作项处理 4 个元素/2对）
    size_t total_vec_rotations = seq_len * n_heads * (head_dim / 4);
    // 计算需要的块数 = (总工作项数 + 每个块的线程数 - 1) / 每个块的线程数
    // (向上取整)
    blocks = (total_vec_rotations + threads - 1) / threads;
    // 将内核指针指向 BF16 特化版本
    kernel_ptr = (void *)rope_kernel<__nv_bfloat16>;
  } else {
    // 对于其他类型 (如 float, __half)，使用通用模板核函数
    // 计算通用版本核函数需要的总工作项数（每个工作项处理 2 个元素/1对）
    size_t total_rotations = seq_len * n_heads * (head_dim / 2);
    // 计算需要的块数
    blocks = (total_rotations + threads - 1) / threads;
    // 将内核指针指向通用模板版本
    kernel_ptr = (void *)rope_kernel<T>;
  }

  // 对每个批次样本应用RoPE
  for (size_t b = 0; b < batch_size; b++) {
    // 计算当前批次样本的数据指针
    T *batch_ptr = x->data_ptr() + b * seq_len * n_heads * head_dim;

    // 启动 CUDA 核函数
    if (kernel_ptr == (void *)rope_kernel<T>) {
      // 启动通用模板 Kernel
      rope_kernel<T><<<blocks, threads, 0, stream>>>(
          batch_ptr, seq_len, n_heads, head_dim, offset, theta);
    } else if (kernel_ptr == (void *)rope_kernel<__nv_bfloat16>) {
      // 启动 BF16 特化 Kernel，注意需要将 data_ptr 转换为 __nv_bfloat16*
      rope_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
          reinterpret_cast<__nv_bfloat16 *>(batch_ptr),  // 类型转换
          seq_len, n_heads, head_dim, offset, theta);
    } else {
      // 理论上不应该执行到这里，因为上面已经覆盖了所有情况
      throw std::runtime_error("内部错误：未能选择有效的 RoPE CUDA 核函数");
    }

    // 检查 CUDA API 调用和核函数启动是否出错
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "CUDA error after RoPE kernel launch: "
                << cudaGetErrorString(err) << std::endl;
      throw std::runtime_error("RoPE CUDA kernel launch failed");
    }
  }
}

// 显式模板实例化：
// 这会告诉编译器为 float 和 __nv_bfloat16 这两种类型生成 RopeCUDAOperator
// 类的完整代码。 如果不显式实例化，链接器可能会找不到这些特定类型的实现。
template class RopeCUDAOperator<float>;
template class RopeCUDAOperator<__nv_bfloat16>;
// 如果需要支持 FP16 (__half)，也需要在这里添加:
// template class RopeCUDAOperator<__half>;

}  // namespace op