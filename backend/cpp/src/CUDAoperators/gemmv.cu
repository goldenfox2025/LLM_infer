#include <cuda_bf16.h>  // For nv_bfloat16, __float2bfloat16, etc.
#include <cuda_fp16.h>  // For half, __float2half, __half2float, __hadd, half2
#include <stdint.h>

#include <cstdio>       // For fprintf, stderr
#include <stdexcept>    // For std::runtime_error, std::invalid_argument
#include <string>       // For std::string, std::to_string
#include <type_traits>  // For std::is_same, std::conditional

#include "cudaOP.cuh"  // For CUDA_OP::Vec, CUDA_OP::Vec_2

namespace cuda_OP {

//----------------------------------------------------------------------------//
// Helper Structs and Functions                                               //
//----------------------------------------------------------------------------//

// Define nv_bfloat162 vector type if needed and supported by architecture
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// Use native type if available
// typedef __nv_bfloat162 nv_bfloat162; // Uncomment if not auto-defined
#else
// Fallback struct if native type is not available or for older architectures
struct nv_bfloat162 {
  nv_bfloat16 x, y;
};
#endif

// Device function to convert bfloat16 to float (needed if not intrinsic)
__device__ inline float bfloat16_to_float(nv_bfloat16 val) {
  unsigned int ui;
  ui = *reinterpret_cast<unsigned short *>(&val);
  ui <<= 16;
  return *reinterpret_cast<float *>(&ui);
}

// Helper to get vector type based on scalar type
template <typename ScalarT>
struct VectorType {
  using type = typename std::conditional<
      std::is_same<ScalarT, float>::value, float2,
      typename std::conditional<
          std::is_same<ScalarT, half>::value, half2,
          typename std::conditional<std::is_same<ScalarT, nv_bfloat16>::value,
                                    nv_bfloat162, void>::type>::type>::type;
  static constexpr bool supported = !std::is_same<type, void>::value;
};

// Warp reduction function (uses float for intermediate accumulation)

__device__ inline float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = 32 / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset, 32);
  }
  return val;
}

// CUDA Error Checking Macro
#define CUDA_CHECK(call)                                                \
  do {                                                                  \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
      fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                 \
      throw std::runtime_error("CUDA error: " +                         \
                               std::string(cudaGetErrorString(err)));   \
    }                                                                   \
  } while (0)

//----------------------------------------------------------------------------//
// GEMMV Kernel                                                        //
//----------------------------------------------------------------------------//
template <typename T, typename type_acc, int block_size>
static __global__ void gemmv(
    const T *x,               // 矩阵
    const T *y,               // 向量
    type_acc *dst,            // 输出
    const int channel_ratio,  // q 头除以 ratio 得到对应的 kv 头
    const int stride_channel_x, const int stride_channel_y,
    const int stride_channel_dst) {
  const int64_t seq_idx = blockIdx.x;
  const int64_t channel = blockIdx.y;
  const int tid = threadIdx.x;
  constexpr int warp_size = 32;

  // 调整 x、y、dst 的指针位置
  x += seq_idx * (gridDim.y / channel_ratio) * stride_channel_x +
       (channel / channel_ratio) * stride_channel_x;
  y += channel * stride_channel_y;
  dst += channel * stride_channel_dst;

  __shared__ float smem[32];
  if (block_size > warp_size) {
    if (tid < warp_size) {
      smem[tid] = 0.0f;
    }
    __syncthreads();
  }

  float sumf = 0.0f;
  // 针对 nv_bfloat16 分支
  if constexpr (std::is_same<T, nv_bfloat16>::value) {
    // 定义每个向量载入单元中包含的 T 数量
    constexpr int vec_unit =
        16 / sizeof(T);  // 例如：nv_bfloat16 是 2 字节，则 vec_unit == 8
    // 将 x 和 y 分别 reinterpret_cast 为 Vec<T, vec_unit> 指针，利用 union 对
    // 16 字节数据进行载入
    const Vec<T, vec_unit> *x8 = reinterpret_cast<const Vec<T, vec_unit> *>(x);
    const Vec<T, vec_unit> *y8 = reinterpret_cast<const Vec<T, vec_unit> *>(y);

    for (int64_t col8 = tid; col8 < stride_channel_y / vec_unit;
         col8 += block_size) {
      Vec<T, vec_unit> xi = x8[col8];
      Vec<T, vec_unit> yi = y8[col8];
      for (int i = 0; i < vec_unit; i++) {
        sumf += float(xi.t[i]) * float(yi.t[i]);
      }
    }

  } else if constexpr (std::is_same<T, float>::value) {
    // 对于 float 类型，使用 float4 进行向量化载入
    const float4 *x4 = reinterpret_cast<const float4 *>(x);
    const float4 *y4 = reinterpret_cast<const float4 *>(y);
    for (int64_t col8 = tid; col8 < stride_channel_y / 4; col8 += block_size) {
      float4 xi = x4[col8];
      float4 yi = y4[col8];
      sumf += xi.x * yi.x + xi.y * yi.y + xi.z * yi.z + xi.w * yi.w;
    }
  } else {
    static_assert(std::is_same<T, void>::value, "unsupported type");
  }

  sumf = warp_reduce_sum(sumf);

  if (block_size > warp_size) {
    if (tid % warp_size == 0) {
      smem[tid / warp_size] = sumf;
    }

    __syncthreads();
    if (tid >= warp_size) {
      return;
    }
    sumf = smem[tid];
    sumf = warp_reduce_sum(sumf);
  }

  if (tid != 0) {
    return;
  }
  dst[seq_idx] = type_acc(sumf * rsqrtf(static_cast<float>(128)));
}

//----------------------------------------------------------------------------//
// Launcher Function                                                          //
//----------------------------------------------------------------------------//

template <typename T, typename AccT>
void launch_gemmv(const T *x,              // 矩阵
                  const T *y,              // 向量
                  AccT *dst,               // 输出
                  const int channel_size,  // 矩阵的列数 ratio之前
                  const int channel_ratio,  // q头除以ratio得到对应的kv头
                  const int row_size,       // 需要多少行乘法 totalseqlen
                  const int stride_channel_x,   // 矩阵的行步长
                  const int stride_channel_y,   // 向量的行步长
                  const int stride_channel_dst  // 输出的行步长
) {
  dim3 grid(row_size, channel_size);
  constexpr int block_size = 128;
  dim3 block(block_size);
  gemmv<T, AccT, block_size>
      <<<grid, block>>>(x, y, dst, channel_ratio, stride_channel_x,
                        stride_channel_y, stride_channel_dst);
}
template void launch_gemmv<float, float>(
    const float *x, const float *y, float *dst, const int channel_size,
    const int channel_ratio, const int row_size, const int stride_channel_x,
    const int stride_channel_y, const int stride_channel_dst);
template void launch_gemmv<nv_bfloat16, nv_bfloat16>(
    const nv_bfloat16 *x, const nv_bfloat16 *y, nv_bfloat16 *dst,
    const int channel_size, const int channel_ratio, const int row_size,
    const int stride_channel_x, const int stride_channel_y,
    const int stride_channel_dst);
template void launch_gemmv<nv_bfloat16, float>(
    const nv_bfloat16 *x, const nv_bfloat16 *y, float *dst,
    const int channel_size, const int channel_ratio, const int row_size,
    const int stride_channel_x, const int stride_channel_y,
    const int stride_channel_dst);
}  // namespace cuda_OP
