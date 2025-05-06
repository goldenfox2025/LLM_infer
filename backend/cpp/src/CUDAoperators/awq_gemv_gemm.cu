
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

// 假设 Tensor 类定义在 "tensor.h" 或其他地方
#include "cudaOP.cuh"  // 替换成你的 Tensor 头文件

// 用于调试输出的辅助函数 (Host 端)
template <typename T>
std::string vec_to_string(const std::vector<T>& vec) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    // 注意：这里直接输出可能对 __half, __nv_bfloat16 等类型不友好
    // 在实际调试中可能需要转换成 float
    ss << float(vec[i]) << (i == vec.size() - 1 ? "" : ", ");
  }
  ss << "]";
  return ss.str();
}

// CUDA 操作命名空间
namespace cuda_OP {
// --- 常量定义 ---
constexpr int BITS = 4;                 // AWQ 量化位数
constexpr int PACK_FACTOR = 32 / BITS;  // 一个 int32 可以打包多少个 4bit 数字
constexpr int WARP_SIZE = 32;           // CUDA Warp 大小
// --- GEMM Kernel (M > 1, N-Major 优化版) ---
// 针对 M > 1 的情况优化，假设权重、scales、zeros 为 N-Major 布局
// 优化点:
// 1. 改进了 sh_inp 的加载方式，让所有线程参与。
// 2. 缓存了当前 K-Tile group 的 scale 和 zero 到共享内存，减少全局内存读取。
// 前提假设: TILE_K <= group_size
template <typename T,   // 输入/输出数据类型 (half, bfloat16, float)
          typename S,   // Scale 数据类型 (通常是 float)
          int TILE_K,   // K 维度分块大小
          int BLOCK_N>  // Block 处理的 N 维度大小 (等于 Block 的线程数)
__global__ void matmul_awq_gemm_kernel_opt(
    const T* __restrict__ inp,        // 输入矩阵 [M, K]
    const int32_t* __restrict__ qwt,  // 量化权重 [N, K/8] (N-Major)
    const S* __restrict__ scl,        // 缩放因子 [N, G_padded] (N-Major)
    const int32_t* __restrict__ zos,  // 零点 [N, G/8] (N-Major)
    T* __restrict__ out,              // 输出矩阵 [M, N]
    int M, int K, int N,
    int group_size,  // AWQ group 大小
    int G_PADDED,    // !!! 新增: Scales 张量的实际第二维度 (Padding) !!!
    const T* __restrict__ bias) {  // 偏置向量 [N] (可选)
  // --- Grid/Block 映射 ---
  int m = blockIdx.y;                  // 输入行 (batch 索引)
  int tid = threadIdx.x;               // Block 内线程 ID (0 to BLOCK_N-1)
  int n = blockIdx.x * BLOCK_N + tid;  // 此线程负责的输出列 n

  // --- 边界检查 ---
  if (m >= M || n >= N) {
    return;
  }
  // --- 静态共享内存声明 ---
  // 注意：共享内存大小在编译时确定
  __shared__ T sh_inp[TILE_K];
  constexpr int K_PACKED_PER_TILE = (TILE_K + PACK_FACTOR - 1) / PACK_FACTOR;
  __shared__ int32_t sh_qwt[BLOCK_N][K_PACKED_PER_TILE];
  __shared__ S sh_scl_tile[BLOCK_N];
  __shared__ int32_t sh_zos_tile[BLOCK_N];

  // --- 常量计算 ---
  const int K_PACKED =
      (K + PACK_FACTOR - 1) / PACK_FACTOR;  // 总的 packed K 维度
  const int G = K / group_size;  // Group 的总数 (假设 K 可被 group_size 整除)
  const int G_PACKED =
      (G + PACK_FACTOR - 1) / PACK_FACTOR;  // Packed Group 的总数

  float acc = 0.0f;  // 累加器

  // --- K 维度分块循环 ---
  for (int kb = 0; kb < K; kb += TILE_K) {
    // --- 优化点 2: sh_inp 加载 (所有线程参与) ---
    __syncthreads();  // 在加载新 Tile 数据前同步
    for (int k_offset = tid; k_offset < TILE_K; k_offset += BLOCK_N) {
      int k_idx = kb + k_offset;
      sh_inp[k_offset] = (k_idx < K) ? inp[m * K + k_idx]
                                     : static_cast<T>(0.0f);  // K 边界检查
    }

    // --- 优化点 1: 加载 Scales & Zeros 到共享内存 ---
    int current_group_idx = kb / group_size;  // Tile 起始位置所属的 group
    if (current_group_idx < G) {
      // 使用传入的 G_PADDED 来计算正确的 stride
      sh_scl_tile[tid] = scl[n * G_PADDED + current_group_idx];

      int packed_g = current_group_idx / PACK_FACTOR;
      sh_zos_tile[tid] =
          zos[n * G_PACKED + packed_g];  // zos 使用 G_PACKED stride
    } else {
      sh_scl_tile[tid] = static_cast<S>(0.0f);
      sh_zos_tile[tid] = 0;
    }

    // --- 加载 Packed Weights 到共享内存 ---
    // 这里的全局内存访问模式是 Strided (跳跃 K_PACKED), 但索引是正确的
    for (int k_packed_offset = 0; k_packed_offset < K_PACKED_PER_TILE;
         ++k_packed_offset) {
      int k_local_start = k_packed_offset * PACK_FACTOR;
      int global_packed_k_idx = (kb + k_local_start) / PACK_FACTOR;
      if (global_packed_k_idx < K_PACKED) {
        sh_qwt[tid][k_packed_offset] =
            qwt[n * K_PACKED + global_packed_k_idx];  // N-Major 访问
      } else {
        sh_qwt[tid][k_packed_offset] = 0;
      }
    }

    __syncthreads();  // 确保所有共享内存加载完成

    // --- 计算核心循环 ---
    for (int t = 0; t < TILE_K; ++t) {
      int current_k = kb + t;     // 当前处理的全局 K 索引
      if (current_k >= K) break;  // K 边界检查

      float iv = static_cast<float>(sh_inp[t]);  // 从共享内存读输入

      // 从共享内存读取 Scale 和 Packed Zero (对应当前 Tile 的 group)
      S cur_s = sh_scl_tile[tid];
      int32_t cur_z_packed = sh_zos_tile[tid];

      // 提取当前 k 对应的 Zero-point
      int g =
          current_k / group_size;  // 重新计算 g (这里假设 TILE_K<=group_size, g
                                   // == current_group_idx)
      int inner_g = g % PACK_FACTOR;
      int shift_z = inner_g * BITS;
      uint32_t z = (cur_z_packed >> shift_z) & ((1 << BITS) - 1);

      // 从共享内存读取 Packed Weight
      int packed_k_in_tile = t / PACK_FACTOR;
      int32_t pw = sh_qwt[tid][packed_k_in_tile];

      // 提取当前 k 对应的 Weight
      // !!! 修正: 使用 current_k 计算 inner_k !!!
      int inner_k = current_k % PACK_FACTOR;
      int shift_w = inner_k * BITS;
      uint32_t w = (pw >> shift_w) & ((1 << BITS) - 1);

      // 反量化并累加 (FMA)
      float scale_val = static_cast<float>(cur_s);
      acc = __fmaf_rn(
          iv, (static_cast<float>(w) - static_cast<float>(z)) * scale_val, acc);
    }
  }  // 结束 K 分块循环

  // --- 写回结果 ---
  if (bias) {
    acc += static_cast<float>(bias[n]);
  }
  out[m * N + n] =
      static_cast<T>(acc);  // N-Major 输出? 不，输出通常是 M-Major [M, N]
}

// --- GEMV Kernel (M = 1, N-Major 优化版) ---
// 专门为 M = 1 优化，假设权重、scales、zeros 为 N-Major 布局
// 使用动态共享内存，因其大小依赖运行时的 K 和 G
template <typename T,        // 输入/输出数据类型
          typename S,        // Scale 数据类型
          int BLOCK_N_GEMV>  // Block 内的线程数 (必须是 WARP_SIZE 的倍数)
__global__ void matmul_awq_gemv_kernel_opt(
    const T* __restrict__ inp,        // 输入向量 [K]
    const int32_t* __restrict__ qwt,  // 权重 [N, K/8]
    const S* __restrict__ scl,        // Scales [N, G_padded]
    const int32_t* __restrict__ zos,  // Zeros [N, G/8]
    T* __restrict__ out,              // 输出向量 [N]
    int K, int N, int group_size,
    int G_PADDED,  // !!! 新增: Scales 张量的实际第二维度 (Padding) !!!
    const T* __restrict__ bias) {  // 偏置向量 [N]
  // --- 常量 ---
  static_assert(BLOCK_N_GEMV % WARP_SIZE == 0,
                "BLOCK_N_GEMV 必须是 WARP_SIZE 的倍数");
  const int G = K / group_size;
  const int K_PACKED = (K + PACK_FACTOR - 1) / PACK_FACTOR;
  const int G_PACKED = (G + PACK_FACTOR - 1) / PACK_FACTOR;
  // G_PADDED 作为参数传入

  // --- Grid/Block/Warp 映射 ---
  const int warps_per_block = BLOCK_N_GEMV / WARP_SIZE;
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;
  const int n = blockIdx.x * warps_per_block + warp_id;  // 此 Warp 负责的列 n

  if (n >= N) return;  // 边界检查

  // --- 动态共享内存声明与指针分配 ---
  extern __shared__ char sh_mem_raw[];
  T* sh_inp = reinterpret_cast<T*>(sh_mem_raw);  // sh_inp [K]
  S* sh_scl =
      reinterpret_cast<S*>(&sh_inp[K]);  // sh_scl [warps_per_block][G_PADDED]
  int32_t* sh_zos = reinterpret_cast<int32_t*>(
      &sh_scl[warps_per_block *
              G_PADDED]);  // sh_zos [warps_per_block][G_PACKED]

  // --- 加载共享内存 (Block 协作) ---
  // 加载 sh_inp
  for (int k_idx = threadIdx.x; k_idx < K; k_idx += BLOCK_N_GEMV) {
    sh_inp[k_idx] = inp[k_idx];
  }
  // 加载 sh_scl
  for (int g = lane; g < G; g += WARP_SIZE) {
    // 使用传入的 G_PADDED 计算 stride 和偏移
    sh_scl[warp_id * G_PADDED + g] = scl[n * G_PADDED + g];
  }
  // 加载 sh_zos
  for (int packed_g = lane; packed_g < G_PACKED; packed_g += WARP_SIZE) {
    sh_zos[warp_id * G_PACKED + packed_g] = zos[n * G_PACKED + packed_g];
  }
  __syncthreads();  // 确保加载完成

  // --- 计算核心 ---
  float acc = 0.0f;
  for (int k = lane; k < K; k += WARP_SIZE) {  // Warp 内线程并行处理 K
    float iv = static_cast<float>(sh_inp[k]);  // 读共享内存

    int g = k / group_size;
    // 使用 G_PADDED 计算共享内存索引
    S current_s = sh_scl[warp_id * G_PADDED + g];

    int packed_g = g / PACK_FACTOR;
    int32_t packed_z_val = sh_zos[warp_id * G_PACKED + packed_g];
    int inner_g = g % PACK_FACTOR;
    int shift_z = inner_g * BITS;
    uint32_t z = (packed_z_val >> shift_z) & ((1 << BITS) - 1);

    // 直接读全局内存权重 (N-Major, Coalesced)
    int packed_k = k / PACK_FACTOR;
    int32_t packed_w_val = qwt[n * K_PACKED + packed_k];
    int inner_k = k % PACK_FACTOR;  // 全局 k 计算 inner_k
    int shift_w = inner_k * BITS;
    uint32_t w = (packed_w_val >> shift_w) & ((1 << BITS) - 1);

    float scale_val = static_cast<float>(current_s);
    acc = __fmaf_rn(
        iv, (static_cast<float>(w) - static_cast<float>(z)) * scale_val, acc);
  }

// --- Warp 内规约 ---
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }

  // --- 写回结果 (Lane 0) ---
  if (lane == 0) {
    if (bias) {
      acc += static_cast<float>(bias[n]);
    }
    out[n] = static_cast<T>(acc);
  }
}
template <int BLOCK_N_GEMV, typename ScaleType = float>
__global__ void matmul_awq_gemv_bf16_vectorized_kernel(  // <--- 重命名 Kernel
    const __nv_bfloat16* __restrict__ inp,               // 输入向量 [K] (BF16)
    const int32_t* __restrict__ qwt,                     // 权重 [N, K/8]
    const ScaleType* __restrict__ scl,  // Scales [N, G_padded] (ScaleType)
    const int32_t* __restrict__ zos,    // Zeros [N, G/8]
    __nv_bfloat16* __restrict__ out,    // 输出向量 [N] (BF16)
    int K, int N, int group_size,
    int G_PADDED,                            // Scales 张量的实际第二维度
    const __nv_bfloat16* __restrict__ bias)  // 偏置向量 [N] (BF16, 可选)
{
  // --- 常量 ---
  static_assert(BLOCK_N_GEMV % WARP_SIZE == 0,
                "BLOCK_N_GEMV 必须是 WARP_SIZE 的倍数");
  const int G = K / group_size;
  const int K_PACKED = (K + PACK_FACTOR - 1) / PACK_FACTOR;
  const int G_PACKED = (G + PACK_FACTOR - 1) / PACK_FACTOR;
  constexpr int K_PER_THREAD = 8;  // 每个线程每次迭代处理 K 的数量
  constexpr int K_PER_WARP =
      WARP_SIZE * K_PER_THREAD;  // Warp 每次迭代处理的数量

  // --- Grid/Block/Warp 映射 ---
  const int warps_per_block = BLOCK_N_GEMV / WARP_SIZE;
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;
  const int n = blockIdx.x * warps_per_block + warp_id;  // 此 Warp 负责的列 n

  if (n >= N) return;  // 边界检查

  // --- 动态共享内存 ---
  extern __shared__ char sh_mem_raw[];
  __nv_bfloat16* sh_inp = reinterpret_cast<__nv_bfloat16*>(sh_mem_raw);
  ScaleType* sh_scl = reinterpret_cast<ScaleType*>(&sh_inp[K]);
  int32_t* sh_zos =
      reinterpret_cast<int32_t*>(&sh_scl[warps_per_block * G_PADDED]);
  constexpr int vec_unit = 16 / sizeof(__nv_bfloat16);
  // --- 加载共享内存 (保持不变) ---
  // 加载 sh_inp (BF16)
  // 注意 默认K是8倍数
  for (int k_idx = threadIdx.x; k_idx < K / vec_unit; k_idx += BLOCK_N_GEMV) {
    reinterpret_cast<float4*>(sh_inp)[k_idx] =
        reinterpret_cast<const float4*>(inp)[k_idx];
  }
  // 加载 sh_scl (Float)
  for (int g = lane; g < G; g += WARP_SIZE) {
    sh_scl[warp_id * G_PADDED + g] = scl[n * G_PADDED + g];
  }
  // 加载 sh_zos (int32_t)
  for (int packed_g = lane; packed_g < G_PACKED; packed_g += WARP_SIZE) {
    sh_zos[warp_id * G_PACKED + packed_g] = zos[n * G_PACKED + packed_g];
  }
  __syncthreads();  // 确保加载完成

  // --- 计算核心 (向量化修改) ---
  float acc = 0.0f;  // 使用 float 累加器

  // K 循环步长改为 K_PER_WARP
  for (int k_block_start = 0; k_block_start < K; k_block_start += K_PER_WARP) {
    // 计算当前线程负责的 8 个 K 值的起始索引
    int k_thread_start = k_block_start + lane * K_PER_THREAD;

    // 加载 1 个 packed weight (包含 8 个 INT4)
    // 这 8 个 K 值共享同一个 packed_k 索引
    int packed_k_idx = k_thread_start / PACK_FACTOR;
    int32_t packed_w_val = 0;
    // 边界检查: 确保 packed_k_idx 在有效范围内才加载
    if (k_thread_start < K && packed_k_idx < K_PACKED) {
      packed_w_val = qwt[n * K_PACKED + packed_k_idx];
    }

    // --- Scale & Zero 加载 (简化版) ---
    // !!! 警告: 假设 k_thread_start 到 k_thread_start + 7 均在同一 group 内 !!!
    // 基于 k_thread_start 确定 group, scale, 和 packed_zero
    int g = k_thread_start / group_size;
    float current_s = 0.0f;
    uint32_t z_vals[K_PER_THREAD];  // 存储解包后的 8 个 zero-point (通常相同)

    // 边界检查: 确保 g 在有效范围内
    if (k_thread_start < K && g < G) {
      current_s = sh_scl[warp_id * G_PADDED + g];  // 从共享内存加载 scale
      int packed_g = g / PACK_FACTOR;
      int32_t packed_z_val =
          sh_zos[warp_id * G_PACKED + packed_g];  // 从共享内存加载 packed zero

      // 解包 zero-point (我们只需要 group `g` 对应的 zero)
      int inner_g = g % PACK_FACTOR;
      int shift_z = inner_g * BITS;
      uint32_t z = (packed_z_val >> shift_z) & ((1 << BITS) - 1);

// 假设组内所有元素的 zero-point 相同 (因为 scale 相同)
#pragma unroll
      for (int i = 0; i < K_PER_THREAD; ++i) {
        z_vals[i] = z;
      }
    } else {
// 如果 k_thread_start 无效或 g 无效，设置 zero 为 0
#pragma unroll
      for (int i = 0; i < K_PER_THREAD; ++i) {
        z_vals[i] = 0;
      }
    }
// --- End Scale & Zero 加载 ---

// --- Unroll K_PER_THREAD (8) 次计算 ---
#pragma unroll
    for (int i = 0; i < K_PER_THREAD; ++i) {
      int current_k = k_thread_start + i;

      // 边界检查: 确保 current_k 在有效范围内
      if (current_k < K) {
        // 1. 加载输入并转换
        float iv = __bfloat162float(sh_inp[current_k]);

        // 2. 解包权重 w
        int inner_k = current_k % PACK_FACTOR;
        int shift_w = inner_k * BITS;
        uint32_t w = (packed_w_val >> shift_w) & ((1 << BITS) - 1);

        // 3. 获取 zero-point z (已解包)
        uint32_t z = z_vals[i];

        // 4. 执行 FMA
        float scale_val = static_cast<float>(current_s);
        acc = __fmaf_rn(
            iv, (static_cast<float>(w) - static_cast<float>(z)) * scale_val,
            acc);
      }
      // 如果 current_k >= K, 则不进行任何操作
    }
  }  // 结束 K 循环

// --- Warp 内规约 (保持不变) ---
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }

  // --- 写回结果 (Lane 0, 保持不变) ---
  if (lane == 0) {
    if (bias) {
      acc += __bfloat162float(bias[n]);
    }
    out[n] = __float2bfloat16_rn(acc);
  }
}
// --- Host 端启动函数 ---
template <typename T, typename ScaleType>
void matmul_quantized_gemv(
    const Tensor<T>& input,           // 输入 [M, K]
    const Tensor<int32_t>& qweight,   // 权重 [N, K/8] (N-Major)
    const Tensor<ScaleType>& scales,  // Scales [N, G_padded] (N-Major)
    const Tensor<int32_t>& zeros,     // Zeros [N, G/8] (N-Major)
    int group_size,                   // Group 大小
    Tensor<T>* output,                // 输出 [M, N]
    cudaStream_t stream,              // CUDA 流
    const Tensor<T>* bias) {          // 偏置 [N] (可选)

  // --- 输入参数检查 ---
  if (input.sizes().size() != 2) throw std::runtime_error("输入张量必须是 2D");
  int M = input.sizes()[0];
  int K = input.sizes()[1];

  if (qweight.sizes().size() != 2)
    throw std::runtime_error("权重张量必须是 2D");
  int N = qweight.sizes()[0];  // N-Major
  int K_PACKED_w = qweight.sizes()[1];

  if (scales.sizes().size() != 2)
    throw std::runtime_error("Scales 张量必须是 2D");
  if (scales.sizes()[0] != N) throw std::runtime_error("Scales N 维度不匹配");
  int G_PADDED = scales.sizes()[1];  // *** 获取 Scales 实际的第二维度 ***

  if (zeros.sizes().size() != 2)
    throw std::runtime_error("Zeros 张量必须是 2D");
  if (zeros.sizes()[0] != N) throw std::runtime_error("Zeros N 维度不匹配");
  int G_PACKED_z = zeros.sizes()[1];

  // ... (保留其他所有维度检查, K % group_size, K % PACK_FACTOR 等) ...
  if (bias && (bias->sizes().size() != 1 || bias->sizes()[0] != N)) {
    throw std::runtime_error("Bias 必须是 1D 且大小为 N (" + std::to_string(N) +
                             ")");
  }
  if (group_size <= 0) {
    throw std::runtime_error("group_size 必须为正数");
  }
  if (K % group_size != 0) {
    throw std::runtime_error("K (" + std::to_string(K) +
                             ") 必须能被 group_size (" +
                             std::to_string(group_size) + ") 整除");
  }
  // ... (保留 K % PACK_FACTOR 和 N % PACK_FACTOR 的警告或错误) ...

  // --- 重新计算维度用于验证和 Kernel 参数 ---
  int G = K / group_size;
  int K_PACKED = (K + PACK_FACTOR - 1) / PACK_FACTOR;
  int G_PACKED = (G + PACK_FACTOR - 1) / PACK_FACTOR;

  // --- 维度验证 ---
  if (K_PACKED_w != K_PACKED) {
    throw std::runtime_error("QWeight 的 K/8 维度 (" +
                             std::to_string(K_PACKED_w) + ") 不匹配，期望 " +
                             std::to_string(K_PACKED));
  }
  if (G_PACKED_z != G_PACKED) {
    throw std::runtime_error("Zeros 的 G/8 维度 (" +
                             std::to_string(G_PACKED_z) + ") 不匹配，期望 " +
                             std::to_string(G_PACKED));
  }
  if (G_PADDED < G) {
    throw std::runtime_error("Scales 的 G 维度 (" + std::to_string(G_PADDED) +
                             ") 小于计算出的 G (" + std::to_string(G) + ")");
  }

  // --- 根据 M 选择 Kernel ---
  if (M == 1) {
    // --- GEMV 路径 (M=1) ---
    constexpr int BLOCK_N_GEMV = 256;  // GEMV Kernel 的 Block 线程数
    constexpr int WARPS_PER_BLOCK = BLOCK_N_GEMV / WARP_SIZE;

    const dim3 grid_gemv((N + WARPS_PER_BLOCK - 1) /
                         WARPS_PER_BLOCK);  // 1D Grid
    const dim3 threads_gemv(BLOCK_N_GEMV);  // 1D Block

    // 计算 GEMV Kernel 所需的动态共享内存大小
    size_t shmem_size_gemv = K * sizeof(T);  // sh_inp
    shmem_size_gemv +=
        WARPS_PER_BLOCK * G_PADDED * sizeof(ScaleType);               // sh_scl
    shmem_size_gemv += WARPS_PER_BLOCK * G_PACKED * sizeof(int32_t);  // sh_zos

#ifndef NDEBUG  // 调试输出
    std::cout << "--- 启动优化版 GEMV Kernel (M=1) ---" << std::endl;
    std::cout << "Grid: (" << grid_gemv.x << "), Block: (" << threads_gemv.x
              << "), SharedMem: " << shmem_size_gemv << std::endl;
#endif
    // 调用 GEMV Kernel, 传递 G_PADDED
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      matmul_awq_gemv_bf16_vectorized_kernel<BLOCK_N_GEMV, ScaleType>
          <<<grid_gemv, threads_gemv, shmem_size_gemv, stream>>>(
              input.data_ptr(), qweight.data_ptr(), scales.data_ptr(),
              zeros.data_ptr(), output->data_ptr(), K, N, group_size,
              G_PADDED,  // !!! 传递 G_PADDED !!!
              bias ? bias->data_ptr() : nullptr);
    } else
      matmul_awq_gemv_kernel_opt<T, ScaleType, BLOCK_N_GEMV>
          <<<grid_gemv, threads_gemv, shmem_size_gemv, stream>>>(
              input.data_ptr(), qweight.data_ptr(), scales.data_ptr(),
              zeros.data_ptr(), output->data_ptr(), K, N, group_size,
              G_PADDED,  // !!! 传递 G_PADDED !!!
              bias ? bias->data_ptr() : nullptr);

  } else {
    // --- GEMM 路径 (M > 1) ---
    constexpr int BLOCK_N_GEMM = 256;  // GEMM Kernel 的 Block 线程数
    constexpr int TILE_K_GEMM = 32;    // K 分块大小

    // 检查前提
    if (TILE_K_GEMM > group_size) {
      std::cerr << "警告: GEMM Kernel 优化假设 TILE_K <= group_size。否则 "
                   "Scale/Zero 缓存逻辑需修改。"
                << std::endl;
    }

    const dim3 threads_gemm(BLOCK_N_GEMM);                           // 1D Block
    const dim3 grid_gemm((N + BLOCK_N_GEMM - 1) / BLOCK_N_GEMM, M);  // 2D Grid

#ifndef NDEBUG  // 调试输出
    std::cout << "--- 启动优化版 GEMM Kernel (M > 1) ---" << std::endl;
    std::cout << "Grid: (" << grid_gemm.x << ", " << grid_gemm.y
              << "), Block: (" << threads_gemm.x << ")" << std::endl;
#endif
    // 调用 GEMM Kernel, 传递 G_PADDED
    matmul_awq_gemm_kernel_opt<T, ScaleType, TILE_K_GEMM, BLOCK_N_GEMM>
        <<<grid_gemm, threads_gemm, 0, stream>>>(  // 使用静态共享内存
            input.data_ptr(), qweight.data_ptr(), scales.data_ptr(),
            zeros.data_ptr(), output->data_ptr(), M, K, N, group_size,
            G_PADDED,  // !!! 传递 G_PADDED !!!
            bias ? bias->data_ptr() : nullptr);
  }

  // --- CUDA 错误检查 ---
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::string error_msg =
        "CUDA kernel launch failed: " + std::string(cudaGetErrorString(err));
    error_msg += " [M=" + std::to_string(M) + ", K=" + std::to_string(K) +
                 ", N=" + std::to_string(N) +
                 ", gs=" + std::to_string(group_size) + "]";
    throw std::runtime_error(error_msg);
  }
}

template void matmul_quantized_gemv<float>(const Tensor<float>&,
                                           const Tensor<int32_t>&,
                                           const Tensor<float>&,
                                           const Tensor<int32_t>&, int,
                                           Tensor<float>*, cudaStream_t,
                                           const Tensor<float>*);
template void matmul_quantized_gemv<__nv_bfloat16>(
    const Tensor<__nv_bfloat16>&, const Tensor<int32_t>&, const Tensor<float>&,
    const Tensor<int32_t>&, int, Tensor<__nv_bfloat16>*, cudaStream_t,
    const Tensor<__nv_bfloat16>*);
template void matmul_quantized_gemv<__nv_bfloat16, __nv_bfloat16>(
    const Tensor<__nv_bfloat16>&, const Tensor<int32_t>&,
    const Tensor<__nv_bfloat16>&, const Tensor<int32_t>&, int,
    Tensor<__nv_bfloat16>*, cudaStream_t, const Tensor<__nv_bfloat16>*);
template void matmul_quantized_gemv<__half>(const Tensor<__half>&,
                                            const Tensor<int32_t>&,
                                            const Tensor<float>&,
                                            const Tensor<int32_t>&, int,
                                            Tensor<__half>*, cudaStream_t,
                                            const Tensor<__half>*);

}  // namespace cuda_OP