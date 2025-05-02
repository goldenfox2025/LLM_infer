// matmul_awq.cu

#include <cuda_bf16.h>     // for __nv_bfloat16 BF16数据类型
#include <cuda_fp16.h>     // for __half FP16数据类型
#include <cuda_runtime.h>  // CUDA运行时API
#include <stdint.h>        // for int32_t, uint32_t 标准整数类型

#include <iomanip>    // for std::setw, std::setprecision 格式化输出
#include <iostream>   // for std::cout 标准输入输出
#include <sstream>    // for std::stringstream 字符串流
#include <stdexcept>  // for std::runtime_error 运行时错误
#include <string>     // for std::string 字符串
#include <vector>     // for std::vector 动态数组

#include "cudaOP.cuh"  // 假设包含 Tensor 类的定义

namespace cuda_OP {
// 辅助函数: 格式化打印张量尺寸 (复用自旧版本)
inline std::string format_sizes(const std::vector<size_t> &sizes) {
  std::stringstream ss;
  ss << "(";
  for (size_t i = 0; i < sizes.size(); ++i) {
    ss << sizes[i] << (i == sizes.size() - 1 ? "" : ", ");
  }
  ss << ")";
  return ss.str();
}

// 辅助函数: 打印张量值用于调试 (复用)
template <typename T>
void debug_print_tensor(const Tensor<T> &tensor, const std::string &name,
                        int max_elements = 10) {
#ifdef DEBUG_AWQ  // 仅在启用 DEBUG_AWQ 时编译
  std::cout << "Tensor " << name << " " << format_sizes(tensor.sizes()) << ":"
            << std::endl;
  std::vector<T> host_data(tensor.numel());  // 在主机上分配内存
  // 将数据从设备复制到主机
  cudaMemcpy(host_data.data(), tensor.data_ptr(), tensor.numel() * sizeof(T),
             cudaMemcpyDeviceToHost);
  // 确定要打印的元素数量
  int num_to_print = std::min(static_cast<int>(tensor.numel()), max_elements);
  std::cout << std::fixed << std::setprecision(4);  // 设置浮点数打印格式
  for (int i = 0; i < num_to_print; ++i) {
    // 处理不同数据类型的打印
    float val;
    if constexpr (std::is_same_v<T, float>) {  // 如果是 float
      val = host_data[i];
    } else if constexpr (std::is_same_v<T, __half>) {         // 如果是 half
      val = __half2float(host_data[i]);                       // 转换为 float
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {  // 如果是 bfloat16
      val = __bfloat162float(host_data[i]);                   // 转换为 float
    } else {                                   // 其他整数类型等
      val = static_cast<float>(host_data[i]);  // 强制转换为 float
    }
    std::cout << "  [" << i << "] = " << val << std::endl;
  }
  if (tensor.numel() > max_elements) {  // 如果元素过多
    std::cout << "  ... (还有 " << tensor.numel() - max_elements << " 个元素)"
              << std::endl;
  }
  std::cout << std::defaultfloat << std::setprecision(6);  // 恢复默认打印格式
#endif
}
//----------------------------------------------------------------------------
// 常量和映射定义
//----------------------------------------------------------------------------
constexpr int BITS = 4;                 // 量化位数 (4-bit)
constexpr int PACK_FACTOR = 32 / BITS;  // 每个 int32 包含的元素数量 (8)

// AWQ 内部物理顺序映射: logical_inner_idx -> physical_inner_idx
// 用于计算解包时的位移。例如，逻辑上第1个元素(索引0)物理上在最低4位(索引0)，
// 逻辑上第2个元素(索引1)物理上在bit 16-19(索引4)，以此类推。
__constant__ const int LOGICAL_TO_PHYSICAL_INNER_IDX[PACK_FACTOR] = {
    0, 4, 1, 5, 2, 6, 3, 7};

//----------------------------------------------------------------------------
// 优化的 CUDA Kernel (AWQ 反量化 + Matmul 核心，使用 Shared Memory Tiling)
//----------------------------------------------------------------------------
template <typename T,  // 输入/输出数据类型 (e.g., __half, float)
          typename ScaleType,  // Scale 数据类型 (e.g., __half, float)
          int TILE_K_ = 32,  // K 维度分块大小 (影响 Shared Memory 和计算粒度)
          int BLOCK_N_ = 256,  // 每个 Block 处理的 N 维度大小 (线程数)
          int BLOCK_M_ = 1>  // 每个 Block 处理的 M 维度大小 (通常为 1)
__global__ void matmul_awq_kernel_optimized(
    const T *__restrict__ inp,  // 输入激活: [M, K], __restrict__
                                // 提示编译器指针不混叠
    const int32_t
        *__restrict__ qwt,  // 量化权重: [K, N / PACK_FACTOR] (AWQ 物理顺序)
    const ScaleType *__restrict__ scl,  // 缩放因子: [NumGroups, N] (逻辑顺序)
    const int32_t *__restrict__ zos,  // 量化零点: [NumGroups, N / PACK_FACTOR]
                                      // (AWQ 物理顺序)
    T *__restrict__ out,        // 输出结果: [M, N]
    const int M,                // 矩阵维度 M
    const int K,                // 矩阵维度 K
    const int N,                // 矩阵维度 N
    const int group_size,       // 量化分组大小
    const T *__restrict__ bias  // 可选的偏置: [N] (可以为 nullptr)
) {
  // --- 编译时常量 ---
  constexpr int TILE_K = TILE_K_;  // K维度瓦片大小
  constexpr int BLOCK_N = BLOCK_N_;  // Block的N维度大小，应与 blockDim.x 匹配
  // constexpr int BLOCK_M = BLOCK_M_;  // 每个 Block 处理的行数，这里假设为 1

  // --- Shared Memory 声明 ---
  // 缓存输入激活 (如果 BLOCK_M > 1, 需要调整大小)
  __shared__ T sh_inp[TILE_K];  // 大小为 TILE_K (因为 BLOCK_M = 1)
  // 缓存压缩后的权重 [TILE_K, BLOCK_N / PACK_FACTOR]
  // 每个 int32 包含 PACK_FACTOR 个 4bit 权重
  __shared__ int32_t sh_qwt[TILE_K][BLOCK_N / PACK_FACTOR];

  // --- 线程索引 ---
  const int tidx = threadIdx.x;  // Block 内的线程 ID (0 到 BLOCK_N-1)

  // 逻辑输出行索引 (每个 Block 处理一行)
  const int m = blockIdx.y;
  // 当前线程负责计算的逻辑输出列索引
  const int n = blockIdx.x * BLOCK_N + tidx;

  // 当前 Block 处理的压缩数据的起始列索引
  const int block_packed_col_start = blockIdx.x * (BLOCK_N / PACK_FACTOR);
  // 线程在 Block 内部压缩数据瓦片中的列偏移量 (0 到 BLOCK_N/PACK_FACTOR - 1)
  // tidx 0..7 对应 offset 0, tidx 8..15 对应 offset 1, ...
  const int block_packed_col_offset = tidx / PACK_FACTOR;
  // (注释掉的旧变量) 当前线程需要从全局内存加载的压缩列索引 (在 cooperative
  // loading 中未使用) const int packed_col_to_load = block_packed_col_start +
  // tidx;

  // --- 预计算 AWQ 重排信息 (对每个线程的逻辑输出列 n 是常量) ---
  // 逻辑索引 (0-7)，表示在 8 个元素的包内的位置
  const int logical_inner_idx = n % PACK_FACTOR;
  // 使用 AWQ 映射表找到物理索引 (0-7)
  const int physical_inner_col_idx =
      LOGICAL_TO_PHYSICAL_INNER_IDX[logical_inner_idx];
  // 根据物理位置计算解包所需的位移量
  const int bit_shift = physical_inner_col_idx * BITS;

  // --- 累加器初始化 ---
  float acc = 0.0f;  // 使用 float 进行累加，以提高精度

  // --- 检查 Grid 边界 (对于超出 M 范围的 Block 提前退出) ---
  if (m >= M) return;
  // 注意: 对于 n >= N 的线程，它们仍然可能参与加载 Shared Memory，
  // 但在计算和写回之前会进行检查并跳过。

  // --- 主循环: 沿 K 维度分块处理 ---
  for (int k_base = 0; k_base < K; k_base += TILE_K) {
    // --- 将数据加载到 Shared Memory ---
    __syncthreads();  // 同步，确保上一块计算完成，准备加载下一块

    // 1. 协作加载输入激活瓦片 (sh_inp)
    // 假设 TILE_K <= BLOCK_N 且 BLOCK_M = 1，每个线程加载一个元素
    if (tidx < TILE_K) {
      const int k_load = k_base + tidx;  // 计算全局 K 索引
      // 检查 K 维度边界
      if (k_load < K) {
        sh_inp[tidx] = inp[m * K + k_load];  // 从全局内存加载
      } else {
        // 如果超出 K 边界 (通常是最后一个瓦片)，用 0 填充
        sh_inp[tidx] = static_cast<T>(0.0f);
      }
    }

    // 2. 协作加载压缩量化权重瓦片 (sh_qwt)
    // 每个线程负责加载 sh_qwt 中的一部分数据
    // 使用线程索引跨步遍历要加载的元素，以实现内存合并访问
    for (int load_idx = tidx; load_idx < TILE_K * (BLOCK_N / PACK_FACTOR);
         load_idx += BLOCK_N) {
      // 计算在 Shared Memory 瓦片内的行索引 (tk_load) 和列索引
      // (packed_col_in_tile)
      const int tk_load =
          load_idx / (BLOCK_N / PACK_FACTOR);  // Shared Memory 行 (0..TILE_K-1)
      const int packed_col_in_tile =
          load_idx % (BLOCK_N / PACK_FACTOR);  // Shared Memory 列

      const int k_global = k_base + tk_load;  // 要加载元素的全局 K 索引
      // 要加载元素的全局压缩列索引
      const int packed_col_global_for_load =
          block_packed_col_start + packed_col_in_tile;

      // 检查全局内存访问边界
      if (k_global < K && packed_col_global_for_load < (N / PACK_FACTOR)) {
        // 从全局内存加载压缩权重到 Shared Memory
        sh_qwt[tk_load][packed_col_in_tile] =
            qwt[k_global * (N / PACK_FACTOR) + packed_col_global_for_load];
      } else {
        // 如果超出边界，用 0 填充 Shared Memory
        sh_qwt[tk_load][packed_col_in_tile] = 0;
      }
    }

    // 等待 Block 内所有线程完成 Shared Memory 加载
    __syncthreads();

    // --- 使用 Shared Memory 进行计算 ---
    // 对于完全超出 N 边界的线程，提前退出此瓦片的计算循环
    if (n >= N) continue;

    // 缓存当前 group 的量化参数，避免重复访存
    int current_group_idx = -1;      // 当前处理的分组索引
    int32_t current_packed_zos = 0;  // 当前分组的压缩零点
    ScaleType current_scale_val =
        static_cast<ScaleType>(0.0f);  // 当前分组和列n的 Scale

    // 循环处理当前加载到 Shared Memory 的 K 维度瓦片
    // #pragma unroll // 可选: 对于小的 TILE_K，可以考虑展开循环，需要性能分析
    for (int tk = 0; tk < TILE_K; ++tk) {
      const int k = k_base + tk;  // 当前全局 K 索引
      // 检查 k 是否超出 K 边界 (因为最后一个瓦片可能被填充)
      if (k >= K) break;

      // 1. 从 Shared Memory 获取输入激活值
      const float inp_val =
          static_cast<float>(sh_inp[tk]);  // 转换为 float 计算

      // 2. 获取量化参数 (Scale & Zero) - 仅在 group 改变时加载
      const int group_idx = k / group_size;  // 计算当前 K 对应的分组索引
      if (group_idx != current_group_idx) {  // 如果分组变化
        current_group_idx = group_idx;       // 更新当前分组索引
        // 安全地计算总分组数
        const int num_groups = (K + group_size - 1) / group_size;

        // 计算全局压缩零点索引 (同一压缩列的线程读取相同 packed_zos)
        const int zos_packed_col_global =
            block_packed_col_start + block_packed_col_offset;
        const int zos_idx =
            group_idx * (N / PACK_FACTOR) + zos_packed_col_global;

        // 检查访问 zeros 的边界
        if (group_idx < num_groups &&
            zos_packed_col_global < (N / PACK_FACTOR)) {
          current_packed_zos = zos[zos_idx];  // 加载压缩零点
        } else {
          current_packed_zos = 0;  // 超出边界则使用 0 填充
        }

        // 计算全局 Scale 索引 (每个逻辑列 n 对应不同的 scale)
        const int scl_idx = group_idx * N + n;  // 使用逻辑列索引 'n'

        // 检查访问 scales 的边界 (n < N 已在上层检查, 只需检查 group_idx)
        if (group_idx < num_groups) {
          current_scale_val = scl[scl_idx];  // 加载 Scale
        } else {
          current_scale_val =
              static_cast<ScaleType>(0.0f);  // 超出边界则使用 0 填充
        }
      }  // 分组参数加载结束

      // 3. 使用 Shared Memory 和缓存的参数反量化权重
      // 3a. 从 Shared Memory 瓦片获取压缩权重
      //    使用 tk 作为行索引, block_packed_col_offset 作为列索引
      const int32_t packed_qwt_tile = sh_qwt[tk][block_packed_col_offset];

      // 3b. 使用预计算的物理索引和缓存的零点解包 4-bit 权重和零点
      //    从 packed_qwt_tile 中提取当前线程对应的 4bit 权重 q_w
      const uint32_t q_w =
          (static_cast<uint32_t>(packed_qwt_tile) >> bit_shift) &
          0x0F;  // 位移和掩码操作
      //    从 current_packed_zos 中提取对应的 4bit 零点 q_z
      const uint32_t q_z =
          (static_cast<uint32_t>(current_packed_zos) >> bit_shift) &
          0x0F;  // 位移和掩码操作

      // 3c. 使用缓存的 scale 计算反量化后的权重
      const float scale_float =
          static_cast<float>(current_scale_val);  // 转为 float
      const float dequant_w =
          (static_cast<float>(q_w) - static_cast<float>(q_z)) *
          scale_float;  // 反量化公式

      // 4. 使用 FMA (Fused Multiply-Add) 累加乘积
      //    __fmaf_rn: round-to-nearest-even 模式的单精度 FMA 指令
      acc = __fmaf_rn(inp_val, dequant_w, acc);

    }  // 结束 tk 循环 (处理 TILE_K)

    // 在下一次 k_base 迭代开始时，循环顶部的 __syncthreads() 会进行同步

  }  // 结束 k_base 循环 (处理所有 K 维度瓦片)

  // --- 添加偏置 (如果提供) ---
  // 在访问 bias 和写入输出前再次检查 'n' 边界
  if (n < N) {
    if (bias != nullptr) {                 // 如果偏置存在
      acc += static_cast<float>(bias[n]);  // 添加偏置 (转为 float 加)
    }

    // --- 将结果存储回全局内存 ---
    out[m * N + n] = static_cast<T>(acc);  // 将累加结果转回输出类型 T 并存储
  }

}  // 优化 Kernel 函数结束

//----------------------------------------------------------------------------
// 封装函数 (调用优化的计算内核) - 包含来自原始 v12 的验证逻辑
//----------------------------------------------------------------------------
template <typename T>  // 模板参数为输入/输出数据类型
void matmul_quantized(
    const Tensor<T> &input,  // 输入张量 [M, K]
    const Tensor<int32_t>
        &qweight,  // 量化权重张量 [K, N / PACK_FACTOR] (AWQ物理顺序)
    const Tensor<float> &scales_input,  // 缩放因子张量 [NumGroups, N]
                                        // (封装层假设为 float, 逻辑顺序)
    const Tensor<int32_t> &
        zeros_input,  // 量化零点张量 [NumGroups, N / PACK_FACTOR] (AWQ物理顺序)
    int group_size,  // 分组大小
    Tensor<T> *output,  // 输出张量指针 [M, N] (需要预先分配好内存和形状)
    cudaStream_t stream,   // CUDA 流
    const Tensor<T> *bias  // 可选的偏置张量 [N] (可以为 nullptr)
) {
  // --- 输入验证、维度推导、形状检查 (来自用户提供的 v12) ---
  const int M = static_cast<int>(input.sizes()[0]);  // 获取维度 M
  const int K = static_cast<int>(input.sizes()[1]);  // 获取维度 K
  int N = 0;                                         // 初始化维度 N
  // 注意: 下面的 N 推断逻辑来自 v12，可能不够健壮
  if (scales_input.sizes().size() == 2)  // 优先从 scales 推断 N
    N = static_cast<int>(scales_input.sizes()[1]);
  else if (output->sizes().size() ==
           2)  // 其次从 output 推断 N (要求 output 已设置形状)
    N = static_cast<int>(output->sizes()[1]);
  else if (qweight.sizes().size() == 2)  // 再次从 qweight 推断 N
    N = static_cast<int>(qweight.sizes()[1]) * PACK_FACTOR;
  else
    // 如果无法推断 N，则抛出错误 (保持 v12 的行为)
    throw std::runtime_error("无法确定维度 N");

  // 基础维度和 group_size 验证 (来自 v12)
  // K 必须是 group_size 的整数倍
  if (K <= 0 || N <= 0 || M <= 0 || group_size <= 0 || K % group_size != 0) {
    throw std::runtime_error("无效的维度或 group_size. M=" + std::to_string(M) +
                             ", K=" + std::to_string(K) +
                             ", N=" + std::to_string(N) +
                             ", group_size=" + std::to_string(group_size));
  }
  const int NumGroups = K / group_size;  // 计算分组数量

  // 形状一致性检查 (来自 v12)
  if (input.sizes().size() != 2 || input.sizes()[0] != M ||
      input.sizes()[1] != K)
    throw std::runtime_error("Input 形状不匹配. 期望 (" + std::to_string(M) +
                             ", " + std::to_string(K) + "), 实际 " +
                             format_sizes(input.sizes()));
  if (qweight.sizes().size() != 2 || qweight.sizes()[0] != K ||
      qweight.sizes()[1] != (N / PACK_FACTOR))
    throw std::runtime_error("QWeight 形状不匹配. 期望 (" + std::to_string(K) +
                             ", " + std::to_string(N / PACK_FACTOR) +
                             "), 实际 " + format_sizes(qweight.sizes()));
  if (scales_input.sizes().size() != 2 ||
      scales_input.sizes()[0] != NumGroups || scales_input.sizes()[1] != N)
    throw std::runtime_error(
        "Scales 形状不匹配. 期望 (" + std::to_string(NumGroups) + ", " +
        std::to_string(N) + "), 实际 " + format_sizes(scales_input.sizes()));
  if (zeros_input.sizes().size() != 2 || zeros_input.sizes()[0] != NumGroups ||
      zeros_input.sizes()[1] != (N / PACK_FACTOR))
    throw std::runtime_error("Zeros 形状不匹配. 期望 (" +
                             std::to_string(NumGroups) + ", " +
                             std::to_string(N / PACK_FACTOR) + "), 实际 " +
                             format_sizes(zeros_input.sizes()));
  // 输出张量检查：理想情况下应确保输出形状是 M,N。
  // v12 的检查方式是如果 output 已分配，则检查其形状是否为 M, N。保持这种风格。

  if (bias &&
      (bias->sizes().size() != 1 || bias->sizes()[0] != N))  // 检查偏置形状
    throw std::runtime_error("Bias 形状不匹配. 期望 (" + std::to_string(N) +
                             "), 实际 " + format_sizes(bias->sizes()));

  // --- Kernel 启动配置 ---
  // 定义瓦片/块维度 (可以调整以优化性能)
  constexpr int TILE_K_VAL = 32;  // K 维度瓦片大小
  constexpr int BLOCK_N_VAL =
      256;  // 每个 Block 的线程数 (必须是 WARP_SIZE=32 和 PACK_FACTOR=8 的倍数)
  constexpr int BLOCK_M_VAL = 1;  // 每个 Block 处理的行数

  // 编译时检查 BLOCK_N_VAL 的有效性
  static_assert(BLOCK_N_VAL % PACK_FACTOR == 0,
                "BLOCK_N 必须是 PACK_FACTOR (8) 的倍数");
  static_assert(BLOCK_N_VAL % 32 == 0, "BLOCK_N 应该是 Warp 大小 (32) 的倍数");

  // 设置 Block 维度 (1D Block，大小为 BLOCK_N_VAL)
  const dim3 block_size(BLOCK_N_VAL);
  // 设置 Grid 维度:
  // X 维度覆盖 N 列 (向上取整)
  // Y 维度覆盖 M 行
  const dim3 grid_size((N + BLOCK_N_VAL - 1) / BLOCK_N_VAL, M);

#ifdef DEBUG_AWQ  // 如果定义了 DEBUG_AWQ
  std::cout << "\n启动 AWQ 优化 Kernel (v14 based on v12)..." << std::endl;
  std::cout << "  Grid 维度: (" << grid_size.x << ", " << grid_size.y << ")"
            << std::endl;
  std::cout << "  Block 维度: (" << block_size.x << ")" << std::endl;
  std::cout << "  Tile K: " << TILE_K_VAL << std::endl;
  std::cout << "  M=" << M << ", K=" << K << ", N=" << N
            << ", group_size=" << group_size << std::endl;
  // 打印输入张量信息
  debug_print_tensor(input, "Input");
  debug_print_tensor(qweight, "QWeight");
  debug_print_tensor(scales_input, "Scales");
  debug_print_tensor(zeros_input, "Zeros");
  if (bias) debug_print_tensor(*bias, "Bias");
#endif

  // --- 选择 Scale 类型 ---
  // 根据接口定义，假设 scales 是 float 类型
  using ScaleType = float;

  // --- 启动优化的 CUDA Kernel ---
  matmul_awq_kernel_optimized<T, ScaleType, TILE_K_VAL, BLOCK_N_VAL,
                              BLOCK_M_VAL>
      <<<grid_size, block_size, 0, stream>>>(  // 配置网格、块、共享内存(0)、流
          input.data_ptr(),                    // 输入数据指针
          qweight.data_ptr(),                  // 量化权重指针
          scales_input.data_ptr(),           // 缩放因子指针
          zeros_input.data_ptr(),            // 量化零点指针
          output->data_ptr(),                // 输出数据指针
          M, K, N,                           // 维度 M, K, N
          group_size,                        // 分组大小
          bias ? bias->data_ptr() : nullptr  // 偏置指针 (如果 bias 非空)
      );

  // --- 错误检查 ---
  cudaError_t launch_err = cudaGetLastError();  // 获取 Kernel 启动错误
  if (launch_err != cudaSuccess) {
    std::cerr << "CUDA kernel 启动错误 (matmul_awq_kernel_optimized): "
              << cudaGetErrorString(launch_err) << std::endl;
    throw std::runtime_error("CUDA kernel 启动错误");
  }

#ifdef DEBUG_AWQ  // 如果定义了 DEBUG_AWQ
  // 可选: 同步流并打印输出以进行调试
  cudaStreamSynchronize(stream);              // 等待 Kernel 执行完毕
  cudaError_t sync_err = cudaGetLastError();  // 获取同步错误
  if (sync_err != cudaSuccess) {
    std::cerr << "Kernel 执行后 CUDA 同步错误: " << cudaGetErrorString(sync_err)
              << std::endl;
  } else {
    debug_print_tensor(*output, "Output", 20);  // 打印输出张量的前 20 个元素
  }
#endif
}

// --- 显式模板实例化 ---
// 为支持的类型 (float, bfloat16, half) 显式实例化 matmul_quantized 函数模板
// 这确保了这些类型的代码会被编译和链接，即使它们没有在其他地方被显式调用。
template void matmul_quantized<float>(const Tensor<float> &,
                                      const Tensor<int32_t> &,
                                      const Tensor<float> &,
                                      const Tensor<int32_t> &, int,
                                      Tensor<float> *, cudaStream_t,
                                      const Tensor<float> *);

template void matmul_quantized<__nv_bfloat16>(
    const Tensor<__nv_bfloat16> &, const Tensor<int32_t> &,
    const Tensor<float> &, const Tensor<int32_t> &, int,
    Tensor<__nv_bfloat16> *, cudaStream_t, const Tensor<__nv_bfloat16> *);

template void matmul_quantized<__half>(const Tensor<__half> &,
                                       const Tensor<int32_t> &,
                                       const Tensor<float> &,
                                       const Tensor<int32_t> &, int,
                                       Tensor<__half> *, cudaStream_t,
                                       const Tensor<__half> *);

}  // namespace cuda_OP
