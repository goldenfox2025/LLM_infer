#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cudaOP.cuh"

template <typename T>
void debugPrintTensor(const Tensor<T>& tensor, const std::string& tensor_name, size_t num_to_print = 10) {
    std::cout << "[Debug] " << tensor_name << ":\n";

    // 1) Print shape
    std::cout << "  shape: [";
    for (auto s : tensor.sizes()) {
        std::cout << s << " ";
    }
    std::cout << "]\n";

    // 2) Print strides
    std::cout << "  strides: [";
    for (auto st : tensor.strides()) {
        std::cout << st << " ";
    }
    std::cout << "]\n";

    // 3) Print device
    std::cout << "  device: ";
    if (tensor.device() == Device::CPU) {
        std::cout << "CPU";
    } else if (tensor.device() == Device::CUDA) {
        std::cout << "CUDA";
    } else {
        std::cout << "UNKNOWN";
    }
    std::cout << "\n";

    // 4) Print elements starting from offset 0
    size_t offset = 0;  // 从开始处打印
    size_t total_elements = tensor.numel();
    size_t n_print = std::min(num_to_print, total_elements - offset);

    std::cout << "  elements from offset " << offset << " (" << n_print << " element(s)): ";

    // Copy from GPU to CPU, then print
    std::vector<T> host_buffer(n_print);
    cudaError_t err =
        cudaMemcpy(host_buffer.data(), tensor.data_ptr() + offset, n_print * sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cout << "  [Error] cudaMemcpy failed\n";
        return;
    }
    for (size_t i = 0; i < n_print; i++) {
        std::cout << static_cast<float>(host_buffer[i]) << " ";
    }
    std::cout << "\n";
}

// 为__nv_bfloat16类型特化debugPrintTensor函数
template <>
void debugPrintTensor<__nv_bfloat16>(const Tensor<__nv_bfloat16>& tensor, const std::string& tensor_name,
                                     size_t num_to_print) {
    std::cout << "[Debug] " << tensor_name << ":\n";

    // 1) Print shape
    std::cout << "  shape: [";
    for (auto s : tensor.sizes()) {
        std::cout << s << " ";
    }
    std::cout << "]\n";

    // 2) Print strides
    std::cout << "  strides: [";
    for (auto st : tensor.strides()) {
        std::cout << st << " ";
    }
    std::cout << "]\n";

    // 3) Print device
    std::cout << "  device: ";
    if (tensor.device() == Device::CPU) {
        std::cout << "CPU";
    } else if (tensor.device() == Device::CUDA) {
        std::cout << "CUDA";
    } else {
        std::cout << "UNKNOWN";
    }
    std::cout << "\n";

    // 4) Print elements starting from offset 0
    size_t offset = 0;  // 从开始处打印
    size_t total_elements = tensor.numel();
    size_t n_print = std::min(num_to_print, total_elements - offset);

    std::cout << "  elements from offset " << offset << " (" << n_print << " element(s)): ";

    // Copy from GPU to CPU, then print
    std::vector<__nv_bfloat16> host_buffer(n_print);
    cudaError_t err = cudaMemcpy(host_buffer.data(), tensor.data_ptr() + offset, n_print * sizeof(__nv_bfloat16),
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cout << "  [Error] cudaMemcpy failed\n";
        return;
    }
    for (size_t i = 0; i < n_print; i++) {
        std::cout << __bfloat162float(host_buffer[i]) << " ";
    }
    std::cout << "\n";
}

// 为__half类型特化debugPrintTensor函数
template <>
void debugPrintTensor<__half>(const Tensor<__half>& tensor, const std::string& tensor_name, size_t num_to_print) {
    std::cout << "[Debug] " << tensor_name << ":\n";

    // 1) Print shape
    std::cout << "  shape: [";
    for (auto s : tensor.sizes()) {
        std::cout << s << " ";
    }
    std::cout << "]\n";

    // 2) Print strides
    std::cout << "  strides: [";
    for (auto st : tensor.strides()) {
        std::cout << st << " ";
    }
    std::cout << "]\n";

    // 3) Print device
    std::cout << "  device: ";
    if (tensor.device() == Device::CPU) {
        std::cout << "CPU";
    } else if (tensor.device() == Device::CUDA) {
        std::cout << "CUDA";
    } else {
        std::cout << "UNKNOWN";
    }
    std::cout << "\n";

    // 4) Print elements starting from offset 0
    size_t offset = 0;  // 从开始处打印
    size_t total_elements = tensor.numel();
    size_t n_print = std::min(num_to_print, total_elements - offset);

    std::cout << "  elements from offset " << offset << " (" << n_print << " element(s)): ";

    // Copy from GPU to CPU, then print
    std::vector<__half> host_buffer(n_print);
    cudaError_t err =
        cudaMemcpy(host_buffer.data(), tensor.data_ptr() + offset, n_print * sizeof(__half), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cout << "  [Error] cudaMemcpy failed\n";
        return;
    }
    for (size_t i = 0; i < n_print; i++) {
        std::cout << __half2float(host_buffer[i]) << " ";
    }
    std::cout << "\n";
}

// CUDA 操作命名空间
namespace cuda_OP {
// --- 常量定义 ---
constexpr int BITS = 4;                 // AWQ 量化位数
constexpr int PACK_FACTOR = 32 / BITS;  // 一个 int32 可以打包多少个 4bit 数字
constexpr int WARP_SIZE = 32;           // CUDA Warp 大小

// gemv 格式的 awq 权重打包是顺序的
// 仅支持bf16
template <typename T,                                                  // 输入/输出数据类型 (half, bfloat16, float)
          typename S,                                                  // Scale 数据类型 (通常是 float)
          int BM,                                                      // M 方向的 Block 大小
          int BN,                                                      // N 方向的 Block 大小
          int BK,                                                      // K 方向的 Block 大小
          int WMMA_M,                                                  // WMMA 指令的 M 方向大小
          int WMMA_N,                                                  // WMMA 指令的 N 方向大小
          int WMMA_K,                                                  // WMMA 指令的 K 方向大小
          int WARP_CNT>                                                // 每个 Block 的 Warp 数量
__global__ void matmul_awq_gemm_kernel_opt(const T* __restrict__ inp,  // 输入矩阵 [M, K]
                                           const int32_t* __restrict__ qwt,  // 量化权重 [N, K/8] (N-Major)
                                           const S* __restrict__ scl,        // 缩放因子 [N, G_padded] (N-Major)
                                           const int32_t* __restrict__ zos,  // 零点 [N, G/8] (N-Major)
                                           T* __restrict__ out,              // 输出矩阵 [M, N]
                                           int M, int K, int N,
                                           int group_size,  // AWQ group 大小
                                           int G_PADDED,    // !!! 新增: Scales 张量的实际第二维度 (Padding) !!!
                                           const T* __restrict__ bias) {
    const int G = K / group_size;
    const int K_PACKED = (K + PACK_FACTOR - 1) / PACK_FACTOR;
    const int G_PACKED = (G + PACK_FACTOR - 1) / PACK_FACTOR;
    const int WarpsN = BN / WMMA_N;

    int warp_id = threadIdx.x / WARP_SIZE;
    int warp_m_id = warp_id / WarpsN;
    int warp_n_id = warp_id % WarpsN;

    int m_global_start = blockIdx.x * BM;
    int n_global_start = blockIdx.y * BN;
    using namespace nvcuda;
    using FragmentA = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T, wmma::row_major>;
    using FragmentB = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T, wmma::row_major>;
    using FragmentC = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;
    __shared__ T smemA[BM * BK];
    __shared__ T smemB[BK * BN];
    __shared__ float smem_out[BM][BN];
    FragmentC fragC;
    wmma::fill_fragment(fragC, 0.0f);
    constexpr int vec_unit = 16 / sizeof(T);
    for (int tile_k = 0; tile_k < K; tile_k += BK) {
        // 加载A矩阵
        for (int load_idx = threadIdx.x; load_idx < BM * BK / vec_unit; load_idx += blockDim.x) {
            int load_row = load_idx / (BK / vec_unit);
            int load_pack_col = load_idx % (BK / vec_unit);
            int global_row = m_global_start + load_row;
            int global_col = tile_k + load_pack_col * vec_unit;
            if (global_row < M) {
                if (global_col + vec_unit <= K) {
                    reinterpret_cast<float4*>(smemA)[load_row * (BK / vec_unit) + load_pack_col] =
                        reinterpret_cast<const float4*>(
                            inp)[(m_global_start + load_row) * (K / vec_unit) + global_col / vec_unit];
                } else if (global_col < K) {
                    for (int i = 0; i < vec_unit; ++i) {
                        if (global_col + i < K) {
                            smemA[load_row * BK + load_pack_col * vec_unit + i] =
                                inp[(m_global_start + load_row) * K + global_col + i];
                        } else {
                            smemA[load_row * BK + load_pack_col * vec_unit + i] = 0.0f;
                        }
                    }
                } else {
                    for (int i = 0; i < vec_unit; ++i) {
                        smemA[load_row * BK + load_pack_col * vec_unit + i] = 0.0f;
                    }
                }
            } else {
                for (int i = 0; i < vec_unit; ++i) {
                    smemA[load_row * BK + load_pack_col * vec_unit + i] = 0.0f;
                }
            }
        }
        // 加载B矩阵

        // BN 最初设置为 128
        // 单个 int32 打包 8 个权重
        // float4 可一次性加载 32 个数据
        // 考虑到复杂性，仍然先实现单加载
        // 或者说 int32 本身即可以理解为向量化加载
        for (int load_idx = threadIdx.x; load_idx < BN * BK / 8; load_idx += blockDim.x) {
            int load_row = load_idx / (BK / 8);
            int load_col = load_idx % (BK / 8);
            int global_row = n_global_start + load_row;
            int global_col = tile_k + load_col * 8;

            int32_t qwt_val = 0;
            if (global_row < N && global_col < K) {
                qwt_val = qwt[global_row * K_PACKED + (tile_k + load_col * 8) / 8];
            }

            int32_t zeros_val = 0;
            if (global_row < N && global_col < K) {
                zeros_val = zos[global_row * G_PACKED + ((tile_k + load_col * 8) / group_size) / 8];
            }

            for (int i = 0; i < 8; ++i) {
                int current_k = global_col + i;
                int inner_k = current_k % PACK_FACTOR;
                int group_idx = current_k / group_size;
                S scale_val = (global_row < N && current_k < K) ? scl[global_row * G_PADDED + group_idx] : S(0);
                int shift_w = inner_k * BITS;
                uint32_t w = (qwt_val >> shift_w) & ((1 << BITS) - 1);
                int inner_g = (current_k / group_size) % PACK_FACTOR;
                int shift_z = inner_g * BITS;
                uint32_t z = (zeros_val >> shift_z) & ((1 << BITS) - 1);
                float temp_val = (static_cast<float>(w) - static_cast<float>(z)) * static_cast<float>(scale_val);
                T dequantized_val = static_cast<T>(temp_val);  // 最后转换

                int n_local = load_row;
                int k_local = load_col * 8 + i;
                if (n_local < BN && k_local < BK) {
                    smemB[k_local * BN + n_local] = dequantized_val;
                }
            }
        }
        __syncthreads();

        for (int k_step = 0; k_step < BK; k_step += WMMA_K) {
            FragmentA fragA_load;
            FragmentB fragB_load;
            const T* smemA_ptr = smemA + (warp_m_id * WMMA_M * BK) + k_step;
            wmma::load_matrix_sync(fragA_load, smemA_ptr, BK);
            const T* smemB_ptr = smemB + (k_step * BN) + (warp_n_id * WMMA_N);
            wmma::load_matrix_sync(fragB_load, smemB_ptr, BN);

            wmma::mma_sync(fragC, fragA_load, fragB_load, fragC);
        }
    }

    const int out_m_base = warp_m_id * WMMA_M;
    const int out_n_base = warp_n_id * WMMA_N;

    wmma::store_matrix_sync(&smem_out[out_m_base][out_n_base], fragC, BN, wmma::mem_row_major);
    __syncthreads();
#pragma unroll
    for (int write_idx = threadIdx.x; write_idx < BM * BN; write_idx += blockDim.x) {
        int m_local = write_idx / BN;
        int load_row = write_idx % BN;

        int m_global = m_global_start + m_local;
        int n_global = n_global_start + load_row;

        if (m_global < M && n_global < N) {
            float result = smem_out[m_local][load_row];
            if (bias) {
                result += static_cast<float>(bias[n_global]);
            }

            out[m_global * N + n_global] = static_cast<T>(result);
        }
    }
}

// --- GEMV Kernel (M = 1, N-Major 优化版) ---
// 专门为 M = 1 优化，假设权重、scales、zeros 为 N-Major 布局
// 使用动态共享内存，因其大小依赖运行时的 K 和 G
template <typename T,                                                  // 输入/输出数据类型
          typename S,                                                  // Scale 数据类型
          int BLOCK_N_GEMV>                                            // Block 内的线程数 (必须是 WARP_SIZE 的倍数)
__global__ void matmul_awq_gemv_kernel_opt(const T* __restrict__ inp,  // 输入向量 [K]
                                           const int32_t* __restrict__ qwt,  // 权重 [N, K/8]
                                           const S* __restrict__ scl,        // Scales [N, G_padded]
                                           const int32_t* __restrict__ zos,  // Zeros [N, G/8]
                                           T* __restrict__ out,              // 输出向量 [N]
                                           int K, int N, int group_size,
                                           int G_PADDED,  // !!! 新增: Scales 张量的实际第二维度 (Padding) !!!
                                           const T* __restrict__ bias) {  // 偏置向量 [N]
    // --- 常量 ---
    static_assert(BLOCK_N_GEMV % WARP_SIZE == 0, "BLOCK_N_GEMV 必须是 WARP_SIZE 的倍数");
    const int G = K / group_size;
    const int K_PACKED = (K + PACK_FACTOR - 1) / PACK_FACTOR;
    const int G_PACKED = (G + PACK_FACTOR - 1) / PACK_FACTOR;
    // G_PADDED 作为参数传入

    // --- Grid/Block/Warp 映射 ---
    const int warps_per_block = BLOCK_N_GEMV / WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int n = blockIdx.x * warps_per_block + warp_id;  // 此 Warp 负责的列 n

    if (n >= N)
        return;  // 边界检查

    // --- 动态共享内存声明与指针分配 ---
    extern __shared__ char sh_mem_raw[];
    T* sh_inp = reinterpret_cast<T*>(sh_mem_raw);  // sh_inp [K]
    S* sh_scl = reinterpret_cast<S*>(&sh_inp[K]);  // sh_scl [warps_per_block][G_PADDED]
    int32_t* sh_zos =
        reinterpret_cast<int32_t*>(&sh_scl[warps_per_block * G_PADDED]);  // sh_zos [warps_per_block][G_PACKED]

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
    for (int k = lane; k < K; k += WARP_SIZE) {    // Warp 内线程并行处理 K
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
        acc = __fmaf_rn(iv, (static_cast<float>(w) - static_cast<float>(z)) * scale_val, acc);
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
    const ScaleType* __restrict__ scl,                   // Scales [N, G_padded] (ScaleType)
    const int32_t* __restrict__ zos,                     // Zeros [N, G/8]
    __nv_bfloat16* __restrict__ out,                     // 输出向量 [N] (BF16)
    int K, int N, int group_size,
    int G_PADDED,                            // Scales 张量的实际第二维度
    const __nv_bfloat16* __restrict__ bias)  // 偏置向量 [N] (BF16, 可选)
{
    // --- 常量 ---
    static_assert(BLOCK_N_GEMV % WARP_SIZE == 0, "BLOCK_N_GEMV 必须是 WARP_SIZE 的倍数");
    const int G = K / group_size;
    const int K_PACKED = (K + PACK_FACTOR - 1) / PACK_FACTOR;
    const int G_PACKED = (G + PACK_FACTOR - 1) / PACK_FACTOR;
    constexpr int K_PER_THREAD = 8;                       // 每个线程每次迭代处理 K 的数量
    constexpr int K_PER_WARP = WARP_SIZE * K_PER_THREAD;  // Warp 每次迭代处理的数量

    // --- Grid/Block/Warp 映射 ---
    const int warps_per_block = BLOCK_N_GEMV / WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int n = blockIdx.x * warps_per_block + warp_id;  // 此 Warp 负责的列 n

    if (n >= N)
        return;  // 边界检查

    // --- 动态共享内存 ---
    extern __shared__ char sh_mem_raw[];
    __nv_bfloat16* sh_inp = reinterpret_cast<__nv_bfloat16*>(sh_mem_raw);
    ScaleType* sh_scl = reinterpret_cast<ScaleType*>(&sh_inp[K]);
    int32_t* sh_zos = reinterpret_cast<int32_t*>(&sh_scl[warps_per_block * G_PADDED]);
    constexpr int vec_unit = 16 / sizeof(__nv_bfloat16);

    // 加载 sh_inp (BF16)
    // 注意 默认K是8倍数
    for (int k_idx = threadIdx.x; k_idx < K / vec_unit; k_idx += BLOCK_N_GEMV) {
        reinterpret_cast<float4*>(sh_inp)[k_idx] = reinterpret_cast<const float4*>(inp)[k_idx];
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

        // !!! 警告: 假设 k_thread_start 到 k_thread_start + 7 均在同一 group 内 !!!
        // 基于 k_thread_start 确定 group, scale, 和 packed_zero
        int g = k_thread_start / group_size;
        float current_s = 0.0f;
        uint32_t z_vals[K_PER_THREAD];  // 存储解包后的 8 个 zero-point (通常相同)

        // 边界检查: 确保 g 在有效范围内
        if (k_thread_start < K && g < G) {
            current_s = sh_scl[warp_id * G_PADDED + g];  // 从共享内存加载 scale
            int packed_g = g / PACK_FACTOR;
            int32_t packed_z_val = sh_zos[warp_id * G_PACKED + packed_g];  // 从共享内存加载 packed zero

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
                acc = __fmaf_rn(iv, (static_cast<float>(w) - static_cast<float>(z)) * scale_val, acc);
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
void matmul_quantized_gemv(const Tensor<T>& input,           // 输入 [M, K]
                           const Tensor<int32_t>& qweight,   // 权重 [N, K/8] (N-Major)
                           const Tensor<ScaleType>& scales,  // Scales [N, G_padded] (N-Major)
                           const Tensor<int32_t>& zeros,     // Zeros [N, G/8] (N-Major)
                           int group_size,                   // Group 大小
                           Tensor<T>* output,                // 输出 [M, N]
                           cudaStream_t stream,              // CUDA 流
                           const Tensor<T>* bias) {          // 偏置 [N] (可选)
    // --- 输入参数检查 ---

    if (input.sizes().size() != 2)
        throw std::runtime_error("输入张量必须是 2D");
    int M = input.sizes()[0];
    int K = input.sizes()[1];

    if (qweight.sizes().size() != 2)
        throw std::runtime_error("权重张量必须是 2D");
    int N = qweight.sizes()[0];  // N-Major
    int K_PACKED_w = qweight.sizes()[1];

    if (scales.sizes().size() != 2)
        throw std::runtime_error("Scales 张量必须是 2D");
    if (scales.sizes()[0] != N)
        throw std::runtime_error("Scales N 维度不匹配");
    int G_PADDED = scales.sizes()[1];  // *** 获取 Scales 实际的第二维度 ***

    if (zeros.sizes().size() != 2)
        throw std::runtime_error("Zeros 张量必须是 2D");
    if (zeros.sizes()[0] != N)
        throw std::runtime_error("Zeros N 维度不匹配");
    int G_PACKED_z = zeros.sizes()[1];

    // ... (保留其他所有维度检查, K % group_size, K % PACK_FACTOR 等) ...
    if (bias && (bias->sizes().size() != 1 || bias->sizes()[0] != N)) {
        throw std::runtime_error("Bias 必须是 1D 且大小为 N (" + std::to_string(N) + ")");
    }
    if (group_size <= 0) {
        throw std::runtime_error("group_size 必须为正数");
    }
    if (K % group_size != 0) {
        throw std::runtime_error("K (" + std::to_string(K) + ") 必须能被 group_size (" + std::to_string(group_size) +
                                 ") 整除");
    }
    // ... (保留 K % PACK_FACTOR 和 N % PACK_FACTOR 的警告或错误) ...

    // --- 重新计算维度用于验证和 Kernel 参数 ---
    int G = K / group_size;
    int K_PACKED = (K + PACK_FACTOR - 1) / PACK_FACTOR;
    int G_PACKED = (G + PACK_FACTOR - 1) / PACK_FACTOR;

    // --- 维度验证 ---
    if (K_PACKED_w != K_PACKED) {
        throw std::runtime_error("QWeight 的 K/8 维度 (" + std::to_string(K_PACKED_w) + ") 不匹配，期望 " +
                                 std::to_string(K_PACKED));
    }
    if (G_PACKED_z != G_PACKED) {
        throw std::runtime_error("Zeros 的 G/8 维度 (" + std::to_string(G_PACKED_z) + ") 不匹配，期望 " +
                                 std::to_string(G_PACKED));
    }
    if (G_PADDED < G) {
        throw std::runtime_error("Scales 的 G 维度 (" + std::to_string(G_PADDED) + ") 小于计算出的 G (" +
                                 std::to_string(G) + ")");
    }

    // --- 根据 M 选择 Kernel ---
    if (M == 1) {
        // --- GEMV 路径 (M=1) ---
        constexpr int BLOCK_N_GEMV = 256;  // GEMV Kernel 的 Block 线程数
        constexpr int WARPS_PER_BLOCK = BLOCK_N_GEMV / WARP_SIZE;

        const dim3 grid_gemv((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);  // 1D Grid
        const dim3 threads_gemv(BLOCK_N_GEMV);                              // 1D Block

        // 计算 GEMV Kernel 所需的动态共享内存大小c
        size_t shmem_size_gemv = K * sizeof(T);                             // sh_inp
        shmem_size_gemv += WARPS_PER_BLOCK * G_PADDED * sizeof(ScaleType);  // sh_scl
        shmem_size_gemv += WARPS_PER_BLOCK * G_PACKED * sizeof(int32_t);    // sh_zos

        // 调用 GEMV Kernel, 传递 G_PADDED
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            matmul_awq_gemv_bf16_vectorized_kernel<BLOCK_N_GEMV, ScaleType>
                <<<grid_gemv, threads_gemv, shmem_size_gemv, stream>>>(input.data_ptr(), qweight.data_ptr(),
                                                                       scales.data_ptr(), zeros.data_ptr(),
                                                                       output->data_ptr(), K, N, group_size,
                                                                       G_PADDED,  // !!! 传递 G_PADDED !!!
                                                                       bias ? bias->data_ptr() : nullptr);
        } else
            matmul_awq_gemv_kernel_opt<T, ScaleType, BLOCK_N_GEMV>
                <<<grid_gemv, threads_gemv, shmem_size_gemv, stream>>>(input.data_ptr(), qweight.data_ptr(),
                                                                       scales.data_ptr(), zeros.data_ptr(),
                                                                       output->data_ptr(), K, N, group_size,
                                                                       G_PADDED,  // !!! 传递 G_PADDED !!!
                                                                       bias ? bias->data_ptr() : nullptr);

    } else {
        constexpr int BM = 16;
        constexpr int BN = 64;
        constexpr int BK = 16;
        constexpr int WMMA_M = 16;
        constexpr int WMMA_N = 16;
        constexpr int WMMA_K = 16;
        constexpr int WARP_CNT = BM / WMMA_M * BN / WMMA_N;
        const dim3 threads_gemm(WARP_CNT * WARP_SIZE);               // 1D Block
        const dim3 grid_gemm((M + BM - 1) / BM, (N + BN - 1) / BN);  // 2D Grid

        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            matmul_awq_gemm_kernel_opt<T, ScaleType, BM, BN, BK, WMMA_M, WMMA_N, WMMA_K, WARP_CNT>
                <<<grid_gemm, threads_gemm, 0, stream>>>(  // 使用静态共享内存
                    input.data_ptr(), qweight.data_ptr(), scales.data_ptr(), zeros.data_ptr(), output->data_ptr(), M, K,
                    N, group_size,
                    G_PADDED,  // !!! 传递 G_PADDED !!!
                    bias ? bias->data_ptr() : nullptr);
        } else {
            throw std::runtime_error("Unsupported input type for AWQ gemv GEMM kernel");
        }
    }

    // --- CUDA 错误检查 ---
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::string error_msg = "CUDA kernel launch failed: " + std::string(cudaGetErrorString(err));
        error_msg += " [M=" + std::to_string(M) + ", K=" + std::to_string(K) + ", N=" + std::to_string(N) +
                     ", gs=" + std::to_string(group_size) + "]";
        throw std::runtime_error(error_msg);
    }
}

template void matmul_quantized_gemv<float>(const Tensor<float>&, const Tensor<int32_t>&, const Tensor<float>&,
                                           const Tensor<int32_t>&, int, Tensor<float>*, cudaStream_t,
                                           const Tensor<float>*);
template void matmul_quantized_gemv<__nv_bfloat16>(const Tensor<__nv_bfloat16>&, const Tensor<int32_t>&,
                                                   const Tensor<float>&, const Tensor<int32_t>&, int,
                                                   Tensor<__nv_bfloat16>*, cudaStream_t, const Tensor<__nv_bfloat16>*);
template void matmul_quantized_gemv<__nv_bfloat16, __nv_bfloat16>(const Tensor<__nv_bfloat16>&, const Tensor<int32_t>&,
                                                                  const Tensor<__nv_bfloat16>&, const Tensor<int32_t>&,
                                                                  int, Tensor<__nv_bfloat16>*, cudaStream_t,
                                                                  const Tensor<__nv_bfloat16>*);
template void matmul_quantized_gemv<__half>(const Tensor<__half>&, const Tensor<int32_t>&, const Tensor<float>&,
                                            const Tensor<int32_t>&, int, Tensor<__half>*, cudaStream_t,
                                            const Tensor<__half>*);

}  // namespace cuda_OP