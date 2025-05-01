// matmul_awq.cu
// AWQ量化矩阵乘法CUDA实现 - v12 (Incorporates AWQ Reordering)
// 目的: 实现AWQ 4bit量化权重的矩阵乘法, 处理AWQ特定的内部顺序
// 处理: Input[M, K], QWeight[K, N/8], Scales[NumGroups, N], Zeros[NumGroups, N/8]
// 输出 = Input * Dequant(QWeight, Scales, Zeros)
// #define DEBUG_AWQ
#include <cuda_runtime.h>
#include <cuda_fp16.h> // for __half
#include <cuda_bf16.h> // for __nv_bfloat16
#include <stdint.h>    // for int32_t, uint32_t
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <iomanip>    // for std::setw, std::setprecision
#include "cudaOP.cuh" // Assuming this contains your Tensor class definition

namespace cuda_OP
{
    // Helper function to print tensor sizes (reuse from previous version)
    inline std::string format_sizes(const std::vector<size_t> &sizes)
    {
        std::stringstream ss;
        ss << "(";
        for (size_t i = 0; i < sizes.size(); ++i)
        {
            ss << sizes[i] << (i == sizes.size() - 1 ? "" : ", ");
        }
        ss << ")";
        return ss.str();
    }

    // Helper function to print tensor values for debugging (reuse)
    template <typename T>
    void debug_print_tensor(const Tensor<T> &tensor, const std::string &name, int max_elements = 10)
    {
#ifdef DEBUG_AWQ
        // ... (implementation from previous version) ...
        std::cout << "Tensor " << name << " " << format_sizes(tensor.sizes()) << ":" << std::endl;
        std::vector<T> host_data(tensor.numel());
        cudaMemcpy(host_data.data(), tensor.data_ptr(), tensor.numel() * sizeof(T), cudaMemcpyDeviceToHost);
        int num_to_print = std::min(static_cast<int>(tensor.numel()), max_elements);
        for (int i = 0; i < num_to_print; ++i)
        {
            std::cout << "  [" << i << "] = " << static_cast<float>(host_data[i]) << std::endl;
        }
        if (tensor.numel() > max_elements)
        {
            std::cout << "  ... (and " << tensor.numel() - max_elements << " more elements)" << std::endl;
        }
#endif
    }
    //----------------------------------------------------------------------------
    // 常量和映射定义
    //----------------------------------------------------------------------------
    constexpr int BITS = 4;
    constexpr int PACK_FACTOR = 32 / BITS; // = 8

    // AWQ 内部物理顺序映射: logical_inner_idx -> physical_inner_idx used for bit shifts
    // Derived from AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
    // physical_inner_idx[logical_inner_idx]
    __constant__ const int LOGICAL_TO_PHYSICAL_INNER_IDX[PACK_FACTOR] = {0, 4, 1, 5, 2, 6, 3, 7};

    //----------------------------------------------------------------------------
    // CUDA Kernel (AWQ 反量化与矩阵乘法核心 - 加入 Reordering)
    //----------------------------------------------------------------------------
    template <typename T, typename ScaleType>
    __global__ void matmul_awq_kernel(
        const T *__restrict__ inp,         // 输入激活: [M, K]
        const int32_t *__restrict__ qwt,   // 量化权重: [K, N / PACK_FACTOR] (AWQ Order)
        const ScaleType *__restrict__ scl, // 缩放因子: [NumGroups, N] (Logical Order)
        const int32_t *__restrict__ zos,   // 量化零点: [NumGroups, N / PACK_FACTOR] (AWQ Order)
        T *__restrict__ out,               // 输出结果: [M, N]
        const int M,
        const int K,
        const int N,
        const int group_size,
        const T *__restrict__ bias) // 可选的偏置: [N]
    {
        // --- 线程索引计算 ---
        const int m = blockIdx.y;
        const int n = blockIdx.x * blockDim.x + threadIdx.x; // Logical output column index

        // --- 网格边界检查 ---
        if (n >= N || m >= M)
            return;

        // --- 累加器初始化 ---
        float acc = 0.0f;

        // --- K 维度计算循环 ---
        for (int k = 0; k < K; ++k)
        {
            // 1. 计算分组索引
            const int group_idx = k / group_size;

            // 2. 获取输入激活值
            const float inp_val = static_cast<float>(inp[m * K + k]);

            // 3. 反量化权重值 (w[k, n]) - incorporating AWQ Reordering
            // 3a. 找到包含目标权重和零点的 packed int32 的列索引
            //    这个索引基于逻辑列 n，因为 QW/ZOS 的列数是 N/PACK_FACTOR
            const int packed_col_idx = n / PACK_FACTOR;

            // 3b. 计算 *逻辑上* n 在其 8 元素块内的索引
            const int logical_inner_idx = n % PACK_FACTOR;

            // 3c. *** AWQ Reordering ***
            //    使用映射找到存储逻辑索引 logical_inner_idx 的 *物理* 内部索引
            const int physical_inner_col_idx = LOGICAL_TO_PHYSICAL_INNER_IDX[logical_inner_idx];

            // 3d. 计算 qwt 和 zos 在全局内存中的线性索引 (使用 packed_col_idx)
            const int qwt_idx = k * (N / PACK_FACTOR) + packed_col_idx;
            const int zos_idx = group_idx * (N / PACK_FACTOR) + packed_col_idx;

            // 3e. 从全局内存加载 packed 权重和零点
            //    添加边界检查 (虽然理论上不应越界)
            int32_t packed_qwt = 0;
            int32_t packed_zos = 0;
            if (qwt_idx < (K * (N / PACK_FACTOR)))
            { // Check upper bound
                packed_qwt = qwt[qwt_idx];
            }
            if (zos_idx < ((K / group_size) * (N / PACK_FACTOR)))
            { // Check upper bound
                packed_zos = zos[zos_idx];
            }

            // 3f. 解包 (Unpack) 得到 n 列对应的 4-bit 权重和零点
            //    使用 ***physical_inner_col_idx*** 进行位移
            const uint32_t q_w = (static_cast<uint32_t>(packed_qwt) >> (physical_inner_col_idx * BITS)) & 0x0F;
            const uint32_t q_z = (static_cast<uint32_t>(packed_zos) >> (physical_inner_col_idx * BITS)) & 0x0F;

            // 3g. 获取缩放因子 Scale (scl[group_idx, n]) - Scales 是按逻辑顺序 n 存储的
            const int scl_idx = group_idx * N + n;
            //    添加边界检查
            float scale_val = 0.0f;
            if (scl_idx < ((K / group_size) * N))
            { // Check upper bound
                scale_val = static_cast<float>(scl[scl_idx]);
            }

            // 3h. 计算反量化后的权重值
            const float dequant_w = (static_cast<float>(q_w) - static_cast<float>(q_z)) * scale_val;

            // 4. 累加乘积
            acc += inp_val * dequant_w;

        } // 结束 K 维度循环

        // --- 添加偏置（如果存在）---
        if (bias != nullptr)
        {
            // 添加边界检查
            if (n < N)
            { // Check bias index bound
                const T bias_val = bias[n];
                acc += static_cast<float>(bias_val);
            }
        }

        // --- 将累加结果写入输出 ---
        // 再次检查输出索引边界
        if ((m * N + n) < (M * N))
        {
            out[m * N + n] = static_cast<T>(acc);
        }

    } // 内核函数结束

    //----------------------------------------------------------------------------
    // 封装函数 (调用计算内核) - 无需修改
    //----------------------------------------------------------------------------
    template <typename T>
    void matmul_quantized(
        const Tensor<T> &input,             // 输入张量 [M, K]
        const Tensor<int32_t> &qweight,     // 量化权重张量 [K, N / PACK_FACTOR]
        const Tensor<float> &scales_input,  // 缩放因子张量 [NumGroups, N] (封装层假设为 float)
        const Tensor<int32_t> &zeros_input, // 量化零点张量 [NumGroups, N / PACK_FACTOR]
        int group_size,                     // 分组大小
        Tensor<T> *output,                  // 输出张量 [M, N]
        cudaStream_t stream,                // CUDA 流
        const Tensor<T> *bias)              // 可选的偏置张量 [N] (可以为nullptr)
    {
        // --- 输入验证、维度推导、形状检查 ---
        const int M = static_cast<int>(input.sizes()[0]);
        const int K = static_cast<int>(input.sizes()[1]);
        int N = 0;
        if (scales_input.sizes().size() == 2)
            N = static_cast<int>(scales_input.sizes()[1]);
        else if (output->sizes().size() == 2)
            N = static_cast<int>(output->sizes()[1]);
        else if (qweight.sizes().size() == 2)
            N = static_cast<int>(qweight.sizes()[1]) * PACK_FACTOR;
        else
            throw std::runtime_error("无法确定维度 N");

        // （省略了详细的交叉验证和错误检查代码，假设与之前版本相同）
        // ... Ensure M, K, N > 0, group_size > 0, K % group_size == 0 ...
        // ... Ensure shape consistency checks pass ...
        if (K <= 0 || N <= 0 || M <= 0 || group_size <= 0 || K % group_size != 0)
        {
            // Basic checks
            throw std::runtime_error("Invalid dimensions or group_size");
        }
        const int NumGroups = K / group_size;
        // Add crucial shape checks from previous version here...
        if (K != static_cast<int>(qweight.sizes()[0]))
            throw std::runtime_error("Input K mismatch qweight K");
        if (N / PACK_FACTOR != static_cast<int>(qweight.sizes()[1]))
            throw std::runtime_error("N/PACK_FACTOR mismatch qweight dim 1");
        if (NumGroups != static_cast<int>(scales_input.sizes()[0]))
            throw std::runtime_error("NumGroups mismatch scales dim 0");
        if (N != static_cast<int>(scales_input.sizes()[1]))
            throw std::runtime_error("N mismatch scales dim 1");
        if (NumGroups != static_cast<int>(zeros_input.sizes()[0]))
            throw std::runtime_error("NumGroups mismatch zeros dim 0");
        if (N / PACK_FACTOR != static_cast<int>(zeros_input.sizes()[1]))
            throw std::runtime_error("N/PACK_FACTOR mismatch zeros dim 1");
        if (M != static_cast<int>(output->sizes()[0]))
            throw std::runtime_error("Input M mismatch output M");
        // if (N != static_cast<int>(output->sizes()[1]))
        //     throw std::runtime_error("Derived N mismatch output N");
        if (bias && N != static_cast<int>(bias->sizes()[0]))
            throw std::runtime_error("Bias N mismatch output N");

        // --- Kernel Launch ---
        const dim3 block_size(256); // Or adjust based on GPU architecture
        const dim3 grid_size((N + block_size.x - 1) / block_size.x, M);

#ifdef DEBUG_AWQ
        // （省略了详细的调试打印代码，假设与之前版本相同）
        std::cout << "\nLaunching AWQ Reordered Kernel..." << std::endl;
        std::cout << "  Grid Size: (" << grid_size.x << ", " << grid_size.y << ")" << std::endl;
        std::cout << "  Block Size: (" << block_size.x << ")" << std::endl;
        std::cout << "  M=" << M << ", K=" << K << ", N=" << N << ", group_size=" << group_size << std::endl;
        // ... other debug prints ...
#endif

        // --- 选择 Scale 类型 ---
        // 仍假设封装函数提供 float scales
        using ScaleType = float;

        // --- 启动 CUDA 内核 ---
        matmul_awq_kernel<T, ScaleType><<<grid_size, block_size, 0, stream>>>(
            input.data_ptr(),
            qweight.data_ptr(),
            scales_input.data_ptr(),
            zeros_input.data_ptr(),
            output->data_ptr(),
            M, K, N, group_size,
            bias ? bias->data_ptr() : nullptr);

        // --- 错误检查 ---
        cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess)
        {
            std::cerr << "CUDA kernel LAUNCH error: " << cudaGetErrorString(launch_err) << std::endl;
            throw std::runtime_error("CUDA kernel launch error");
        }

        // 可选同步检查
        // #ifdef DEBUG_CUDA_SYNC ... #endif
    }

    // --- 显式模板实例化 ---
    // (保持与之前版本相同)
    template void matmul_quantized<float>(const Tensor<float> &, const Tensor<int32_t> &, const Tensor<float> &, const Tensor<int32_t> &, int, Tensor<float> *, cudaStream_t, const Tensor<float> *);
    template void matmul_quantized<__nv_bfloat16>(const Tensor<__nv_bfloat16> &, const Tensor<int32_t> &, const Tensor<float> &, const Tensor<int32_t> &, int, Tensor<__nv_bfloat16> *, cudaStream_t, const Tensor<__nv_bfloat16> *);
    template void matmul_quantized<__half>(const Tensor<__half> &, const Tensor<int32_t> &, const Tensor<float> &, const Tensor<int32_t> &, int, Tensor<__half> *, cudaStream_t, const Tensor<__half> *);

} // namespace cuda_OP