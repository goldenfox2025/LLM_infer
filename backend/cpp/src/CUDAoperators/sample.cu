#ifndef CUDA_OP_CUH
#define CUDA_OP_CUH

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <float.h>

#include <limits>
#include <stdexcept>
#include <string>  // 用于 std::to_string
#include <vector>

// 包含 CUB 头文件
#include <cub/cub.cuh>

// 假设这些是项目特定的头文件 (忽略其具体内容)
#include "CudaMemoryPool.hpp"
#include "cudaOP.cuh"
#include "tensor.hpp"

// --- 常量定义 (cudaOP.cuh定义) ---
// 示例: 定义 Kernel 2 中共享内存数组的最大大小
// #define MAX_TOPK 256 // 重要: 必须定义此宏, 否则 Kernel 2 无法编译!
// 假设 MAX_TOPK 在其他地方定义
#define MAX_TOPK 1024
// --- 检查 CUDA 错误的宏 (cudaOP.cuh定义) ---
// #define CUDA_CHECK(call) ... // 重要: 必须定义此宏!
// 假设 CUDA_CHECK 在其他地方定义

namespace cuda_OP {

// Kernel 1 (融合): 缩放 Logits 并初始化索引 (多块执行)
// 功能: 将输入的 logits 除以 temperature 并初始化一个从 0 到 vocab_size-1
// 的索引数组。
template <typename T>
__global__ void scale_logits_and_init_indices_kernel(const T* __restrict__ logits,  // 输入: 原始 logits (设备指针)
                                                     T* d_scaled_logits,            // 输出: 缩放后的 logits (设备指针)
                                                     int* d_indices,                // 输出: 初始化的索引数组 (设备指针)
                                                     size_t vocab_size,             // 输入: 词汇表大小
                                                     float temperature              // 输入: 温度系数
) {
    // 使用网格跨步循环处理所有词汇
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < vocab_size; i += stride) {
        // 1. 缩放 Logits
        float logit_f;
        // 使用 __ldg 进行缓存的全局内存读取
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            logit_f = __bfloat162float(__ldg(&logits[i]));
        } else {
            logit_f = static_cast<float>(__ldg(&logits[i]));
        }
        float scaled_logit_f = logit_f / temperature;  // 应用温度缩放

        // 写回缩放后的值 (根据类型转换)
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            d_scaled_logits[i] = __float2bfloat16(scaled_logit_f);
        } else {
            d_scaled_logits[i] = static_cast<T>(scaled_logit_f);
        }

        // 2. 初始化索引
        d_indices[i] = i;
    }
}

// Kernel 2: 从 Top-K 结果中进行最终采样 (单块执行)
// 功能: 对排序后的 Top-K logits 进行 softmax 和加权随机采样。
// T: 数据类型 (float 或 __nv_bfloat16)
// BLOCK_DIM_X: CUDA 块的大小 (用于 CUB 和并行计算)
template <typename T, int BLOCK_DIM_X>
__global__ void sample_from_sorted_topk_kernel(const T* __restrict__ d_sorted_topk_logits,     // 输入: 排序后的 Top-K
                                                                                               // logits (设备指针)
                                               const int* __restrict__ d_sorted_topk_indices,  // 输入: 排序后的 Top-K
                                                                                               // 索引 (设备指针)
                                               size_t k,  // 输入: Top-K 中的 'k' 值 (必须 <= MAX_TOPK)
                                               const float* __restrict__ d_max_val_ptr,  // 输入: 所有缩放后 logits
                                                                                         // 的最大值 (设备指针)
                                               curandState* states,       // 输入/输出: cuRAND 状态 (用于随机数生成)
                                               uint32_t* d_sampled_index  // 输出: 最终采样得到的索引 (设备指针)
) {
    // CUB 块内归约，用于计算 exp 值的总和
    using BlockReduce = cub::BlockReduce<float, BLOCK_DIM_X>;

    // 共享内存: 用于 CUB 临时存储和存储 Top-K 的 exp(logit - max_val) 值
    __shared__ union SharedStorage {
        typename BlockReduce::TempStorage reduce_storage;  // CUB Reduce 所需的存储
        // 联合体确保内存复用。需要足够空间存放 MAX_TOPK 个 float 值。
        struct Combined {
            typename BlockReduce::TempStorage reduce_storage;
            float exp_vals[MAX_TOPK];  // 存储 Top-K 指数的数组 (大小由 MAX_TOPK 决定)
        } combined;
    } shared_storage;

    int tid = threadIdx.x;  // 当前线程 ID

    // 线程 0 读取最大值并存入共享内存
    __shared__ float max_val_shared;
    if (tid == 0) {
        max_val_shared = *d_max_val_ptr;
    }
    __syncthreads();  // 确保所有线程都能读到 max_val_shared

    // --- 并行计算 exp(logit - max_val) ---
    float thread_exp_sum = 0.0f;  // 每个线程的局部 exp 值累加和
    // 线程协作计算前 k 个值的 exp
    for (int i = tid; i < k; i += BLOCK_DIM_X) {
        T scaled_logit_T = d_sorted_topk_logits[i];
        float scaled_logit_f;
        // 类型转换
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            scaled_logit_f = __bfloat162float(scaled_logit_T);
        } else {
            scaled_logit_f = static_cast<float>(scaled_logit_T);
        }
        // 计算 exp(logit - max)，减去 max 防止上溢
        float exp_val = expf(scaled_logit_f - max_val_shared);

        // 将计算出的 exp 值存入共享内存，供后续采样使用
        if (i < MAX_TOPK) {  // 检查边界，确保不越界写入共享数组
            shared_storage.combined.exp_vals[i] = exp_val;
        }
        // 累加到线程局部和
        thread_exp_sum += exp_val;
    }
    __syncthreads();  // 确保所有 exp_vals
                      // 都已写入共享内存，并且所有线程都完成了计算

    // --- 使用 CUB 在块内归约求和 ---
    // CUB 调用: BlockReduce::Sum
    // 输入: thread_exp_sum (每个线程计算的部分 exp 和)
    // 输出: block_total_exp_sum (块内所有线程的 exp 总和)
    // 作用: 高效地计算块内所有线程的部分和的总和。
    // 临时存储: shared_storage.combined.reduce_storage (在共享内存中)
    float block_total_exp_sum = BlockReduce(shared_storage.combined.reduce_storage).Sum(thread_exp_sum);
    // 此刻，块内所有线程的 block_total_exp_sum 都持有相同的总和值

    // --- 线程 0 执行加权采样 ---
    if (tid == 0) {
        float total_exp_sum = block_total_exp_sum;  // 获取总和
        curandState localState = states[0];         // 获取 cuRAND 状态

        uint32_t selected_final_index = 0;  // 初始化采样结果

        // 处理特殊情况：如果总和过小或 k=0，则默认选择第一个 top-k 元素 (或 0)
        if (total_exp_sum <= 1e-9f || k == 0) {
            if (k > 0) {  // 如果 k>0 但总和接近 0，选择概率最高的那个
                selected_final_index = static_cast<uint32_t>(d_sorted_topk_indices[0]);
            } else {  // 如果 k=0 (理论上不应发生，因为前面有检查)，返回 0
                selected_final_index = 0;
            }
        } else {
            // 生成一个 [0, total_exp_sum) 范围内的随机数
            float r = curand_uniform(&localState) * total_exp_sum;
            float cumulative = 0.0f;  // 累积概率

            // 线性扫描共享内存中的 exp 值进行加权采样
            selected_final_index = static_cast<uint32_t>(d_sorted_topk_indices[0]);  // 默认值
            float* s_exp_vals = shared_storage.combined.exp_vals;                    // 指向共享内存中的 exp 数组
            for (int i = 0; i < k; ++i) {
                // 从共享内存读取预先计算好的 exp 值
                cumulative += s_exp_vals[i];
                // 如果累积和超过随机阈值 r，则选择当前索引
                if (cumulative >= r) {
                    selected_final_index = static_cast<uint32_t>(d_sorted_topk_indices[i]);
                    break;  // 找到后即退出循环
                }
            }
        }
        // 将最终选定的索引写入输出指针
        *d_sampled_index = selected_final_index;
        // 更新 cuRAND 状态
        states[0] = localState;
    }
}

// CUB TransformIterator 的辅助 Functor
// 功能: 在 CUB 操作中动态地将输入类型 Tin (如 bfloat16) 转换为 float。
template <typename Tin>
struct ConvertToFloatFunctor {
    __device__ __forceinline__ float operator()(const Tin& x) const {
        if constexpr (std::is_same_v<Tin, __nv_bfloat16>) {
            return __bfloat162float(x);  // bfloat16 转 float
        } else {
            return static_cast<float>(x);  // 其他类型直接转 float
        }
    }
};

// 主采样函数
// 功能: 对输入的 logits 执行 Top-K 采样。
// 输入:
//   - logits: 输入的 logits 张量 (T 类型, 形状 [seq_len, vocab_size], 必须在
//   CUDA 设备上)
//   - temperature: 温度系数，用于缩放 logits
//   - top_p: Top-P 采样的概率阈值 (当前代码中未使用)
//   - top_k: Top-K 采样的 K 值
//   - d_states: 指向设备端 cuRAND 状态的指针
//   - stream: CUDA 流
// 返回:
//   - 指向设备端存储最终采样结果 (单个 uint32_t 索引) 的指针。注意：结果仍在
//   GPU 上。
template <typename T>
uint32_t* sample(Tensor<T>&& logits, float temperature,
                 float top_p,  // top_p 未在此实现中使用
                 size_t top_k, curandState* d_states, cudaStream_t stream) {
    // --- 输入验证 ---
    if (logits.device() != Device::CUDA) {
        throw std::runtime_error("输入张量必须在 CUDA 设备上");
    }
    // Top-K 采样至少需要 k=1
    if (top_k == 0) {
        throw std::runtime_error("top_k 必须至少为 1");
    }

    const auto& shape = logits.sizes();
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
        throw std::runtime_error("输入张量必须是二维且维度非零 [seq_len, vocab_size]");
    }

    const size_t seq_len = shape[0];
    const size_t vocab_size = shape[1];

    // 如果 K 大于词汇表大小，则将其限制为词汇表大小
    if (top_k > vocab_size) {
        top_k = vocab_size;
    }
    // 检查 K 是否超过 Kernel 2 中共享内存的限制 (MAX_TOPK)
    if (top_k > MAX_TOPK) {
        throw std::runtime_error("请求的 top_k (" + std::to_string(top_k) + ") 超过了 Kernel 2 的 MAX_TOPK 限制 (" +
                                 std::to_string(MAX_TOPK) + ")，无法在共享内存中分配。");
    }

    // 获取指向最后一个 token 的 logits 数据的设备指针
    const T* d_logits_ptr = logits.data_ptr() + (seq_len - 1) * vocab_size;

    // --- 内存管理 ---
    auto& pool = GlobalCudaMemoryPool::instance();  // 获取全局内存池实例
    // 分配临时设备内存
    T* d_scaled_logits = static_cast<T*>(pool.allocate(vocab_size * sizeof(T)));         // 存储缩放后的 logits
    float* d_max_val = static_cast<float*>(pool.allocate(sizeof(float)));                // 存储缩放后 logits 的最大值
    int* d_indices = static_cast<int*>(pool.allocate(vocab_size * sizeof(int)));         // 存储原始索引
    T* d_sorted_logits = static_cast<T*>(pool.allocate(vocab_size * sizeof(T)));         // 存储排序后的 logits
    int* d_sorted_indices = static_cast<int*>(pool.allocate(vocab_size * sizeof(int)));  // 存储排序后的索引
    // 使用tagged memory分配采样结果，确保每次都是相同的固定地址
    uint32_t* d_sampled_index = static_cast<uint32_t*>(pool.allocate(sizeof(uint32_t), "sample_output"));

    // CUB 临时存储指针和大小 (初始为 nullptr 和 0)
    void* d_reduce_temp_storage = nullptr;
    size_t reduce_temp_storage_bytes = 0;
    void* d_sort_temp_storage = nullptr;
    size_t sort_temp_storage_bytes = 0;

    // --- 步骤 1 (融合 Kernel): 缩放 Logits 并初始化索引 ---
    const int scale_init_block_size = 256;  // 定义块大小
    const int scale_init_grid_size =        // 计算网格大小
        (vocab_size + scale_init_block_size - 1) / scale_init_block_size;
    // 启动 Kernel 1
    scale_logits_and_init_indices_kernel<T><<<scale_init_grid_size, scale_init_block_size, 0, stream>>>(
        d_logits_ptr, d_scaled_logits, d_indices, vocab_size, temperature);
    CUDA_CHECK(cudaGetLastError());  // 检查核函数启动错误

    // --- 步骤 2: 查找最大缩放 Logit (使用 CUB Device Reduce) ---
    // 创建一个转换迭代器，在计算 Max 时将 T 动态转换为 float
    cub::TransformInputIterator<float, ConvertToFloatFunctor<T>, const T*> itr(d_scaled_logits,
                                                                               ConvertToFloatFunctor<T>());
    // 第一次调用 CUB Reduce: 获取所需的临时存储大小
    // CUB 调用: DeviceReduce::Max (第一次调用)
    // 输入: itr (转换迭代器), vocab_size (元素数量)
    // 输出: reduce_temp_storage_bytes (所需的临时存储大小)
    // 作用: 计算执行 Max 操作所需的临时设备内存大小。
    CUDA_CHECK(
        cub::DeviceReduce::Max(d_reduce_temp_storage, reduce_temp_storage_bytes, itr, d_max_val, vocab_size, stream));
    // 分配所需的临时存储
    d_reduce_temp_storage = pool.allocate(reduce_temp_storage_bytes);
    // 第二次调用 CUB Reduce: 执行 Max 操作
    // CUB 调用: DeviceReduce::Max (第二次调用)
    // 输入: itr (转换迭代器), vocab_size, d_reduce_temp_storage (临时存储指针)
    // 输出: d_max_val (指向最大值的设备指针)
    // 作用: 在设备上计算输入范围内的最大值。
    CUDA_CHECK(
        cub::DeviceReduce::Max(d_reduce_temp_storage, reduce_temp_storage_bytes, itr, d_max_val, vocab_size, stream));
    CUDA_CHECK(cudaGetLastError());  // 检查 CUB 调用错误

    // --- 步骤 3: 按 Logit 值降序排序 (Logit, Index) 对 (使用 CUB Device Radix
    // Sort) --- 第一次调用 CUB Sort: 获取所需的临时存储大小 CUB 调用:
    // DeviceRadixSort::SortPairsDescending (第一次调用) 输入: d_scaled_logits
    // (键), d_indices (值), vocab_size 输出: sort_temp_storage_bytes
    // (所需的临时存储大小) 作用: 计算执行降序键值对排序所需的临时设备内存大小。
    CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(d_sort_temp_storage, sort_temp_storage_bytes, d_scaled_logits,
                                                         d_sorted_logits, d_indices, d_sorted_indices, vocab_size, 0,
                                                         sizeof(T) * 8, stream));  // sizeof(T)*8 是 T 类型的位数
    // 分配所需的临时存储
    d_sort_temp_storage = pool.allocate(sort_temp_storage_bytes);
    // 第二次调用 CUB Sort: 执行排序操作
    // CUB 调用: DeviceRadixSort::SortPairsDescending (第二次调用)
    // 输入: d_scaled_logits, d_indices, vocab_size, d_sort_temp_storage
    // (临时存储指针) 输出: d_sorted_logits (排序后的键), d_sorted_indices
    // (排序后的值) 作用: 根据键 (logits) 对键值对 (logit, index) 进行降序排序。
    CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(d_sort_temp_storage, sort_temp_storage_bytes, d_scaled_logits,
                                                         d_sorted_logits, d_indices, d_sorted_indices, vocab_size, 0,
                                                         sizeof(T) * 8, stream));
    CUDA_CHECK(cudaGetLastError());  // 检查 CUB 调用错误

    // --- 步骤 4: 从 Top-K 结果中进行最终加权采样 (单块 Kernel) ---
    // 为采样核函数选择块大小 (必须与核函数模板参数匹配)
    const int sample_block_size = 128;  // 示例块大小

    // 计算 Kernel 2 所需的共享内存大小
    // 需要 CUB Reduce 的临时存储大小 + 存储 exp 值的数组大小
    // 注意: 精确计算 CUB Reduce 存储需要模板特化，这里用 sizeof 估算
    size_t reduce_storage_size_est = sizeof(cub::BlockReduce<float,
                                                             sample_block_size>::TempStorage);  // CUB Reduce 存储
    size_t exp_values_size = MAX_TOPK * sizeof(float);                     // 存储 exp 值的数组大小 (基于 MAX_TOPK)
    size_t sample_shared_mem = reduce_storage_size_est + exp_values_size;  // 总共享内存需求

    // 检查所需共享内存是否超过设备限制
    int max_shared_mem_per_block = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_shared_mem_per_block, cudaDevAttrMaxSharedMemoryPerBlock, 0));
    if (sample_shared_mem > max_shared_mem_per_block) {
        throw std::runtime_error("计算出的所需共享内存 (" + std::to_string(sample_shared_mem) + ") 超过设备限制 (" +
                                 std::to_string(max_shared_mem_per_block) +
                                 ")。检查 MAX_TOPK 或减小 sample_block_size。");
    }

    // 启动 Kernel 2 (单块，块大小为 sample_block_size)
    // 模板参数 <T, sample_block_size> 必须与核函数定义匹配
    sample_from_sorted_topk_kernel<T, sample_block_size><<<1, sample_block_size, sample_shared_mem, stream>>>(
        d_sorted_logits,   // 排序后的 top-k logits
        d_sorted_indices,  // 排序后的 top-k indices
        top_k,             // 实际使用的 k 值 (已确保 <= MAX_TOPK)
        d_max_val,         // 最大 logit 值指针
        d_states,          // cuRAND 状态
        d_sampled_index    // 输出采样的索引
    );
    CUDA_CHECK(cudaGetLastError());  // 检查核函数启动错误

    // --- 结果处理 ---
    // 注意: 此函数返回的是指向设备内存中结果的指针 `d_sampled_index`。
    // 如果需要将结果复制回主机，需要在此处添加 cudaMemcpy 调用。
    // uint32_t h_result = 0;
    // CUDA_CHECK(cudaMemcpy(&h_result, d_sampled_index, sizeof(uint32_t),
    // cudaMemcpyDeviceToHost));

    // --- 释放临时内存 ---
    // 将所有临时分配的设备内存返还给内存池
    pool.free(d_scaled_logits);
    pool.free(d_max_val);
    pool.free(d_indices);
    pool.free(d_sorted_logits);
    pool.free(d_sorted_indices);
    // pool.free(d_sampled_index); // 不释放，因为它是返回值
    pool.free(d_reduce_temp_storage);
    pool.free(d_sort_temp_storage);

    // 返回指向设备端采样结果的指针
    return d_sampled_index;
}
#include "common.hpp"
// 批量采样函数
// 功能: 对输入的每个序列位置进行采样，返回指向设备端采样结果的指针数组
// 输入:
//   - logits: 输入的 logits 张量 (T 类型, 形状 [seq_len, vocab_size], 必须在
//   CUDA 设备上)
//   - temperature: 温度系数，用于缩放 logits
//   - top_p: Top-P 采样的概率阈值 (当前代码中未使用)
//   - top_k: Top-K 采样的 K 值
//   - d_states: 指向设备端 cuRAND 状态的指针
//   - stream: CUDA 流
// 返回:
//   - 包含指向设备端存储采样结果的指针数组，每个指针对应一个序列位置的采样结果
template <typename T>
std::vector<uint32_t*> sample_batch(Tensor<T>&& logits, float temperature,
                                    float top_p,  // top_p 未在此实现中使用
                                    size_t top_k, curandState* d_states, cudaStream_t stream) {
    // --- 输入验证 ---
    if (logits.device() != Device::CUDA) {
        throw std::runtime_error("输入张量必须在 CUDA 设备上");
    }
    // Top-K 采样至少需要 k=1
    if (top_k == 0) {
        throw std::runtime_error("top_k 必须至少为 1");
    }

    const auto& shape = logits.sizes();
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
        throw std::runtime_error("输入张量必须是二维且维度非零 [seq_len, vocab_size]");
    }

    const size_t seq_len = shape[0];
    const size_t vocab_size = shape[1];

    // 如果 K 大于词汇表大小，则将其限制为词汇表大小
    if (top_k > vocab_size) {
        top_k = vocab_size;
    }
    // 检查 K 是否超过 Kernel 2 中共享内存的限制 (MAX_TOPK)
    if (top_k > MAX_TOPK) {
        throw std::runtime_error("请求的 top_k (" + std::to_string(top_k) + ") 超过了 Kernel 2 的 MAX_TOPK 限制 (" +
                                 std::to_string(MAX_TOPK) + ")，无法在共享内存中分配。");
    }

    // 创建结果向量，存储每个序列位置的采样结果指针
    std::vector<uint32_t*> result_tokens;
    result_tokens.reserve(seq_len);

    // 为每个序列位置创建一个视图并调用sample函数
    for (size_t i = 0; i < seq_len; i++) {
        // 创建当前位置logits的视图
        // 使用slice方法，需要为每个维度提供start和end向量
        std::vector<size_t> start = {i, 0};
        std::vector<size_t> end = {i + 1, vocab_size};
        Tensor<T> logit_view = logits.slice(start, end);
        uint32_t* token_ptr = sample(std::move(logit_view), temperature, top_p, top_k, d_states, stream);
        result_tokens.push_back(token_ptr);
    }

    return result_tokens;
}

// 采样函数的变体，将结果写入指定的GPU内存位置
template <typename T>
void sample_to_fixed(Tensor<T>&& input, uint32_t* output_ptr, float temperature, float top_p, size_t top_k,
                     curandState* d_states, cudaStream_t stream) {
    // 使用原始sample函数获取采样结果
    uint32_t* sampled_ptr = sample(std::move(input), temperature, top_p, top_k, d_states, stream);

    // 将结果复制到指定的输出位置
    cudaMemcpyAsync(output_ptr, sampled_ptr, sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream);

    // 释放原始sample函数分配的内存
    cudaFree(sampled_ptr);
}

// 批量采样函数的变体，将结果写入指定的GPU内存位置数组
template <typename T>
void sample_batch_to_fixed(Tensor<T>&& logits, uint32_t* output_ptr, float temperature, float top_p, size_t top_k,
                           curandState* d_states, cudaStream_t stream) {
    // --- 输入验证 ---
    if (logits.device() != Device::CUDA) {
        throw std::runtime_error("输入张量必须在 CUDA 设备上");
    }
    // Top-K 采样至少需要 k=1
    if (top_k == 0) {
        throw std::runtime_error("top_k 必须至少为 1");
    }

    const auto& shape = logits.sizes();
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
        throw std::runtime_error("输入张量必须是二维且维度非零 [seq_len, vocab_size]");
    }

    const size_t seq_len = shape[0];
    const size_t vocab_size = shape[1];

    // 为每个序列位置创建一个视图并调用sample_to_fixed函数
    for (size_t i = 0; i < seq_len; i++) {
        // 创建当前位置logits的视图
        std::vector<size_t> start = {i, 0};
        std::vector<size_t> end = {i + 1, vocab_size};
        Tensor<T> logit_view = logits.slice(start, end);

        // 调用单个token的sample_to_fixed函数，将结果写入output_ptr[i]
        sample_to_fixed(std::move(logit_view), output_ptr + i, temperature, top_p, top_k, d_states, stream);
    }
}

// --- 模板显式实例化 ---
// 为 float 和 __nv_bfloat16 类型实例化 sample 函数
template uint32_t* sample<float>(Tensor<float>&&, float, float, size_t, curandState*, cudaStream_t);
template uint32_t* sample<__nv_bfloat16>(Tensor<__nv_bfloat16>&&, float, float, size_t, curandState*, cudaStream_t);

// 为 float 和 __nv_bfloat16 类型实例化 sample_to_fixed 函数
template void sample_to_fixed<float>(Tensor<float>&&, uint32_t*, float, float, size_t, curandState*, cudaStream_t);
template void sample_to_fixed<__nv_bfloat16>(Tensor<__nv_bfloat16>&&, uint32_t*, float, float, size_t, curandState*,
                                             cudaStream_t);

// 为 float 和 __nv_bfloat16 类型实例化 sample_batch 函数
template std::vector<uint32_t*> sample_batch<float>(Tensor<float>&&, float, float, size_t, curandState*, cudaStream_t);
template std::vector<uint32_t*> sample_batch<__nv_bfloat16>(Tensor<__nv_bfloat16>&&, float, float, size_t, curandState*,
                                                            cudaStream_t);

// 为 float 和 __nv_bfloat16 类型实例化 sample_batch_to_fixed 函数
template void sample_batch_to_fixed<float>(Tensor<float>&&, uint32_t*, float, float, size_t, curandState*,
                                           cudaStream_t);
template void sample_batch_to_fixed<__nv_bfloat16>(Tensor<__nv_bfloat16>&&, uint32_t*, float, float, size_t,
                                                   curandState*, cudaStream_t);

}  // namespace cuda_OP

#endif  // CUDA_OP_CUH