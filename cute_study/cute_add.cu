/***************************************************************************************************
 * CuTe Add Kernel Study
 *
 * 这是一个学习CuTe库的简单示例，实现向量加法操作
 * CuTe (CUDA Templates for Linear Algebra Subroutines) 是CUTLASS 3.0的核心组件
 **************************************************************************************************/

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

// CuTe核心头文件
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/fill.hpp>

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

using namespace cute;

/**
 * CuTe向量加法内核
 *
 * @param tensor_a 输入张量A
 * @param tensor_b 输入张量B
 * @param tensor_c 输出张量C = A + B
 * @param size 向量大小
 */
template<typename T>
__global__ void cute_add_kernel(T* a, T* b, T* c, int size) {
    using namespace cute;

    // 创建CuTe张量视图
    // make_tensor创建一个张量，参数为：数据指针，形状，步长
    auto tensor_a = make_tensor(a, make_shape(size), make_stride(1));
    auto tensor_b = make_tensor(b, make_shape(size), make_stride(1));
    auto tensor_c = make_tensor(c, make_shape(size), make_stride(1));

    // 计算线程布局
    // 每个线程处理一个元素
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查
    if (tid < size) {
        // 使用CuTe的张量访问语法
        tensor_c(tid) = tensor_a(tid) + tensor_b(tid);
    }
}

/**
 * 更高级的CuTe向量加法内核 - 使用线程块分片
 */
template<typename T, int ThreadsPerBlock = 256>
__global__ void cute_add_tiled_kernel(T* a, T* b, T* c, int size) {
    using namespace cute;

    // 定义线程块大小
    constexpr int kThreadsPerBlock = ThreadsPerBlock;

    // 创建全局张量视图
    auto tensor_a = make_tensor(a, make_shape(size));
    auto tensor_b = make_tensor(b, make_shape(size));
    auto tensor_c = make_tensor(c, make_shape(size));

    // 定义线程布局
    auto thread_layout = make_layout(make_shape(kThreadsPerBlock));

    // 计算当前线程块处理的数据范围
    int block_start = blockIdx.x * kThreadsPerBlock;
    int thread_id = threadIdx.x;
    int global_id = block_start + thread_id;

    // 使用CuTe的local_partition进行数据分片
    if (global_id < size) {
        tensor_c(global_id) = tensor_a(global_id) + tensor_b(global_id);
    }
}

/**
 * CPU参考实现
 */
template<typename T>
void cpu_add(const T* a, const T* b, T* c, int size) {
    for (int i = 0; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
}

/**
 * 验证结果
 */
template<typename T>
bool verify_result(const T* gpu_result, const T* cpu_result, int size, T tolerance = 1e-5) {
    for (int i = 0; i < size; ++i) {
        if (abs(gpu_result[i] - cpu_result[i]) > tolerance) {
            printf("Mismatch at index %d: GPU = %f, CPU = %f\n",
                   i, (float)gpu_result[i], (float)cpu_result[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    // 解析命令行参数
    int size = 1024 * 1024;  // 默认1M个元素
    if (argc > 1) {
        size = atoi(argv[1]);
    }

    printf("CuTe向量加法示例\n");
    printf("向量大小: %d 个元素\n", size);
    printf("数据类型: float\n\n");

    using T = float;

    // 分配主机内存
    std::vector<T> h_a(size);
    std::vector<T> h_b(size);
    std::vector<T> h_c_gpu(size);
    std::vector<T> h_c_cpu(size);

    // 初始化数据
    for (int i = 0; i < size; ++i) {
        h_a[i] = static_cast<T>(rand()) / RAND_MAX;
        h_b[i] = static_cast<T>(rand()) / RAND_MAX;
    }

    // 分配设备内存
    T *d_a, *d_b, *d_c;
    size_t bytes = size * sizeof(T);

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // 拷贝数据到设备
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    // 配置内核启动参数
    const int threads_per_block = 256;
    const int blocks = (size + threads_per_block - 1) / threads_per_block;

    printf("内核配置: %d blocks, %d threads per block\n", blocks, threads_per_block);

    // 创建CUDA事件用于计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 启动CuTe内核
    CUDA_CHECK(cudaEventRecord(start));
    cute_add_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_c, size);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // 计算执行时间
    float gpu_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));

    // 检查内核执行错误
    CUDA_CHECK(cudaGetLastError());

    // 拷贝结果回主机
    CUDA_CHECK(cudaMemcpy(h_c_gpu.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    // CPU参考计算
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_add(h_a.data(), h_b.data(), h_c_cpu.data(), size);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();

    // 验证结果
    bool correct = verify_result(h_c_gpu.data(), h_c_cpu.data(), size);

    // 输出结果
    printf("\n=== 性能结果 ===\n");
    printf("GPU时间 (CuTe): %.3f ms\n", gpu_time);
    printf("CPU时间: %.3f ms\n", cpu_time);
    printf("加速比: %.2fx\n", cpu_time / gpu_time);
    printf("带宽: %.2f GB/s\n", (3.0 * bytes) / (gpu_time * 1e6));  // 读A,B + 写C
    printf("结果验证: %s\n", correct ? "通过" : "失败");

    // 清理资源
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return correct ? 0 : 1;
}
