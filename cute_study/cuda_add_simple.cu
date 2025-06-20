/***************************************************************************************************
 * 朴素CUDA向量加法 - 用于与CuTe版本对比
 *
 * 这是标准的CUDA实现，不使用任何CuTe特性
 **************************************************************************************************/

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

// CUDA错误检查宏
#define CUDA_CHECK(call)                                                                                  \
    do {                                                                                                  \
        cudaError_t error = call;                                                                         \
        if (error != cudaSuccess) {                                                                       \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while (0)

/**
 * GPU预热内核
 */
template <typename T>
__global__ void warmup_kernel(T* data, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        data[tid] = data[tid] * T(1.001) + T(0.001);
    }
}

/**
 * 朴素CUDA向量加法内核
 */
template <typename T>
__global__ void cuda_add_kernel(const T* A, const T* B, T* C, int size) {
    // 标准CUDA实现
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        C[tid] = A[tid] + B[tid];
    }
}

/**
 * CPU参考实现
 */
template <typename T>
void cpu_add(const T* a, const T* b, T* c, int size) {
    for (int i = 0; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
}

/**
 * 验证结果
 */
template <typename T>
bool verify_result(const T* gpu_result, const T* cpu_result, int size, T tolerance = T(1e-5)) {
    for (int i = 0; i < size; ++i) {
        if (abs(gpu_result[i] - cpu_result[i]) > tolerance) {
            printf("Mismatch at index %d: GPU = %f, CPU = %f\n", i, (float)gpu_result[i], (float)cpu_result[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    using Element = float;

    // 解析命令行参数
    int size = 1024 * 1024;  // 默认1M个元素
    if (argc > 1) {
        size = atoi(argv[1]);
    }

    printf("朴素CUDA向量加法示例（用于与CuTe对比）\n");
    printf("向量大小: %d 个元素\n", size);
    printf("数据类型: float\n");
    printf("使用标准CUDA实现，无CuTe抽象\n\n");

    // 分配主机内存
    std::vector<Element> h_A(size);
    std::vector<Element> h_B(size);
    std::vector<Element> h_C_gpu(size);
    std::vector<Element> h_C_cpu(size);

    // 初始化数据
    srand(42);  // 固定种子确保与CuTe版本一致
    for (int i = 0; i < size; ++i) {
        h_A[i] = static_cast<Element>(rand()) / RAND_MAX;
        h_B[i] = static_cast<Element>(rand()) / RAND_MAX;
    }

    // 分配设备内存
    Element *d_A, *d_B, *d_C, *d_warmup;
    size_t bytes = size * sizeof(Element);

    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMalloc(&d_warmup, bytes));

    // 拷贝数据到设备
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_warmup, h_A.data(), bytes, cudaMemcpyHostToDevice));

    // 配置内核启动参数
    const int threads_per_block = 256;
    const int blocks = (size + threads_per_block - 1) / threads_per_block;

    printf("内核配置: %d blocks, %d threads per block\n", blocks, threads_per_block);

    // === GPU预热 ===
    printf("\n正在进行GPU预热（使用专用缓冲区）...\n");
    for (int i = 0; i < 3; ++i) {
        warmup_kernel<<<blocks, threads_per_block>>>(d_warmup, size);
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("预热轮次 %d/3 完成\n", i + 1);
    }
    printf("GPU预热完成，计算数据未受影响\n\n");

    // 创建CUDA事件用于计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 启动朴素CUDA内核
    printf("=== 测试朴素CUDA向量加法内核 ===\n");
    CUDA_CHECK(cudaEventRecord(start));
    cuda_add_kernel<<<blocks, threads_per_block>>>(d_A, d_B, d_C, size);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // 计算执行时间
    float gpu_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));

    // 检查内核执行错误
    CUDA_CHECK(cudaGetLastError());

    // 拷贝结果回主机
    CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    // CPU参考计算
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_add(h_A.data(), h_B.data(), h_C_cpu.data(), size);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();

    // 验证结果
    bool correct = verify_result(h_C_gpu.data(), h_C_cpu.data(), size);

    // 输出结果
    printf("\n=== 性能结果 ===\n");
    printf("GPU时间 (朴素CUDA): %.3f ms\n", gpu_time);
    printf("CPU时间: %.3f ms\n", cpu_time);
    printf("加速比: %.2fx\n", cpu_time / gpu_time);
    printf("带宽: %.2f GB/s\n", (3.0f * size * sizeof(Element)) / (gpu_time * 1e6));
    printf("结果验证: %s\n", correct ? "✅ 通过" : "❌ 失败");

    // 显示实现特性
    printf("\n=== 实现特性 ===\n");
    printf("✅ 标准CUDA内核实现\n");
    printf("✅ 直接指针访问 C[tid] = A[tid] + B[tid]\n");
    printf("✅ 简单线程ID计算\n");
    printf("✅ 无额外抽象层开销\n");
    printf("✅ GPU预热机制正确实现\n");
    printf("✅ 标准cudaMalloc内存管理\n");

    // 清理资源
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_warmup));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return correct ? 0 : 1;
}