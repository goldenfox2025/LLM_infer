/***************************************************************************************************
 * 朴素CUDA矩阵乘法 - 用于与CuTe版本对比
 *
 * 这是标准的CUDA矩阵乘法实现，不使用任何CuTe特性
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
 * 朴素CUDA矩阵乘法内核 - 全局内存版本
 */
template <typename T>
__global__ void cuda_matmul_kernel(const T* A, const T* B, T* C, int M, int N, int K) {
    // 标准CUDA实现，每个线程计算一个C[i][j]
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        T sum = T(0);
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

/**
 * 朴素CUDA矩阵乘法内核 - 共享内存版本
 */
template <typename T, int TILE_SIZE>
__global__ void cuda_matmul_shared_kernel(const T* A, const T* B, T* C, int M, int N, int K) {
    __shared__ T As[TILE_SIZE][TILE_SIZE];
    __shared__ T Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    T sum = T(0);

    for (int phase = 0; phase < (K + TILE_SIZE - 1) / TILE_SIZE; ++phase) {
        // 加载数据到共享内存
        if (row < M && phase * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + phase * TILE_SIZE + tx];
        } else {
            As[ty][tx] = T(0);
        }

        if (col < N && phase * TILE_SIZE + ty < K) {
            Bs[ty][tx] = B[(phase * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = T(0);
        }

        __syncthreads();

        // 计算部分乘积
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * CPU参考实现
 */
template <typename T>
void cpu_matmul(const T* A, const T* B, T* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            T sum = T(0);
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/**
 * 验证结果
 */
template <typename T>
bool verify_result(const T* gpu_result, const T* cpu_result, int size, T tolerance = T(1e-3)) {
    int errors = 0;
    for (int i = 0; i < size; ++i) {
        if (abs(gpu_result[i] - cpu_result[i]) > tolerance) {
            if (errors < 10) {  // 只显示前10个错误
                printf("Mismatch at index %d: GPU = %f, CPU = %f\n", i, (float)gpu_result[i], (float)cpu_result[i]);
            }
            errors++;
        }
    }
    if (errors > 0) {
        printf("Total errors: %d / %d\n", errors, size);
    }
    return errors == 0;
}

int main(int argc, char** argv) {
    using Element = float;

    // 解析命令行参数
    int M = 512, N = 512, K = 512;
    if (argc >= 2)
        M = atoi(argv[1]);
    if (argc >= 3)
        N = atoi(argv[2]);
    if (argc >= 4)
        K = atoi(argv[3]);

    printf("朴素CUDA矩阵乘法示例（用于与CuTe对比）\n");
    printf("矩阵大小: A(%dx%d) × B(%dx%d) = C(%dx%d)\n", M, K, K, N, M, N);
    printf("数据类型: float\n");
    printf("使用标准CUDA实现，无CuTe抽象\n\n");

    // 分配主机内存
    size_t size_A = M * K * sizeof(Element);
    size_t size_B = K * N * sizeof(Element);
    size_t size_C = M * N * sizeof(Element);

    std::vector<Element> h_A(M * K);
    std::vector<Element> h_B(K * N);
    std::vector<Element> h_C_gpu(M * N);
    std::vector<Element> h_C_cpu(M * N);

    // 初始化数据
    srand(42);
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = static_cast<Element>(rand()) / RAND_MAX - 0.5f;
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = static_cast<Element>(rand()) / RAND_MAX - 0.5f;
    }

    // 分配设备内存
    Element *d_A, *d_B, *d_C, *d_warmup;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    CUDA_CHECK(cudaMalloc(&d_warmup, size_C));

    // 拷贝数据到设备
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));

    // GPU预热
    const int WARMUP_TILE_SIZE = 16;
    dim3 warmup_block(16, 16);
    dim3 warmup_grid((N + warmup_block.x - 1) / warmup_block.x, (M + warmup_block.y - 1) / warmup_block.y);

    printf("正在进行GPU预热...\n");
    for (int i = 0; i < 3; ++i) {
        cuda_matmul_shared_kernel<Element, WARMUP_TILE_SIZE>
            <<<warmup_grid, warmup_block>>>(d_A, d_B, d_warmup, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("预热轮次 %d/3 完成\n", i + 1);
    }
    printf("GPU预热完成\n\n");

    // 创建CUDA事件用于计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 测试全局内存版本
    printf("=== 测试朴素CUDA矩阵乘法内核（全局内存）===\n");
    dim3 block1(16, 16);
    dim3 grid1((N + block1.x - 1) / block1.x, (M + block1.y - 1) / block1.y);

    CUDA_CHECK(cudaEventRecord(start));
    cuda_matmul_kernel<<<grid1, block1>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_time_global;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_global, start, stop));
    CUDA_CHECK(cudaGetLastError());

    // 测试共享内存版本
    printf("=== 测试朴素CUDA矩阵乘法内核（共享内存）===\n");
    const int TILE_SIZE = 16;
    dim3 block2(TILE_SIZE, TILE_SIZE);
    dim3 grid2((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    CUDA_CHECK(cudaEventRecord(start));
    cuda_matmul_shared_kernel<Element, TILE_SIZE><<<grid2, block2>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_time_shared;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_shared, start, stop));
    CUDA_CHECK(cudaGetLastError());

    // 拷贝结果回主机
    CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, size_C, cudaMemcpyDeviceToHost));

    // CPU参考计算（小矩阵才计算，避免太慢）
    float cpu_time = 0;
    bool run_cpu = (M <= 1024 && N <= 1024 && K <= 1024);

    if (run_cpu) {
        printf("=== CPU参考计算 ===\n");
        auto cpu_start = std::chrono::high_resolution_clock::now();
        cpu_matmul(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        cpu_time = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
    }

    // 计算GFLOPS
    double gflops = (2.0 * M * N * K) * 1e-9;

    // 输出结果
    printf("\n=== 性能结果 ===\n");
    printf("GPU时间 (全局内存): %.3f ms, %.2f GFLOPS\n", gpu_time_global, gflops / (gpu_time_global * 1e-3));
    printf("GPU时间 (共享内存): %.3f ms, %.2f GFLOPS\n", gpu_time_shared, gflops / (gpu_time_shared * 1e-3));

    if (run_cpu) {
        printf("CPU时间: %.3f ms, %.2f GFLOPS\n", cpu_time, gflops / (cpu_time * 1e-3));
        printf("加速比 (全局内存): %.2fx\n", cpu_time / gpu_time_global);
        printf("加速比 (共享内存): %.2fx\n", cpu_time / gpu_time_shared);

        // 验证结果
        bool correct = verify_result(h_C_gpu.data(), h_C_cpu.data(), M * N);
        printf("结果验证: %s\n", correct ? "✅ 通过" : "❌ 失败");
    }

    printf("共享内存优化提升: %.2fx\n", gpu_time_global / gpu_time_shared);

    // 显示实现特性
    printf("\n=== 实现特性 ===\n");
    printf("✅ 朴素CUDA矩阵乘法实现\n");
    printf("✅ 全局内存版本：直接访问全局内存\n");
    printf("✅ 共享内存版本：使用分块优化\n");
    printf("✅ 2D线程块布局\n");
    printf("✅ 标准cudaMalloc内存管理\n");
    printf("✅ GPU预热机制\n");

    // 清理资源
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_warmup));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}