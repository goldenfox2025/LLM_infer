/***************************************************************************************************
 * cuBLAS BF16 矩阵乘法 - 高性能基准测试
 *
 * 基于backend/cpp/src/CUDAoperators/matmul.cu的cuBLAS实现
 * 用于与CuTe版本进行性能对比
 **************************************************************************************************/

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
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

// cuBLAS错误检查宏
inline void checkCublasStatus(cublasStatus_t status, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        char errorMsg[256];
        snprintf(errorMsg, sizeof(errorMsg), "cuBLAS error %d at %s:%d", static_cast<int>(status), file, line);
        fprintf(stderr, "%s\n", errorMsg);
        throw std::runtime_error(errorMsg);
    }
}
#define CHECK_CUBLAS(call) checkCublasStatus(call, __FILE__, __LINE__)

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
 * 基于cuBLAS的BF16矩阵乘法包装函数
 * 基于backend/cpp/src/CUDAoperators/matmul.cu实现
 */
void cublas_bf16_matmul(cublasHandle_t handle, int m, int n, int k, const __nv_bfloat16* A, int lda,
                        const __nv_bfloat16* B, int ldb, __nv_bfloat16* C, int ldc, cudaStream_t stream = 0) {
    // 设置流
    CHECK_CUBLAS(cublasSetStream(handle, stream));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 基于matmul.cu的正确实现
    // 目标计算： C = A * B^T (这里B是K×N，需要转置为N×K)
    // 利用转换： C^T = B * A^T
    // GEMM 调用参数：m = N, n = M, k = K
    cublasStatus_t status = cublasGemmEx(handle,
                                         CUBLAS_OP_T,                   // 转置B
                                         CUBLAS_OP_N,                   // 不转置A
                                         n,                             // m = N
                                         m,                             // n = M
                                         k,                             // k = K
                                         &alpha,                        // alpha
                                         B,                             // B矩阵 (原始B)
                                         CUDA_R_16BF,                   // B的数据类型
                                         ldb,                           // ldb = K
                                         A,                             // A矩阵 (原始A)
                                         CUDA_R_16BF,                   // A的数据类型
                                         lda,                           // lda = K
                                         &beta,                         // beta
                                         C,                             // C矩阵
                                         CUDA_R_16BF,                   // C的数据类型
                                         ldc,                           // ldc = N
                                         CUBLAS_COMPUTE_32F_FAST_TF32,  // 计算类型：启用TF32加速
                                         CUBLAS_GEMM_DEFAULT            // 算法选择
    );

    CHECK_CUBLAS(status);
}

/**
 * CPU参考实现：BF16矩阵乘法
 * A: [M, K], B: [N, K], C: [M, N]
 * 计算: C = A * B^T (因为B是[N,K]布局)
 */
void cpu_matmul_bf16(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                // A[m, k] * B[n, k] (注意B是[N,K]布局)
                float a_val = __bfloat162float(A[m * K + k]);
                float b_val = __bfloat162float(B[n * K + k]);  // B[n,k] 不是 B[k,n]
                sum += a_val * b_val;
            }
            C[m * N + n] = __float2bfloat16(sum);
        }
    }
}

/**
 * 验证结果
 */
bool verify_result_bf16(const __nv_bfloat16* gpu_result, const __nv_bfloat16* cpu_result, int size,
                        float tolerance = 1e-2f) {
    int errors = 0;
    float max_diff = 0.0f;

    for (int i = 0; i < size; ++i) {
        float gpu_val = __bfloat162float(gpu_result[i]);
        float cpu_val = __bfloat162float(cpu_result[i]);
        float diff = fabs(gpu_val - cpu_val);
        max_diff = fmax(max_diff, diff);

        if (diff > tolerance) {
            if (errors < 10) {
                printf("Mismatch at index %d: GPU = %f, CPU = %f, diff = %f\n", i, gpu_val, cpu_val, diff);
            }
            errors++;
        }
    }

    printf("最大误差: %f, 错误数量: %d/%d\n", max_diff, errors, size);
    return errors == 0;
}

int main(int argc, char** argv) {
    // 解析命令行参数
    int M = 512, N = 512, K = 512;
    if (argc >= 2)
        M = atoi(argv[1]);
    if (argc >= 3)
        N = atoi(argv[2]);
    if (argc >= 4)
        K = atoi(argv[3]);

    printf("cuBLAS BF16矩阵乘法基准测试（修复版）\n");
    printf("矩阵大小: A(%dx%d) × B(%dx%d) = C(%dx%d)\n", M, K, N, K, M, N);
    printf("数据类型: __nv_bfloat16\n");
    printf("计算类型: CUBLAS_COMPUTE_32F_FAST_TF32\n");
    printf("布局: A[M,K], B[N,K] (基于matmul.cu)\n\n");

    // 初始化cuBLAS
    static cublasHandle_t handle = nullptr;
    static std::once_flag init_flag;
    static std::mutex handle_mutex;

    std::call_once(init_flag, [&]() {
        std::lock_guard<std::mutex> lock(handle_mutex);
        cublasStatus_t status = cublasCreate(&handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "FATAL ERROR: cublasCreate failed: %d\n", status);
            exit(1);
        }

        std::atexit([]() {
            std::lock_guard<std::mutex> lock(handle_mutex);
            if (handle != nullptr) {
                cublasDestroy(handle);
                handle = nullptr;
            }
        });
    });

    if (handle == nullptr) {
        fprintf(stderr, "Error: cuBLAS handle was not initialized correctly.\n");
        return 1;
    }

    // 使用thrust分配内存 - 正确的布局
    thrust::host_vector<__nv_bfloat16> h_A(M * K);  // A: [M, K]
    thrust::host_vector<__nv_bfloat16> h_B(N * K);  // B: [N, K] 修正！
    thrust::host_vector<__nv_bfloat16> h_C_gpu(M * N);
    thrust::host_vector<__nv_bfloat16> h_C_cpu(M * N);

    // 初始化数据 - 使用较小的数值避免溢出
    srand(42);
    for (int i = 0; i < M * K; ++i) {
        float val = (rand() % 21 - 10) / 10.0f;  // [-1, 1]
        h_A[i] = __float2bfloat16(val);
    }
    for (int i = 0; i < N * K; ++i) {            // 修正！N*K而不是K*N
        float val = (rand() % 21 - 10) / 10.0f;  // [-1, 1]
        h_B[i] = __float2bfloat16(val);
    }

    thrust::device_vector<__nv_bfloat16> d_A = h_A;
    thrust::device_vector<__nv_bfloat16> d_B = h_B;
    thrust::device_vector<__nv_bfloat16> d_C(M * N);
    thrust::device_vector<__nv_bfloat16> d_warmup(M * N);

    // 创建CUDA事件用于计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 首先计算CPU参考结果（小矩阵才计算）
    float cpu_time = 0;
    bool run_cpu = (M <= 1024 && N <= 1024 && K <= 1024);

    if (run_cpu) {
        printf("=== 计算CPU参考结果 ===\n");
        auto cpu_start = std::chrono::high_resolution_clock::now();
        cpu_matmul_bf16(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        cpu_time = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
        printf("CPU计算完成，耗时: %.3f ms\n\n", cpu_time);
    }

    // GPU预热
    printf("=== GPU预热 ===\n");
    const int threads_per_block = 256;
    const int blocks = (M * N + threads_per_block - 1) / threads_per_block;

    for (int i = 0; i < 3; ++i) {
        warmup_kernel<<<blocks, threads_per_block>>>(thrust::raw_pointer_cast(d_warmup.data()), M * N);
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("预热轮次 %d/3 完成\n", i + 1);
    }
    printf("GPU预热完成\n\n");

    // 测试cuBLAS BF16 GEMM
    printf("=== 测试cuBLAS BF16矩阵乘法 ===\n");

    // 内存布局参数 - 基于matmul.cu的正确布局
    int lda = K;  // A: M×K, 行优先
    int ldb = K;  // B: N×K, 行优先
    int ldc = N;  // C: M×N, 行优先

    // 清零输出矩阵
    thrust::fill(d_C.begin(), d_C.end(), __float2bfloat16(0.0f));

    {
        std::lock_guard<std::mutex> lock(handle_mutex);

        CUDA_CHECK(cudaEventRecord(start));

        cublas_bf16_matmul(handle, M, N, K, thrust::raw_pointer_cast(d_A.data()), lda,
                           thrust::raw_pointer_cast(d_B.data()), ldb, thrust::raw_pointer_cast(d_C.data()), ldc);

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
    }

    float gpu_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));

    // 拷贝结果回主机
    h_C_gpu = d_C;

    // 验证结果
    bool correct = true;
    if (run_cpu) {
        correct = verify_result_bf16(h_C_gpu.data(), h_C_cpu.data(), M * N);
        printf("结果验证: %s\n", correct ? "✅ 通过" : "❌ 失败");
    } else {
        printf("矩阵过大，跳过CPU验证\n");
    }

    // 计算GFLOPS
    double gflops = (2.0 * M * N * K) * 1e-9;

    // 输出性能结果
    printf("\n=== 性能结果 ===\n");
    printf("GPU时间 (cuBLAS BF16): %.3f ms, %.2f GFLOPS\n", gpu_time, gflops / (gpu_time * 1e-3));

    if (run_cpu) {
        printf("CPU时间: %.3f ms, %.2f GFLOPS\n", cpu_time, gflops / (cpu_time * 1e-3));
        printf("加速比: %.2fx\n", cpu_time / gpu_time);
    }

    // 显示实现特性
    printf("\n=== 实现特性 ===\n");
    printf("✅ cuBLAS BF16 GEMM with TF32 acceleration\n");
    printf("✅ 数据类型: __nv_bfloat16 (16-bit brain floating point)\n");
    printf("✅ 计算类型: CUBLAS_COMPUTE_32F_FAST_TF32\n");
    printf("✅ 内存布局: A[M,K], B[N,K], C[M,N] (基于matmul.cu)\n");
    printf("✅ 矩阵操作: C = A * B^T (通过转置实现)\n");
    printf("✅ GPU预热机制\n");
    printf("✅ 使用thrust内存管理\n");

    // 清理资源
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return correct ? 0 : 1;
}