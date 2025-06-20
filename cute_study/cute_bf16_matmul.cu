/***************************************************************************************************
 * CuTe BF16 矩阵乘法 - 高级实现
 *
 * 基于CUTLASS官方示例sgemm_sm80.cu的高级CuTe特性
 * 使用TiledMMA和更复杂的优化技术
 **************************************************************************************************/

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
#include <vector>

// 只包含CuTe核心头文件，如官方示例所示
#include <cute/tensor.hpp>

// CUDA错误检查宏
#define CUDA_CHECK(call)                                                                                  \
    do {                                                                                                  \
        cudaError_t error = call;                                                                         \
        if (error != cudaSuccess) {                                                                       \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while (0)

using namespace cute;

// 类型别名
using BF16 = __nv_bfloat16;

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
 * 简单的CuTe BF16矩阵乘法内核
 */
template <typename T>
__global__ void cute_bf16_matmul_simple_kernel(const T* A, const T* B, T* C, int M, int N, int K) {
    using namespace cute;

    // 定义正确的stride，基于官方示例
    auto stride_A = make_stride(K, Int<1>{});  // (K, 1) - 行优先
    auto stride_B = make_stride(K, Int<1>{});  // (K, 1) - B为[N,K]行优先
    auto stride_C = make_stride(N, Int<1>{});  // (N, 1) - 行优先

    // 创建全局内存张量，使用正确的stride
    Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), stride_A);  // (M, K)
    Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(N, K), stride_B);  // (N, K) 修正！
    Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), stride_C);  // (M, N)

    // 线程到矩阵元素的映射
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;  // 使用float累加以提高精度
        for (int k = 0; k < K; ++k) {
            // BF16计算，累加到float
            // A[row, k] * B[col, k] (B是[N,K]布局)
            sum += __bfloat162float(mA(row, k)) * __bfloat162float(mB(col, k));
        }
        mC(row, col) = __float2bfloat16(sum);
    }
}

/**
 * 高级CuTe BF16矩阵乘法内核 - 基于sgemm_sm80.cu模式
 * 使用TiledMMA和更复杂的优化
 */
template <typename T, int TILE_M, int TILE_N, int TILE_K>
__global__ void cute_bf16_matmul_advanced_kernel(const T* A, const T* B, T* C, int M, int N, int K) {
    using namespace cute;

    // 定义正确的stride
    auto stride_A = make_stride(K, Int<1>{});
    auto stride_B = make_stride(K, Int<1>{});  // B为[N,K]行优先
    auto stride_C = make_stride(N, Int<1>{});

    // 创建全局内存张量
    Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), stride_A);
    Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(N, K), stride_B);  // (N, K) 修正！
    Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), stride_C);

    // 定义CTA tiler，类似官方示例
    auto cta_tiler = make_shape(Int<TILE_M>{}, Int<TILE_N>{}, Int<TILE_K>{});

    // 获取当前thread block的坐标
    auto cta_coord = make_coord(blockIdx.y, blockIdx.x, _);

    // 使用local_tile获取当前CTA的数据块，类似官方示例
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});  // (TILE_M, TILE_K, k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});  // 改为B[N,K]对应的取块方式
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});  // (TILE_M, TILE_N)

    // 简化的TiledMMA模式 - 线程级计算
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 每个线程处理的子块
    int elements_per_thread_m = TILE_M / blockDim.y;
    int elements_per_thread_n = TILE_N / blockDim.x;

    for (int local_m = 0; local_m < elements_per_thread_m; ++local_m) {
        for (int local_n = 0; local_n < elements_per_thread_n; ++local_n) {
            int gm = ty * elements_per_thread_m + local_m;
            int gn = tx * elements_per_thread_n + local_n;

            if (gm < TILE_M && gn < TILE_N) {
                float sum = 0.0f;

                // K维度循环，类似官方示例的k-tile遍历
                for (int k_tile = 0; k_tile < size<2>(gA); ++k_tile) {
                    for (int k_local = 0; k_local < TILE_K; ++k_local) {
                        if (gm < size<0>(gA) && gn < size<0>(gB) && k_local < size<1>(gA) && k_local < size<1>(gB)) {
                            auto a_val = gA(gm, k_local, k_tile);
                            auto b_val = gB(gn, k_local, k_tile);  // B[n,k]访问模式
                            sum += __bfloat162float(a_val) * __bfloat162float(b_val);
                        }
                    }
                }

                if (gm < size<0>(gC) && gn < size<1>(gC)) {
                    gC(gm, gn) = __float2bfloat16(sum);
                }
            }
        }
    }
}

/**
 * NT GEMM版本 - 基于官方示例的通用模式
 */
template <class TA, class TB, class TC>
void cute_gemm_nt_bf16(int m, int n, int k, TA const* A, int ldA, TB const* B, int ldB, TC* C, int ldC,
                       cudaStream_t stream = 0) {
    using namespace cute;

    // 定义形状 (动态)
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);

    // 定义NT步幅 (混合)，类似官方示例
    auto dA = make_stride(Int<1>{}, ldA);  // (dM, dK)
    auto dB = make_stride(Int<1>{}, ldB);  // (dN, dK)
    auto dC = make_stride(Int<1>{}, ldC);  // (dM, dN)

    // 定义CTA tile大小 (静态)
    auto bM = Int<64>{};
    auto bN = Int<64>{};
    auto bK = Int<16>{};

    // 简化的线程布局
    dim3 threadsPerBlock(16, 16);  // 256线程
    dim3 numBlocks((N + bN - 1) / bN, (M + bM - 1) / bM);

    // 调用简单内核
    cute_bf16_matmul_simple_kernel<TC><<<numBlocks, threadsPerBlock, 0, stream>>>(A, B, C, M, N, K);
}

/**
 * CPU参考实现 (BF16)
 * A: [M, K], B: [N, K], C: [M, N]
 * 计算: C = A * B^T (因为B是[N,K]布局)
 */
void cpu_matmul_bf16(const BF16* A, const BF16* B, BF16* C, int M, int N, int K) {
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
bool verify_result_bf16(const BF16* gpu_result, const BF16* cpu_result, int size, float tolerance = 1e-2f) {
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

    printf("CuTe BF16矩阵乘法示例（高级版 - 基于CUTLASS官方示例）\n");
    printf("矩阵大小: A(%dx%d) × B(%dx%d) = C(%dx%d)\n", M, K, N, K, M, N);
    printf("数据类型: __nv_bfloat16\n");
    printf("基于sgemm_sm80.cu的高级CuTe特性\n");
    printf("布局: A[M,K], B[N,K] (修复版)\n\n");

    // 使用thrust分配内存，如官方示例
    thrust::host_vector<BF16> h_A(M * K);
    thrust::host_vector<BF16> h_B(N * K);  // 修正！B为[N,K]
    thrust::host_vector<BF16> h_C_gpu_simple(M * N);
    thrust::host_vector<BF16> h_C_gpu_advanced(M * N);
    thrust::host_vector<BF16> h_C_cpu(M * N);

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

    thrust::device_vector<BF16> d_A = h_A;
    thrust::device_vector<BF16> d_B = h_B;
    thrust::device_vector<BF16> d_C_simple(M * N);
    thrust::device_vector<BF16> d_C_advanced(M * N);

    // 创建CUDA事件用于计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float cpu_time = 0;
    bool run_cpu = true;

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
    thrust::device_vector<BF16> d_warmup(M * N);
    const int threads_per_block = 256;
    const int blocks = (M * N + threads_per_block - 1) / threads_per_block;

    for (int i = 0; i < 3; ++i) {
        warmup_kernel<<<blocks, threads_per_block>>>(thrust::raw_pointer_cast(d_warmup.data()), M * N);
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("预热轮次 %d/3 完成\n", i + 1);
    }
    printf("GPU预热完成\n\n");

    // 测试简单CuTe BF16内核
    printf("=== 测试简单CuTe BF16矩阵乘法内核 ===\n");
    dim3 block1(16, 16);
    dim3 grid1((N + block1.x - 1) / block1.x, (M + block1.y - 1) / block1.y);

    // 清零输出矩阵
    thrust::fill(d_C_simple.begin(), d_C_simple.end(), __float2bfloat16(0.0f));

    CUDA_CHECK(cudaEventRecord(start));
    cute_bf16_matmul_simple_kernel<<<grid1, block1>>>(thrust::raw_pointer_cast(d_A.data()),
                                                      thrust::raw_pointer_cast(d_B.data()),
                                                      thrust::raw_pointer_cast(d_C_simple.data()), M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_time_simple;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_simple, start, stop));
    CUDA_CHECK(cudaGetLastError());

    // 拷贝结果回主机
    h_C_gpu_simple = d_C_simple;

    // 验证简单版本结果
    bool simple_correct = true;
    if (run_cpu) {
        simple_correct = verify_result_bf16(h_C_gpu_simple.data(), h_C_cpu.data(), M * N);
        printf("简单版本结果验证: %s\n", simple_correct ? "✅ 通过" : "❌ 失败");
    }

    // 测试高级CuTe BF16内核
    printf("\n=== 测试高级CuTe BF16矩阵乘法内核 ===\n");
    const int TILE_M = 64, TILE_N = 64, TILE_K = 16;
    dim3 block2(16, 16);
    dim3 grid2((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    // 清零输出矩阵
    thrust::fill(d_C_advanced.begin(), d_C_advanced.end(), __float2bfloat16(0.0f));

    CUDA_CHECK(cudaEventRecord(start));
    cute_bf16_matmul_advanced_kernel<BF16, TILE_M, TILE_N, TILE_K>
        <<<grid2, block2>>>(thrust::raw_pointer_cast(d_A.data()), thrust::raw_pointer_cast(d_B.data()),
                            thrust::raw_pointer_cast(d_C_advanced.data()), M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_time_advanced;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_advanced, start, stop));
    CUDA_CHECK(cudaGetLastError());

    // 拷贝结果回主机
    h_C_gpu_advanced = d_C_advanced;

    // 验证高级版本结果
    bool advanced_correct = true;
    if (run_cpu) {
        advanced_correct = verify_result_bf16(h_C_gpu_advanced.data(), h_C_cpu.data(), M * N);
        printf("高级版本结果验证: %s\n", advanced_correct ? "✅ 通过" : "❌ 失败");
    }

    // 计算GFLOPS
    double gflops = (2.0 * M * N * K) * 1e-9;

    // 输出性能结果
    printf("\n=== 性能结果 ===\n");
    printf("GPU时间 (CuTe简单BF16): %.3f ms, %.2f GFLOPS\n", gpu_time_simple, gflops / (gpu_time_simple * 1e-3));
    printf("GPU时间 (CuTe高级BF16): %.3f ms, %.2f GFLOPS\n", gpu_time_advanced, gflops / (gpu_time_advanced * 1e-3));

    if (run_cpu) {
        printf("CPU时间: %.3f ms, %.2f GFLOPS\n", cpu_time, gflops / (cpu_time * 1e-3));
        printf("加速比 (简单): %.2fx\n", cpu_time / gpu_time_simple);
        printf("加速比 (高级): %.2fx\n", cpu_time / gpu_time_advanced);
    }

    // 清理资源
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return (simple_correct && advanced_correct) ? 0 : 1;
}