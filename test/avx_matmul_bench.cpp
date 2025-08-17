#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <map>

#include "avx_operators.hpp"
#include "tensor.hpp"

// 矩阵乘法函数签名
using GemmFunc = std::function<Tensor<float>(const Tensor<float>&, const Tensor<float>&)>;

// 矩阵乘法版本注册器
struct GemmVersion {
    std::string name;
    GemmFunc func;
    std::string description;
};

static std::map<std::string, GemmVersion> gemm_registry;

// 注册一个矩阵乘法版本
static void register_gemm(const std::string& name, GemmFunc func, const std::string& description) {
    gemm_registry[name] = {name, func, description};
}

// 验证矩阵布局: A[M,K] x B[N,K] 其中K维度连续
static void verify_matrix_layout(const Tensor<float>& a, const Tensor<float>& b, 
                                size_t M, size_t K, size_t N) {
    const auto& a_shape = a.sizes();
    const auto& b_shape = b.sizes();
    
    // 验证A是MK布局 (K维度连续)
    if (a_shape[a_shape.size()-1] != K) {
        throw std::runtime_error("Matrix A: K dimension must be contiguous (rightmost)");
    }
    if (a_shape[a_shape.size()-2] != M) {
        throw std::runtime_error("Matrix A: M dimension mismatch");
    }
    
    // 验证B是NK布局 (K维度连续)  
    if (b_shape[b_shape.size()-1] != K) {
        throw std::runtime_error("Matrix B: K dimension must be contiguous (rightmost)");
    }
    if (b_shape[b_shape.size()-2] != N) {
        throw std::runtime_error("Matrix B: N dimension mismatch");
    }
    
    std::cout << "Layout verified: A[M=" << M << ",K=" << K << "], B[N=" << N << ",K=" << K << "] ✓\n";
}

// 朴素 CPU 矩阵乘: A[M,K] x B[N,K] (注意 B 行主存储但按 n,k 访问以匹配 avx_operators 的布局)
static Tensor<float> naive_matmul(const Tensor<float>& a, const Tensor<float>& b) {
    const auto& as = a.sizes();
    const auto& bs = b.sizes();
    if (as.size() < 2 || bs.size() < 2) {
        throw std::runtime_error("naive_matmul: both tensors must have at least 2 dimensions");
    }
    size_t a_rank = as.size();
    size_t b_rank = bs.size();
    size_t M = as[a_rank - 2];
    size_t K = as[a_rank - 1];
    size_t N = bs[b_rank - 1];
    size_t K2 = bs[b_rank - 2];
    if (K != K2) {
        throw std::runtime_error("naive_matmul: inner dimensions do not match");
    }

    // 检查批次维一致
    std::vector<size_t> batch_dims_a(as.begin(), as.end() - 2);
    std::vector<size_t> batch_dims_b(bs.begin(), bs.end() - 2);
    if (batch_dims_a != batch_dims_b) {
        throw std::runtime_error("naive_matmul: batch dimensions must be the same");
    }
    size_t batch_size = 1;
    for (auto d : batch_dims_a)
        batch_size *= d;

    std::vector<size_t> out_shape = batch_dims_a;
    out_shape.push_back(M);
    out_shape.push_back(N);
    Tensor<float> out(out_shape);

    const float* Aall = a.data_ptr();
    const float* Ball = b.data_ptr();
    float* Call = out.data_ptr();

    size_t a_batch_stride = M * K;
    size_t b_batch_stride = N * K;
    size_t c_batch_stride = M * N;

    for (size_t bidx = 0; bidx < batch_size; ++bidx) {
        const float* A = Aall + bidx * a_batch_stride;
        const float* B = Ball + bidx * b_batch_stride;
        float* C = Call + bidx * c_batch_stride;

        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; ++n) {
                float sum = 0.f;
                for (size_t k = 0; k < K; ++k) {
                    sum += A[m * K + k] * B[n * K + k];
                }
                C[m * N + n] = sum;
            }
        }
    }
    return out;
}

static double elapsed_ms(std::function<void()> fn, int iters) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i)
        fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = t1 - t0;
    return ms.count() / iters;
}

static Tensor<float> make_random_tensor(std::vector<size_t> shape, unsigned seed) {
    size_t total = 1;
    for (auto d : shape)
        total *= d;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(total);
    for (size_t i = 0; i < total; ++i)
        data[i] = dist(rng);
    return Tensor<float>(std::move(data), shape);
}

// 比较两个张量并报告精度指标
static bool compare_and_report(const Tensor<float>& reference, const Tensor<float>& test, 
                              const std::string& test_name, double tolerance = 1e-4) {
    if (reference.sizes() != test.sizes()) {
        throw std::runtime_error("Output shapes mismatch");
    }
    size_t n = reference.numel();
    const float* ref_ptr = reference.data_ptr();
    const float* test_ptr = test.data_ptr();
    double max_abs = 0.0, mae = 0.0, mse = 0.0;
    size_t mismatches = 0;
    
    for (size_t i = 0; i < n; ++i) {
        double diff = std::abs(double(ref_ptr[i]) - double(test_ptr[i]));
        max_abs = std::max(max_abs, diff);
        mae += diff;
        mse += diff * diff;
        if (diff > tolerance) mismatches++;
    }
    mae /= std::max<size_t>(1, n);
    mse = std::sqrt(mse / std::max<size_t>(1, n));
    
    bool passed = (max_abs <= tolerance);
    std::cout << std::fixed << std::setprecision(6);
    std::cout << test_name << " accuracy: ";
    std::cout << "max_abs=" << max_abs << ", mae=" << mae << ", rmse=" << mse;
    std::cout << ", mismatches=" << mismatches << "/" << n;
    std::cout << " [" << (passed ? "PASS" : "FAIL") << "]\n";
    
    return passed;
}

// 对单个矩阵乘法函数进行基准测试
static void benchmark_gemm(const std::string& name, GemmFunc func, 
                          const Tensor<float>& A, const Tensor<float>& B,
                          const Tensor<float>& reference, 
                          size_t B_batch, size_t M, size_t K, size_t N,
                          int warmup, int iters) {
    std::cout << "\n=== " << name << " ===" << std::endl;
    
    // 精度测试
    auto result = func(A, B);
    bool accuracy_ok = compare_and_report(reference, result, name);
    
    if (!accuracy_ok) {
        std::cout << "❌ Accuracy test failed for " << name << ", skipping performance test\n";
        return;
    }
    
    // 预热
    for (int i = 0; i < warmup; ++i) {
        volatile auto t = func(A, B);
        (void)t;
    }
    
    // 性能测试
    double time_ms = elapsed_ms([&]() {
        volatile auto t = func(A, B);
        (void)t;
    }, iters);
    
    // 计算指标
    double flops = 2.0 * double(B_batch) * double(M) * double(N) * double(K);
    double gflops = flops / (time_ms * 1e6);
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Performance: " << time_ms << " ms, " << gflops << " GFLOP/s\n";
}

int main(int argc, char** argv) {
    // 解析维度参数: B M K N [预热次数] [迭代次数]
    size_t B = 1, M = 1024, K = 1024, N = 1024;
    int warmup = 3, iters = 10;
    if (argc >= 5) {
        B = std::stoul(argv[1]);
        M = std::stoul(argv[2]);
        K = std::stoul(argv[3]);
        N = std::stoul(argv[4]);
    }
    if (argc >= 6) warmup = std::stoi(argv[5]);
    if (argc >= 7) iters = std::stoi(argv[6]);

    std::cout << "=== AVX GEMM Benchmark ===" << std::endl;
    std::cout << "B=" << B << ", M=" << M << ", K=" << K << ", N=" << N 
              << ", warmup=" << warmup << ", iters=" << iters << "\n\n";

    // 注册矩阵乘法实现
    register_gemm("naive_cpu", naive_matmul, "朴素CPU实现 (参考基准)");
    register_gemm("avx_v1", avx_OP::matmul, "当前AVX实现");
    register_gemm("avx_v2", avx_OP::matmul_v2_fixed, "8x16缓存行优化");
    register_gemm("avx_v3", avx_OP::matmul_v4, "V2 + 顺序8x16 + OpenMP (137+ GFLOP/s)");

    // 创建正确布局的矩阵: A[B,M,K], B[B,N,K]
    std::vector<size_t> a_shape = {B, M, K};
    std::vector<size_t> b_shape = {B, N, K};

    auto A = make_random_tensor(a_shape, /*seed=*/123);
    auto B_ = make_random_tensor(b_shape, /*seed=*/321);

    // 验证矩阵布局
    verify_matrix_layout(A, B_, M, K, N);

    // 使用朴素实现生成参考结果
    std::cout << "\nGenerating reference result..." << std::endl;
    auto reference = naive_matmul(A, B_);

    // 对所有注册的矩阵乘法实现进行基准测试
    std::cout << "\n=== Benchmarking All GEMM Implementations ===" << std::endl;
    
    std::vector<std::pair<std::string, double>> results;
    
    for (const auto& [name, version] : gemm_registry) {
        if (name == "naive_cpu") {
            // 对参考实现的特殊处理
            std::cout << "\n=== " << name << " (reference) ===" << std::endl;
            std::cout << "跳过精度测试 (这是参考实现)\n";
            
            // 预热
            for (int i = 0; i < warmup; ++i) {
                volatile auto t = version.func(A, B_);
                (void)t;
            }
            
            double time_ms = elapsed_ms([&]() {
                volatile auto t = version.func(A, B_);
                (void)t;
            }, iters);
            
            double flops = 2.0 * double(B) * double(M) * double(N) * double(K);
            double gflops = flops / (time_ms * 1e6);
            
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "Performance: " << time_ms << " ms, " << gflops << " GFLOP/s\n";
            results.push_back({name, gflops});
        } else {
            benchmark_gemm(name, version.func, A, B_, reference, B, M, K, N, warmup, iters);
            
            // 计算GFLOPS用于比较
            double time_ms = elapsed_ms([&]() {
                volatile auto t = version.func(A, B_);
                (void)t;
            }, iters);
            double flops = 2.0 * double(B) * double(M) * double(N) * double(K);
            double gflops = flops / (time_ms * 1e6);
            results.push_back({name, gflops});
        }
    }

    // 总结
    std::cout << "\n=== 性能总结 ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    
    double baseline_gflops = 0;
    for (const auto& [name, gflops] : results) {
        if (name == "naive_cpu") {
            baseline_gflops = gflops;
            break;
        }
    }
    
    for (const auto& [name, gflops] : results) {
        double speedup = (baseline_gflops > 0) ? gflops / baseline_gflops : 1.0;
        std::cout << std::setw(12) << name << ": " << std::setw(8) << gflops 
                  << " GFLOP/s  (speedup: " << std::setw(6) << speedup << "x)\n";
    }

    std::cout << "\n=== 添加新版本的说明 ===" << std::endl;
    std::cout << "1. 在avx_operators.hpp中实现新的矩阵乘法函数\n";
    std::cout << "2. 在main()中添加注册: register_gemm(\"avx_v2\", avx_OP::matmul_v2, \"描述\")\n";
    std::cout << "3. 重新编译运行 - 会自动测试精度和性能\n";

    return 0;
}