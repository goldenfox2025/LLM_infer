#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "avx_operators.hpp"
#include "tensor.hpp"

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

static void compare_and_report(const Tensor<float>& C1, const Tensor<float>& C2) {
    if (C1.sizes() != C2.sizes()) {
        throw std::runtime_error("Output shapes mismatch");
    }
    size_t n = C1.numel();
    const float* p1 = C1.data_ptr();
    const float* p2 = C2.data_ptr();
    double max_abs = 0.0, mae = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = std::abs(double(p1[i]) - double(p2[i]));
        max_abs = std::max(max_abs, diff);
        mae += diff;
    }
    mae /= std::max<size_t>(1, n);
    std::cout << std::fixed << std::setprecision(6) << "max_abs_err=" << max_abs << ", mae=" << mae << "\n";
}

int main(int argc, char** argv) {
    // 解析尺寸，默认 M=N=K=1024，批次1
    size_t B = 1, M = 1024, K = 1024, N = 1024;
    int warmup = 3, iters = 10;
    if (argc >= 5) {
        B = std::stoul(argv[1]);
        M = std::stoul(argv[2]);
        K = std::stoul(argv[3]);
        N = std::stoul(argv[4]);
    }
    if (argc >= 6)
        warmup = std::stoi(argv[5]);
    if (argc >= 7)
        iters = std::stoi(argv[6]);

    std::cout << "B=" << B << ", M=" << M << ", K=" << K << ", N=" << N << ", warmup=" << warmup << ", iters=" << iters
              << "\n";

    std::vector<size_t> a_shape = {B, M, K};
    std::vector<size_t> b_shape = {B, N, K};  // 与 avx 实现一致：B[n, k]

    auto A = make_random_tensor(a_shape, /*seed=*/123);
    auto B_ = make_random_tensor(b_shape, /*seed=*/321);

    // 先计算一次以生成基准结果
    auto C_naive_ref = naive_matmul(A, B_);
    auto C_avx = avx_OP::matmul(A, B_);
    std::cout << "Validate results (single run): ";
    compare_and_report(C_naive_ref, C_avx);

    // 预热
    for (int i = 0; i < warmup; ++i) {
        auto t1 = naive_matmul(A, B_);
        auto t2 = avx_OP::matmul(A, B_);
        (void)t1;
        (void)t2;
    }

    // 基准 - 朴素
    double naive_ms = elapsed_ms(
        [&]() {
            volatile auto t = naive_matmul(A, B_);
            (void)t;
        },
        iters);
    // 基准 - AVX
    double avx_ms = elapsed_ms(
        [&]() {
            volatile auto t = avx_OP::matmul(A, B_);
            (void)t;
        },
        iters);

    // 计算 GFLOPS： 2*M*N*K*B / time
    double flops = 2.0 * double(B) * double(M) * double(N) * double(K);
    double naive_gflops = flops / (naive_ms * 1e6);
    double avx_gflops = flops / (avx_ms * 1e6);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "naive: " << naive_ms << " ms, " << naive_gflops << " GFLOP/s\n";
    std::cout << "  avx: " << avx_ms << " ms, " << avx_gflops << " GFLOP/s\n";
    std::cout << "speedup(avx/naive): " << (naive_ms / avx_ms) << "x\n";

    return 0;
}