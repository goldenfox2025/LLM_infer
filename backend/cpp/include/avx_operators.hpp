#pragma once
#include <immintrin.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "tensor.hpp"

namespace avx_OP {

inline Tensor<float> matmul(const Tensor<float>& a, const Tensor<float>& b) {
  const auto& shape_a = a.sizes();
  const auto& shape_b = b.sizes();

  if (shape_a.size() < 2 || shape_b.size() < 2) {
    throw std::runtime_error(
        "matmul_avx: both tensors must have at least 2 dimensions");
  }
  size_t a_rank = shape_a.size();
  size_t b_rank = shape_b.size();

  size_t m = shape_a[a_rank - 2];
  size_t k = shape_a[a_rank - 1];
  size_t n = shape_b[b_rank - 1];
  size_t k2 = shape_b[b_rank - 2];
  if (k != k2) {
    throw std::runtime_error("matmul_avx: inner dimensions do not match");
  }

  std::vector<size_t> batch_dims(shape_a.begin(), shape_a.end() - 2);
  std::vector<size_t> batch_dims_b(shape_b.begin(), shape_b.end() - 2);
  if (batch_dims != batch_dims_b) {
    throw std::runtime_error("matmul_avx: batch dimensions must be the same");
  }
  size_t batch_size = 1;
  for (auto d : batch_dims) {
    batch_size *= d;
  }

  std::vector<size_t> result_shape = batch_dims;
  result_shape.push_back(m);
  result_shape.push_back(n);
  size_t result_elements = 1;
  for (auto d : result_shape) {
    result_elements *= d;
  }
  std::vector<float> result_data(result_elements, 0.0f);
  Tensor<float> result(std::move(result_data), result_shape);

  size_t a_batch_stride = m * k;
  size_t b_batch_stride = n * k;
  size_t res_batch_stride = m * n;

  const float* a_ptr_all = a.data_ptr();
  const float* b_ptr_all = b.data_ptr();
  float* res_ptr_all = result.data_ptr();

  for (size_t b = 0; b < batch_size; b++) {
    const float* a_ptr = a_ptr_all + b * a_batch_stride;
    const float* b_ptr = b_ptr_all + b * b_batch_stride;
    float* res_ptr = res_ptr_all + b * res_batch_stride;
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
        const float* A_row = a_ptr + i * k;
        const float* B_row = b_ptr + j * k;
        __m256 vsum = _mm256_setzero_ps();
        size_t p = 0;
        // 每次处理8个 float
        for (; p + 8 <= k; p += 8) {
          __m256 va = _mm256_loadu_ps(A_row + p);
          __m256 vb = _mm256_loadu_ps(B_row + p);
          vsum = _mm256_fmadd_ps(va, vb, vsum);
        }
        float sum = 0.0f;
        __m128 vlow = _mm256_castps256_ps128(vsum);
        __m128 vhigh = _mm256_extractf128_ps(vsum, 1);
        __m128 vsum128 = _mm_add_ps(vlow, vhigh);
        vsum128 = _mm_hadd_ps(vsum128, vsum128);
        vsum128 = _mm_hadd_ps(vsum128, vsum128);
        _mm_store_ss(&sum, vsum128);
        for (; p < k; p++) {
          sum += A_row[p] * B_row[p];
        }
        res_ptr[i * n + j] = sum;
      }
    }
  }
  return result;
}

}  // namespace avx_OP
