#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "tensor.hpp"

namespace OP {
// RoPE：对张量最后两个维度进行旋转编码，假定 x 的 shape 为 [seq_len, n_heads,
// head_dim]
template <typename T>
void rope(Tensor<T>* x, size_t offset, float theta) {
  const auto& sizes = x->sizes();
  if (sizes.size() < 3) {
    throw std::runtime_error("rope: tensor must be at least 3D");
  }
  const size_t seq_len = sizes[0];
  const size_t n_heads = sizes[1];
  const size_t head_dim = sizes[2];
  const size_t dim_half = head_dim / 2;
  for (size_t s = 0; s < seq_len; s++) {
    for (size_t h = 0; h < n_heads; h++) {
      T* head_ptr = x->data_ptr() + s * n_heads * head_dim + h * head_dim;
      for (size_t i = 0; i < dim_half; i++) {
        float freq = 1.0f / powf(theta, (2.0f * i) / head_dim);
        float val = (s + offset) * freq;
        float cos_val = cosf(val);
        float sin_val = sinf(val);
        const T x0 = head_ptr[i];
        const T x1 = head_ptr[i + dim_half];
        head_ptr[i] = x0 * cos_val - x1 * sin_val;
        head_ptr[i + dim_half] = x0 * sin_val + x1 * cos_val;
      }
    }
  }
}

// 假定 x 与 out 均为二维张量
template <typename T>
void rms_norm(Tensor<T>* out, const Tensor<T>* x, const Tensor<T>* weight,
              float eps) {
  const auto& sizes = x->sizes();
  if (sizes.size() != 2) {
    throw std::runtime_error("rms_norm: only supports 2D tensors");
  }
  const size_t rows = sizes[0];
  const size_t cols = sizes[1];
  for (size_t i = 0; i < rows; i++) {
    T sum_squares = 0;
    for (size_t j = 0; j < cols; j++) {
      T val = x->data_ptr()[i * cols + j];
      sum_squares += val * val;
    }
    T rms = sqrtf(sum_squares / cols + eps);
    for (size_t j = 0; j < cols; j++) {
      out->data_ptr()[i * cols + j] =
          x->data_ptr()[i * cols + j] / rms * weight->data_ptr()[j];
    }
  }
}

// SiLU激活函数：逐元素操作
template <typename T>
void silu(Tensor<T>* out, const Tensor<T>* x) {
  size_t total = x->numel();
  for (size_t i = 0; i < total; i++) {
    T val = x->data_ptr()[i];
    out->data_ptr()[i] = val / (1 + expf(-val));
  }
}

// sample: 从 logits 中采样下一个 token，支持温度、top-k 和 top-p 采样。
// 参数说明：
// - logits: 形状假定为 [1, vocab_size] 的二维张量。
// - temperature: 温度缩放因子。
// - top_p: nucleus 采样参数（累计概率阈值，取值通常小于 1.0，如 0.9）。
// - top_k: 仅保留概率最高的 top_k 个候选（若为 0 则不进行 top-k 过滤）。
inline uint32_t sample(const Tensor<float>* logits, float temperature,
                       float top_p, size_t top_k) {
  if (logits->device() == Device::CUDA) {
    throw std::runtime_error("sample: logits must be on CPU");
  }

  // 检查 logits 形状，要求为 [1, vocab_size]
  const auto& shape = logits->sizes();

  if (shape.size() != 2 || shape[0] != 1) {
    throw std::runtime_error(
        "sample: logits must be a 2D tensor with shape [1, vocab_size]");
  }
  size_t vocab_size = shape[1];

  const float* logits_ptr = logits->data_ptr();
  if (logits_ptr == nullptr) {
    std::cerr << "[OP::sample] 错误: logits_ptr 为空!" << std::endl;
    throw std::runtime_error("logits_ptr is null");
  }

  std::vector<float> scaled_logits(vocab_size);

  for (size_t i = 1; i < vocab_size; i++) {
    scaled_logits[i] = logits_ptr[i] / temperature;
  }

  float max_logit =
      *std::max_element(scaled_logits.begin(), scaled_logits.end());

  std::vector<float> exp_logits(vocab_size);  // 直接构造指定大小的vector

  float sum_exp = 0.0f;
  for (size_t i = 0; i < vocab_size; i++) {
    exp_logits[i] = std::exp(scaled_logits[i] - max_logit);
    sum_exp += exp_logits[i];
  }
  std::vector<float> probs(vocab_size);
  for (size_t i = 0; i < vocab_size; i++) {
    probs[i] = exp_logits[i] / sum_exp;
  }

  // 调试打印：初始 softmax 后前 5 的 token
  // {
  //   std::vector<uint32_t> all_indices(vocab_size);
  //   std::iota(all_indices.begin(), all_indices.end(), 0);
  //   std::sort(
  //       all_indices.begin(), all_indices.end(),
  //       [&](uint32_t a, uint32_t b) { return logits_ptr[a] > logits_ptr[b];
  //       });
  //   std::cout << "[OP::sample] 初始 softmax 前 5:" << std::endl;
  //   for (size_t i = 0; i < 5 && i < all_indices.size(); i++) {
  //     uint32_t idx = all_indices[i];
  //     std::cout << "  idx = " << idx << ", prob = " << logits_ptr[idx]
  //               << std::endl;
  //   }
  // }

  // 构造候选 token 下标的数组（初始时全部候选）
  // std::cout << "[OP::sample] 初始化候选索引..." << std::endl;
  std::vector<uint32_t> indices(vocab_size);
  for (size_t i = 0; i < vocab_size; i++) {
    indices[i] = static_cast<uint32_t>(i);
  }

  // 先应用 top-k 过滤：保留概率最高的 top_k 个候选
  if (top_k > 0 && top_k < vocab_size) {
    // std::cout << "[OP::sample] 应用 top-k 过滤 (k=" << top_k << ")..."
    //           << std::endl;
    std::sort(indices.begin(), indices.end(),
              [&](uint32_t a, uint32_t b) { return probs[a] > probs[b]; });
    indices.resize(top_k);
    // 经过 top-k 过滤后的前 5
    // std::cout << "[OP::sample] top-k 过滤后前 5:" << std::endl;
    // for (size_t i = 0; i < 5 && i < indices.size(); i++) {
    //   uint32_t idx = indices[i];
    //   std::cout << "  idx = " << idx << ", prob = " << probs[idx] <<
    //   std::endl;
    // }
  }

  // 再应用 top-p (nucleus) 过滤
  if (top_p < 1.0f) {
    // std::cout << "[OP::sample] 应用 top-p 过滤 (p=" << top_p << ")..."
    //           << std::endl;
    std::sort(indices.begin(), indices.end(),
              [&](uint32_t a, uint32_t b) { return probs[a] > probs[b]; });
    float cumulative = 0.0f;
    std::vector<uint32_t> top_p_indices;
    for (uint32_t idx : indices) {
      cumulative += probs[idx];
      top_p_indices.push_back(idx);
      if (cumulative >= top_p) {
        break;
      }
    }
    indices = top_p_indices;
    // std::cout << "[OP::sample] top-p 过滤后前 5:" << std::endl;
    // for (size_t i = 0; i < 5 && i < indices.size(); i++) {
    //   uint32_t idx = indices[i];
    //   std::cout << "  idx = " << idx << ", prob = " << probs[idx] <<
    //   std::endl;
    // }
  }

  // 重新归一化候选 token 的概率
  // std::cout << "[OP::sample] 重新归一化概率..." << std::endl;
  float prob_sum = 0.0f;
  for (uint32_t idx : indices) {
    prob_sum += probs[idx];
  }
  std::vector<float> renorm_probs;
  for (uint32_t idx : indices) {
    renorm_probs.push_back(probs[idx] / prob_sum);
  }

  // 调试打印：归一化后的候选 token（前 5）
  // std::cout << "[OP::sample] 归一化后的候选分布:" << std::endl;
  // for (size_t i = 0; i < 5 && i < indices.size(); i++) {
  //   uint32_t idx = indices[i];
  //   std::cout << "  idx = " << idx << ", norm_prob = " << renorm_probs[i]
  //             << std::endl;
  // }

  // 使用随机数生成器和离散分布进行采样
  // std::cout << "[OP::sample] 开始随机采样..." << std::endl;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> dist(renorm_probs.begin(), renorm_probs.end());
  uint32_t chosen = indices[dist(gen)];
  // std::cout << "[OP::sample] 采样结果: " << chosen << std::endl;
  // std::cout << "[OP::sample] ====== 采样完成 ======\n" << std::endl;
  return chosen;
}
// 逐元素乘法
template <typename T>
void multiply(Tensor<T>* out, const Tensor<T>* a, const Tensor<T>* b) {
  // 获取形状和 strides 信息
  const auto& shape = a->sizes();
  const auto& a_strides = a->strides();
  const auto& b_strides = b->strides();
  const auto& out_strides = out->strides();
  size_t ndim = shape.size();

  // 总的元素个数
  size_t total = a->numel();

  // 用一个 vector 记录当前多维下标
  std::vector<size_t> index(ndim, 0);

  // 遍历所有元素（总次数为 total）
  for (size_t count = 0; count < total; count++) {
    size_t offset_a = 0;
    size_t offset_b = 0;
    size_t offset_out = 0;

    // 根据当前下标 index 和 strides 计算各个 Tensor 的数据偏移
    for (size_t d = 0; d < ndim; d++) {
      offset_a += index[d] * a_strides[d];
      offset_b += index[d] * b_strides[d];
      offset_out += index[d] * out_strides[d];
    }

    // 进行对应位置的相乘
    out->data_ptr()[offset_out] =
        a->data_ptr()[offset_a] * b->data_ptr()[offset_b];

    // 更新多维下标 index：从最后一个维度开始进位
    for (int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
      index[d]++;
      if (index[d] < shape[d]) {
        break;  // 本维度未达到上界，退出进位
      } else {
        // 本维度溢出，重置为 0，继续向前进位
        index[d] = 0;
      }
    }
  }
}

// Gather：根据索引从 table 中查找行，复制到 out 中；假定 table 为二维张量
template <typename T>
void gather(Tensor<T>* out, const Tensor<uint32_t>* indices,
            const Tensor<T>* table) {
  const size_t num_indices = indices->numel();
  const auto& table_sizes = table->sizes();
  if (table_sizes.size() < 2) {
    throw std::runtime_error("gather: table tensor must be at least 2D");
  }
  const size_t embedding_dim = table_sizes[1];
  for (size_t i = 0; i < num_indices; i++) {
    const uint32_t idx = indices->data_ptr()[i];
    const T* src = table->data_ptr() + idx * embedding_dim;
    T* dst = out->data_ptr() + i * embedding_dim;
    std::copy(src, src + embedding_dim, dst);
  }
}

#include <cmath>
#include <limits>
#include <stdexcept>

template <typename T>
void softmax(Tensor<T>* out, const Tensor<T>* x, int dim, bool mask = false,
             int heads = 1, int offset = 0) {
  const auto& shape = x->sizes();
  if (shape.empty()) {
    throw std::runtime_error("softmax: input tensor is empty!");
  }
  size_t rank = shape.size();
  if (dim < 0) {
    dim += static_cast<int>(rank);
  }
  if (dim < 0 || static_cast<size_t>(dim) >= rank) {
    throw std::runtime_error("softmax: dimension out of range");
  }
  // 计算 outer, N, inner
  // N 为 softmax 维度的大小
  size_t N = shape[dim];
  size_t outer = 1;
  for (size_t i = 0; i < static_cast<size_t>(dim); i++) {
    outer *= shape[i];
  }
  size_t inner = 1;
  for (size_t i = dim + 1; i < rank; i++) {
    inner *= shape[i];
  }
  // 获取在 softmax 维度上的 stride
  size_t stride = x->strides()[dim];
  const T* x_ptr = x->data_ptr();
  T* out_ptr = out->data_ptr();
  // outer 循环对应于 [seq_len * n_q_h]（例如在 att_scores 中）
  for (size_t o = 0; o < outer; o++) {
    // 如果 mask 为 true，则计算该 slice 对应的 query token 序号
    int valid_length = static_cast<int>(N);
    if (mask) {
      // 假设 outer = seq_len * n_q_h，则 query index 为：
      size_t query_index = o / heads;
      // 如果有 KVCache，则有效长度 = offset + query_index + 1；否则为
      // query_index + 1
      valid_length = (offset > 0 ? static_cast<int>(offset + query_index)
                                 : static_cast<int>(query_index)) +
                     1;
      if (valid_length > static_cast<int>(N))
        valid_length = static_cast<int>(N);
    }
    for (size_t i = 0; i < inner; i++) {
      size_t base_index = (o * N * inner) + i;
      // 数值稳定性：仅在有效区域内寻找最大值，其它位置当作 -1e9
      T max_val = -1e9;
      for (size_t j = 0; j < N; j++) {
        size_t idx = base_index + j * stride;
        // 如果 mask 且 j 超出有效范围，则认为该位置为 -1e9
        T val =
            (mask && static_cast<int>(j) >= valid_length) ? -1e9 : x_ptr[idx];
        if (val > max_val) {
          max_val = val;
        }
      }
      // 计算 exponent 并累加 sum
      T sum = 0;
      for (size_t j = 0; j < N; j++) {
        size_t idx = base_index + j * stride;
        T val =
            (mask && static_cast<int>(j) >= valid_length) ? -1e9 : x_ptr[idx];
        T exp_val = std::exp(val - max_val);
        out_ptr[idx] = exp_val;
        sum += exp_val;
      }

      // 归一化
      if (sum > 0) {
        for (size_t j = 0; j < N; j++) {
          size_t idx = base_index + j * stride;
          out_ptr[idx] /= sum;
        }
      } else {
        // 极端情况下（例如全为 -∞），全部置 0
        for (size_t j = 0; j < N; j++) {
          size_t idx = base_index + j * stride;
          out_ptr[idx] = 0;
        }
      }
    }
  }
}

// Matmul：支持广播并适应多维矩阵乘法。
// 设 a 的形状为 [..., m, k]，b 的形状为 [..., k, n]，则结果形状为广播后的批次
// shape + [m, n].
template <typename T>
Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b) {
  const auto& shape_a = a.sizes();
  const auto& shape_b = b.sizes();
  if (shape_a.size() < 2 || shape_b.size() < 2) {
    throw std::runtime_error(
        "matmul: both tensors must have at least 2 dimensions");
  }
  size_t a_rank = shape_a.size();
  size_t b_rank = shape_b.size();

  // a 的形状 [*, m, k], b 的形状 [*, k, n]
  size_t m = shape_a[a_rank - 2];
  size_t k = shape_a[a_rank - 1];
  size_t k2 = shape_b[b_rank - 2];
  size_t n = shape_b[b_rank - 1];
  if (k != k2) {
    throw std::runtime_error("matmul: inner dimensions do not match");
  }

  // 提取批次维度
  std::vector<size_t> a_batch(shape_a.begin(), shape_a.end() - 2);
  std::vector<size_t> b_batch(shape_b.begin(), shape_b.end() - 2);

  // 计算广播后的批次数组
  size_t max_batch_rank = std::max(a_batch.size(), b_batch.size());

  std::vector<size_t> broadcast_batch(max_batch_rank, 1);

  for (size_t i = 0; i < max_batch_rank; i++) {
    size_t a_dim = 1, b_dim = 1;
    // 从后往前对齐
    if (i < a_batch.size()) {
      // a_batch.size() - 1, a_batch.size() - 2 ...
      // 这里直接对齐: i -> max_batch_rank - i - 1
      a_dim = a_batch[a_batch.size() - 1 - i];
    }
    if (i < b_batch.size()) {
      b_dim = b_batch[b_batch.size() - 1 - i];
    }
    if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
      throw std::runtime_error(
          "matmul: batch dimensions are not broadcastable");
    }
    broadcast_batch[max_batch_rank - 1 - i] = std::max(a_dim, b_dim);
  }

  // 结果形状 = broadcast_batch + [m, n]
  std::vector<size_t> result_shape = broadcast_batch;
  result_shape.push_back(m);
  result_shape.push_back(n);
  size_t result_elements = 1;
  for (auto d : result_shape) {
    result_elements *= d;
  }
  std::vector<T> result_data(result_elements, T(0));
  Tensor<T> result(std::move(result_data), result_shape);

  // 计算总 batch_size，用于遍历所有批次
  size_t batch_size = 1;
  for (auto d : broadcast_batch) {
    batch_size *= d;
  }

  // 保存 a 与 b 批次部分的 strides
  // a_batch_rank = shape_a.size() - 2
  // b_batch_rank = shape_b.size() - 2
  std::vector<size_t> a_batch_strides(a_batch.size());
  std::vector<size_t> b_batch_strides(b_batch.size());

  {
    const auto& a_strides = a.strides();
    for (size_t i = 0; i < a_batch.size(); i++) {
      a_batch_strides[i] = a_strides[i];
    }
    const auto& b_strides = b.strides();
    for (size_t i = 0; i < b_batch.size(); i++) {
      b_batch_strides[i] = b_strides[i];
    }
  }

  // 针对最后两维 (m, k) (k, n)  同样要读取 strides
  // a 倒数第2维 => 行stride, 倒数第1维 => 列stride
  size_t a_row_stride = a.strides()[a_rank - 2];
  size_t a_col_stride = a.strides()[a_rank - 1];
  size_t b_row_stride = b.strides()[b_rank - 2];
  size_t b_col_stride = b.strides()[b_rank - 1];

  // result 同理：其 rank = max_batch_rank + 2
  // 倒数第2维 => m => 行stride, 倒数第1维 => n => 列stride
  {
    // 这时我们可以做新的 strides 读取:
    const auto& r_strides = result.strides();
  }
  // 迭代 broadcast 的索引：与 broadcast_batch 同维度
  std::vector<size_t> index(max_batch_rank, 0);

  // 取出 result 的 strides，一会要用到
  const auto& res_strides = result.strides();

  // 遍历 batch_size 个批次
  for (size_t count = 0; count < batch_size; count++) {
    size_t a_offset = 0;
    {
      int offset_index = (int)max_batch_rank - (int)a_batch.size();
      for (int i = 0; i < (int)a_batch.size(); i++) {
        // 若该维 a_batch[i] == 1 则重复使用 index[offset_index + i] = 0
        size_t idx = (a_batch[i] == 1) ? 0 : index[offset_index + i];
        a_offset += idx * a_batch_strides[i];
      }
    }

    size_t b_offset = 0;
    {
      int offset_index = (int)max_batch_rank - (int)b_batch.size();
      for (int i = 0; i < (int)b_batch.size(); i++) {
        size_t idx = (b_batch[i] == 1) ? 0 : index[offset_index + i];
        b_offset += idx * b_batch_strides[i];
      }
    }

    // 计算 result 对应批次的 offset
    size_t res_offset = 0;
    {
      // result 的前 max_batch_rank 个维度对应 batch
      for (size_t i = 0; i < max_batch_rank; i++) {
        res_offset += index[i] * res_strides[i];
      }
    }

    // 执行 2D 矩阵乘法，但用 strides 访问
    const T* a_ptr = a.data_ptr() + a_offset;
    const T* b_ptr = b.data_ptr() + b_offset;
    T* res_ptr = result.data_ptr() + res_offset;

    // 取 result 在最后2维的行、列 stride
    size_t r_row_stride = res_strides[max_batch_rank];      // 对应 m
    size_t r_col_stride = res_strides[max_batch_rank + 1];  // 对应 n

    // 朴素乘法： O(m*n*k)
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
        T sum = 0;
        for (size_t p = 0; p < k; p++) {
          // a 在 (i, p) => i*a_row_stride + p*a_col_stride
          // b 在 (p, j) => p*b_row_stride + j*b_col_stride
          size_t a_idx = i * a_row_stride + p * a_col_stride;
          size_t b_idx = p * b_row_stride + j * b_col_stride;
          sum += a_ptr[a_idx] * b_ptr[b_idx];
        }
        // result 在 (i, j) => i*r_row_stride + j*r_col_stride
        size_t r_idx = i * r_row_stride + j * r_col_stride;
        res_ptr[r_idx] = sum;
      }
    }

    // 增量下一个批次的 index
    for (int d = (int)max_batch_rank - 1; d >= 0; d--) {
      index[d]++;
      if (index[d] < broadcast_batch[d]) {
        break;
      } else {
        index[d] = 0;
      }
    }
  }
  return result;
}

template Tensor<float> matmul<float>(const Tensor<float>&,
                                     const Tensor<float>&);
template void rope<float>(Tensor<float>*, size_t, float);
template void rms_norm<float>(Tensor<float>*, const Tensor<float>*,
                              const Tensor<float>*, float);
template void silu<float>(Tensor<float>*, const Tensor<float>*);
template void multiply<float>(Tensor<float>*, const Tensor<float>*,
                              const Tensor<float>*);
template void gather<float>(Tensor<float>*, const Tensor<uint32_t>*,
                            const Tensor<float>*);
template void softmax<float>(Tensor<float>*, const Tensor<float>*, int, bool,
                             int, int);

}  // namespace OP
