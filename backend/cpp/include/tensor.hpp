#pragma once
#include <algorithm>
#include <cassert>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

template <typename T>
class Tensor {
 public:
  // 默认构造函数
  Tensor()
      : data_(std::make_shared<std::vector<T>>()), offset_(0), length_(0) {}
  // 从已有数据和形状构造（要求数据至少包含所有元素）
  Tensor(std::shared_ptr<std::vector<T>> data, const std::vector<size_t>& shape)
      : data_(data), shape_(shape), offset_(0) {
    length_ = 1;
    for (size_t dim : shape_) {
      length_ *= dim;
    }
    if (length_ > data_->size()) {
      throw std::runtime_error("Data size does not match tensor shape");
    }
    strides_ = compute_strides(shape_);
  }

  // 从初始化列表构造，分配新的数据
  Tensor(std::initializer_list<size_t> shape) : shape_(shape), offset_(0) {
    length_ = 1;
    for (size_t dim : shape_) {
      length_ *= dim;
    }
    data_ = std::make_shared<std::vector<T>>(length_);
    strides_ = compute_strides(shape_);
  }
  Tensor operator+(const Tensor& other) {
    if (shape_ != other.shape_) {
      throw std::runtime_error("Tensor shape mismatch");
    }
    Tensor result(shape_);
    for (size_t i = 0; i < length_; ++i) {
      result.data_ptr()[i] = data_ptr()[i] + other.data_ptr()[i];
    }
    return result;
  }
  // 从右值 vector 数据和形状构造
  Tensor(std::vector<T>&& data, const std::vector<size_t>& shape)
      : data_(std::make_shared<std::vector<T>>(std::move(data))),
        shape_(shape),
        offset_(0) {
    length_ = 1;
    for (size_t dim : shape_) {
      length_ *= dim;
    }
    if (length_ > data_->size()) {
      throw std::runtime_error("Data size does not match tensor shape");
    }
    strides_ = compute_strides(shape_);
  }

  // 从 shape 构造，分配新的数据
  Tensor(const std::vector<size_t>& shape) : shape_(shape), offset_(0) {
    length_ = 1;
    for (size_t dim : shape_) {
      length_ *= dim;
    }
    data_ = std::make_shared<std::vector<T>>(length_);
    strides_ = compute_strides(shape_);
  }

  // 拷贝构造函数
  Tensor(const Tensor& other)
      : data_(other.data_),
        shape_(other.shape_),
        strides_(other.strides_),
        offset_(other.offset_),
        length_(other.length_) {}

  // 赋值运算符
  Tensor& operator=(const Tensor& other) {
    if (this != &other) {
      data_ = other.data_;
      shape_ = other.shape_;
      strides_ = other.strides_;
      offset_ = other.offset_;
      length_ = other.length_;
    }
    return *this;
  }

  // 深拷贝：返回一个拥有独立数据的 Tensortemplate<typename T>
  Tensor<T> clone() const {
    // 分配一个新的 vector，新内存大小为当前 tensor 的元素个数
    std::vector<T> new_data(length_);
    size_t rank = shape_.size();
    // 如果是标量（rank == 0）或只有一个元素，则直接拷贝即可
    if (rank == 0 || length_ == 1) {
      new_data[0] = (*data_)[offset_];
    } else {
      // 使用多重索引迭代，将非连续数据按照 row-major 顺序排列到 new_data 中
      // 这里采用“反向”求余法将连续下标转换成各维的坐标（假设标准连续布局为
      // row-major）
      std::vector<size_t> indices(rank, 0);
      for (size_t i = 0; i < length_; i++) {
        size_t remainder = i;
        // 根据 shape 计算标准连续布局下的多维索引（从最后一维到第一维）
        for (size_t d = rank; d > 0; d--) {
          indices[d - 1] = remainder % shape_[d - 1];
          remainder /= shape_[d - 1];
        }
        // 根据当前 tensor 的 strides 计算真实存储中对应的偏移
        size_t src_index = offset_;
        for (size_t d = 0; d < rank; d++) {
          src_index += indices[d] * strides_[d];
        }
        new_data[i] = (*data_)[src_index];
      }
    }

    // 构造一个新的 tensor，使用 shape_ 信息自动计算出标准连续的 strides
    Tensor<T> result(std::move(new_data), shape_);
    // 将 offset 重置为 0，length 保持不变
    result.offset_ = 0;
    result.length_ = length_;
    // 此时 result.strides_ 是标准连续布局（一般由构造函数自动计算）
    return result;
  }

  // 返回数据指针（类似于 libtorch 的 data_ptr()）
  const T* data_ptr() const { return data_->data() + offset_; }
  T* data_ptr() { return data_->data() + offset_; }

  // 返回张量尺寸（等同于 torch 的 sizes()）
  const std::vector<size_t>& sizes() const { return shape_; }

  // 返回元素总数（等同于 torch 的 numel()）
  size_t numel() const { return length_; }

  // 就地填充所有元素（类似于 torch 的 fill_()）
  void fill_(const T& value) {
    T* ptr = data_ptr();
    for (size_t i = 0; i < length_; ++i) {
      ptr[i] = value;
    }
  }

  // 返回当前张量的步长（等同于 torch 的 strides()）
  const std::vector<size_t>& strides() const { return strides_; }

  // 返回一个新的 view，其形状改变为 new_shape（要求元素总数不变）
  Tensor<T> view(const std::vector<size_t>& new_shape) const {
    size_t new_numel = 1;
    for (size_t dim : new_shape) {
      new_numel *= dim;
    }
    if (new_numel != length_) {
      throw std::runtime_error(
          "view: new shape must have the same number of elements");
    }
    Tensor<T> result(*this);
    result.shape_ = new_shape;
    result.strides_ = compute_strides(new_shape);
    return result;
  }

  // 转置操作：交换指定的两个维度，返回一个新的 tensor view（类似于 torch 的
  // transpose）
  Tensor<T> transpose(int dim0, int dim1) const {
    if (dim0 < 0) {
      dim0 += shape_.size();
    }
    if (dim1 < 0) {
      dim1 += shape_.size();
    }
    if (dim0 >= shape_.size() || dim1 >= shape_.size()) {
      throw std::runtime_error("transpose: dimension index out of range");
    }
    Tensor<T> result(*this);
    std::swap(result.shape_[dim0], result.shape_[dim1]);
    std::swap(result.strides_[dim0], result.strides_[dim1]);
    return result;
  }

 private:
  // 计算连续张量的步长
  static std::vector<size_t> compute_strides(const std::vector<size_t>& shape) {
    std::vector<size_t> strides(shape.size());
    size_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
      strides[i] = stride;
      stride *= shape[i];
    }
    return strides;
  }

  std::shared_ptr<std::vector<T>> data_;
  std::vector<size_t> shape_;
  std::vector<size_t> strides_;
  size_t offset_;
  size_t length_;
};
