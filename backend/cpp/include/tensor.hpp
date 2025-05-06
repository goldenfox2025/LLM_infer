#pragma once
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "CudaMemoryPool.hpp"  // 该头中定义了 CudaMemoryPool，提供静态方法 allocate/free

enum class Device { CPU, CUDA };

// 前向声明 Tensor 类模板（用于友元声明）
template <typename T>
class Tensor;

// 声明类型转换辅助函数，专用于 Tensor<__nv_bfloat16> 到 Tensor<float> 的转换
template <typename FromType, typename ToType>
Tensor<ToType> tensor_convert(const Tensor<FromType>& src);

template <typename T>
class Tensor {
 private:
  // 检查 CUDA 错误的静态内联函数（可在 const 成员中调用）
  static inline void checkCudaError(cudaError_t error) {
    if (error != cudaSuccess) {
      throw std::runtime_error("CUDA error: " +
                               std::string(cudaGetErrorString(error)));
    }
  }

 public:
  // 友元声明，允许任意 Tensor<U> 访问 Tensor<T> 的私有成员
  template <typename U>
  friend class Tensor;

  // 将类型转换辅助函数声明为友元
  template <typename FromType, typename ToType>
  friend Tensor<ToType> tensor_convert(const Tensor<FromType>& src);

  // 默认构造函数（CPU 模式，空 tensor）
  Tensor()
      : data_(std::make_shared<std::vector<T>>()),
        offset_(0),
        length_(0),
        device_(Device::CPU),
        gpu_data_(nullptr, [](T* ptr) { /* no-op */ }) {}

  // 从已有数据和形状构造（要求数据至少包含所有元素，CPU 模式）
  Tensor(std::shared_ptr<std::vector<T>> data, const std::vector<size_t>& shape)
      : data_(data),
        shape_(shape),
        offset_(0),
        length_(1),
        device_(Device::CPU),
        gpu_data_(nullptr, [](T* ptr) { /* no-op */ }) {
    for (size_t dim : shape_) {
      length_ *= dim;
    }
    if (length_ > data_->size()) {
      throw std::runtime_error("Data size does not match tensor shape");
    }
    strides_ = compute_strides(shape_);
  }

  // 从 initializer_list 构造，分配新的数据，并指定设备
  // is_prefill: 是否是prefill阶段的内存分配
  // tag: 内存标签，用于固定内存分配
  Tensor(std::initializer_list<size_t> shape, Device device = Device::CPU,
         bool is_prefill = false, const std::string& tag = "")
      : shape_(shape),
        offset_(0),
        length_(1),
        device_(device),
        gpu_data_(nullptr, [](T* ptr) { /* no-op */ }),
        tag_(tag) {
    for (size_t dim : shape_) {
      length_ *= dim;
    }
    strides_ = compute_strides(shape_);
    if (device_ == Device::CPU) {
      data_ = std::make_shared<std::vector<T>>(length_);
      gpu_data_.reset();
    } else if (device_ == Device::CUDA) {
      data_.reset();
      // 通过内存池申请 GPU 内存
      T* gpu_ptr = nullptr;
      if (!tag_.empty()) {
        // 使用标签分配固定内存
        gpu_ptr = static_cast<T*>(GlobalCudaMemoryPool::allocate_tagged(
            tag_, length_ * sizeof(T), is_prefill));
      } else {
        // 常规内存分配
        gpu_ptr = static_cast<T*>(GlobalCudaMemoryPool::instance().allocate(
            length_ * sizeof(T), is_prefill));
      }
      gpu_data_ = std::shared_ptr<T>(
          gpu_ptr, [](T* ptr) { GlobalCudaMemoryPool::instance().free(ptr); });
    } else {
      throw std::runtime_error("Invalid device specified");
    }
  }

  // 这个仅限gpu_ptr已经被分配了内存，用于sample返回gpu指针的从尝试版本
  Tensor(T* gpu_ptr, const std::vector<size_t>& shape, Device device)
      : shape_(shape),
        offset_(0),
        length_(1),
        device_(Device::CUDA),
        data_(nullptr),
        gpu_data_(gpu_ptr,
                  [](T* ptr) { GlobalCudaMemoryPool::instance().free(ptr); }) {
    if (device == Device::CPU) {
      throw std::runtime_error(
          "Invalid device specified in Tensor constructor");
    }
    for (size_t dim : shape_) {
      length_ *= dim;
    }
    strides_ = compute_strides(shape_);
  }
  // 从右值 vector 数据和形状构造（CPU 模式）
  Tensor(std::vector<T>&& data, const std::vector<size_t>& shape)
      : data_(std::make_shared<std::vector<T>>(std::move(data))),
        shape_(shape),
        offset_(0),
        length_(1),
        device_(Device::CPU),
        gpu_data_(nullptr, [](T* ptr) { /* no-op */ }) {
    for (size_t dim : shape_) {
      length_ *= dim;
    }
    if (length_ > data_->size()) {
      throw std::runtime_error("Data size does not match tensor shape");
    }
    strides_ = compute_strides(shape_);
  }

  // 从右值 vector 数据、形状和 device 构造
  Tensor(std::vector<T>&& data, const std::vector<size_t>& shape, Device device)
      : shape_(shape),
        offset_(0),
        length_(1),
        device_(device),
        gpu_data_(nullptr, [](T* ptr) { /* no-op */ }) {
    for (size_t dim : shape_) {
      length_ *= dim;
    }
    strides_ = compute_strides(shape_);
    if (device_ == Device::CPU) {
      // CPU 模式：直接保存数据
      data_ = std::make_shared<std::vector<T>>(std::move(data));
    } else if (device_ == Device::CUDA) {
      // GPU 模式：通过内存池申请 GPU 内存并拷贝数据
      data_.reset();
      T* gpu_ptr = static_cast<T*>(
          GlobalCudaMemoryPool::instance().allocate(length_ * sizeof(T)));
      checkCudaError(cudaMemcpyAsync(gpu_ptr, data.data(), length_ * sizeof(T),
                                     cudaMemcpyHostToDevice));
      gpu_data_ = std::shared_ptr<T>(
          gpu_ptr, [](T* ptr) { GlobalCudaMemoryPool::instance().free(ptr); });
    } else {
      throw std::runtime_error("Invalid device specified");
    }
  }

  // 从 shape 构造，分配新的数据（CPU 模式缺省）
  Tensor(const std::vector<size_t>& shape)
      : shape_(shape),
        offset_(0),
        length_(1),
        device_(Device::CPU),
        gpu_data_(nullptr, [](T* ptr) { /* no-op */ }) {
    for (size_t dim : shape_) {
      length_ *= dim;
    }
    data_ = std::make_shared<std::vector<T>>(length_);
    strides_ = compute_strides(shape_);
  }

  // 从 shape 和 device 构造
  // is_prefill: 是否是prefill阶段的内存分配
  // tag: 内存标签，用于固定内存分配
  Tensor(const std::vector<size_t>& shape, Device device,
         bool is_prefill = false, const std::string& tag = "")
      : shape_(shape),
        offset_(0),
        length_(1),
        device_(device),
        gpu_data_(nullptr, [](T* ptr) { /* no-op */ }),
        tag_(tag) {
    for (size_t dim : shape_) {
      length_ *= dim;
    }
    strides_ = compute_strides(shape_);
    if (device_ == Device::CPU) {
      data_ = std::make_shared<std::vector<T>>(length_);
      gpu_data_.reset();
    } else if (device_ == Device::CUDA) {
      data_.reset();
      // 通过内存池申请 GPU 内存
      T* gpu_ptr = nullptr;
      if (!tag_.empty()) {
        // 使用标签分配固定内存
        gpu_ptr = static_cast<T*>(GlobalCudaMemoryPool::allocate_tagged(
            tag_, length_ * sizeof(T), is_prefill));
      } else {
        // 常规内存分配
        gpu_ptr = static_cast<T*>(GlobalCudaMemoryPool::instance().allocate(
            length_ * sizeof(T), is_prefill));
      }
      gpu_data_ = std::shared_ptr<T>(
          gpu_ptr, [](T* ptr) { GlobalCudaMemoryPool::instance().free(ptr); });
    } else {
      throw std::runtime_error("Invalid device specified");
    }
  }

  // 拷贝构造（CUDA 下采用浅拷贝，即共享 gpu_data_）
  Tensor(const Tensor& other)
      : data_(other.data_),
        shape_(other.shape_),
        strides_(other.strides_),
        offset_(other.offset_),
        length_(other.length_),
        device_(other.device_) {
    if (device_ == Device::CUDA) {
      gpu_data_ = other.gpu_data_;
    } else {
      gpu_data_.reset();
    }
  }

  // 赋值运算符（CUDA 下采用浅拷贝）
  Tensor& operator=(const Tensor& other) {
    if (this != &other) {
      shape_ = other.shape_;
      strides_ = other.strides_;
      offset_ = other.offset_;
      length_ = other.length_;
      device_ = other.device_;
      if (device_ == Device::CUDA) {
        gpu_data_ = other.gpu_data_;
        data_.reset();
      } else {
        data_ = other.data_;
        gpu_data_.reset();
      }
    }
    return *this;
  }

  // 转换为 Tensor<float>（若当前类型 T 不是 float，则每个元素转换为 float）
  Tensor<float> to_float() const {
    if constexpr (std::is_same_v<T, float>) {
      return *this;
    } else {
      return tensor_convert<T, float>(*this);
    }
  }
  int nbytes() const { return sizeof(T) * length_; }
  // 返回数据指针（const 版本）
  const T* data_ptr() const {
    if (device_ == Device::CPU) {
      return data_->data() + offset_;
    } else {
      return gpu_data_.get() + offset_;
    }
  }
  // 返回数据指针（非 const 版本）
  T* data_ptr() {
    if (device_ == Device::CPU) {
      return data_->data() + offset_;
    } else {
      return gpu_data_.get() + offset_;
    }
  }

  // 返回张量尺寸
  const std::vector<size_t>& sizes() const { return shape_; }

  // 返回元素总数
  size_t numel() const { return length_; }

  // 填充张量
  void fill_(const T& value) {
    if (device_ == Device::CPU) {
      T* ptr = data_ptr();
      for (size_t i = 0; i < length_; ++i) {
        ptr[i] = value;
      }
    } else {
      std::vector<T> cpu_data(length_);
      std::fill(cpu_data.begin(), cpu_data.end(), value);
      checkCudaError(cudaMemcpy(gpu_data_.get(), cpu_data.data(),
                                length_ * sizeof(T), cudaMemcpyHostToDevice));
    }
  }

  // 返回 strides
  const std::vector<size_t>& strides() const { return strides_; }

  // view：返回一个共享底层数据的新张量（不拷贝数据，仅修改元信息）// 对于非
  // const 左值：直接修改自身
  Tensor<T>& view(const std::vector<size_t>& new_shape) & {
    size_t new_numel = 1;
    for (size_t dim : new_shape) {
      new_numel *= dim;
    }
    if (new_numel != length_) {
      std::cerr << "[Tensor::view] Error: New shape's number of elements ("
                << new_numel << ") does not match original (" << length_ << ")"
                << std::endl;
      throw std::runtime_error(
          "view: New shape must have same number of elements");
    }
    shape_ = new_shape;
    strides_ = compute_strides(new_shape);
    return *this;
  }

  // 对于非 const 右值：移动后构造一个新张量返回
  Tensor<T> view(const std::vector<size_t>& new_shape) && {
    size_t new_numel = 1;
    for (size_t dim : new_shape) {
      new_numel *= dim;
    }
    if (new_numel != length_) {
      std::cerr << "[Tensor::view] Error: New shape's number of elements ("
                << new_numel << ") does not match original (" << length_ << ")"
                << std::endl;
      throw std::runtime_error(
          "view: New shape must have same number of elements");
    }
    Tensor<T> result = std::move(*this);
    result.shape_ = new_shape;
    result.strides_ = compute_strides(new_shape);
    return result;
  }

  // 新增 const 左值版本：对于 const 对象返回一个新
  // Tensor（拷贝），而不是修改原对象
  Tensor<T> view(const std::vector<size_t>& new_shape) const& {
    size_t new_numel = 1;
    for (size_t dim : new_shape) {
      new_numel *= dim;
    }
    if (new_numel != length_) {
      std::cerr << "[Tensor::view] Error: New shape's number of elements ("
                << new_numel << ") does not match original (" << length_ << ")"
                << std::endl;
      throw std::runtime_error(
          "view: New shape must have same number of elements");
    }
    Tensor<T> result = *this;  // 拷贝当前对象
    result.shape_ = new_shape;
    result.strides_ = compute_strides(new_shape);
    return result;
  }

  // transpose：交换两个维度
  Tensor<T> transpose(int dim0, int dim1) const {
    if (dim0 < 0) dim0 += shape_.size();
    if (dim1 < 0) dim1 += shape_.size();
    if (dim0 >= shape_.size() || dim1 >= shape_.size()) {
      throw std::runtime_error("transpose: dimension index out of range");
    }
    Tensor<T> result(*this);
    std::swap(result.shape_[dim0], result.shape_[dim1]);
    std::swap(result.strides_[dim0], result.strides_[dim1]);
    return result;
  }

  Tensor<T>& slice_inplace(const std::vector<size_t>& start,
                           const std::vector<size_t>& end) & {
    if (start.size() != shape_.size() || end.size() != shape_.size()) {
      throw std::runtime_error(
          "slice: start and end must have same dimensions as tensor");
    }
    std::vector<size_t> new_shape(shape_.size());
    for (size_t i = 0; i < shape_.size(); i++) {
      if (start[i] >= shape_[i] || end[i] > shape_[i] || start[i] >= end[i]) {
        throw std::runtime_error("slice: invalid start or end indices");
      }
      new_shape[i] = end[i] - start[i];
    }
    size_t new_offset = offset_;
    for (size_t i = 0; i < shape_.size(); i++) {
      new_offset += start[i] * strides_[i];
    }
    size_t new_length = 1;
    for (size_t dim : new_shape) {
      new_length *= dim;
    }
    this->shape_ = new_shape;
    this->strides_ = strides_;
    this->offset_ = new_offset;
    this->length_ = new_length;
    this->device_ = device_;
    if (device_ == Device::CPU) {
      this->data_ = data_;
      this->gpu_data_.reset();
    } else {
      this->data_.reset();
      this->gpu_data_ = gpu_data_;
    }
    return *this;
  }

  // slice：提取张量的一部分，仍共享底层数据
  Tensor<T> slice(const std::vector<size_t>& start,
                  const std::vector<size_t>& end) const {
    if (start.size() != shape_.size() || end.size() != shape_.size()) {
      throw std::runtime_error(
          "slice: start and end must have same dimensions as tensor");
    }
    std::vector<size_t> new_shape(shape_.size());
    for (size_t i = 0; i < shape_.size(); i++) {
      if (start[i] >= shape_[i] || end[i] > shape_[i] || start[i] >= end[i]) {
        throw std::runtime_error("slice: invalid start or end indices");
      }
      new_shape[i] = end[i] - start[i];
    }
    size_t new_offset = offset_;
    for (size_t i = 0; i < shape_.size(); i++) {
      new_offset += start[i] * strides_[i];
    }
    size_t new_length = 1;
    for (size_t dim : new_shape) {
      new_length *= dim;
    }
    Tensor<T> result;
    result.shape_ = new_shape;
    result.strides_ = strides_;
    result.offset_ = new_offset;
    result.length_ = new_length;
    result.device_ = device_;
    if (device_ == Device::CPU) {
      result.data_ = data_;
      result.gpu_data_.reset();
    } else {
      result.data_.reset();
      result.gpu_data_ = gpu_data_;
    }
    return result;
  }

  // 重载加法运算符
  Tensor operator+(const Tensor& other) {
    if (shape_ != other.shape_) {
      throw std::runtime_error("Tensor shape mismatch");
    }
    if (device_ != other.device_) {
      throw std::runtime_error("Tensors must be on same device for +");
    }
    Tensor result(shape_);
    if (device_ == Device::CPU) {
      for (size_t i = 0; i < length_; ++i) {
        result.data_ptr()[i] = data_ptr()[i] + other.data_ptr()[i];
      }
    } else {
      std::vector<T> host_data(length_);
      std::vector<T> host_data_other(length_);
      checkCudaError(cudaMemcpy(host_data.data(), gpu_data_.get(),
                                length_ * sizeof(T), cudaMemcpyDeviceToHost));
      checkCudaError(cudaMemcpy(host_data_other.data(), other.gpu_data_.get(),
                                length_ * sizeof(T), cudaMemcpyDeviceToHost));
      std::vector<T> cpu_result(length_);
      for (size_t i = 0; i < length_; ++i) {
        cpu_result[i] = host_data[i] + host_data_other[i];
      }
      result.cuda();
      checkCudaError(cudaMemcpy(result.gpu_data_.get(), cpu_result.data(),
                                length_ * sizeof(T), cudaMemcpyHostToDevice));
    }
    return result;
  }
  Tensor<T> squeeze(size_t dim) {
    if (dim >= shape_.size()) {
      std::cerr << "Dimension " << dim << " is out of range." << std::endl;
      return *this;
    }
    if (shape_[dim] != 1) {
      std::cout << "Cannot squeeze dimension " << dim << " because its size is "
                << shape_[dim] << " (not 1)." << std::endl;
      return *this;
    }
    shape_.erase(shape_.begin() + dim);
    strides_.erase(strides_.begin() + dim);
    return *this;
  }

  // 转换到 CUDA：将数据从 CPU 拷贝到 GPU，通过内存池申请 GPU 内存
  // is_prefill: 是否是prefill阶段的内存分配
  // tag: 内存标签，用于固定内存分配
  Tensor<T>& cuda(bool is_prefill = false, const std::string& tag = "") {
    if (device_ == Device::CUDA) return *this;

    // 如果提供了新标签，则使用新标签
    if (!tag.empty()) {
      tag_ = tag;
    }

    // 通过内存池申请 GPU 内存
    T* gpu_ptr = nullptr;
    if (!tag_.empty()) {
      // 使用标签分配固定内存
      gpu_ptr = static_cast<T*>(GlobalCudaMemoryPool::allocate_tagged(
          tag_, length_ * sizeof(T), is_prefill));
    } else {
      // 常规内存分配
      gpu_ptr = static_cast<T*>(GlobalCudaMemoryPool::instance().allocate(
          length_ * sizeof(T), is_prefill));
    }

    checkCudaError(cudaMemcpy(gpu_ptr, data_ptr(), length_ * sizeof(T),
                              cudaMemcpyHostToDevice));
    data_.reset();
    gpu_data_ = std::shared_ptr<T>(
        gpu_ptr, [](T* ptr) { GlobalCudaMemoryPool::instance().free(ptr); });
    device_ = Device::CUDA;
    return *this;
  }

  // 转换到 CPU：将数据从 GPU 拷贝到 CPU
  Tensor<T>& cpu() {
    if (device_ == Device::CPU) return *this;
    data_ = std::make_shared<std::vector<T>>(length_);
    checkCudaError(cudaMemcpy(data_->data(), gpu_data_.get(),
                              length_ * sizeof(T), cudaMemcpyDeviceToHost));
    gpu_data_.reset();
    device_ = Device::CPU;
    return *this;
  }

  // 返回当前设备
  Device device() const { return device_; }
  size_t offset() const { return offset_; }

  // 获取内存标签
  const std::string& tag() const { return tag_; }

  // 设置内存标签（仅在CUDA模式下有效）
  void set_tag(const std::string& tag) {
    if (device_ != Device::CUDA) {
      return;  // 仅CUDA模式支持标签
    }

    // 如果已有标签相同，则不做任何操作
    if (tag_ == tag) {
      return;
    }

    // 如果当前没有标签，但要设置标签，需要重新分配内存
    if (tag_.empty() && !tag.empty()) {
      // 保存当前数据
      std::vector<T> host_data(length_);
      checkCudaError(cudaMemcpy(host_data.data(), gpu_data_.get(),
                                length_ * sizeof(T), cudaMemcpyDeviceToHost));

      // 使用新标签分配内存
      T* gpu_ptr = static_cast<T*>(
          GlobalCudaMemoryPool::allocate_tagged(tag, length_ * sizeof(T)));

      // 复制数据
      checkCudaError(cudaMemcpy(gpu_ptr, host_data.data(), length_ * sizeof(T),
                                cudaMemcpyHostToDevice));

      // 更新指针和标签
      gpu_data_ = std::shared_ptr<T>(
          gpu_ptr, [](T* ptr) { GlobalCudaMemoryPool::instance().free(ptr); });
    }

    // 更新标签
    tag_ = tag;
  }

 private:
  // 计算 strides
  static std::vector<size_t> compute_strides(const std::vector<size_t>& shape) {
    std::vector<size_t> strides(shape.size());
    size_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
      strides[i] = stride;
      stride *= shape[i];
    }
    return strides;
  }

  std::shared_ptr<std::vector<T>> data_;  // CPU 数据
  std::shared_ptr<T>
      gpu_data_;  // GPU 数据（使用 shared_ptr 管理，并传入自定义删除器）
  std::vector<size_t> shape_;
  std::vector<size_t> strides_;
  size_t offset_;
  size_t length_;
  Device device_;
  std::string tag_;  // 内存标签，用于固定内存分配
};

// 实现类型转换辅助函数
template <typename FromType, typename ToType>
Tensor<ToType> tensor_convert(const Tensor<FromType>& src) {
  // 构造与当前形状和设备相同的结果张量
  Tensor<ToType> result(src.shape_, src.device_);
  result.offset_ = src.offset_;
  result.length_ = src.length_;
  result.strides_ = src.strides_;
  if (src.device_ == Device::CPU) {
    // CPU 模式：分配新的 vector，并逐元素转换
    auto new_data = std::make_shared<std::vector<ToType>>(src.length_);
    const FromType* src_ptr = src.data_ptr();
    for (size_t i = 0; i < src.length_; ++i) {
      (*new_data)[i] = static_cast<ToType>(src_ptr[i]);
    }
    result.data_ = new_data;
  } else {
    // CUDA 模式：先将数据拷贝到 host，再转换后重新分配 GPU 内存
    std::vector<FromType> host_data(src.length_);
    const FromType* src_ptr = src.data_ptr();
    cudaError_t err =
        cudaMemcpy(host_data.data(), src_ptr, src.length_ * sizeof(FromType),
                   cudaMemcpyDeviceToHost);
    result.checkCudaError(err);
    std::vector<ToType> host_data_float(src.length_);
    for (size_t i = 0; i < src.length_; ++i) {
      host_data_float[i] = static_cast<ToType>(host_data[i]);
    }
    ToType* gpu_ptr = static_cast<ToType*>(
        Tensor<FromType>::pool.allocate(src.length_ * sizeof(ToType)));
    err = cudaMemcpy(gpu_ptr, host_data_float.data(),
                     src.length_ * sizeof(ToType), cudaMemcpyHostToDevice);
    result.checkCudaError(err);
    result.gpu_data_ = std::shared_ptr<ToType>(
        gpu_ptr, [](ToType* ptr) { Tensor<FromType>::pool.free(ptr); });
  }
  return result;
}
