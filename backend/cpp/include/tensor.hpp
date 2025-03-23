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
enum class Device { CPU, CUDA };
template <typename T>
class Tensor {
 private:
  // 静态内联函数：检查 CUDA 错误（可在 const 成员中调用）
  static inline void checkCudaError(cudaError_t error) {
    if (error != cudaSuccess) {
      throw std::runtime_error("CUDA error: " +
                               std::string(cudaGetErrorString(error)));
    }
  }

  // 自定义删除器：将 cudaFree 包装为返回 void 的形式
  static inline void myCudaFree(T* p) { cudaFree(p); }

 public:
  // 默认构造函数
  Tensor()
      : data_(std::make_shared<std::vector<T>>()),
        offset_(0),
        length_(0),
        device_(Device::CPU),
        gpu_data_(nullptr, myCudaFree) {}

  // 从已有数据和形状构造（要求数据至少包含所有元素）
  Tensor(std::shared_ptr<std::vector<T>> data, const std::vector<size_t>& shape)
      : data_(data),
        shape_(shape),
        offset_(0),
        length_(1),
        device_(Device::CPU),
        gpu_data_(nullptr, myCudaFree) {
    for (size_t dim : shape_) {
      length_ *= dim;
    }
    if (length_ > data_->size()) {
      throw std::runtime_error("Data size does not match tensor shape");
    }
    strides_ = compute_strides(shape_);
  }

  // 从 initializer_list 构造，分配新的数据，并指定设备
  Tensor(std::initializer_list<size_t> shape, Device device = Device::CPU)
      : shape_(shape),
        offset_(0),
        length_(1),
        device_(device),
        gpu_data_(nullptr, myCudaFree) {
    for (size_t dim : shape_) {
      length_ *= dim;
    }
    strides_ = compute_strides(shape_);
    if (device_ == Device::CPU) {
      data_ = std::make_shared<std::vector<T>>(length_);
      gpu_data_.reset();
    } else if (device_ == Device::CUDA) {
      data_.reset();
      T* gpu_ptr;
      cudaError_t error = cudaMalloc(&gpu_ptr, length_ * sizeof(T));
      checkCudaError(error);
      gpu_data_ = std::shared_ptr<T>(gpu_ptr, myCudaFree);
    } else {
      throw std::runtime_error("Invalid device specified");
    }
  }

  // 从右值 vector 数据和形状构造（CPU 缺省）
  Tensor(std::vector<T>&& data, const std::vector<size_t>& shape)
      : data_(std::make_shared<std::vector<T>>(std::move(data))),
        shape_(shape),
        offset_(0),
        length_(1),
        device_(Device::CPU),
        gpu_data_(nullptr, myCudaFree) {
    for (size_t dim : shape_) {
      length_ *= dim;
    }
    if (length_ > data_->size()) {
      throw std::runtime_error("Data size does not match tensor shape");
    }
    strides_ = compute_strides(shape_);
  }

  // 新增：从右值 vector 数据、形状和 device 构造
  Tensor(std::vector<T>&& data, const std::vector<size_t>& shape, Device device)
      : shape_(shape),
        offset_(0),
        length_(1),
        device_(device),
        gpu_data_(nullptr, myCudaFree) {
    for (size_t dim : shape_) {
      length_ *= dim;
    }
    strides_ = compute_strides(shape_);
    if (device_ == Device::CPU) {
      // CPU 模式，直接将 data 存入 std::vector
      data_ = std::make_shared<std::vector<T>>(std::move(data));
    } else if (device_ == Device::CUDA) {
      // GPU 模式，拷贝 data 到 GPU
      data_.reset();  // 不在 CPU 中保存数据
      T* gpu_ptr;
      checkCudaError(cudaMalloc(&gpu_ptr, length_ * sizeof(T)));
      checkCudaError(cudaMemcpy(gpu_ptr, data.data(), length_ * sizeof(T),
                                cudaMemcpyHostToDevice));
      gpu_data_ = std::shared_ptr<T>(gpu_ptr, myCudaFree);
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
        gpu_data_(nullptr, myCudaFree) {
    for (size_t dim : shape_) {
      length_ *= dim;
    }
    data_ = std::make_shared<std::vector<T>>(length_);
    strides_ = compute_strides(shape_);
  }

  // 拷贝构造函数（CUDA 下采用浅拷贝，即共享 gpu_data_）
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
    // 构造与当前形状和设备相同的结果张量
    Tensor<float> result(this->shape_, this->device_);
    result.offset_ = this->offset_;
    result.length_ = this->length_;
    result.strides_ = this->strides_;
    if (device_ == Device::CPU) {
      // CPU 模式：data_ 保存有效数据
      // 分配一个新的 vector，并逐元素转换为 float
      auto new_data = std::make_shared<std::vector<float>>(length_);
      const T* src = this->data_ptr();
      for (size_t i = 0; i < length_; ++i) {
        (*new_data)[i] = static_cast<float>(src[i]);
      }
      result.data_ = new_data;
    } else {
      // CUDA 模式：gpu_data_ 保存数据
      std::vector<T> host_data(length_);
      cudaError_t err = cudaMemcpy(host_data.data(), gpu_data_.get(),
                                   length_ * sizeof(T), cudaMemcpyDeviceToHost);
      // 使用类内的 checkCudaError 检查错误
      checkCudaError(err);
      std::vector<float> host_data_float(length_);
      for (size_t i = 0; i < length_; ++i) {
        host_data_float[i] = static_cast<float>(host_data[i]);
      }
      // 分配新的 GPU 内存用于存放 float 数据
      float* gpu_ptr;
      err = cudaMalloc(&gpu_ptr, length_ * sizeof(float));
      checkCudaError(err);
      err = cudaMemcpy(gpu_ptr, host_data_float.data(), length_ * sizeof(float),
                       cudaMemcpyHostToDevice);
      checkCudaError(err);
      result.gpu_data_ = std::shared_ptr<float>(gpu_ptr, myCudaFree);
    }
    return result;
  }

  // 返回数据指针
  const T* data_ptr() const {
    if (device_ == Device::CPU) {
      return data_->data() + offset_;
    } else {
      return gpu_data_.get() + offset_;
    }
  }
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

  // 填充
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

  // view：返回一个共享底层数据的新张量（不拷贝数据，仅修改元信息）
  Tensor<T> view(const std::vector<size_t>& new_shape) const {
    // 计算新形状的元素总数
    size_t new_numel = 1;
    for (size_t dim : new_shape) {
      new_numel *= dim;
    }
    if (new_numel != length_) {
      std::cerr << "[Tensor::view] 错误: 新形状的元素数量 (" << new_numel
                << ") 与原形状元素数量 (" << length_ << ") 不匹配" << std::endl;
      throw std::runtime_error("view: 新形状必须具有相同数量的元素");
    }
    // 直接复制本张量（共享底层数据），仅更新形状与 strides
    Tensor<T> result = *this;
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
    result.strides_ = strides_;  // 保持原始 strides
    result.offset_ = new_offset;
    result.length_ = new_length;
    result.device_ = device_;
    if (device_ == Device::CPU) {
      result.data_ = data_;  // 共享 CPU 数据
      result.gpu_data_.reset();
    } else {
      result.data_.reset();
      result.gpu_data_ = gpu_data_;  // 共享 GPU 数据（引用计数增加）
    }
    return result;
  }

  // 重载加法
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

  // 转换到 CUDA
  Tensor<T>& cuda() {
    if (device_ == Device::CUDA) return *this;
    T* gpu_ptr;
    checkCudaError(cudaMalloc(&gpu_ptr, length_ * sizeof(T)));
    checkCudaError(cudaMemcpy(gpu_ptr, data_ptr(), length_ * sizeof(T),
                              cudaMemcpyHostToDevice));
    data_.reset();
    gpu_data_ = std::shared_ptr<T>(gpu_ptr, myCudaFree);
    device_ = Device::CUDA;
    return *this;
  }

  // 转换到 CPU
  Tensor<T>& cpu() {
    if (device_ == Device::CPU) return *this;
    data_ = std::make_shared<std::vector<T>>(length_);
    checkCudaError(cudaMemcpy(data_->data(), gpu_data_.get(),
                              length_ * sizeof(T), cudaMemcpyDeviceToHost));
    gpu_data_.reset();
    device_ = Device::CPU;
    return *this;
  }

  // 返回设备
  Device device() const { return device_; }

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
};
