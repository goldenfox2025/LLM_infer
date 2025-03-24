#pragma once
#include "tensor.hpp"

// Forward declaration
template <typename T>
class Tensor;

// 非模板基类KVCache接口，用于向上转型
class KVCacheBase {
 public:
  virtual ~KVCacheBase() = default;
  
  // 添加基类的纯虚函数接口
  virtual void resize(size_t new_size) = 0;
  virtual void clear() = 0;
  virtual size_t size() const = 0;
  virtual Device device() const = 0;
  

  // 添加获取n_layers和head_dim的方法
  virtual size_t get_n_layers() const = 0;
  virtual size_t get_head_dim() const = 0;
  virtual size_t get_max_seq_len() const = 0;


};
