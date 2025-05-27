# QwenModel CUDA图推理切换

## 简介

现在QwenModel支持在CUDA图推理和常规CUDA推理之间切换，通过修改一个简单的成员变量即可。

## 使用方法

### 切换推理模式

在 `backend/cpp/include/qwen.hpp` 文件中，找到第250行：

```cpp
// 推理模式控制 - 默认使用常规CUDA推理，手动修改此值来测试CUDA图
bool use_cuda_graph_ = false;  // 改为true来启用CUDA图推理
```

**切换到CUDA图推理：**
```cpp
bool use_cuda_graph_ = true;   // 启用CUDA图推理
```

**切换回常规CUDA推理：**
```cpp
bool use_cuda_graph_ = false;  // 使用常规CUDA推理
```

### 重新编译

修改后需要重新编译：
```bash
./build.sh
```

### 运行测试

```bash
# 运行聊天程序
python frontend/chat.py --model_type qwen_bf16

# 或者运行推理测试
./run.sh
```

## 状态确认

模型初始化时会显示当前的CUDA图状态：

```
QwenModel Info:
  Vocab size: 151936
  Layers: 28
  Heads: 12
  KV Heads: 2
  Hidden size: 1536
  ...
  CUDA Graph: Enabled    # 或 Disabled
```

## 两种推理模式对比

### 常规CUDA推理 (`use_cuda_graph_ = false`)
- **优点**：稳定可靠，兼容性好
- **缺点**：每次推理都有CUDA kernel启动开销
- **适用**：调试、开发、稳定性要求高的场景

### CUDA图推理 (`use_cuda_graph_ = true`)
- **优点**：减少kernel启动开销，理论上性能更好
- **缺点**：初始化时间长，可能有兼容性问题
- **适用**：性能测试、生产环境优化

## 注意事项

1. **初始化时间**：CUDA图推理首次运行时需要较长时间来捕获计算图
2. **内存使用**：CUDA图推理可能使用更多GPU内存
3. **调试建议**：如果遇到问题，先切换到常规CUDA推理进行调试
4. **仅支持QwenModel**：目前只有QwenModel支持此功能

## 快速测试

1. 修改 `qwen.hpp` 中的 `use_cuda_graph_` 值
2. 运行 `./build.sh` 重新编译
3. 运行 `python frontend/chat.py --model_type qwen_bf16`
4. 观察模型信息中的 "CUDA Graph" 状态
5. 测试推理性能和正确性

这样你就可以方便地在两种推理模式之间切换进行测试了！
