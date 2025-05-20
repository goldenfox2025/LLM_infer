# Logits分布可视化工具

本工具用于可视化比较目标模型和草稿模型在投机解码过程中的logits分布。

## 修改内容

我们对以下文件进行了修改和创建：

1. **backend/cpp/include/common.hpp**：
   - 添加了`saveTensorToFile`函数及其特化版本，用于将tensor保存到文件
   - 特化版本支持`float`、`__nv_bfloat16`和`__half`类型

2. **backend/cpp/src/speculative_decoder.cpp**：
   - 在`generate_draft_tokens_with_probs_gpu`函数中添加代码保存草稿模型的logits
   - 在`verify_draft_tokens_greedy`和`verify_draft_tokens_prob_ratio`函数中添加代码保存目标模型的logits

3. **test_speculative_decoding.py**：
   - 添加创建logits数据存储目录的代码
   - 在脚本结束时添加提示，告知用户可以运行可视化脚本

4. **visualize_logits.py**（新增）：
   - 用于读取保存的logits数据，并生成可视化图表
   - 提供了多种分析功能：Top-K token概率比较和KL散度计算

## 文件格式

保存的tensor文件使用以下简单的二进制格式：

1. 形状信息：
   - 维度数量（1个uint64）
   - 每个维度的大小（N个uint64，N为维度数量）

2. 数据部分：
   - 所有元素的浮点值（float32）
   - 对于半精度类型（__nv_bfloat16或__half），会被转换为float32存储

## 使用说明

### 1. 编译修改后的C++代码

```bash
cd backend/cpp
mkdir -p build
cd build
cmake ..
make -j
```

### 2. 运行投机解码测试脚本

```bash
python test_speculative_decoding.py
```

这将运行标准解码和投机解码过程，并收集logits数据。数据将保存在以下目录：

- `./logits_data/target/`：目标模型的logits
- `./logits_data/draft/`：草稿模型的logits

### 3. 运行可视化脚本

```bash
python visualize_logits.py
```

这将读取保存的logits数据，并生成以下可视化图表：

- 每个token位置的Top-K token概率比较
- 各个token位置的KL散度图

可视化结果将保存在`./logits_data/visualizations/`目录中。

## 可视化结果解读

### Top-K Token概率比较

每个可视化包含4个子图：

1. **左上**：目标模型的Top-K个token及其概率
2. **右上**：草稿模型对目标模型Top-K个token的概率
3. **左下**：草稿模型的Top-K个token及其概率
4. **右下**：目标模型对草稿模型Top-K个token的概率

通过比较这些分布，我们可以看出：
- 目标模型和草稿模型在预测top token时的一致性
- 草稿模型的预测与目标模型的偏差程度

### KL散度图

KL散度（Kullback-Leibler散度）衡量两个概率分布的差异。图表显示了各个token位置的KL(目标||草稿)散度值。

- 较低的KL散度意味着草稿模型的预测更接近目标模型
- 较高的KL散度表示草稿模型的预测与目标模型相差较大

在投机解码中，较低的KL散度通常意味着较高的投机成功率。

## 注意事项

1. 确保有足够的磁盘空间存储logits数据，对于大词表模型（如Qwen3），每个token位置的logits可能占用数十MB。

2. 可视化脚本需要安装以下Python依赖：
   - numpy
   - matplotlib
   - transformers

3. 可视化过程可能需要较长时间，特别是在token位置较多时。

4. 默认显示top-10的token，可以通过修改`visualize_logits.py`中的参数调整。 