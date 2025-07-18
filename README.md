# LLM_infer Engine: 高性能推理引擎探索与实践

**日期: 2025年7月18日**

## 项目简介

`LLM_infer` 是一个旨在探索和实现大语言模型（LLM）本地化、高性能端侧推理的实验性项目。本项目的核心目标是通过自研推理引擎、应用量化技术以及实践前沿的解码算法，深入理解并优化LLM的推理性能。

在本项目中，我们依次完成了以下三个核心阶段的探索与评估。所有性能数据均为**五次测试中的最优结果**，以确保展示的是引擎在理想状态下的峰值性能。

---

## 1. 核心引擎性能对决：`LLM_infer` vs. `llama.cpp`

为了评估 `LLM_infer` 引擎的基础性能，我们将其与业界知名的 `llama.cpp` 在同等条件下进行了性能对比。

**测试条件:**
* **模型:** Qwen2 1.5B-Instruct-BF16
* **硬件:** RTX4070Laptop
* **参数:** 启用Flash Attention 输出长度限制为201（包括prefill输出的1个token在内）`enabled`, `top-k = 20`, `top-p = disabled`

| 性能指标 (Performance Metric) | `llama.cpp` | `LLM_infer` |
| :--- | :--- | :--- |
| **Prompt处理 (Prefill)** | 23 tokens / 21.60 ms | **23 tokens / 17.94 ms** |
| **Prompt处理速度 (Prefill Speed)** | 1065.01 tokens/s | **1282.05 tokens/s (1.20x)** |
| **内容生成 (Decoding)** | 200 tokens / 2883.04 ms | **200 tokens / 2640.19 ms** |
| **内容生成速度 (Decoding Speed)**| 69.37 tokens/s | **75.75 tokens/s (1.09x)** |

**结论：**
测试结果表明，`LLM_infer` 引擎在核心推理任务上表现出更高的性能。在处理输入提示的Prefill阶段，其速度是`llama.cpp`的 **1.20倍**；在持续生成内容的Decoding阶段，速度实现了 **1.09倍** 的提升。

---

## 2. 量化技术探索：BF16 vs. AWQ 性能分析

在基础引擎之上，我们进一步探索了AWQ量化技术对性能的增益。

| 性能指标 (Performance Metric) | Qwen2 1.5B (BF16) | Qwen2 1.5B (AWQ) |
| :--- | :--- | :--- |
| **Prompt处理 (Prefill)** | 23 tokens / 24.26 ms | **23 tokens / 50.73 ms** |
| **Prompt处理速度 (Prefill Speed)** | **956.31 tokens/s** | 453.38 tokens/s (0.47x) |
| **内容生成 (Decoding)** | 200 tokens / 3104.76 ms | **200 tokens / 1980.37 ms** |
| **内容生成速度 (Decoding Speed)**| 64.42 tokens/s | **100.99 tokens/s (1.57x)** |

**结论分析：**
* **解码性能显著提升：** AWQ量化将核心的解码速度提升至 **1.57倍**。这得益于低位宽整数运算的高效性以及模型体积减小带来的显存带宽优化。
* **Prefill性能说明：** 本次测试中，AWQ版本的Prefill阶段性能有所下降。其主要原因是该阶段所依赖的GEMM实现为**朴素的WMMA版本**，优化尚不充分。我们相信在后续针对性地优化GEMM核后，此处的性能将得到大幅改善。

即使Prefill存在优化空间，AWQ在解码环节带来的巨大增益依然证明了其在提升LLM推理性能中的核心价值。

---

### 3. 前沿技术实践：投机解码性能评估

#### 3.1 实验概述

我们评估了 `Qwen3 0.6B AWQ` 草稿模型对 `Qwen3 1.7B AWQ` 目标模型的加速效果。本节将基于实测数据，量化分析其性能表现。

#### 3.2 核心性能数据分析

要理解投机解码的效益，关键在于比较其单次迭代成本与收益。从日志中，我们提取了三个核心耗时数据：

1.  **目标模型单Token解码耗时 (`T_baseline`)**: 约 **9ms**。
    * 这是标准自回归解码（无投机）生成一个 token 的时间，是我们的性能基线。

2.  **草稿生成耗时 (`T_draft`)**: 约 **30ms**。
    * 这是草稿模型一次性生成 6-8 个候选 token 的总时间。

3.  **目标模型验证耗时 (`T_verify`)**: 约 **55ms**。
    * 这是目标模型一次并行处理全部草稿并且完成验证的开销。

投机解码一次迭代的总成本是 `T_draft + T_verify ≈ 30ms + 55ms = 85ms`。

要实现加速，这次迭代的收益（`N_accepted` 个 token）必须高于基线成本：

$$T_{draft} + T_{verify} < N_{accepted} \times T_{baseline}$$
$$85ms < N_{accepted} \times 9ms$$

解得：
$$N_{accepted} > 9.4$$

**结论：** 这意味着，平均每次迭代至少需要成功**接受 9 个**以上的草稿 token，投机解码才能实现正向收益。

然而完全无法达到这样的接受率。