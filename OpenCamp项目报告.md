OpenCamp课程项目报告
一、项目概述
本项目基于Python与C++重构OpenCamp课程原RUST版本，采用Prefill/Decode阶段分离策略，支持多轮对话。系统通过模块化设计实现前后端解耦，提供Web交互界面及高性能推理能力。
本项目开发及运行环境为WSL2。
二、技术架构设计
1. 系统分层
Interface层：前后端通信桥梁。

Backend层：推理引擎以及KV缓存管理。

Frontend层：Python Web交互界面，使用flask实现接口api，SSE实现流式显示文本。可通过交互实时调整温度、TOPK等参数。

2. 核心策略
阶段解耦：Prefill与Decode独立实现(未来可分别优化)。

混合加速：AVX指令集加速与常规CPU算子并存。

异步处理：多线程+队列实现生成与传输并行化。

三、后端实现(CPP)
1. 核心模块
模块文件	功能描述
avx_operators.hpp	AVX指令集加速矩阵运算(内存连续场景)
operators.hpp	通用CPU算子(支持非连续内存)
inference.hpp	推理引擎实现/KV缓存管理
llama.hpp	LLAMA模型结构定义/权重加载
tensor.hpp	张量数据结构实现

inference.cpp  推理引擎等具体实现
llama_decode.cpp  decode阶段实现
llama_prefill.cpp  prefill阶段实现


四、加速效果

已经实现的效果：
√可交互的 UI 或网络服务 API（web_ui）；
√适配一种加速软件栈后端，Nvidia、AMD、国产芯片或 OpenCL、九源后端均可（目前适配英特尔芯片加速指令）；

![第一轮对话(AVX)](e2.png) 
这是AVX加速矩阵乘法的效果。

![第一轮对话(普通CPP)](e1.png) 
这是AVX加速矩阵乘法的效果。

随着KV缓存增加，二者速度差值变小。在输出第一个token的时候，AVX的速度超过了10 token/s，而未加速版本是8 token/s的级别。avx_operators.hpp为具体实现。

![多轮对话](e3.png) 

本项目支持多轮对话。


还未实现的效果：
×多会话管理以及历史会话回滚；
×多线程分布式推理优化，附加性能对比；


下一步计划：
[]多线程分布式推理
[]完成CUDA后端适配
[]多会话管理
