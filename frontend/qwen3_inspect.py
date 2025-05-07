#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen3 模型检查工具

该脚本使用 transformers 库加载 Qwen3-1.7B 模型，
打印每一层的张量形状，并执行推理。
"""

import os
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, List, Tuple, Optional, Union

def print_header(title: str) -> None:
    """打印带有分隔线的标题"""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)

def print_section(title: str) -> None:
    """打印带有分隔线的小节标题"""
    print("\n" + "-" * 80)
    print(f" {title} ".center(80, "-"))
    print("-" * 80)

def print_tensor_info(name: str, tensor: torch.Tensor) -> None:
    """打印张量的详细信息"""
    print(f"{name}:")
    print(f"  - Shape: {tensor.shape}")
    print(f"  - Dtype: {tensor.dtype}")
    print(f"  - Device: {tensor.device}")
    print(f"  - Min/Max: {tensor.min().item():.6f} / {tensor.max().item():.6f}")
    print(f"  - Mean/Std: {tensor.mean().item():.6f} / {tensor.std().item():.6f}")

def inspect_model_structure(model: AutoModelForCausalLM) -> None:
    """检查并打印模型结构和每层的张量形状"""
    print_header("模型结构概览")

    # 打印模型的基本信息
    print(f"模型类型: {model.__class__.__name__}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"模型设备: {next(model.parameters()).device}")

    # 打印模型配置
    config = model.config
    print_section("模型配置")
    print(f"隐藏层大小 (hidden_size): {config.hidden_size}")
    print(f"层数 (num_hidden_layers): {config.num_hidden_layers}")
    print(f"注意力头数 (num_attention_heads): {config.num_attention_heads}")
    print(f"KV头数 (num_key_value_heads): {config.num_key_value_heads}")
    print(f"中间层大小 (intermediate_size): {config.intermediate_size}")
    print(f"头维度 (head_dim): {config.head_dim}")
    print(f"最大位置嵌入 (max_position_embeddings): {config.max_position_embeddings}")
    print(f"词表大小 (vocab_size): {config.vocab_size}")
    print(f"RMS归一化epsilon (rms_norm_eps): {config.rms_norm_eps}")
    print(f"RoPE theta: {config.rope_theta}")

    # 检查嵌入层
    print_section("嵌入层")
    embed_tokens = model.model.embed_tokens
    print_tensor_info("embed_tokens.weight", embed_tokens.weight)

    # 只详细检查第一个Transformer层
    print_section(f"Transformer层详细信息 (共 {len(model.model.layers)} 层)")
    layer = model.model.layers[0]

    # 检查注意力模块
    print("\n注意力模块:")
    print(f"  - q_proj.weight: {layer.self_attn.q_proj.weight.shape}")
    print(f"  - k_proj.weight: {layer.self_attn.k_proj.weight.shape}")
    print(f"  - v_proj.weight: {layer.self_attn.v_proj.weight.shape}")
    print(f"  - o_proj.weight: {layer.self_attn.o_proj.weight.shape}")

    # 检查MLP模块
    print("\nMLP模块:")
    print(f"  - gate_proj.weight: {layer.mlp.gate_proj.weight.shape}")
    print(f"  - up_proj.weight: {layer.mlp.up_proj.weight.shape}")
    print(f"  - down_proj.weight: {layer.mlp.down_proj.weight.shape}")

    # 检查归一化层并打印权重值
    print("\n归一化层:")
    print_tensor_info("input_layernorm.weight", layer.input_layernorm.weight)
    print_tensor_info("post_attention_layernorm.weight", layer.post_attention_layernorm.weight)

    # 检查最终层归一化和LM头
    print_section("最终层归一化和LM头")
    print_tensor_info("norm.weight", model.model.norm.weight)
    print(f"lm_head.weight: {model.lm_head.weight.shape}")

    # 检查是否共享权重
    print_section("权重共享检查")
    is_shared = torch.equal(model.model.embed_tokens.weight, model.lm_head.weight)
    print(f"嵌入层和LM头是否共享权重: {is_shared}")

    # 打印所有层的名称和形状
    print_section("所有层的名称和形状")
    for name, param in model.named_parameters():
        if 'layers' not in name or 'layers.0' in name:  # 只显示第一层的详细信息
            print(f"{name}: {param.shape}")

def run_inference(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str) -> str:
    """运行推理并返回生成的文本"""
    print_header("运行推理")

    # 准备输入
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    # 应用聊天模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    print(f"输入文本: {text}")

    # 编码输入
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    print(f"输入形状: {inputs.input_ids.shape}")

    # 记录开始时间
    start_time = time.time()

    # 生成文本
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,  # 减少生成的token数量，加快执行速度
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    # 计算生成时间
    generation_time = time.time() - start_time

    # 解码输出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 打印生成信息
    print(f"\n生成时间: {generation_time:.2f}秒")
    print(f"生成的token数量: {outputs.shape[1] - inputs.input_ids.shape[1]}")
    print(f"每秒生成token数: {(outputs.shape[1] - inputs.input_ids.shape[1]) / generation_time:.2f}")

    return generated_text

def inspect_attention_patterns(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str) -> None:
    """检查注意力模式"""
    print_header("注意力模式检查")

    # 准备输入
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 启用输出注意力权重
    try:
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_attentions=True,
                return_dict=True
            )

        # 检查注意力权重
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            attentions = outputs.attentions  # 这是一个元组，每个元素对应一层的注意力权重

            print(f"注意力权重数量: {len(attentions)}")
            for i, attn in enumerate(attentions):
                print(f"层 {i+1} 注意力权重形状: {attn.shape}")
                # 形状通常是 [batch_size, num_heads, seq_length, seq_length]
        else:
            print("模型没有输出注意力权重。这可能是因为模型架构不支持或未启用此功能。")
    except Exception as e:
        print(f"检查注意力模式时出错: {e}")

def main():
    try:
        # 模型路径
        model_path = "/home/LLM_infer/models/Qwen3-1.7B"

        # 加载模型和分词器
        print_header("加载模型和分词器")
        print(f"模型路径: {model_path}")

        # 加载分词器
        print("加载分词器...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception as e:
            print(f"从本地加载分词器失败: {e}")
            print("尝试从Hugging Face直接加载分词器...")
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)

        # 加载模型
        print("加载模型...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True
            )
        except ValueError as e:
            print(f"使用标准方法加载失败: {e}")
            print("尝试使用 trust_remote_code=True 和 revision='main' 参数...")

            # 尝试从Hugging Face直接加载
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen3-1.7B",
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
                revision="main"
            )

        # 检查模型结构
        inspect_model_structure(model)

        # 运行简单推理
        prompt = "请简单介绍一下量子计算"
        generated_text = run_inference(model, tokenizer, prompt)

        # 打印生成的文本
        print_section("生成的文本")
        print(generated_text)

        # 检查注意力模式
        inspect_attention_patterns(model, tokenizer, "人工智能是什么？")

    except Exception as e:
        import traceback
        print_header("执行过程中出现错误")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {e}")
        print("\n详细错误信息:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
