#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
from pathlib import Path
from safetensors import safe_open
import torch

def analyze_mapping(model_path):
    model_path = Path(model_path)
    print(f"分析模型权重映射: {model_path}")
    
    # 检查模型配置
    config_path = model_path / "config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            num_layers = config.get("num_hidden_layers", 0)
            print(f"模型层数: {num_layers}")
    
    # 检查权重文件
    safetensors_path = model_path / "model.safetensors"
    if not os.path.exists(safetensors_path):
        print(f"错误: 找不到权重文件 {safetensors_path}")
        return
    
    # 定义映射表
    weight_mapping = {
        "self_attn.q_proj": "wq",
        "self_attn.k_proj": "wk",
        "self_attn.v_proj": "wv",
        "self_attn.o_proj": "wo",
        "mlp.gate_proj": "w_gate",
        "mlp.up_proj": "w_up",
        "mlp.down_proj": "w_down"
    }
    
    # 读取权重
    with safe_open(safetensors_path, framework="pt") as f:
        all_keys = list(f.keys())
        
        # 筛选AWQ量化权重
        qweight_keys = [k for k in all_keys if ".qweight" in k]
        scales_keys = [k for k in all_keys if ".scales" in k]
        qzeros_keys = [k for k in all_keys if ".qzeros" in k]
        
        # 检查所有层的所有权重类型
        for layer in range(num_layers):
            print(f"\n层 {layer} 的权重:")
            for weight_type, dst_prefix in weight_mapping.items():
                # 原始名称模式
                orig_pattern = f"model.layers.{layer}.{weight_type}"
                
                # 目标名称
                target_name = f"{dst_prefix}{layer}"
                
                # 查找匹配的权重
                qweight_found = any(k.startswith(orig_pattern) and k.endswith(".qweight") for k in qweight_keys)
                scales_found = any(k.startswith(orig_pattern) and k.endswith(".scales") for k in scales_keys)
                qzeros_found = any(k.startswith(orig_pattern) and k.endswith(".qzeros") for k in qzeros_keys)
                
                status = "✓" if (qweight_found and scales_found and qzeros_found) else "✗"
                
                print(f"  {orig_pattern} -> {target_name}: {status}")
                print(f"    qweight: {'找到' if qweight_found else '未找到'}")
                print(f"    scales: {'找到' if scales_found else '未找到'}")
                print(f"    qzeros: {'找到' if qzeros_found else '未找到'}")
                
                # 如果找到，展示一些样本数据
                if qweight_found:
                    qweight_key = next(k for k in qweight_keys if k.startswith(orig_pattern) and k.endswith(".qweight"))
                    tensor = f.get_tensor(qweight_key)
                    print(f"    形状: {tensor.shape}, 类型: {tensor.dtype}")

# 实现权重转换过程的模拟，跟踪键名变化
def simulate_processing(model_path):
    model_path = Path(model_path)
    print(f"\n模拟权重处理过程: {model_path}")
    
    # 读取权重
    weight_keys = []
    with safe_open(model_path / "model.safetensors", framework="pt") as f:
        weight_keys = list(f.keys())
    
    # 模拟权重映射
    transformed_weights = {}
    
    for key in weight_keys:
        if "model.layers." in key:
            # 提取层索引
            layer_parts = key.split(".")
            layer_idx = -1
            for i, part in enumerate(layer_parts):
                if part == "layers" and i+1 < len(layer_parts):
                    layer_idx = int(layer_parts[i+1])
                    break
            
            if layer_idx >= 0:
                # 处理注意力权重
                if "self_attn" in key:
                    if "q_proj" in key:
                        if ".qweight" in key:
                            transformed_weights[f"wq{layer_idx}"] = key
                        elif ".scales" in key:
                            transformed_weights[f"wq{layer_idx}.scales"] = key
                        elif ".qzeros" in key:
                            transformed_weights[f"wq{layer_idx}.qzeros"] = key
                    elif "k_proj" in key:
                        if ".qweight" in key:
                            transformed_weights[f"wk{layer_idx}"] = key
                        elif ".scales" in key:
                            transformed_weights[f"wk{layer_idx}.scales"] = key
                        elif ".qzeros" in key:
                            transformed_weights[f"wk{layer_idx}.qzeros"] = key
                    elif "v_proj" in key:
                        if ".qweight" in key:
                            transformed_weights[f"wv{layer_idx}"] = key
                        elif ".scales" in key:
                            transformed_weights[f"wv{layer_idx}.scales"] = key
                        elif ".qzeros" in key:
                            transformed_weights[f"wv{layer_idx}.qzeros"] = key
                    elif "o_proj" in key:
                        if ".qweight" in key:
                            transformed_weights[f"wo{layer_idx}"] = key
                        elif ".scales" in key:
                            transformed_weights[f"wo{layer_idx}.scales"] = key
                        elif ".qzeros" in key:
                            transformed_weights[f"wo{layer_idx}.qzeros"] = key
    
    # 输出所有o_proj映射结果
    print("\no_proj权重映射:")
    for i in range(28):  # 假设有28层
        target_key = f"wo{i}"
        if target_key in transformed_weights:
            print(f"  {target_key} <- {transformed_weights[target_key]}")
        else:
            print(f"  {target_key}: 未找到映射")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python awq_debug.py <模型路径>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    analyze_mapping(model_path)
    simulate_processing(model_path) 