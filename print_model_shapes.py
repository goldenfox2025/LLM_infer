#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
from collections import OrderedDict
import json

def print_tensor_shapes(model_path):
    """
    加载模型并打印所有权重张量的形状
    
    Args:
        model_path: 模型文件夹的路径
    """
    print(f"正在加载模型: {model_path}")
    
    # 检查模型文件夹是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型路径 {model_path} 不存在")
        sys.exit(1)
    
    # 检查是否存在PyTorch权重文件
    pytorch_files = [f for f in os.listdir(model_path) if f.endswith('.pt') or f.endswith('.bin') or f.endswith('.pth')]
    safetensors_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
    
    try:
        # 尝试以不同方式加载模型
        weights = None
        
        # 方法1：尝试直接加载整个模型
        if len(pytorch_files) > 0:
            weight_file = os.path.join(model_path, pytorch_files[0])
            print(f"尝试加载PyTorch权重文件: {weight_file}")
            try:
                weights = torch.load(weight_file, map_location='cpu')
            except Exception as e:
                print(f"直接加载PyTorch文件失败: {e}")
        
        # 方法2：尝试从safetensors加载
        if weights is None and len(safetensors_files) > 0:
            try:
                from safetensors.torch import load_file
                weight_file = os.path.join(model_path, safetensors_files[0])
                print(f"尝试加载Safetensors权重文件: {weight_file}")
                weights = load_file(weight_file)
            except Exception as e:
                print(f"加载Safetensors文件失败: {e}")
        
        # 方法3：尝试使用transformers库加载
        if weights is None:
            try:
                from transformers import AutoModel
                print("尝试使用transformers库加载模型")
                model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16)
                weights = model.state_dict()
            except Exception as e:
                print(f"使用transformers加载失败: {e}")
        
        # 如果所有方法都失败，退出
        if weights is None:
            print("所有加载方法均失败，请检查模型文件")
            sys.exit(1)
        
        # 打印权重信息
        print("\n===================== 模型权重形状信息 =====================")
        
        # 创建有序字典以按键名排序
        sorted_weights = OrderedDict()
        
        # 检查是否是state_dict格式或其他格式
        if isinstance(weights, dict):
            weight_dict = weights
            if 'state_dict' in weights:  # 某些模型将权重封装在'state_dict'键下
                weight_dict = weights['state_dict']
            
            for key, tensor in weight_dict.items():
                sorted_weights[key] = tensor
        else:
            print(f"警告: 模型不是标准dict格式，而是 {type(weights)}")
            sys.exit(1)
        
        # 按字母顺序打印权重形状
        for i, (key, tensor) in enumerate(sorted(sorted_weights.items())):
            print(f"{i+1:4d}. {key:80s} | 形状: {tuple(tensor.shape)}")
            
            # 特别检查lm_head和嵌入表的形状
            if 'lm_head' in key or 'embed' in key:
                print(f"      -> 注意: 该权重形状为 [{tensor.shape[0]}, {tensor.shape[1]}]" +
                      f" ({'NK格式' if tensor.shape[0] > tensor.shape[1] else 'KN格式'})")
        
        print("\n===================== 统计信息 =====================")
        total_params = sum(p.numel() for p in sorted_weights.values())
        print(f"权重数量: {len(sorted_weights)}")
        print(f"参数总数: {total_params:,}")
        print(f"参数总数 (十亿): {total_params / 1e9:.2f}B")
        
        # 保存结果到JSON文件
        shapes_info = {k: list(v.shape) for k, v in sorted_weights.items()}
        with open(f"{os.path.basename(model_path)}_shapes.json", 'w') as f:
            json.dump(shapes_info, f, indent=2)
        
        print(f"\n形状信息已保存至: {os.path.basename(model_path)}_shapes.json")
        
    except Exception as e:
        print(f"程序出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    model_path = "/home/LLM_infer/models/Qwen3-1.7B"
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    print_tensor_shapes(model_path) 