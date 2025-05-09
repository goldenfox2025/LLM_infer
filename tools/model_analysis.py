#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
from pathlib import Path
from safetensors import safe_open

def analyze_model(model_path):
    model_path = Path(model_path)
    print(f"分析模型: {model_path}")
    
    # 检查模型配置
    config_path = model_path / "config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            print("模型配置:")
            for key, value in config.items():
                if isinstance(value, dict):
                    print(f"  {key}: {{{len(value)} items}}")
                elif isinstance(value, list):
                    print(f"  {key}: [{len(value)} items]")
                else:
                    print(f"  {key}: {value}")
    
    # 检查权重文件
    safetensors_path = model_path / "model.safetensors"
    weight_keys = []
    
    if os.path.exists(safetensors_path):
        print(f"\n权重文件: {safetensors_path}")
        
        # 使用safetensors打开文件
        with safe_open(safetensors_path, framework="pt") as f:
            # 获取所有键名
            all_keys = f.keys()
            weight_keys = list(all_keys)
            
            # 统计键的数量和分类
            print(f"权重总数: {len(weight_keys)}")
            
            # 分析权重类型
            attention_keys = [k for k in weight_keys if "self_attn" in k]
            mlp_keys = [k for k in weight_keys if "mlp" in k]
            embedding_keys = [k for k in weight_keys if "embed" in k]
            layernorm_keys = [k for k in weight_keys if "layernorm" in k]
            
            print(f"注意力权重数量: {len(attention_keys)}")
            print(f"MLP权重数量: {len(mlp_keys)}")
            print(f"嵌入权重数量: {len(embedding_keys)}")
            print(f"层归一化权重数量: {len(layernorm_keys)}")
            
            # 检查量化权重
            qweight_keys = [k for k in weight_keys if ".qweight" in k]
            scales_keys = [k for k in weight_keys if ".scales" in k]
            qzeros_keys = [k for k in weight_keys if ".qzeros" in k]
            
            print(f"\n量化权重 (.qweight): {len(qweight_keys)}")
            print(f"量化缩放 (.scales): {len(scales_keys)}")
            print(f"量化零点 (.qzeros): {len(qzeros_keys)}")
            
            # 检查各层的输出投影权重
            o_proj_keys = [k for k in weight_keys if "self_attn.o_proj" in k]
            print(f"\n输出投影权重: {len(o_proj_keys)}")
            
            # 获取层数
            layer_counts = set()
            for k in weight_keys:
                if "model.layers." in k:
                    parts = k.split(".")
                    for i, part in enumerate(parts):
                        if part == "layers" and i+1 < len(parts):
                            layer_counts.add(parts[i+1])
            
            print(f"模型层数: {len(layer_counts)}")
            
            # 打印特定的关键权重
            print("\n关键权重样本:")
            for prefix in ["model.layers.0.self_attn.o_proj", "model.embed_tokens"]:
                matching_keys = [k for k in weight_keys if k.startswith(prefix)]
                for k in matching_keys[:5]:  # 最多展示5个
                    tensor = f.get_tensor(k)
                    print(f"  {k}: 形状{tensor.shape}, 类型{tensor.dtype}")
            
            # 打印所有权重键名
            print("\n所有权重键名:")
            for key in sorted(weight_keys):
                tensor = f.get_tensor(key)
                print(f"  {key}: 形状{tensor.shape}, 类型{tensor.dtype}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python model_analysis.py <模型路径>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    analyze_model(model_path) 