#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import argparse
import os
import numpy as np
from pathlib import Path

def check_scales(model_path):
    """检查AWQ模型权重中的scales值并输出详细统计信息"""
    print(f"加载模型: {model_path}")
    
    # 尝试加载模型
    try:
        weights = torch.load(model_path, map_location="cpu")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    print(f"模型加载成功，共有 {len(weights)} 个键")
    
    # 查找所有scales权重
    scales_keys = [k for k in weights.keys() if ".scales" in k]
    
    print(f"找到 {len(scales_keys)} 个scales权重")
    
    # 检查每个scales张量
    for key in sorted(scales_keys):
        scale = weights[key]
        if not isinstance(scale, torch.Tensor):
            print(f"{key}: 不是Tensor类型")
            continue
        
        # 基本信息
        dtype = scale.dtype
        shape = scale.shape
        device = scale.device
        
        # 统计信息
        with torch.no_grad():
            scale_np = scale.float().cpu().numpy()
            total_elements = scale_np.size
            non_zero = np.count_nonzero(scale_np)
            non_zero_percent = (non_zero / total_elements) * 100
            
            # 详细统计
            min_val = scale_np.min() if total_elements > 0 else "N/A"
            max_val = scale_np.max() if total_elements > 0 else "N/A"
            mean_val = scale_np.mean() if total_elements > 0 else "N/A"
            
            # 高精度检查微小值
            almost_zero = np.sum(np.abs(scale_np) < 1e-6)
            
        print("\n" + "="*50)
        print(f"键名: {key}")
        print(f"类型: {dtype}, 形状: {shape}, 设备: {device}")
        print(f"元素总数: {total_elements}")
        print(f"非零元素: {non_zero} ({non_zero_percent:.4f}%)")
        print(f"几乎为零元素 (<1e-6): {almost_zero} ({almost_zero/total_elements*100:.4f}%)")
        print(f"最小值: {min_val}")
        print(f"最大值: {max_val}")
        print(f"平均值: {mean_val}")
        
        # 打印样本值
        print("样本值:")
        
        # 如果是2D张量，打印第一行的前10个值
        if len(shape) == 2:
            first_row = scale_np[0, :min(10, shape[1])]
            print(f"第一行前10个值: {first_row}")
            
            # 随机抽样其他行的值
            if shape[0] > 1:
                random_indices = np.random.choice(shape[0], min(5, shape[0]), replace=False)
                for idx in random_indices:
                    if idx == 0:  # 已经打印过第一行
                        continue
                    row_sample = scale_np[idx, :min(10, shape[1])]
                    print(f"第{idx}行前10个值: {row_sample}")
        
        # 统计小值分布
        if total_elements > 0:
            small_vals_count = {}
            thresholds = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
            
            for t in thresholds:
                count = np.sum((scale_np > 0) & (scale_np < t))
                small_vals_count[t] = count
            
            print("小值分布:")
            for t, count in small_vals_count.items():
                print(f"  0 < x < {t}: {count} ({count/total_elements*100:.4f}%)")

def main():
    parser = argparse.ArgumentParser(description="检查AWQ模型权重中的scales值")
    parser.add_argument("model_path", help="模型权重文件路径")
    args = parser.parse_args()
    
    check_scales(args.model_path)

if __name__ == "__main__":
    main() 