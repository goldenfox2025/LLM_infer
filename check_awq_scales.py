#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import argparse
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_scales(model_path, fix=False, fix_scale_factor=0.05, output_dir=None):
    """分析AWQ模型权重中的scales值，并可选择进行修复"""
    print(f"加载模型: {model_path}")
    
    # 创建输出目录
    if fix and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
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
    
    # 问题统计
    zero_scales_count = 0
    total_analyzed = 0
    fixed_count = 0
    
    # 存储统计数据用于绘图
    all_scale_values = []
    all_max_values = []
    
    # 检查每个scales张量
    for key in sorted(scales_keys):
        total_analyzed += 1
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
            
            # 收集所有非零scale值用于绘图
            flat_scales = scale_np.flatten()
            non_zero_scales = flat_scales[flat_scales != 0]
            all_scale_values.extend(non_zero_scales)
            
            total_elements = scale_np.size
            non_zero = np.count_nonzero(scale_np)
            non_zero_percent = (non_zero / total_elements) * 100
            
            # 详细统计
            min_val = scale_np.min() if total_elements > 0 else "N/A"
            max_val = scale_np.max() if total_elements > 0 else "N/A"
            mean_val = scale_np.mean() if total_elements > 0 else "N/A"
            
            # 记录最大值用于绘图
            if isinstance(max_val, (int, float)) and max_val > 0:
                all_max_values.append(max_val)
            
            # 高精度检查微小值
            almost_zero = np.sum(np.abs(scale_np) < 1e-6)
            
            # 判断是否存在问题
            has_problem = non_zero == 0 or non_zero_percent < 5
            
            if has_problem:
                zero_scales_count += 1
            
        print("\n" + "="*50)
        print(f"键名: {key}")
        print(f"类型: {dtype}, 形状: {shape}, 设备: {device}")
        print(f"元素总数: {total_elements}")
        print(f"非零元素: {non_zero} ({non_zero_percent:.4f}%)")
        print(f"几乎为零元素 (<1e-6): {almost_zero} ({almost_zero/total_elements*100:.4f}%)")
        print(f"最小值: {min_val}")
        print(f"最大值: {max_val}")
        print(f"平均值: {mean_val}")
        print(f"存在问题: {'是' if has_problem else '否'}")
        
        # 如果scales全部或几乎全部为0，并且请求修复
        if fix and has_problem:
            print(f"正在修复: {key}")
            # 创建一个正态分布的随机scale，以fix_scale_factor为均值，0.01为标准差
            random_scales = torch.normal(mean=fix_scale_factor, std=0.01, size=shape, dtype=torch.float32)
            # 确保所有值都大于0
            random_scales = torch.clamp(random_scales, min=0.01, max=0.2)
            # 替换原来的scales
            weights[key] = random_scales.to(dtype)
            fixed_count += 1
            print(f"已修复: 使用均值为{fix_scale_factor}的随机正态分布")
    
    # 打印总结
    print("\n" + "="*50)
    print(f"分析总结:")
    print(f"总共分析: {total_analyzed} 个scales tensor")
    print(f"问题scales数量: {zero_scales_count} ({zero_scales_count/total_analyzed*100:.2f}%)")
    
    if fix:
        print(f"已修复: {fixed_count} 个scales tensor")
        
        # 保存修复后的模型
        if output_dir:
            output_path = os.path.join(output_dir, Path(model_path).name)
            print(f"正在保存修复后的模型: {output_path}")
            try:
                torch.save(weights, output_path)
                print(f"模型已保存到: {output_path}")
            except Exception as e:
                print(f"保存模型失败: {e}")
    
    # 绘制scale值分布图
    if all_scale_values:
        plt.figure(figsize=(10, 6))
        
        # 绘制所有scale值的直方图
        plt.subplot(1, 2, 1)
        plt.hist(all_scale_values, bins=50, alpha=0.75)
        plt.title('Scale值分布')
        plt.xlabel('Scale值')
        plt.ylabel('频率')
        
        # 绘制scale最大值分布
        plt.subplot(1, 2, 2)
        plt.hist(all_max_values, bins=30, alpha=0.75)
        plt.title('Scale最大值分布')
        plt.xlabel('最大scale值')
        plt.ylabel('频率')
        
        plt.tight_layout()
        
        # 保存图表
        if output_dir:
            chart_path = os.path.join(output_dir, "scale_distribution.png")
            plt.savefig(chart_path)
            print(f"分布图已保存到: {chart_path}")
        
        # 显示图表
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="分析和修复AWQ模型权重中的scales值")
    parser.add_argument("model_path", help="模型权重文件路径")
    parser.add_argument("--fix", action="store_true", help="是否修复问题scales")
    parser.add_argument("--scale-factor", type=float, default=0.05, help="修复时使用的scale因子均值")
    parser.add_argument("--output-dir", type=str, default="fixed_model", help="修复后模型的输出目录")
    args = parser.parse_args()
    
    analyze_scales(args.model_path, args.fix, args.scale_factor, args.output_dir)

if __name__ == "__main__":
    main() 