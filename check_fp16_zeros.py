#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np

def test_precision_conversion():
    """测试float16到bfloat16转换中可能出现的精度损失"""
    
    print("===== 测试float16到bfloat16转换中的精度问题 =====")
    
    # 创建一些特定的小值
    test_values = [
        0.1,           # 普通小数
        0.01,          # 更小的值
        0.001,         # 千分之一
        0.0001,        # 万分之一
        1e-5,          # 十万分之一
        1e-6,          # 百万分之一
        1e-7,          # 千万分之一
        1e-8,          # 亿分之一
        0.0            # 零值
    ]
    
    # 创建一个浮点tensor
    print("\n原始float32值:")
    for val in test_values:
        print(f"{val:.10f}")
    
    # 转换到float16
    fp16_tensor = torch.tensor(test_values, dtype=torch.float16)
    
    # 打印float16值
    print("\nfloat16值:")
    fp16_np = fp16_tensor.numpy()
    for val in fp16_np:
        print(f"{val:.10f}")
    
    # 转换到bfloat16 (通过先转为float32)
    bf16_tensor = fp16_tensor.to(torch.bfloat16)
    
    # 打印bfloat16值
    print("\nbfloat16值:")
    bf16_np = bf16_tensor.to(torch.float32).numpy()  # 必须先转回float32才能打印
    for val in bf16_np:
        print(f"{val:.10f}")
    
    # 检查精度损失
    print("\n精度损失分析:")
    print("原始值 -> float16 -> bfloat16")
    for i, orig in enumerate(test_values):
        fp16_val = float(fp16_np[i])
        bf16_val = float(bf16_np[i])
        
        fp16_diff = abs(orig - fp16_val)
        bf16_diff = abs(orig - bf16_val)
        
        fp16_percent = (fp16_diff / max(abs(orig), 1e-10)) * 100
        bf16_percent = (bf16_diff / max(abs(orig), 1e-10)) * 100
        
        print(f"{orig:.8f} -> {fp16_val:.8f} (损失: {fp16_percent:.2f}%) -> {bf16_val:.8f} (损失: {bf16_percent:.2f}%)")
    
    # 测试二进制表示
    print("\n二进制表示比较:")
    for i, val in enumerate(test_values):
        if val == 0.0:  # 跳过零值
            continue
            
        fp32 = torch.tensor([val], dtype=torch.float32)
        fp16 = torch.tensor([val], dtype=torch.float16)
        bf16 = torch.tensor([val], dtype=torch.bfloat16)
        
        # 获取二进制表示
        fp32_hex = hex(fp32.numpy().view(np.uint32)[0])
        fp16_bits = fp16.numpy().view(np.uint16)[0]
        fp16_hex = hex(fp16_bits)
        
        # bfloat16需要先转为float32
        bf16_as_fp32 = bf16.to(torch.float32)
        bf16_bits = bf16_as_fp32.numpy().view(np.uint32)[0] >> 16  # 只保留高16位
        bf16_hex = hex(bf16_bits)
        
        print(f"值 {val}:")
        print(f"  float32: {fp32_hex}")
        print(f"  float16: {fp16_hex}")
        print(f"  bfloat16: {bf16_hex}")
    
    # 特别测试 - 创建一个随机范围内的浮点数组
    print("\n=== 随机小值测试 ===")
    # 生成0到0.2范围内的100个随机值
    torch.manual_seed(42)  # 固定随机种子以便重现
    random_small = torch.rand(100) * 0.2
    
    # 转换并比较
    fp16_random = random_small.to(torch.float16)
    bf16_from_fp16 = fp16_random.to(torch.bfloat16)
    bf16_from_fp32 = random_small.to(torch.bfloat16)
    
    # 转回float32以便比较
    fp16_as_fp32 = fp16_random.to(torch.float32)
    bf16_from_fp16_as_fp32 = bf16_from_fp16.to(torch.float32)
    bf16_from_fp32_as_fp32 = bf16_from_fp32.to(torch.float32)
    
    # 计算变为0的比例
    zeros_in_fp16 = (fp16_as_fp32 == 0).sum().item()
    zeros_in_bf16_from_fp16 = (bf16_from_fp16_as_fp32 == 0).sum().item()
    zeros_in_bf16_from_fp32 = (bf16_from_fp32_as_fp32 == 0).sum().item()
    
    print(f"原始值中为0的数量: 0 / 100 (0%)")
    print(f"float16中变为0的数量: {zeros_in_fp16} / 100 ({zeros_in_fp16}%)")
    print(f"float16->bfloat16中变为0的数量: {zeros_in_bf16_from_fp16} / 100 ({zeros_in_bf16_from_fp16}%)")
    print(f"float32->bfloat16中变为0的数量: {zeros_in_bf16_from_fp32} / 100 ({zeros_in_bf16_from_fp32}%)")
    
    # 检查非零值的分布情况
    non_zero_orig = random_small[random_small > 0]
    non_zero_fp16 = fp16_as_fp32[fp16_as_fp32 > 0]
    non_zero_bf16_from_fp16 = bf16_from_fp16_as_fp32[bf16_from_fp16_as_fp32 > 0]
    
    print(f"\n非零值的范围:")
    print(f"原始值: 最小={non_zero_orig.min().item():.8f}, 最大={non_zero_orig.max().item():.8f}")
    
    if len(non_zero_fp16) > 0:
        print(f"float16: 最小={non_zero_fp16.min().item():.8f}, 最大={non_zero_fp16.max().item():.8f}")
    else:
        print("float16: 没有非零值")
        
    if len(non_zero_bf16_from_fp16) > 0:
        print(f"float16->bfloat16: 最小={non_zero_bf16_from_fp16.min().item():.8f}, 最大={non_zero_bf16_from_fp16.max().item():.8f}")
    else:
        print("float16->bfloat16: 没有非零值")

if __name__ == "__main__":
    test_precision_conversion() 