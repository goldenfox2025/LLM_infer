#!/usr/bin/env python3
"""
专门对比embedding向量的脚本
"""

import os
import struct
import numpy as np
import sys

def read_tensor_from_binary(filename: str) -> np.ndarray:
    """从二进制文件读取张量数据"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"文件不存在: {filename}")
    
    with open(filename, 'rb') as f:
        # 读取维度数量
        ndim = struct.unpack('Q', f.read(8))[0]  # size_t = uint64
        
        # 读取各维度大小
        shape = []
        for _ in range(ndim):
            dim = struct.unpack('Q', f.read(8))[0]
            shape.append(dim)
        
        # 读取数据类型大小
        dtype_size = struct.unpack('Q', f.read(8))[0]
        
        # 根据数据类型大小确定numpy数据类型
        if dtype_size == 4:
            if filename.endswith('input_token.bin'):
                dtype = np.uint32
            else:
                dtype = np.float32
        elif dtype_size == 2:
            dtype = np.float16  # 对于bfloat16，我们用float16近似
        else:
            raise ValueError(f"不支持的数据类型大小: {dtype_size}")
        
        # 读取张量数据
        total_elements = np.prod(shape)
        data = f.read(total_elements * dtype_size)
        
        # 转换为numpy数组
        if dtype_size == 2:
            # 对于bfloat16，需要特殊处理
            raw_data = np.frombuffer(data, dtype=np.uint16)
            # 简单的bfloat16到float32转换（不完全准确，但足够对比）
            tensor_data = raw_data.astype(np.float32) / 256.0
        else:
            tensor_data = np.frombuffer(data, dtype=dtype)
        
        return tensor_data.reshape(shape)

def compare_embedding_vectors(cuda_file: str, graph_file: str):
    """对比CUDA和图推理的embedding向量"""
    print(f"=== 对比embedding向量 ===")
    print(f"CUDA文件: {cuda_file}")
    print(f"图文件: {graph_file}")
    print()
    
    try:
        cuda_embedding = read_tensor_from_binary(cuda_file)
        graph_embedding = read_tensor_from_binary(graph_file)
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return False
    
    print(f"CUDA embedding形状: {cuda_embedding.shape}, 数据类型: {cuda_embedding.dtype}")
    print(f"图 embedding形状: {graph_embedding.shape}, 数据类型: {graph_embedding.dtype}")
    
    if cuda_embedding.shape != graph_embedding.shape:
        print("❌ embedding形状不匹配!")
        return False
    
    # 展平张量以便处理
    flat_cuda = cuda_embedding.flatten()
    flat_graph = graph_embedding.flatten()
    
    # 计算差异
    diff = np.abs(flat_cuda - flat_graph)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    # 相对误差
    rel_diff = diff / (np.abs(flat_cuda) + 1e-10)
    max_rel_diff = np.max(rel_diff)
    
    # 检查是否在容差范围内
    is_close = np.allclose(flat_cuda, flat_graph, rtol=1e-5, atol=1e-8)
    
    print(f"\n=== 差异统计 ===")
    print(f"最大绝对差异: {max_diff:.6e}")
    print(f"平均绝对差异: {mean_diff:.6e}")
    print(f"最大相对差异: {max_rel_diff:.6e}")
    print(f"是否在容差范围内: {'✅ 是' if is_close else '❌ 否'}")
    print(f"总元素数: {len(flat_cuda)}")
    
    if is_close:
        print("\n✅ embedding向量基本相同，权重表访问正常")
        print("   问题可能在gather操作或后续处理")
        return True
    else:
        print("\n❌ embedding向量差异很大，权重表访问有问题")
        
        # 找到最大差异的位置
        max_diff_idx = np.argmax(diff)
        print(f"\n=== 最大差异位置 ===")
        print(f"位置索引: {max_diff_idx}")
        print(f"CUDA值: {flat_cuda[max_diff_idx]:.6f}")
        print(f"图值: {flat_graph[max_diff_idx]:.6f}")
        print(f"绝对差异: {diff[max_diff_idx]:.6e}")
        
        # 打印前10个元素对比
        print(f"\n=== 前10个元素对比 ===")
        print("索引\t\tCUDA\t\t图\t\t绝对差异")
        print("-" * 60)
        for i in range(min(10, len(flat_cuda))):
            print(f"{i:6d}\t\t{flat_cuda[i]:12.6f}\t{flat_graph[i]:12.6f}\t{diff[i]:12.6e}")
        
        return False

def main():
    if len(sys.argv) != 3:
        print("用法: python compare_embedding.py <token_id> <token_id>")
        print("例如: python compare_embedding.py 9707 1")
        print("这将对比:")
        print("  cuda/debug_cuda_token_<token_id>_embedding.bin")
        print("  graph/debug_graph_token_<token_id>_embedding.bin")
        return
    
    token_id1 = sys.argv[1]
    token_id2 = sys.argv[2]
    
    # 对比第一个token
    cuda_file1 = f"cuda/debug_cuda_token_{token_id1}_embedding.bin"
    graph_file1 = f"graph/debug_graph_token_{token_id1}_embedding.bin"
    
    if os.path.exists(cuda_file1) and os.path.exists(graph_file1):
        print(f"🔍 对比token {token_id1}的embedding向量")
        result1 = compare_embedding_vectors(cuda_file1, graph_file1)
        print("\n" + "="*80 + "\n")
    else:
        print(f"❌ token {token_id1}的文件不存在:")
        print(f"   {cuda_file1}: {'存在' if os.path.exists(cuda_file1) else '不存在'}")
        print(f"   {graph_file1}: {'存在' if os.path.exists(graph_file1) else '不存在'}")
        result1 = False
    
    # 对比第二个token（如果不同）
    if token_id1 != token_id2:
        cuda_file2 = f"cuda/debug_cuda_token_{token_id2}_embedding.bin"
        graph_file2 = f"graph/debug_graph_token_{token_id2}_embedding.bin"
        
        if os.path.exists(cuda_file2) and os.path.exists(graph_file2):
            print(f"🔍 对比token {token_id2}的embedding向量")
            result2 = compare_embedding_vectors(cuda_file2, graph_file2)
        else:
            print(f"❌ token {token_id2}的文件不存在:")
            print(f"   {cuda_file2}: {'存在' if os.path.exists(cuda_file2) else '不存在'}")
            print(f"   {graph_file2}: {'存在' if os.path.exists(graph_file2) else '不存在'}")
            result2 = False
    else:
        result2 = True
    
    # 总结
    print("\n" + "="*80)
    print("🎯 总结:")
    if result1 and result2:
        print("✅ 所有embedding向量都匹配，权重表访问正常")
        print("   问题在于gather操作之后的处理")
    else:
        print("❌ embedding向量不匹配，权重表访问有问题")
        print("   需要检查图推理是否使用了正确的权重表")

if __name__ == "__main__":
    main()
