#!/usr/bin/env python3
"""
对比两个二进制张量文件，显示详细的差异信息
"""

import os
import struct
import numpy as np
import argparse
import glob

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
        elif dtype_size == 4 and filename.endswith('input_token.bin'):
            # 对于uint32类型的输入token，直接读取为uint32
            tensor_data = np.frombuffer(data, dtype=np.uint32)
        else:
            tensor_data = np.frombuffer(data, dtype=dtype)

        return tensor_data.reshape(shape)

def compare_two_files(file1: str, file2: str):
    """对比两个张量文件"""
    print(f"=== 对比文件 ===")
    print(f"文件1: {file1}")
    print(f"文件2: {file2}")
    print()

    try:
        tensor1 = read_tensor_from_binary(file1)
        tensor2 = read_tensor_from_binary(file2)
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return

    print(f"张量1形状: {tensor1.shape}, 数据类型: {tensor1.dtype}")
    print(f"张量2形状: {tensor2.shape}, 数据类型: {tensor2.dtype}")

    if tensor1.shape != tensor2.shape:
        print("❌ 张量形状不匹配!")
        return

    # 展平张量以便处理
    flat1 = tensor1.flatten()
    flat2 = tensor2.flatten()

    # 计算差异
    diff = np.abs(flat1 - flat2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    # 相对误差
    rel_diff = diff / (np.abs(flat1) + 1e-10)
    max_rel_diff = np.max(rel_diff)

    # 检查是否在容差范围内
    is_close = np.allclose(flat1, flat2, rtol=1e-5, atol=1e-8)

    print(f"\n=== 差异统计 ===")
    print(f"最大绝对差异: {max_diff:.6e}")
    print(f"平均绝对差异: {mean_diff:.6e}")
    print(f"最大相对差异: {max_rel_diff:.6e}")
    print(f"是否在容差范围内: {'✅ 是' if is_close else '❌ 否'}")
    print(f"总元素数: {len(flat1)}")

    # 找到最大差异的位置
    max_diff_idx = np.argmax(diff)

    print(f"\n=== 最大差异位置 ===")
    print(f"位置索引: {max_diff_idx}")
    if len(tensor1.shape) > 1:
        multi_idx = np.unravel_index(max_diff_idx, tensor1.shape)
        print(f"多维索引: {multi_idx}")
    print(f"张量1值: {flat1[max_diff_idx]:.6f}")
    print(f"张量2值: {flat2[max_diff_idx]:.6f}")
    print(f"绝对差异: {diff[max_diff_idx]:.6e}")
    print(f"相对差异: {rel_diff[max_diff_idx]:.6e}")

    # 打印最大差异附近10个元素
    print(f"\n=== 最大差异附近10个元素 ===")
    start_idx = max(0, max_diff_idx - 5)
    end_idx = min(len(flat1), max_diff_idx + 6)

    print("索引\t\t张量1\t\t张量2\t\t绝对差异\t相对差异")
    print("-" * 80)
    for i in range(start_idx, end_idx):
        marker = " *** " if i == max_diff_idx else "     "
        print(f"{i:6d}{marker}\t{flat1[i]:12.6f}\t{flat2[i]:12.6f}\t{diff[i]:12.6e}\t{rel_diff[i]:12.6e}")

    # 打印前10个元素
    print(f"\n=== 前10个元素对比 ===")
    print("索引\t\t张量1\t\t张量2\t\t绝对差异\t相对差异")
    print("-" * 80)
    for i in range(min(10, len(flat1))):
        print(f"{i:6d}\t\t{flat1[i]:12.6f}\t{flat2[i]:12.6f}\t{diff[i]:12.6e}\t{rel_diff[i]:12.6e}")

    # 如果差异很大，打印更多统计信息
    if max_diff > 1e-3:
        print(f"\n=== 大差异元素统计 ===")
        large_diff_mask = diff > 1e-3
        large_diff_count = np.sum(large_diff_mask)
        print(f"差异 > 1e-3 的元素数量: {large_diff_count} ({large_diff_count/len(flat1)*100:.2f}%)")

        if large_diff_count > 0:
            large_diff_indices = np.where(large_diff_mask)[0]
            print(f"前5个大差异元素的位置: {large_diff_indices[:5].tolist()}")

def compare_all_files():
    """自动对比所有对应的文件"""
    # 查找所有graph文件
    graph_files = glob.glob("debug_graph_*.bin")
    cuda_files = glob.glob("cuda/debug_cuda_*.bin")

    if not graph_files:
        print("未找到graph文件 (debug_graph_*.bin)")
        return

    if not cuda_files:
        print("未找到cuda文件 (cuda/debug_cuda_*.bin)")
        return

    print(f"找到 {len(graph_files)} 个graph文件和 {len(cuda_files)} 个cuda文件")

    # 提取文件名模式进行匹配
    graph_patterns = {}
    for f in graph_files:
        pattern = f.replace("debug_graph_", "").replace(".bin", "")
        graph_patterns[pattern] = f

    cuda_patterns = {}
    for f in cuda_files:
        pattern = f.replace("cuda/debug_cuda_", "").replace(".bin", "")
        cuda_patterns[pattern] = f

    # 找到匹配的文件对
    common_patterns = set(graph_patterns.keys()) & set(cuda_patterns.keys())

    if not common_patterns:
        print("未找到匹配的文件对")
        print(f"Graph模式: {list(graph_patterns.keys())[:5]}...")
        print(f"CUDA模式: {list(cuda_patterns.keys())[:5]}...")
        return

    print(f"找到 {len(common_patterns)} 对匹配的文件\n")

    # 按模式排序并逐一对比
    for pattern in sorted(common_patterns):
        graph_file = graph_patterns[pattern]
        cuda_file = cuda_patterns[pattern]

        print("=" * 100)
        print(f"对比模式: {pattern}")
        compare_two_files(graph_file, cuda_file)
        print()

def main():
    parser = argparse.ArgumentParser(description='对比两个张量二进制文件')
    parser.add_argument('--file1', help='第一个文件路径')
    parser.add_argument('--file2', help='第二个文件路径')
    parser.add_argument('--auto', action='store_true', help='自动对比所有匹配的文件')

    args = parser.parse_args()

    if args.auto:
        compare_all_files()
    elif args.file1 and args.file2:
        compare_two_files(args.file1, args.file2)
    else:
        # 默认对比embedding文件
        file1 = "debug_graph_embedding.bin"
        file2 = "cuda/debug_cuda_embedding.bin"

        if os.path.exists(file1) and os.path.exists(file2):
            compare_two_files(file1, file2)
        else:
            print(f"默认文件不存在:")
            print(f"  {file1}: {'存在' if os.path.exists(file1) else '不存在'}")
            print(f"  {file2}: {'存在' if os.path.exists(file2) else '不存在'}")
            print("\n使用方法:")
            print("  python compare_tensors.py --file1 file1.bin --file2 file2.bin")
            print("  python compare_tensors.py --auto  # 自动对比所有匹配的文件")

if __name__ == "__main__":
    main()
