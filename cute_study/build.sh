#!/bin/bash

# CuTe学习项目构建脚本

set -e  # 遇到错误立即退出

echo "=== CuTe学习项目构建脚本 ==="

# 显示CUDA版本
echo "CUDA版本信息:"
nvcc --version

# 检查CUTLASS路径
CUTLASS_PATH="../cutlass"
if [ -d "$CUTLASS_PATH" ]; then
    echo "找到CUTLASS库: $CUTLASS_PATH"
else
    # 尝试当前目录的cutlass
    CUTLASS_PATH="cutlass"
    if [ -d "$CUTLASS_PATH" ]; then
        echo "找到CUTLASS库: $CUTLASS_PATH"
    else
        # 尝试根目录的cutlass
        CUTLASS_PATH="../cutlass"
        if [ -d "$CUTLASS_PATH" ]; then
            echo "找到CUTLASS库: $CUTLASS_PATH"
        else
            echo "错误: 找不到CUTLASS库，请确保cutlass目录存在"
            echo "当前目录内容:"
            ls -la
            exit 1
        fi
    fi
fi

# 创建构建目录
mkdir -p build
cd build

echo "开始编译程序..."

# 编译选项
NVCC_FLAGS="-std=c++17 --expt-relaxed-constexpr -O2 -I${CUTLASS_PATH}/include"
NVCC_FLAGS="$NVCC_FLAGS -gencode arch=compute_70,code=sm_70"
NVCC_FLAGS="$NVCC_FLAGS -gencode arch=compute_75,code=sm_75"
NVCC_FLAGS="$NVCC_FLAGS -gencode arch=compute_80,code=sm_80"

# BF16和cuBLAS需要额外的库链接
NVCC_FLAGS_CUBLAS="$NVCC_FLAGS -lcublas"

echo "编译选项: $NVCC_FLAGS"

# 编译CuTe向量加法
echo "编译 cute_add..."
nvcc $NVCC_FLAGS -o cute_add ../cute_add.cu
if [ $? -eq 0 ]; then
    echo "✅ cute_add 编译成功"
else
    echo "❌ cute_add 编译失败"
    exit 1
fi

# 编译朴素CUDA向量加法
echo "编译 cuda_add_simple..."
nvcc $NVCC_FLAGS -o cuda_add_simple ../cuda_add_simple.cu
if [ $? -eq 0 ]; then
    echo "✅ cuda_add_simple 编译成功"
else
    echo "❌ cuda_add_simple 编译失败"
    exit 1
fi

# 编译CuTe矩阵乘法
echo "编译 cute_matmul..."
nvcc $NVCC_FLAGS -o cute_matmul ../cute_matmul.cu
if [ $? -eq 0 ]; then
    echo "✅ cute_matmul 编译成功"
else
    echo "❌ cute_matmul 编译失败"
    exit 1
fi

# 编译朴素CUDA矩阵乘法
echo "编译 cuda_matmul_simple..."
nvcc $NVCC_FLAGS -o cuda_matmul_simple ../cuda_matmul_simple.cu
if [ $? -eq 0 ]; then
    echo "✅ cuda_matmul_simple 编译成功"
else
    echo "❌ cuda_matmul_simple 编译失败"
    exit 1
fi

# 编译cuBLAS BF16矩阵乘法
echo "编译 cublas_bf16_matmul..."
nvcc $NVCC_FLAGS_CUBLAS -o cublas_bf16_matmul ../cublas_bf16_matmul.cu
if [ $? -eq 0 ]; then
    echo "✅ cublas_bf16_matmul 编译成功"
else
    echo "❌ cublas_bf16_matmul 编译失败"
    exit 1
fi

# 编译CuTe BF16矩阵乘法
echo "编译 cute_bf16_matmul..."
nvcc $NVCC_FLAGS -o cute_bf16_matmul ../cute_bf16_matmul.cu
if [ $? -eq 0 ]; then
    echo "✅ cute_bf16_matmul 编译成功"
else
    echo "❌ cute_bf16_matmul 编译失败"
    exit 1
fi

echo ""
echo "=== 构建完成 ==="
echo "生成的可执行文件:"
ls -la cute_add cuda_add_simple cute_matmul cuda_matmul_simple cublas_bf16_matmul cute_bf16_matmul

echo ""
echo "使用方法:"
echo "  ./cute_add [vector_size]              - CuTe向量加法"
echo "  ./cuda_add_simple [vector_size]       - 朴素CUDA向量加法"
echo "  ./cute_matmul [M] [N] [K]             - CuTe矩阵乘法"
echo "  ./cuda_matmul_simple [M] [N] [K]      - 朴素CUDA矩阵乘法"
echo "  ./cublas_bf16_matmul [M] [N] [K]      - cuBLAS BF16矩阵乘法"
echo "  ./cute_bf16_matmul [M] [N] [K]        - CuTe BF16矩阵乘法"
