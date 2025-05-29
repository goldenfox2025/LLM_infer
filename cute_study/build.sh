#!/bin/bash

# CuTe学习项目构建脚本

set -e  # 遇到错误立即退出

echo "=== CuTe学习项目构建脚本 ==="

# 检查CUDA是否安装
if ! command -v nvcc &> /dev/null; then
    echo "错误: 未找到nvcc，请确保CUDA已正确安装"
    exit 1
fi

# 显示CUDA版本
echo "CUDA版本信息:"
nvcc --version

# 检查CUTLASS是否存在
if [ ! -d "../cutlass" ]; then
    echo "错误: 未找到CUTLASS库，请确保cutlass目录在上级目录中"
    exit 1
fi

echo "找到CUTLASS库: ../cutlass"

# 创建构建目录
BUILD_DIR="build"
if [ -d "$BUILD_DIR" ]; then
    echo "清理旧的构建目录..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "配置CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

echo "开始编译..."
make -j$(nproc)

echo ""
echo "=== 构建完成 ==="
echo "可执行文件位置: $BUILD_DIR/cute_add"
echo ""
echo "运行示例:"
echo "  cd $BUILD_DIR"
echo "  ./cute_add          # 使用默认参数(1M个元素)"
echo "  ./cute_add 1000000  # 指定元素数量"
echo ""
