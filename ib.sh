#!/bin/bash

# 增量构建脚本 - 只重新编译修改的部分

# 设置颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 设置构建目录
BUILD_DIR="build"
SOURCE_DIR="."

# 检查构建目录是否存在
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}构建目录不存在，创建新的构建目录...${NC}"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake -DCMAKE_BUILD_TYPE=Release ..
    cd ..
fi

# 进入构建目录
cd "$BUILD_DIR"

# 运行 CMake 生成构建文件（不会重新配置，除非 CMakeLists.txt 有变化）
echo -e "${GREEN}正在更新构建文件...${NC}"
cmake ..

# 使用 make 的增量构建功能
echo -e "${GREEN}正在增量构建项目...${NC}"
make -j$(nproc)

# 检查构建结果
if [ $? -eq 0 ]; then
    echo -e "${GREEN}构建成功！${NC}"

    # 创建符号链接到 Python 模块
    # MODULE_FILE=$(find . -name "model_bridge*.so" | head -n 1)
    # if [ -n "$MODULE_FILE" ]; then
    #     echo -e "${GREEN}创建符号链接到 Python 模块: $MODULE_FILE${NC}"
    #     ln -sf "$PWD/$MODULE_FILE" "./model_bridge.so"
    # else
    #     echo -e "${YELLOW}未找到 Python 模块文件${NC}"
    # fi

    # exit 0
else
    echo -e "${RED}构建失败！${NC}"
    exit 1
fi
