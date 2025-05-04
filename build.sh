#!/bin/bash
set -e

# 拉 CUTLASS（如果没拉）
if [ ! -d "cutlass" ]; then
  echo "📥 Cloning CUTLASS..."
  git clone --recursive https://github.com/NVIDIA/cutlass.git --depth=1
fi

echo "🧹 Cleaning build..."
rm -rf build && mkdir build
cd build

echo "🛠️  Running CMake..."
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())')"

echo "🔨 Building..."
make -j$(nproc)

echo "✅ Build Done"
