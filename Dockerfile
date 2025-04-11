# --- Stage 1: Builder ---
# 使用包含 CUDA 12.1.1 开发工具的 Ubuntu 22.04 镜像
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

# 设置非交互式，避免 apt-get 提问
ENV DEBIAN_FRONTEND=noninteractive

# 更新包列表并安装编译依赖 (添加了 git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    python3.12 \
    python3.12-dev \
    python3-pip \
 # 如果你的 C++ 代码依赖其他系统库, 在这里添加
 && rm -rf /var/lib/apt/lists/*

# 将 python3.12 设为默认 python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --set python3 /usr/bin/python3.12

# 升级 pip 并安装 Python 依赖 (pybind11)
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir pybind11
    # 如果需要 PyTorch 等其他库，在这里添加

# 设置构建工作目录
WORKDIR /build_src

# 复制 CMakeLists.txt 和所有需要的源代码/资源目录
# 优先复制 CMakeLists.txt
COPY CMakeLists.txt CMakeLists.txt
# 复制源代码和资源目录
COPY backend/ backend/
COPY cutlass/ cutlass/        # <--- 添加 cutlass 目录
COPY frontend/ frontend/      # <--- 添加 frontend 目录
COPY interface/ interface/
# 注意：readme.md 和 Dockerfile 不需要复制到镜像内

# 创建构建目录并运行 CMake 和 Make
# 使用 Release 模式进行优化编译
RUN mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        # CMake 在此环境中应该能自动找到通过 pip 安装的 pybind11
    && make -j$(nproc) # 使用所有 CPU 核心并行编译

# --- Stage 2: Runtime ---
# 使用更小的 CUDA runtime 镜像
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 安装运行所需的最小依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    # 如果你的 C++ 库运行时链接了其他系统库, 在这里安装
 && rm -rf /var/lib/apt/lists/*

# (可选) 如果需要直接调用 python3 命令，再次设置默认值
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --set python3 /usr/bin/python3.12

# (可选) 安装运行 Python 脚本所需的库 (例如 frontend 里的脚本可能需要)
# RUN python3 -m pip install --no-cache-dir numpy pandas Flask # 示例

# 设置最终应用的工作目录
WORKDIR /app

# 从 builder 阶段复制编译好的 .so 文件
COPY --from=builder /build_src/build/model_bridge*.so ./

# 从 builder 阶段复制运行时需要的 frontend 文件
# 假设你的 Python 入口脚本在 frontend 目录中
COPY --from=builder /build_src/frontend/ ./frontend/
# 如果 frontend 目录根下有主脚本，可以这样复制:
# COPY --from=builder /build_src/frontend/main.py .

# 如果运行时还需要 backend 或 cutlass 中的非代码文件 (例如数据、配置)，也需要复制
# COPY --from=builder /build_src/backend/data/ ./data/ # 示例

# 设置 PYTHONPATH，让 Python 能找到当前目录下的 .so 文件和 frontend 目录
ENV PYTHONPATH=/app:/app/frontend:${PYTHONPATH}

# 设置容器启动时执行的默认命令s
# 假设你的启动脚本是 frontend/main.py
# CMD ["python3", "frontend/main.py"]
# 为了方便测试，先设置为启动 bash:
CMD ["/bin/bash"]