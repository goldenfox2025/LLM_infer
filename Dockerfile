# --- Stage 1: Builder ---
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    python3.12 \
    python3.12-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --set python3 /usr/bin/python3.12

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir pybind11

WORKDIR /build_src

COPY CMakeLists.txt CMakeLists.txt
COPY backend/ backend/
COPY frontend/ frontend/
COPY interface/ interface/

# 拉取 CUTLASS（固定版本更稳定）
RUN git clone --recursive https://github.com/NVIDIA/cutlass.git --depth=1

RUN mkdir build && cd build && \
    cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc)

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