# --- 单阶段开发镜像 Dockerfile ---
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# 安装系统依赖和开发工具
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    git \
    software-properties-common \
    curl \
    python3.10 \
    python3.10-dev \
    python3.10-distutils && \
    rm -rf /var/lib/apt/lists/*

# 设置 Python 3.10 为默认版本
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/bin/python3.10

# 安装 pip 并升级 pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    python3 -m pip install --no-cache-dir --upgrade pip

# 安装开发所需的 Python 库（这里使用清华大学的镜像可以加速下载）
RUN python3 -m pip install --no-cache-dir --ignore-installed --timeout=600 -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
    pybind11 safetensors tokenizers flask transformers torch

# 设置工作目录为 /build_src，存放源码和编译过程
WORKDIR /build_src

# 复制工程源码到容器中
COPY CMakeLists.txt CMakeLists.txt
COPY backend/ backend/
COPY frontend/ frontend/
COPY interface/ interface/

# 拉取 CUTLASS（固定版本更稳定）
RUN git clone --recursive https://github.com/NVIDIA/cutlass.git 

# 编译阶段：自动获取 pybind11 的 CMake 配置路径传给 cmake，然后编译工程
RUN mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -Dpybind11_DIR="$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())')" && \
    make -j$(nproc)

# 整理产物：将生成的 .so 文件和 frontend 代码复制到 /app 下以便运行和调试
# RUN mkdir -p /app && \
#     cp build/model_bridge*.so /app/ && \
#     cp -r frontend /app/

# 设置最终工作目录为 /app，并设置 PYTHONPATH，保证 Python 能找到模块
WORKDIR /app
ENV PYTHONPATH=/app:/app/frontend:${PYTHONPATH}

# 进入开发容器后启动 bash
CMD ["/bin/bash"]
