# --- 单阶段开发镜像 Dockerfile ---
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# 安装系统依赖和开发工具，包括 wget（后续下载工具用到）
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    git \
    software-properties-common \
    curl \
    wget \
    python3.10 \
    python3.10-dev \
    python3.10-distutils && \
    rm -rf /var/lib/apt/lists/*

# 安装 Nsight Systems（nsys）
RUN mkdir -p /opt && cd /opt && \
    wget https://developer.download.nvidia.com/devtools/nsight-systems/nsight-systems-linux-public-2024.1.1.124-41846084.run -O nsys.run && \
    chmod +x nsys.run && \
    ./nsys.run --accept-eula --silent --install-dir=/opt/nsight-systems && \
    ln -sf /opt/nsight-systems/host-linux-x64/nsys /usr/local/bin/nsys

# 安装 Nsight Compute（ncu）
RUN mkdir -p /opt && cd /opt && \
    wget https://developer.download.nvidia.com/devtools/nsight-compute/NsightCompute-linux-public-2023.3.0.4.run -O ncu.run && \
    chmod +x ncu.run && \
    ./ncu.run --accept-eula --silent --install-dir=/opt/nsight-compute && \
    ln -sf /opt/nsight-compute/host-linux-x64/ncu /usr/local/bin/ncu

# 设置 Python 3.10 为默认版本
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/bin/python3.10

# 安装 pip 并升级 pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    python3 -m pip install --no-cache-dir --upgrade pip

# 安装开发所需的 Python 库（使用清华大学镜像加速下载）
RUN python3 -m pip install --no-cache-dir --ignore-installed --timeout=600 -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
    pybind11 safetensors tokenizers flask transformers torch

# 设置工作目录为 /app。建议在启动容器时将你的本地源代码目录挂载到 /app
WORKDIR /app
ENV PYTHONPATH=/app:${PYTHONPATH}

# 进入容器后启动 bash
CMD ["/bin/bash"]
