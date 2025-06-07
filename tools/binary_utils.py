#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二进制张量文件读取工具。
"""

import os
import struct
import numpy as np


def read_tensor_from_binary(filename: str) -> np.ndarray:
    """从二进制文件读取张量数据"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"文件不存在: {filename}")

    with open(filename, "rb") as f:
        # 读取维度数量
        ndim = struct.unpack("Q", f.read(8))[0]

        # 读取各维度大小
        shape = [struct.unpack("Q", f.read(8))[0] for _ in range(ndim)]

        # 读取数据类型大小
        dtype_size = struct.unpack("Q", f.read(8))[0]

        # 根据数据类型大小确定numpy数据类型
        if dtype_size == 4:
            # 特殊处理输入token文件
            dtype = np.uint32 if filename.endswith("input_token.bin") else np.float32
        elif dtype_size == 2:
            dtype = np.float16  # bfloat16使用float16近似
        else:
            raise ValueError(f"不支持的数据类型大小: {dtype_size}")

        # 读取张量数据
        total_elements = int(np.prod(shape))
        data = f.read(total_elements * dtype_size)

        if dtype_size == 2:
            # bfloat16 -> float32 简易转换
            raw = np.frombuffer(data, dtype=np.uint16)
            tensor_data = raw.astype(np.float32) / 256.0
        else:
            tensor_data = np.frombuffer(data, dtype=dtype)

    return tensor_data.reshape(shape)
