"""
推理系统的设备配置模块，提供设置和获取默认设备的函数。
"""

import model_bridge

def set_device(device: str) -> bool:
    """
    设置模型执行的默认设备。

    参数：
        device: 目标设备（"cuda" 或 "cpu"）

    返回：
        bool: 设置成功返回 True，否则返回 False
    """
    return model_bridge.set_default_device(device)

def get_device() -> str:
    """
    获取当前默认设备。

    返回：
        str: 当前默认设备（"cuda" 或 "cpu"）
    """
    return model_bridge.get_default_device()

def is_cuda_available() -> bool:
    """
    检查 CUDA 是否可用。

    返回：
        bool: CUDA 可用返回 True，否则返回 False
    """
    # 当前默认设备为 "cuda" 时视为可用
    return get_device() == 'cuda'
