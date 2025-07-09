#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
投机解码测试脚本
"""

import os
import sys
import time
import queue
import threading
from typing import List, Callable
import torch
from pathlib import Path
from safetensors import safe_open

# 导入tokenizer
from transformers import AutoTokenizer

def load_qwen3_model(model_path, keep_bf16=True, is_awq=True):
    """加载 Qwen3 模型及配置，可选保持 BF16 精度或加载 AWQ 量化模型"""
    import json

    model_path = Path(model_path)
    weights = {}

    # 首先检查和加载索引文件
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        print(f"找到模型索引文件: {index_path}")
        with open(index_path, 'r') as f:
            index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})

        # 按文件分组权重，以便一次性加载每个文件的所有权重
        weights_by_file = {}
        for key, file_name in weight_map.items():
            if file_name not in weights_by_file:
                weights_by_file[file_name] = []
            weights_by_file[file_name].append(key)

        # 加载每个文件中的权重
        for file_name, keys in weights_by_file.items():
            file_path = model_path / file_name
            print(f"从文件加载权重: {file_path}")

            with safe_open(file_path, framework="pt") as f:
                for key in keys:
                    tensor = f.get_tensor(key)

                    # 如果是AWQ量化模型，需要区分量化权重和非量化权重
                    if is_awq:
                        # 检查是否是量化权重（qweight、scales、qzeros）
                        if any(suffix in key for suffix in [".qweight", ".scales", ".qzeros"]):
                            # 量化权重保持原始格式
                            weights[key] = tensor
                            print(f"加载AWQ量化张量 {key}，形状 {tensor.shape}，数据类型 {tensor.dtype}")
                        else:
                            # 非量化权重根据keep_bf16参数决定是否转换
                            if tensor.dtype == torch.bfloat16 and keep_bf16:
                                weights[key] = tensor
                                print(f"加载AWQ非量化bf16张量 {key}，形状 {tensor.shape}")
                            else:
                                weights[key] = tensor.to(torch.float32)
                                print(f"加载AWQ非量化fp32张量 {key}，形状 {weights[key].shape}")
                    # 非AWQ模型处理
                    elif tensor.dtype == torch.bfloat16 and keep_bf16:
                        weights[key] = tensor
                        print(f"加载bf16张量 {key}，形状 {tensor.shape}")
                    else:
                        weights[key] = tensor.to(torch.float32)
                        print(f"加载张量 {key}，形状 {weights[key].shape}")
    else:
        # 尝试从单个文件加载，保持向后兼容性
        safetensors_path = model_path / "model.safetensors"
        if not safetensors_path.exists():
            raise FileNotFoundError(f"未找到模型权重文件: 既没有索引文件 {index_path}，也没有单一权重文件 {safetensors_path}")

        print(f"未找到索引文件，尝试从单一文件加载: {safetensors_path}")
        with safe_open(safetensors_path, framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)

                # 如果是AWQ量化模型，需要区分量化权重和非量化权重
                if is_awq:
                    # 检查是否是量化权重（qweight、scales、qzeros）
                    if any(suffix in key for suffix in [".qweight", ".scales", ".qzeros"]):
                        # 量化权重保持原始格式
                        weights[key] = tensor
                        print(f"加载AWQ量化张量 {key}，形状 {tensor.shape}，数据类型 {tensor.dtype}")
                    else:
                        # 非量化权重根据keep_bf16参数决定是否转换
                        if tensor.dtype == torch.bfloat16 and keep_bf16:
                            weights[key] = tensor
                            print(f"加载AWQ非量化bf16张量 {key}，形状 {tensor.shape}")
                        else:
                            weights[key] = tensor.to(torch.float32)
                            print(f"加载AWQ非量化fp32张量 {key}，形状 {weights[key].shape}")
                # 非AWQ模型处理
                elif tensor.dtype == torch.bfloat16 and keep_bf16:
                    weights[key] = tensor
                    print(f"加载bf16张量 {key}，形状 {tensor.shape}")
                else:
                    weights[key] = tensor.to(torch.float32)
                    print(f"加载张量 {key}，形状 {weights[key].shape}")

    # 加载配置文件
    config_path = model_path / "config.json"
    print(f"正在读取配置文件: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_str = f.read()
            print(f"原始配置内容的前100个字符: {config_str[:100]}...")
            config = json.loads(config_str)
            print("成功加载配置文件")
    except Exception as e:
        print(f"加载配置文件时出错: {e}")
        raise

    # 打印原始配置内容
    print("原始配置内容:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 创建C++接口所需的配置字典
    cpp_config = {}

    # 直接复制所有原始配置（确保所有键都存在）
    for key, value in config.items():
        cpp_config[key] = value

    # 特殊映射键名，同时保留原键名和映射后的键名
    key_mapping = {
        "num_hidden_layers": "n_layers",
        "num_attention_heads": "n_heads",
        "num_key_value_heads": "n_kv_heads"
    }

    # 添加映射后的键名
    for orig_key, new_key in key_mapping.items():
        if orig_key in config:
            cpp_config[new_key] = config[orig_key]
            print(f"添加映射键: {orig_key} -> {new_key}: {config[orig_key]}")


    # 添加量化相关配置
    if is_awq:
        cpp_config["quant_type"] = 1

        # 检查是否有AWQ相关配置
        if "quantization_config" in config:
            quant_config = config["quantization_config"]
            if "group_size" in quant_config:
                cpp_config["group_size"] = quant_config["group_size"]
                print(f"使用配置中的group_size: {cpp_config['group_size']}")
            else:
                cpp_config["group_size"] = 128  # 默认值
                print(f"使用默认group_size: 128")
        else:
            cpp_config["group_size"] = 128  # 默认值
            print(f"使用默认group_size: 128")
    else:
        cpp_config["quant_type"] = 0

    # 打印最终配置
    print("最终配置:")
    for key, value in cpp_config.items():
        print(f"  {key}: {value}")

    return cpp_config, weights

def create_callback(q):
    """创建回调函数，将生成的token放入队列"""
    def callback(token_id):
        q.put(token_id)
    return callback

def main():
    # 创建logits数据存储目录
    logits_dirs = ["./logits_data", "./logits_data/target", "./logits_data/draft", "./logits_data/visualizations"]
    for dir_path in logits_dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("已创建logits数据存储目录")

    # 设置模型路径（写死）
    target_model_path = "./models/Qwen3-1.7B-AWQ"
    draft_model_path = "./models/Qwen3-0.6B-AWQ"
    print(f"目标模型路径: {target_model_path}")
    print(f"草稿模型路径: {draft_model_path}")

    # 导入C++模型桥接接口
    sys.path.append("./build")
    from model_bridge import (
        init_model, generate_text_stream, set_default_device,
        init_speculative_decoder, generate_text_stream_speculative
    )

    # 设置设备
    print("设置设备为: cuda")
    set_default_device("cuda")

    # 检查初始GPU内存状态
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            memory_info = result.stdout.strip().split(', ')
            total_mem, used_mem, free_mem = map(int, memory_info)
            print(f"【GPU内存】总计: {total_mem} MB, 已用: {used_mem} MB, 空闲: {free_mem} MB")
            if free_mem < 3000:  # 如果空闲内存少于3GB
                print("警告: GPU空闲内存可能不足，建议释放其他CUDA程序")
    except:
        print("无法获取GPU内存信息")

    # 加载tokenizer
    print("【调试】开始加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(target_model_path)
    print("【调试】tokenizer加载完成")

    # 加载目标模型
    print(f"【调试】开始加载目标模型: {target_model_path}")
    target_config, target_weights = load_qwen3_model(target_model_path)
    print("【调试】目标模型权重加载完成，开始初始化...")

    # 初始化目标模型
    model_type = "qwen3_awq"  # 假设使用AWQ量化模型
    print(f"【调试】开始初始化目标模型，类型: {model_type}")
    if not init_model(target_config, target_weights, model_type):
        print("目标模型初始化失败")
        return
    print("【调试】目标模型初始化成功")

    # 加载草稿模型
    print(f"【调试】开始加载草稿模型: {draft_model_path}")
    draft_config, draft_weights = load_qwen3_model(draft_model_path)
    print("【调试】草稿模型权重加载完成，开始初始化投机解码器...")
    model_type = "qwen3_awq"  
    # 初始化投机解码器
    spec_length = 8
    print(f"【调试】开始初始化投机解码器，投机长度: {spec_length}")
    if not init_speculative_decoder(draft_config, draft_weights, model_type, spec_length):
        print("投机解码器初始化失败")
        return
    print("【调试】投机解码器初始化成功")

    # 设置系统提示和用户输入 - 使用简短的提示便于观察
    system_prompt = "你是一个有用的AI助手。"
    user_input = "给我讲述一个有趣的黑暗之魂故事。"  # 使用简单的问题，便于观察和分析

    # 使用chat.py中的方式构建对话模板
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    # 应用聊天模板
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print(f"系统提示: {system_prompt}")
    print(f"用户输入: {user_input}")
    print(f"构建的对话模板: {prompt[:100]}...")

    # 编码输入
    input_ids = tokenizer.encode(prompt)
    print(f"输入token数: {len(input_ids)}")

    # 使用标准解码
    print("\n=== 使用标准解码 ===")
    q = queue.Queue()
    callback = create_callback(q)

    start_time = time.time()

    def run_standard_generation():
        try:
            generate_text_stream(
                input_ids,
                callback,
                max_length=60,  # 减少生成长度，便于观察和分析
                temperature=0.8,  
                top_p=0.9,
                top_k=50
            )
        except Exception as e:
            print(f"标准解码出错: {e}")
        finally:
            q.put(None)  # 生成结束标记

    thread = threading.Thread(target=run_standard_generation)
    thread.start()

    # 在主线程中累积token并解码
    output_ids = []
    last_output = ""
    print("输出: ", end="", flush=True)

    try:
        while True:
            token = q.get(timeout=60)  # 添加超时，避免无限等待
            if token is None:
                break

            output_ids.append(token)
            try:
                new_text = tokenizer.decode(output_ids)
                diff = new_text[len(last_output):]  # 只获取新增的文本部分
                last_output = new_text

                if diff:
                    print(diff, end="", flush=True)
            except Exception as e:
                print(f"解码错误: {e}", end="", flush=True)
    except queue.Empty:
        print("\n等待超时，强制结束生成")

    print()
    standard_time = time.time() - start_time
    standard_tokens = len(output_ids)
    print(f"标准解码生成了 {standard_tokens} 个token，耗时 {standard_time:.2f} 秒")
    if standard_tokens > 0:
        print(f"标准解码速度: {standard_tokens / standard_time:.2f} tokens/s")
    else:
        print("标准解码未生成任何token")

    # 使用投机解码
    print("\n=== 使用投机解码 ===")
    q = queue.Queue()
    callback = create_callback(q)

    start_time = time.time()

    def run_speculative_generation():
        try:
            generate_text_stream_speculative(
                input_ids,
                callback,
                max_length=60,  # 减少生成长度，便于观察和分析
                temperature=0.8,
                top_p=0.9,
                top_k=50
            )
        except Exception as e:
            print(f"投机解码出错: {e}")
        finally:
            q.put(None)  # 生成结束标记

    thread = threading.Thread(target=run_speculative_generation)
    thread.start()

    # 在主线程中累积token并解码
    output_ids = []
    last_output = ""
    print("输出: ", end="", flush=True)

    try:
        while True:
            token = q.get(timeout=60)  # 添加超时，避免无限等待
            if token is None:
                break

            output_ids.append(token)
            try:
                new_text = tokenizer.decode(output_ids)
                diff = new_text[len(last_output):]  # 只获取新增的文本部分
                last_output = new_text

                if diff:
                    print(diff, end="", flush=True)
            except Exception as e:
                print(f"解码错误: {e}", end="", flush=True)
    except queue.Empty:
        print("\n等待超时，强制结束生成")

    print()
    spec_time = time.time() - start_time
    spec_tokens = len(output_ids)
    print(f"投机解码生成了 {spec_tokens} 个token，耗时 {spec_time:.2f} 秒")
    if spec_tokens > 0:
        print(f"投机解码速度: {spec_tokens / spec_time:.2f} tokens/s")
    else:
        print("投机解码未生成任何token")

    # 性能比较
    print(f"\n=== 性能比较 ===")
    if standard_tokens > 0 and spec_tokens > 0:
        speedup = standard_time / spec_time
        print(f"加速比: {speedup:.2f}x")
    else:
        print("无法计算加速比：标准解码或投机解码未生成足够的token")
    
    # 提示用户可以运行可视化脚本
    print("\n数据收集完成！您可以运行以下命令查看logits分布可视化：")
    print("python visualize_logits.py")

if __name__ == "__main__":
    main()
