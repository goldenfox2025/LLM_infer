#!/usr/bin/env python3
import sys
import json
import threading
import queue
import time
from pathlib import Path
from safetensors import safe_open
from tokenizers import Tokenizer  
import torch


def load_model(model_path: str, keep_bf16=True):
    """
    加载模型权重和配置文件，可选择保持bf16精度
    """
    model_path = Path(model_path)
    weights = {}
    # 加载模型权重
    with safe_open(model_path / "model.safetensors", framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            # 如果设置了保持bf16并且张量是bf16类型，保持原始格式不转换
            if tensor.dtype == torch.bfloat16 and keep_bf16:
                weights[key] = tensor  # 保持bf16格式
                print(f"Loaded bf16 tensor {key} with shape {tensor.shape}")
            else:
                # 对于非bf16张量或者未启用keep_bf16选项，转换为float32
                weights[key] = tensor.to(torch.float32)
                print(f"Loaded tensor {key} with shape {weights[key].shape}")
    
    # 加载配置文件
    with open(model_path / "config.json", 'r') as f:
        config = json.load(f)
        # 确保核心配置项存在
        expected_keys = [
            "vocab_size", "hidden_size", "num_hidden_layers", 
            "num_attention_heads", "num_key_value_heads",
            "intermediate_size", "max_position_embeddings",
            "rms_norm_eps", "rope_theta"
        ]
        for key in expected_keys:
            if key not in config:
                print(f"Warning: {key} not found in config")
                
        print("Config loaded:", config)
        
    return config, weights

def load_tokenizer(model_path: str):
    """
    加载tokenizer
    """
    tokenizer_path = Path(model_path) / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print("Tokenizer loaded from:", tokenizer_path)
    return tokenizer

# -------------------------------
# 回调函数：累积 token 并输出文本差量及速度信息
# -------------------------------
def create_callback(tokenizer, q: queue.Queue):
    accumulated_tokens = []
    last_output = ""  # 记录上一次完整解码后的文本
    last_token_time = None  # 用于计算 token 生成速度
    start_time = time.time()  # 记录开始时间
    total_tokens = 0  # 记录生成的总 token 数

    def token_callback(token):
        nonlocal last_output, last_token_time, total_tokens, start_time

        current_time = time.time()
        speed = None
        if last_token_time is not None:
            delta = current_time - last_token_time
            speed = 1.0 / delta if delta > 0 else 0.0
        last_token_time = current_time

        accumulated_tokens.append(token)
        total_tokens += 1
        new_text = tokenizer.decode(accumulated_tokens)
        diff = new_text[len(last_output):]
        last_output = new_text
        if diff:
            total_time = current_time - start_time
            avg_speed = total_tokens / total_time if total_time > 0 else 0.0
            # 将差量、速度等信息发送到队列
            q.put(json.dumps({
                "diff": diff,
                "speed": speed if speed is not None else 0.0,
                "total_time": total_time,
                "total_tokens": total_tokens,
                "avg_speed": avg_speed
            }))
    return token_callback


if __name__ == "__main__":
    # 指定模型路径
    MODEL_PATH = "/home/LLM_infer/models/Qwen2.5-1.5B-Instruct"
    # 加载模型时保持bf16精度
    config, weights = load_model(MODEL_PATH, keep_bf16=True)
    tokenizer = load_tokenizer(MODEL_PATH)

    # 计算并打印模型大小（参数数量和内存占用）
    total_params = sum(t.numel() for t in weights.values())
    total_bytes = sum(t.element_size() * t.numel() for t in weights.values())
    print("\nModel size: {} parameters, {:.2f} MB".format(total_params, total_bytes / (1024 * 1024)))

    # 打印模型配置信息
    print("\nModel Configuration:")
    print(f"Model Type: Qwen2.5")
    print(f"Precision: BF16")
    print(f"Hidden Size: {config['hidden_size']}")
    print(f"Num Attention Heads: {config['num_attention_heads']}")
    print(f"Num Key Value Heads: {config['num_key_value_heads']}")
    print(f"Head Dimension: {config['hidden_size'] // config['num_attention_heads']}")
    print(f"Num Layers: {config['num_hidden_layers']}")
    
    # 添加模型桥接模块路径
    sys.path.append("/home/LLM_infer/build")
    from model_bridge import init_model, generate_text_stream

    # 初始化模型，指定类型为qwen_bf16
    if not init_model(config, weights, "qwen_bf16"):
        print("Model initialization failed.", file=sys.stderr)
        exit(1)
    print("\nModel initialized successfully.\n")

    # 定义系统提示词，使用Qwen2.5的对话格式
    system_prompt = "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
    first_chat = True

    print("Enter 'quit' to exit.\n")
    while True:
        user_message = input("User: ").strip()
        if user_message.lower() in {"quit", "exit"}:
            break
        if not user_message:
            continue

        # 构造对话文本，使用Qwen2.5的格式
        if first_chat:
            # 第一次对话包含系统提示词
            conversation = f"{system_prompt}<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
            first_chat = False
        else:
            # 后续对话只包含用户输入
            conversation = f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        
        encoded = tokenizer.encode(conversation)
        input_ids = encoded.ids

        # 使用队列和回调函数传递生成信息
        q = queue.Queue()
        token_callback = create_callback(tokenizer, q)

        # 后台线程调用生成函数
        def run_generation():
            generate_text_stream(
                input_ids,
                token_callback,
                max_length=2000,     # Qwen2.5支持更长的序列
                temperature=0.7,     # 可调整温度
                top_p=0.9,
                top_k=10
            )
            # 生成结束后，发送特殊标记结束
            q.put(None)

        thread = threading.Thread(target=run_generation)
        thread.start()

        print("Assistant: ", end="", flush=True)
        token_speed = 0.0
        total_time = 0.0
        total_tokens = 0
        avg_speed = 0.0
        while True:
            msg = q.get()
            if msg is None:
                break
            try:
                data = json.loads(msg)
            except Exception as e:
                continue
            if "diff" in data:
                print(data["diff"], end="", flush=True)
            if "speed" in data:
                token_speed = data["speed"]
            if "total_time" in data:
                total_time = data["total_time"]
            if "total_tokens" in data:
                total_tokens = data["total_tokens"]
            if "avg_speed" in data:
                avg_speed = data["avg_speed"]
        print("\n")
        # 输出生成结束后的 token 速度统计
        print(f"Token Speed: {token_speed:.2f} tokens/sec")
        print(f"Total Time: {total_time:.2f} seconds")
        print(f"Total Tokens: {total_tokens}")
        print(f"Average Speed: {avg_speed:.2f} tokens/sec\n")
