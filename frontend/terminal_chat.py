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


def load_model(model_path: str):
    model_path = Path(model_path)
    weights = {}
    # 加载模型权重（假设保存在 model.safetensors 中）
    with safe_open(model_path / "model.safetensors", framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            # 如果是 bfloat16，则转换为 float32
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            weights[key] = tensor
            print(f"Loaded tensor {key} with shape {weights[key].shape}")
    
    # 加载配置文件
    with open(model_path / "config.json", 'r') as f:
        config = json.load(f)
        # 若缺少 embedding_table，则采用 lm_head 权重
        if "model.embed_tokens.weight" not in weights:
            config["tie_word_embeddings"] = True
            weights["model.embed_tokens.weight"] = weights["lm_head.weight"]
        print("Config loaded:", config)
        
    return config, weights

def load_tokenizer(model_path: str):
    """
    假设模型文件夹下存在 tokenizer.json 文件
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
    # 指定模型路径（请根据实际情况修改）
    MODEL_PATH = "/home/LLM_infer/models/tinyllama"
    config, weights = load_model(MODEL_PATH)
    tokenizer = load_tokenizer(MODEL_PATH)

    # 计算并打印模型大小（参数数量和内存占用）
    total_params = sum(t.numel() for t in weights.values())
    total_bytes = sum(t.element_size() * t.numel() for t in weights.values())
    print("\nModel size: {} parameters, {:.2f} MB".format(total_params, total_bytes / (1024 * 1024)))

    # 打印模型配置信息
    print("\nModel Configuration:")
    print(f"Hidden Size: {config['hidden_size']}")
    print(f"Num Attention Heads: {config['num_attention_heads']}")
    print(f"Num Key Value Heads: {config['num_key_value_heads']}")
    print(f"Head Dimension: {config['hidden_size'] // config['num_attention_heads']}")
    for layer in range(config.get('num_hidden_layers', 0)):
        print(f"\nLayer {layer} Attention Tensors:")
        q_proj = weights[f'model.layers.{layer}.self_attn.q_proj.weight']
        k_proj = weights[f'model.layers.{layer}.self_attn.k_proj.weight']
        v_proj = weights[f'model.layers.{layer}.self_attn.v_proj.weight']
        o_proj = weights[f'model.layers.{layer}.self_attn.o_proj.weight']
        print(f"Q projection shape: {q_proj.shape}")
        print(f"K projection shape: {k_proj.shape}")
        print(f"V projection shape: {v_proj.shape}")
        print(f"O projection shape: {o_proj.shape}")

    # 添加模型桥接模块路径（请根据实际情况修改）
    sys.path.append("/home/LLM_infer/build")
    from model_bridge import init_model, generate_text_stream

    # 初始化模型
    if not init_model(config, weights):
        print("Model initialization failed.", file=sys.stderr)
        exit(1)
    print("\nModel initialized successfully.\n")

    # 定义系统提示词，仅在第一次对话时使用
    system_prompt = "<|system|>\nYou are a friendly helper.</s>\n"
    first_chat = True

    print("Enter 'quit' to exit.\n")
    while True:
        user_message = input("User: ").strip()
        if user_message.lower() in {"quit", "exit"}:
            break
        if not user_message:
            continue

        # 构造对话文本
        if first_chat:
            # 第一次对话包含系统提示词
            conversation = system_prompt + f"<|user|>\n{user_message}</s>\n<|assistant|>\n"
            first_chat = False
        else:
            # 后续对话只包含用户输入
            conversation = f"<|user|>\n{user_message}</s>\n<|assistant|>\n"
        
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
                max_length=200,      # 根据需要调整生成最大 token 数
                temperature=0.7,     # 可调整温度
                top_p=1.0,
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
