#!/usr/bin/env python3
import sys
import json
import threading
import queue
import time
import argparse
from pathlib import Path
from safetensors import safe_open
import torch

# 检查是否安装了可选依赖
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

# 检查是否安装了tokenizers库
TOKENIZERS_AVAILABLE = False
try:
    from tokenizers import Tokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    pass

# -------------------------------
# 模型加载相关函数
# -------------------------------

def load_llama_model(model_path: str):
    """加载Llama模型及配置"""
    model_path = Path(model_path)
    weights = {}
    
    # 加载模型权重
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
    
    return config, weights, "llama"

def load_qwen_model(model_path: str, keep_bf16=True):
    """加载Qwen模型及配置，可选保持BF16精度"""
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
    
    model_type = "qwen_bf16" if keep_bf16 else "qwen"
    return config, weights, model_type

def load_tokenizer(model_path: str, model_type: str):
    """根据模型类型加载对应的tokenizer"""
    model_path = Path(model_path)
    
    if model_type.startswith("qwen"):
        if not TRANSFORMERS_AVAILABLE:
            print("Error: transformers library required for Qwen models")
            exit(1)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"Qwen tokenizer loaded from: {model_path}")
    else:  # llama
        if not TOKENIZERS_AVAILABLE:
            print("Error: tokenizers library required for Llama models")
            exit(1)
        tokenizer_path = model_path / "tokenizer.json"
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        print(f"Llama tokenizer loaded from: {tokenizer_path}")
    
    return tokenizer

# -------------------------------
# 回调函数：累积 token 并输出文本差量及速度信息
# -------------------------------
def create_callback(tokenizer, q: queue.Queue, model_type: str):
    """创建用于处理生成 token 的回调函数，使用单调时钟计算时间"""
    accumulated_tokens = []
    last_output = ""  # 记录上一次完整解码后的文本
    start_time = None  # 第一个 token 到来时初始化
    last_token_time = None  # 用于计算 token 生成速度
    total_tokens = 0  # 记录生成的 token 总数

    def token_callback(token):
        nonlocal last_output, last_token_time, total_tokens, start_time

        current_time = time.monotonic()
        # 第一个 token 到来时初始化 start_time
        if start_time is None:
            start_time = current_time

        # 计算瞬时 token 生成速度
        speed = 0.0
        if last_token_time is not None:
            delta = current_time - last_token_time
            speed = 1.0 / delta if delta > 0 else 0.0
        last_token_time = current_time

        accumulated_tokens.append(token)
        total_tokens += 1

        # 解码 token 得到新文本
        new_text = tokenizer.decode(accumulated_tokens)
        diff = new_text[len(last_output):]
        last_output = new_text

        total_time = current_time - start_time
        avg_speed = total_tokens / total_time if total_time > 0 else 0.0

        if diff:
            q.put(json.dumps({
                "diff": diff,
                "speed": speed,
                "total_time": total_time,
                "total_tokens": total_tokens,
                "avg_speed": avg_speed
            }))

    return token_callback


# -------------------------------
# 终端聊天实现
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description='LLaMA/Qwen 模型聊天')
    parser.add_argument('--model_path', type=str, default="../models/Qwen2.5-1.5B-Instruct", help='模型路径')
    parser.add_argument('--model_type', type=str, default="qwen", choices=['llama', 'qwen', 'qwen_bf16'], help='模型类型')
    parser.add_argument('--system_prompt', type=str, default="You are a helpful AI assistant.", help='系统提示词')
    parser.add_argument('--max_length', type=int, default=2048, help='生成文本的最大长度')
    parser.add_argument('--temperature', type=float, default=0.7, help='生成温度')
    parser.add_argument('--top_p', type=float, default=0.9, help='top-p采样阈值')
    parser.add_argument('--top_k', type=int, default=50, help='top-k采样阈值')
    args = parser.parse_args()

    # 加载模型和tokenizer
    if args.model_type == "llama":
        config, weights, model_type = load_llama_model(args.model_path)
    elif args.model_type in ["qwen", "qwen_bf16"]:
        config, weights, model_type = load_qwen_model(args.model_path, keep_bf16=args.model_type == "qwen_bf16")
    else:
        print(f"Unsupported model type: {args.model_type}")
        exit(1)
    
    tokenizer = load_tokenizer(args.model_path, model_type)

    # 计算并打印模型大小
    total_params = sum(t.numel() for t in weights.values())
    total_bytes = sum(t.element_size() * t.numel() for t in weights.values())
    print("\nModel size: {} parameters, {:.2f} MB".format(total_params, total_bytes / (1024 * 1024)))

    # 打印模型配置信息
    print("\nModel Configuration:")
    print(f"Model Type: {model_type}")
    if model_type.startswith("qwen"):
        precision = "BF16" if model_type == "qwen_bf16" else "FP32"
        print(f"Precision: {precision}")
    print(f"Hidden Size: {config['hidden_size']}")
    print(f"Num Attention Heads: {config['num_attention_heads']}")
    print(f"Num Key Value Heads: {config['num_key_value_heads']}")
    print(f"Head Dimension: {config['hidden_size'] // config['num_attention_heads']}")

    # 导入C++模型桥接接口
    sys.path.append("/home/LLM_infer/build")
    from model_bridge import init_model, generate_text_stream

    # 初始化模型
    if not init_model(config, weights, model_type):
        print("Model initialization failed.", file=sys.stderr)
        exit(1)
    print("\nModel initialized successfully.\n")

    # 记录是否是首次对话
    first_chat = True
    
    print("聊天已启动。输入'quit'或'exit'退出。\n")
    
    while True:
        user_message = input("User: ").strip()
        if user_message.lower() in {"quit", "exit"}:
            break
        if not user_message:
            continue

        # 根据模型类型构造对话文本 - 只传输当前消息，不传历史记录（kvcache由本地维护）
        if model_type.startswith("qwen"):
            if first_chat:
                # 首次对话，包含系统提示词
                messages = [
                    {"role": "system", "content": args.system_prompt},
                    {"role": "user", "content": user_message}
                ]
                first_chat = False
            else:
                # 后续对话只包含用户消息
                messages = [
                    {"role": "user", "content": user_message}
                ]
            
            # 使用 apply_chat_template 应用聊天模板
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 对文本进行编码
            model_inputs = tokenizer([text], return_tensors="pt")
            input_ids = model_inputs["input_ids"][0].tolist()
        else:  # llama
            if first_chat:
                # 首次对话包含系统提示词
                conversation = f"<|system|>\n{args.system_prompt}</s><|user|>\n\n{user_message}</s>\n<|assistant|>\n"
                first_chat = False
            else:
                # 后续对话只包含用户消息
                conversation = f"{user_message}</s>\n:<|assistant|>\n"
            
            # 对文本进行编码 - 处理不同类型的tokenizer
            if isinstance(tokenizer, Tokenizer):  # tokenizers库的Tokenizer
                encoded = tokenizer.encode(conversation)
                input_ids = encoded.ids
            else:  # transformers库的tokenizer
                model_inputs = tokenizer([conversation], return_tensors="pt")
                input_ids = model_inputs["input_ids"][0].tolist()
        
        # 生成回复
        q = queue.Queue()
        callback = create_callback(tokenizer, q, model_type)
        
        # 启动生成线程
        def run_generation():
            from model_bridge import generate_text_stream
            generate_text_stream(
                input_ids,
                callback,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k
            )
            # 生成结束后，发送特殊标记
            q.put(None)
        
        thread = threading.Thread(target=run_generation)
        thread.start()
        
        print("Assistant: ", end="", flush=True)
        assistant_response = ""
        token_speed = 0.0
        total_time = 0.0
        total_tokens = 0
        avg_speed = 0.0
        
        # 处理生成的回复
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
                assistant_response += data["diff"]
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

if __name__ == "__main__":
    main()
