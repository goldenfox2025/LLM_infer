#!/usr/bin/env python3
import sys
import json
import threading
import queue
import time
from pathlib import Path
from safetensors import safe_open
from tokenizers import Tokenizer  # pip install tokenizers
import torch

from flask import Flask, request, jsonify, Response, stream_with_context

# -------------------------------
# 模型加载相关函数
# -------------------------------

def load_model(model_path: str):
    model_path = Path(model_path)
    weights = {}
    with safe_open(model_path / "model.safetensors", framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            # 如果是 bfloat16，则先转换为 float32
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            weights[key] = tensor
            print(f"Loaded tensor {key} with shape {weights[key].shape}")
    
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
# Flask WebUI 部分
# -------------------------------

app = Flask(__name__)

# 指定模型路径（请根据实际情况修改）
MODEL_PATH = "/home/LLM_infer/models/chat"
config, weights = load_model(MODEL_PATH)
tokenizer = load_tokenizer(MODEL_PATH)

# 添加模块路径，也请根据实际情况修改
sys.path.append("/home/LLM_infer/build")
from model_bridge import init_model, generate_text_stream

# 打印一下，方便对比
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

# 初始化模型
if not init_model(config, weights):
    print("Model initialization failed.", file=sys.stderr)
    exit(1)
print("\nModel initialized successfully.")

# -------------------------------
# 回调函数：累积 token 并输出转换后的文本差量及生成速度
# 当 token == -1 时，发送特殊标记给前端，通知禁用输入，并提示用户点击重启按钮
# -------------------------------

def create_callback(tokenizer, q: queue.Queue):
    accumulated_tokens = []
    last_output = ""  # 记录上一次完整解码后的文本
    last_token_time = None  # 用于计算 token 生成速度

    def token_callback(token):
        nonlocal last_output, last_token_time

        # 如果 token 为 -1，则说明生成过程中出现错误或需要中断
        # if token == -1:
        #     q.put(json.dumps({
        #         "disable_input": True,
        #         "restart": True,
        #         "diff": "<br><b>Generation stopped. Please restart the conversation.</b>"
        #     }))
        #     return

        current_time = time.time()
        speed = None
        if last_token_time is not None:
            delta = current_time - last_token_time
            # 避免极端情况
            speed = 1.0 / delta if delta > 0 else 0.0
        last_token_time = current_time

        accumulated_tokens.append(token)
        new_text = tokenizer.decode(accumulated_tokens)
        diff = new_text[len(last_output):]
        last_output = new_text
        if diff:
            # 替换换行符为 <br> 用于 HTML 显示
            diff = diff.replace("\n", "<br>")
            # 将文本差量和生成速度一起放入队列（用 JSON 封装）
            q.put(json.dumps({"diff": diff, "speed": speed if speed is not None else 0.0}))
    return token_callback

# -------------------------------
# Web API 接口
# -------------------------------

@app.route("/")
def index():

    return """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>Chat Web UI</title>
      <style>
        body {
          background: #f4f7f9;
          font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
          margin: 0;
          padding: 0;
        }
        .container {
          max-width: 800px;
          margin: 40px auto;
          background: #fff;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
          border-radius: 8px;
          overflow: hidden;
        }
        .header {
          background: #4A90E2;
          color: #fff;
          padding: 20px;
          text-align: center;
          font-size: 1.5em;
        }
        .chat-area {
          padding: 20px;
          height: 500px;
          overflow-y: auto;
          border-bottom: 1px solid #eee;
        }
        .parameters {
          padding: 10px 20px;
          background: #fafafa;
          display: flex;
          gap: 10px;
          align-items: center;
        }
        .input-area {
          display: flex;
          padding: 10px 20px;
          background: #fafafa;
        }
        .input-area input[type="text"] {
          flex: 1;
          padding: 10px;
          font-size: 1em;
          border: 1px solid #ccc;
          border-radius: 4px;
        }
        .input-area button {
          margin-left: 10px;
          padding: 10px 20px;
          background: #4A90E2;
          border: none;
          color: #fff;
          border-radius: 4px;
          cursor: pointer;
          font-size: 1em;
        }
        .input-area button:hover {
          background: #3a78c2;
        }
        .message {
          margin-bottom: 15px;
          line-height: 1.5;
        }
        .message.user {
          text-align: right;
        }
        .message.user .bubble {
          background: #4A90E2;
          color: #fff;
          display: inline-block;
          padding: 10px 15px;
          border-radius: 15px 15px 0 15px;
        }
        .message.assistant .bubble {
          background: #e5e5ea;
          color: #000;
          display: inline-block;
          padding: 10px 15px;
          border-radius: 15px 15px 15px 0;
        }
      </style>
    </head>
    <body>
      <div class="container">
        <div class="header">Chat with the Assistant</div>
        <div id="chat" class="chat-area"></div>
        <div id="token_speed" style="padding: 10px 20px; font-size: 0.9em; color: #666;">Token Speed: N/A</div>
        <!-- Restart Chat 按钮，默认隐藏 -->
        <div style="padding: 0 20px;">
          <button id="restart_button" style="display:none; margin: 10px 0;" onclick="restartChat()">Restart Chat</button>
        </div>
        <div class="parameters">
          <label>Temperature: <input type="number" step="0.1" id="temperature" value="0.7"></label>
          <label>TopK: <input type="number" id="topk" value="10"></label>
          <label>TopP: <input type="number" step="0.1" id="topp" value="1"></label>
        </div>
        <div class="input-area">
          <input type="text" id="user_input" placeholder="Type your message" />
          <button onclick="sendMessage()">Send</button>
        </div>
      </div>
      <script>
        function addMessage(cls, html) {
          const chat = document.getElementById('chat');
          const div = document.createElement('div');
          div.className = 'message ' + cls;
          const bubble = document.createElement('div');
          bubble.className = 'bubble';
          bubble.innerHTML = html;
          div.appendChild(bubble);
          chat.appendChild(div);
          chat.scrollTop = chat.scrollHeight;
          return bubble;
        }
        
        async function sendMessage() {
          const input = document.getElementById('user_input');
          const message = input.value.trim();
          if (!message) return;
          addMessage("user", message);
          input.value = '';
          
          // 新建一个容器用于助手回复
          const replyContainer = addMessage("assistant", "");
          
          // 获取用户设置的参数
          const temperature = document.getElementById('temperature').value;
          const topk = document.getElementById('topk').value;
          const topp = document.getElementById('topp').value;
          
          // 使用 EventSource 连接流式接口，并将参数传递过去
          const evtSource = new EventSource("/chat_stream?message=" + encodeURIComponent(message) +
                                             "&temperature=" + encodeURIComponent(temperature) +
                                             "&topk=" + encodeURIComponent(topk) +
                                             "&topp=" + encodeURIComponent(topp));
          evtSource.onmessage = function(event) {
            try {
              const data = JSON.parse(event.data);
              // 如果收到禁用输入的标记，则禁用所有输入，并显示重启按钮
              if (data.disable_input) {
                document.getElementById('user_input').disabled = true;
                document.getElementById('temperature').disabled = true;
                document.getElementById('topk').disabled = true;
                document.getElementById('topp').disabled = true;
                document.querySelector('.input-area button').disabled = true;
                document.getElementById('restart_button').style.display = "inline-block";
              }
              if (data.diff) {
                replyContainer.innerHTML += data.diff;
              }
              if (data.speed !== undefined && data.speed !== null) {
                document.getElementById('token_speed').innerText = "Token Speed: " + data.speed.toFixed(2) + " tokens/sec";
              }
            } catch (e) {
              console.error("Failed to parse event data:", event.data);
            }
          };
          evtSource.onerror = function(err) {
            console.error("EventSource failed:", err);
            evtSource.close();
          };
        }
        
        // 重启对话的函数：调用后端 /restart 接口，清空对话记录并重新启用输入
        function restartChat() {
          const restartBtn = document.getElementById('restart_button');
          restartBtn.disabled = true;
          fetch('/restart', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
               if (data.status === "ok") {
                   // 清空对话区域
                   document.getElementById('chat').innerHTML = "";
                   document.getElementById('token_speed').innerText = "Token Speed: N/A";
                   // 重新启用输入控件
                   document.getElementById('user_input').disabled = false;
                   document.getElementById('temperature').disabled = false;
                   document.getElementById('topk').disabled = false;
                   document.getElementById('topp').disabled = false;
                   document.querySelector('.input-area button').disabled = false;
                   // 隐藏重启按钮
                   restartBtn.style.display = "none";
               } else {
                   alert("Restart failed: " + data.message);
               }
               restartBtn.disabled = false;
            })
            .catch(err => {
               console.error("Error restarting chat:", err);
               restartBtn.disabled = false;
            });
        }
      </script>
    </body>
    </html>
    """

@app.route("/chat_stream")
def chat_stream():
    user_message = request.args.get("message", "")
    if not user_message:
        return jsonify({"reply": ""})
    
    # 获取用户传入的参数，若未传则使用默认值
    try:
        temperature = float(request.args.get("temperature", "0.7"))
    except ValueError:
        temperature = 0.7
    try:
        topk = int(request.args.get("topk", "10"))
    except ValueError:
        topk = 10
    try:
        topp = float(request.args.get("topp", "1"))
    except ValueError:
        topp = 1.0
    
    # 初始化对话历史：包含系统提示和用户输入
    conversation = "<|im_start|>system\nYou're a helpful assistant.\n<|im_end|>\n"
    conversation += f"<|im_start|>user\n{user_message}\n<|im_end|>\n"
    conversation += "<|im_start|>assistant\n"
    
    # 编码对话
    encoded = tokenizer.encode(conversation)
    input_ids = encoded.ids

    # 使用队列在线程和主线程间传递转换后的文本片段及速度信息
    q = queue.Queue()
    # 创建回调函数，传入 tokenizer 和队列
    token_callback = create_callback(tokenizer, q)
    
    # 后台线程调用生成函数
    def run_generation():
        generate_text_stream(
            input_ids,
            token_callback,
            max_length=100,
            temperature=temperature,
            top_p=topp,
            top_k=topk
        )
        # 生成结束后放一个特殊标记，表示结束
        q.put(None)
    
    thread = threading.Thread(target=run_generation)
    thread.start()
    
    @stream_with_context
    def event_stream():
        while True:
            msg = q.get()
            if msg is None:
                break
            # SSE 格式要求每条消息以 "data: " 开头，并以 "\n\n" 结束
            yield f"data: {msg}\n\n"
    
    return Response(event_stream(), mimetype="text/event-stream")

# -------------------------------
# 重启对话接口：重新实例化 model、tokenizer 及推理引擎
# -------------------------------
@app.route("/restart", methods=["POST"])
def restart():
    global config, weights, tokenizer
    try:
        config, weights = load_model(MODEL_PATH)
        tokenizer = load_tokenizer(MODEL_PATH)
        from model_bridge import init_model  # 重新加载推理引擎
        if not init_model(config, weights):
            return jsonify({"status": "failed", "message": "Model initialization failed."}), 500
    except Exception as e:
        return jsonify({"status": "failed", "message": str(e)}), 500
    return jsonify({"status": "ok", "message": "Model and chat restarted."})

if __name__ == "__main__":
    # 启动 Flask 服务
    app.run(host="0.0.0.0", port=5000, debug=True)
