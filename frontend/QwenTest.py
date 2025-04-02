# # import time
# # import torch
# # from transformers import AutoModelForCausalLM, AutoTokenizer

# # def print_tensor_info(name, tensor):
# #     """打印张量的基本信息"""
# #     if isinstance(tensor, tuple):
# #         print(f"{name} is a tuple with {len(tensor)} elements")
# #         for i, t in enumerate(tensor):
# #             if isinstance(t, torch.Tensor):
# #                 print(f"  - Element {i}: shape {t.shape}, dtype {t.dtype}")
# #             else:
# #                 print(f"  - Element {i}: {type(t)}")
# #     elif isinstance(tensor, torch.Tensor):
# #         print(f"{name}: shape {tensor.shape}, dtype {tensor.dtype}")
# #         # 打印前几个值作为样本
# #         if tensor.numel() > 0:
# #             flat_tensor = tensor.view(-1)
# #             sample = flat_tensor[:min(5, flat_tensor.numel())]
# #             print(f"  Sample values: {sample}")
# #     else:
# #         print(f"{name}: {type(tensor)}")

# # def inspect_layer(model, layer_idx=15):
# #     """详细检查模型的特定层"""
# #     print(f"\n==== DETAILED INSPECTION OF LAYER {layer_idx} ====")
    
# #     # 获取目标层
# #     layer = model.model.layers[layer_idx]
    
# #     # 1. 检查层中的模块
# #     print("\n=== MODULES ===")
# #     print(f"Layer structure: {layer}")
    
# #     # 2. 检查权重维度
# #     print("\n=== WEIGHT DIMENSIONS ===")
# #     # 检查自注意力权重
# #     attn = layer.self_attn
# #     for name, param in attn.named_parameters():
# #         print(f"self_attn.{name}: {param.shape}, {param.dtype}")
        
# #     # 检查MLP权重
# #     mlp = layer.mlp
# #     for name, param in mlp.named_parameters():
# #         print(f"mlp.{name}: {param.shape}, {param.dtype}")
    
# #     # 检查LayerNorm权重
# #     for name, param in layer.input_layernorm.named_parameters():
# #         print(f"input_layernorm.{name}: {param.shape}, {param.dtype}")
    
# #     for name, param in layer.post_attention_layernorm.named_parameters():
# #         print(f"post_attention_layernorm.{name}: {param.shape}, {param.dtype}")
    
# #     # 3. 检查旋转位置编码
# #     print("\n=== ROTARY POSITION EMBEDDING ===")
# #     rotary_emb = model.model.rotary_emb
# #     print(f"RoPE module: {rotary_emb}")
# #     if hasattr(rotary_emb, 'inv_freq'):
# #         print(f"inv_freq: {rotary_emb.inv_freq.shape}, {rotary_emb.inv_freq.dtype}")
# #         print(f"Sample inv_freq: {rotary_emb.inv_freq[:5]}")
    
# #     # 打印RoPE的属性
# #     for name in dir(rotary_emb):
# #         if not name.startswith('__') and not callable(getattr(rotary_emb, name)):
# #             attr = getattr(rotary_emb, name)
# #             if isinstance(attr, torch.Tensor):
# #                 print(f"rotary_emb.{name}: {attr.shape}, {attr.dtype}")
# #             elif not callable(attr):
# #                 print(f"rotary_emb.{name}: {attr}")

# # def trace_forward_computation(model, layer_idx=15):
# #     """追踪一层的前向计算流程"""
# #     hooks = []
# #     computation_trace = {}
    
# #     def make_forward_hook(name):
# #         def hook(module, inputs, outputs):
# #             computation_trace[name] = {
# #                 'inputs': [x.clone() if isinstance(x, torch.Tensor) else x for x in inputs],
# #                 'outputs': outputs.clone() if isinstance(outputs, torch.Tensor) else outputs
# #             }
# #         return hook
    
# #     # 为每个关键组件添加钩子
# #     layer = model.model.layers[layer_idx]
    
# #     # 规范化层
# #     hooks.append(layer.input_layernorm.register_forward_hook(
# #         make_forward_hook(f'layer_{layer_idx}.input_layernorm')))
# #     hooks.append(layer.post_attention_layernorm.register_forward_hook(
# #         make_forward_hook(f'layer_{layer_idx}.post_attention_layernorm')))
    
# #     # 注意力子层
# #     hooks.append(layer.self_attn.register_forward_hook(
# #         make_forward_hook(f'layer_{layer_idx}.self_attn')))
# #     hooks.append(layer.self_attn.q_proj.register_forward_hook(
# #         make_forward_hook(f'layer_{layer_idx}.self_attn.q_proj')))
# #     hooks.append(layer.self_attn.k_proj.register_forward_hook(
# #         make_forward_hook(f'layer_{layer_idx}.self_attn.k_proj')))
# #     hooks.append(layer.self_attn.v_proj.register_forward_hook(
# #         make_forward_hook(f'layer_{layer_idx}.self_attn.v_proj')))
# #     hooks.append(layer.self_attn.o_proj.register_forward_hook(
# #         make_forward_hook(f'layer_{layer_idx}.self_attn.o_proj')))
    
# #     # MLP子层
# #     hooks.append(layer.mlp.register_forward_hook(
# #         make_forward_hook(f'layer_{layer_idx}.mlp')))
# #     hooks.append(layer.mlp.gate_proj.register_forward_hook(
# #         make_forward_hook(f'layer_{layer_idx}.mlp.gate_proj')))
# #     hooks.append(layer.mlp.up_proj.register_forward_hook(
# #         make_forward_hook(f'layer_{layer_idx}.mlp.up_proj')))
# #     hooks.append(layer.mlp.down_proj.register_forward_hook(
# #         make_forward_hook(f'layer_{layer_idx}.mlp.down_proj')))
    
# #     # RoPE
# #     if hasattr(model.model, 'rotary_emb'):
# #         hooks.append(model.model.rotary_emb.register_forward_hook(
# #             make_forward_hook('rotary_emb')))
    
# #     return hooks, computation_trace

# # def print_computation_trace(trace):
# #     """打印计算追踪结果"""
# #     print("\n=== COMPUTATION TRACE ===")
# #     for name, data in trace.items():
# #         print(f"\n--- {name} ---")
        
# #         print("Inputs:")
# #         for i, inp in enumerate(data['inputs']):
# #             if isinstance(inp, torch.Tensor):
# #                 print(f"  Input {i}: shape {inp.shape}, dtype {inp.dtype}")
# #                 flat_inp = inp.view(-1)
# #                 sample = flat_inp[:min(5, flat_inp.numel())]
# #                 print(f"  Sample values: {sample}")
# #             else:
# #                 print(f"  Input {i}: {type(inp)}")
        
# #         outputs = data['outputs']
# #         print("Outputs:")
# #         if isinstance(outputs, torch.Tensor):
# #             print(f"  shape {outputs.shape}, dtype {outputs.dtype}")
# #             flat_out = outputs.view(-1)
# #             sample = flat_out[:min(5, flat_out.numel())]
# #             print(f"  Sample values: {sample}")
# #         elif isinstance(outputs, tuple):
# #             print(f"  is a tuple with {len(outputs)} elements")
# #             for i, out in enumerate(outputs):
# #                 if isinstance(out, torch.Tensor):
# #                     print(f"    Element {i}: shape {out.shape}, dtype {out.dtype}")
# #                     flat_out = out.view(-1)
# #                     sample = flat_out[:min(5, flat_out.numel())]
# #                     print(f"    Sample values: {sample}")
# #                 else:
# #                     print(f"    Element {i}: {type(out)}")
# #         else:
# #             print(f"  {type(outputs)}")

# # def main():
# #     model_name = "/home/LLM_infer/models/Qwen2.5-1.5B-Instruct"

# #     # 加载模型和 tokenizer
# #     print("Loading model and tokenizer...")
# #     model = AutoModelForCausalLM.from_pretrained(
# #         model_name,
# #         torch_dtype="auto",
# #         device_map="auto"
# #     )
# #     tokenizer = AutoTokenizer.from_pretrained(model_name)

# #     # 打印模型结构
# #     print("\nModel Structure Overview:")
# #     print(model)
    
# #     # 检查某一层的详细信息
# #     inspect_layer(model, layer_idx=15)
    
# #     # 添加前向计算流程追踪钩子
# #     hooks, computation_trace = trace_forward_computation(model, layer_idx=15)
    
# #     # 准备测试用 prompt (更短以减少输出)
# #     prompt = "你好"
# #     messages = [
# #         {"role": "system", "content": "You are Qwen."},
# #         {"role": "user", "content": prompt},
# #     ]
# #     text = tokenizer.apply_chat_template(
# #         messages,
# #         tokenize=False,
# #         add_generation_prompt=True,
# #     )
# #     model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# #     # 运行生成测试，只生成1个token以便分析计算流程
# #     print("\nRunning forward pass to track computation flow...")
# #     with torch.no_grad():
# #         generated_ids = model.generate(
# #             **model_inputs,
# #             max_new_tokens=1,  # 只生成1个token
# #         )
    
# #     # 分析和打印计算流程
# #     print_computation_trace(computation_trace)
    
# #     # 移除所有钩子
# #     for hook in hooks:
# #         hook.remove()
    
# #     # 打印生成的文本
# #     generated_ids = [
# #         output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# #     ]
# #     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# #     print("\nResponse:")
# #     print(response)

# # if __name__ == "__main__":
# #     main()
# import time
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer

# def main():
#     model_name = "/home/LLM_infer/models/Qwen2.5-1.5B-Instruct"

#     # 加载模型和 tokenizer
#     print("Loading model and tokenizer...")
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype="auto",
#         device_map="auto"
#     )
#     tokenizer = AutoTokenizer.from_pretrained(model_name)

#     # 准备测试用 prompt
#     prompt = "讲个故事。"
#     messages = [
#         {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#         {"role": "user", "content": prompt},
#     ]
#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True,
#     )
#     model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

#     # 运行生成测试，并测量生成时间
#     print("Running generation test...")
#     start_time = time.time()
#     generated_ids = model.generate(
#         **model_inputs,
#         max_new_tokens=512,
#     )
#     elapsed_time = time.time() - start_time
#     # 截取生成部分的 token
#     generated_ids = [
#         output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#     ]
#     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     print("Response:")
#     print(response)
#     # 计算速度
#     speed = len(generated_ids[0]) / elapsed_time
#     print(f"Speed: {speed:.2f} tokens/sec")
#     print(f"Generation time: {elapsed_time:.2f} seconds")

#     # 打印模型结构
#     print("\nModel Structure:")
#     print(model)

# if __name__ == "__main__":
#     main()

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

def main():
    model_name = "/home/LLM_infer/models/Qwen2.5-1.5B-Instruct"

    # 加载模型和 tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 准备测试用 prompt
    prompt = "讲个故事"
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    # 计算输出总token及其时间
    import time

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 创建流式输出 streamer

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 运行生成测试，并测量生成时间
 
    start_time = time.time()
    print("Running generation test (streaming)...")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7,  # 温度参数，值越低生成越确定；值越高生成更多样化
        top_k=40,         # top-k 策略，只从概率最高的 40 个 token 中采样
        streamer=streamer,
    )

    elapsed_time = time.time() - start_time
    
    # 截取生成部分的 token，用于计算速度
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # 计算生成速度
    speed = len(generated_ids[0]) / elapsed_time
    print(f"\nSpeed: {speed:.2f} tokens/sec")
    print(f"Generation time: {elapsed_time:.2f} seconds")

    # # 打印模型结构
    # print("\nModel Structure:")
    # print(model)

if __name__ == "__main__":
    main()