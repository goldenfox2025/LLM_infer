import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

def main():
    model_name = "/home/LLM_infer/models/Qwen2.5-1.5B"

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
        {"role": "system", "content": "You are a helpful assistant."},
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
        max_new_tokens=200,
        temperature=1,  # 温度参数，值越低生成越确定；值越高生成更多样化
        top_k=40,         # top-k 策略，只从概率最高的 40 个 token 中采样
        streamer=streamer,
    )

    elapsed_time = time.time() - start_time
    
    # 截取生成部分的 token，用于计算速度
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # 计算生成速度和总token数
    
    speed = len(generated_ids[0]) / elapsed_time
    print(f"Token count: {len(generated_ids[0])}")
    print(f"\nSpeed: {speed:.2f} tokens/sec")
    print(f"Generation time: {elapsed_time:.2f} seconds")

    # # 打印模型结构
    # print("\nModel Structure:")
    # print(model)

if __name__ == "__main__":
    main()


