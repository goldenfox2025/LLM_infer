import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    prompt = "讲个故事。"
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 运行生成测试，并测量生成时间
    print("Running generation test...")
    start_time = time.time()
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
    )
    elapsed_time = time.time() - start_time
    # 截取生成部分的 token
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("Response:")
    print(response)
    print(f"Generation time: {elapsed_time:.2f} seconds")

    # 打印模型结构
    print("\nModel Structure:")
    print(model)

if __name__ == "__main__":
    main()
