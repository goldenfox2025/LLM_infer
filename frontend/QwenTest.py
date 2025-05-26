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


    # 这般确实不错。那么问题来了。我们有问题其一：rope接受offset，offset却非写于固定内存。请你尝试于 @qwen.hpp  @qwen.cpp  ，让model维护一个固定内存，每次启动图之前，先行修改内存里的offset。这offset自然是参考 forward cuda——这个函数是完全正确的函数。目前图推理已经可行，所以现在是为正确考虑。问题其二：kvcache，是一个连续的大块内存，然后，每次计算时，由内核直接写入对应的区域。由于目前的matmul调用的是cublas（量化模式倒是可以调用自己的matmul，但那个暂时先不考虑，或者留出开发空间以后支持），目前来看，唯一的方法，是每次让matmul将数据写入到一个固定的内存上，然后，cudamemcpy再将其复制到kvcache对应的位置（可这似乎如果能用cutlass的话，是不是就更加方便？也就是直接通过索引来决定matmul写入的位置，但cublas还是没法这样干，看起来kbuf必须是独立的一个固定内存，而不能从kslice取得。不过修改至少要为此留出空间）。问题之三： @flash_attention_variable.cu  @gather_fa_variable.cu  这两个，接受kvcache1的一大块内存，然后根绝索引来分段处理。只是这种分段，或许不太好做，你看看如何让需要的信息处于一个固定的内存地址，然后新建一个flash decode for gprah cu，并且在  @cudaOP.cuh  里实现？当然，由于这个是根据长度，变动切分的段数，更为麻烦，你可以考虑 @gather_fa.cu  @flash_attention.cu  ，基于这两个修改。我给你的代码你都要读。问题你要分析如何解决。