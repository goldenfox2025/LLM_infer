#!/usr/bin/env python3
"""
测试脚本：固定输入"hi"来对比CUDA推理和图推理
"""

import sys
import os
sys.path.append('.')

import model_bridge

def test_with_hi():
    """使用固定输入"hi"进行测试"""
    
    # 初始化模型
    config_path = "models/qwen2.5-1.5b-instruct/config.json"
    weights_path = "models/qwen2.5-1.5b-instruct"
    model_type = "qwen_bf16"
    
    print("初始化模型...")
    model_bridge.init_model(config_path, weights_path, model_type)
    
    # 固定输入"hi"
    input_text = "hi"
    print(f"输入文本: '{input_text}'")
    
    # 生成回复
    def callback(token_id):
        print(f"生成token: {token_id}")
    
    # 将文本转换为token IDs（这里简化处理，实际应该用tokenizer）
    # 假设"hi"对应的token ID，这里用一个示例值
    input_ids = [6151]  # 这是"hi"的大概token ID，具体需要查tokenizer
    
    print("开始生成...")
    model_bridge.generate_text_stream(
        input_ids=input_ids,
        callback=callback,
        max_length=10,  # 只生成几个token
        temperature=1.0,
        top_p=0.9,
        top_k=50
    )
    
    print("生成完成！")
    print("检查生成的文件:")
    print("- cuda/debug_cuda_input_token.bin")
    print("- cuda/debug_cuda_embedding.bin")
    print("- 其他cuda/debug_cuda_*.bin文件")

if __name__ == "__main__":
    test_with_hi()
