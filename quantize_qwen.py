#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用AutoAWQ量化Qwen2.5-1.5B模型 (尝试移除 version='GEMM')
"""

import torch
import os
import shutil # 导入 shutil 用于删除旧目录
from transformers import AutoTokenizer
# 确保已安装: pip install autoawq
try:
    from awq import AutoAWQForCausalLM
except ImportError:
    print("错误: 请安装 autoawq (pip install autoawq)")
    exit()

def main():
    # 模型路径
    model_path = "/home/keta/code/LLM_infer/models/Qwen2.5-1.5B-Instruct"
    quant_path = "/home/keta/code/LLM_infer/quantization_test/Qwen2.5-1.5B-AWQ"

    # --- 重要：删除旧的量化结果以确保重新生成 ---
    if os.path.exists(quant_path):
        print(f"检测到旧的量化目录: {quant_path}")
        try:
            shutil.rmtree(quant_path)
            print("已删除旧的量化目录。")
        except OSError as e:
            print(f"错误：无法删除旧目录 {quant_path}: {e}")
            print("请手动删除该目录后重试。")
            return # 停止执行，防止覆盖不完整

    # 检查量化模型是否已存在 (理论上在上面已经被删除了)
    model_exists = False # 因为我们已经删除了

    # 加载分词器
    print(f"正在加载分词器...")
    # 总是从原始模型加载，因为量化模型目录刚被删除
    print(f"从原始模型路径加载分词器: {model_path}")
    try:
        # 某些模型 (如 Qwen) 可能需要 trust_remote_code=True
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"加载分词器失败: {e}")
        return


    print("进行新的量化...")
    # 创建输出目录
    os.makedirs(quant_path, exist_ok=True)

    # 加载原始模型
    print(f"正在加载原始模型: {model_path}")
    try:
        # 添加 trust_remote_code=True
        model = AutoAWQForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)
    except Exception as e:
         print(f"加载原始模型失败: {e}")
         return

    # 准备校准数据
    print("准备校准数据")
    print("使用本地校准数据")
    samples = [
        "人工智能是一种能够模拟人类智能的计算机系统，它可以学习、推理和自我完善。人工智能的应用范围非常广泛，包括自然语言处理、计算机视觉、机器人技术等多个领域。随着技术的不断发展，人工智能正在改变我们的生活和工作方式。",
        "量子计算利用量子力学原理，如叠加和纠缠，来处理信息，有望解决传统计算机难以解决的问题。量子计算机可以同时处理多种状态，这使得它们在某些特定任务上比传统计算机快得多。科学家们正在努力克服量子计算面临的技术挑战，如量子退相干和错误校正等问题。",
        "机器学习是人工智能的一个子领域，它使计算机系统能够从数据中学习，而无需明确编程。机器学习算法可以分为监督学习、无监督学习和强化学习等多种类型。这些算法已经在图像识别、语音识别、推荐系统等多个领域取得了显著成功。",
        "深度学习是机器学习的一种方法，它使用多层神经网络来提取数据中的高级特征。深度学习模型，如卷积神经网络和循环神经网络，已经在计算机视觉和自然语言处理等领域取得了突破性进展。这些模型需要大量的数据和计算资源来训练，但它们的性能通常超过传统的机器学习方法。",
        "自然语言处理是人工智能的一个分支，专注于使计算机理解、解释和生成人类语言。自然语言处理技术包括文本分类、情感分析、机器翻译和问答系统等。近年来，基于Transformer架构的大型语言模型，如GPT和BERT，极大地提高了自然语言处理的能力。",
        "计算机视觉是人工智能的一个领域，它使计算机能够从图像或视频中获取信息并理解视觉世界。计算机视觉技术包括图像分类、目标检测、图像分割和人脸识别等。这些技术已经在自动驾驶、医疗诊断和安全监控等领域得到了广泛应用。",
        "强化学习是一种机器学习方法，它通过与环境交互并从反馈中学习来优化决策。在强化学习中，智能体通过尝试不同的行动并观察结果来学习最佳策略。这种方法已经在游戏、机器人控制和资源管理等领域取得了成功。",
        "神经网络是一种受人脑启发的计算模型，由相互连接的节点（神经元）组成，用于模式识别和决策。神经网络可以学习复杂的非线性关系，这使它们在处理图像、语音和文本等数据时非常有效。深度神经网络包含多个隐藏层，能够学习数据的层次表示。"
    ] * 16  # 重复样本以增加数量

    print(f"准备了{len(samples)}个校准样本")

    # --- 量化配置修改 ---
    print("设置量化配置 (移除了 version='GEMM')")
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        # "version": "GEMM"  # <--- 移除或注释掉这一行
    }
    # ---------------------

    # 执行量化
    print("开始量化模型...")
    try:
        model.quantize(tokenizer, quant_config=quant_config, calib_data=samples)
    except Exception as e:
        print(f"量化过程中出错: {e}")
        raise

    # 保存量化模型
    print(f"保存量化模型到: {quant_path}")
    try:
        # 保存模型和分词器
        model.save_quantized(quant_path)
        tokenizer.save_pretrained(quant_path) # 再次保存以防万一

        # 简单的验证
        quant_model_file = os.path.join(quant_path, "model.safetensors") # AWQ 倾向于保存为 safetensors
        if not os.path.exists(quant_model_file):
             quant_model_file = os.path.join(quant_path, "pytorch_model.bin") # 检查备用名

        if os.path.exists(quant_model_file):
             print("量化模型权重文件已找到！")
        else:
             print("警告：未找到模型权重文件 (model.safetensors 或 pytorch_model.bin)，保存可能失败")

        if os.path.exists(os.path.join(quant_path, "tokenizer.json")):
             print("分词器文件已找到！")
        else:
             print("警告：未找到分词器文件 (tokenizer.json)，保存可能失败")

        print(f"量化模型保存在: {os.path.abspath(quant_path)}")
        print("下次可以直接从此路径加载量化模型")
    except Exception as e:
        print(f"保存模型时出错: {e}")
        raise

    # 加载量化模型进行测试 (可选，但推荐)
    print("加载新量化的模型进行测试...")
    try:
        # 再次加载以确认量化模型本身可用
        model_quant = AutoAWQForCausalLM.from_quantized(quant_path, device_map="auto", trust_remote_code=True)
        print("新量化模型加载成功！")
    except Exception as e:
        print(f"加载新量化模型失败: {e}")
        print("量化或保存过程可能仍有问题，请检查日志。")
        return # 如果加载失败，后续测试无意义

    # 测试生成 (可选，但推荐)
    prompt = "讲个关于机器人学习绘画的短故事。"
    print(f"\n测试提示: {prompt}")
    messages = [{"role": "user", "content": prompt}] # 简化消息格式
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    try:
        device = next(model_quant.parameters()).device
    except StopIteration: # 处理可能没有 parameters 的情况
         try:
             device = next(model_quant.buffers()).device
         except StopIteration:
             device = "cpu" # 默认回退到 CPU
             print("警告：无法确定模型设备，假设为 CPU")

    inputs = tokenizer(text, return_tensors="pt").to(device)
    print(f"输入已发送到设备: {inputs.input_ids.device}")


    print("开始生成文本...")
    try:
        with torch.no_grad():
            # 使用与之前相同的生成参数
            outputs = model_quant.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id # 明确指定 EOS token ID
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"生成成功！生成的文本:\n{generated_text}")
    except Exception as e:
        print(f"使用新量化模型生成文本时出错: {e}")


if __name__ == "__main__":
    main()