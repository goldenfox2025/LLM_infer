#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用AutoAWQ量化Qwen2.5-1.5B模型，并检查GEMV和GEMM格式的张量形状
"""

import torch
import os
import shutil
from transformers import AutoTokenizer
# 确保已安装: pip install autoawq
try:
    from awq import AutoAWQForCausalLM
    # 导入 AutoAWQ 的量化线性层类
    from awq.modules.linear import WQLinear_GEMM, WQLinear_GEMV
except ImportError:
    print("错误: 请安装 autoawq (pip install autoawq)")
    print("可能还需要: pip install torch transformers accelerate safetensors") # 补充依赖
    exit()
    


def main():
    # 模型路径
    model_path = "./models/Qwen3-4B"
    quant_path = "./models/Qwen3-4B-AWQ"

    # --- 删除旧目录 ---
    if os.path.exists(quant_path):
        print(f"检测到旧的量化目录: {quant_path}")
        try:
            shutil.rmtree(quant_path)
            print("已删除旧的量化目录。")
        except OSError as e:
            print(f"错误：无法删除旧目录 {quant_path}: {e}")
            print("请手动删除该目录后重试。")
            return

    # 加载分词器
    print(f"正在加载分词器...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"加载分词器失败: {e}")
        return

    print("进行新的量化...")
    os.makedirs(quant_path, exist_ok=True)

    # 加载原始模型
    print(f"正在加载原始模型: {model_path}")
    try:
        model = AutoAWQForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)
    except Exception as e:
         print(f"加载原始模型失败: {e}")
         return

    # 准备校准数据 (与之前相同)
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
    ] * 16
    print(f"准备了{len(samples)}个校准样本")

    # --- 量化配置修改 ---
    print("设置量化配置 ")
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMV"  
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
        model.save_quantized(quant_path)
        tokenizer.save_pretrained(quant_path)
        print(f"量化模型已尝试保存在: {os.path.abspath(quant_path)}")
    except Exception as e:
        print(f"保存模型时出错: {e}")
        raise

    # --- 关键步骤：加载量化模型并检查内部形状 ---
    print("\n加载新量化的模型以检查内部张量形状...")
    try:
        model_quant = AutoAWQForCausalLM.from_quantized(quant_path, device_map="auto", trust_remote_code=True)
        print("量化模型加载成功！现在开始检查层...")

        print("\n--- 开始检查 Quantized Layers (寻找 WQLinear_GEMV 和 WQLinear_GEMM) ---")
        found_gemv = 0
        found_gemm = 0
        max_layers_to_print_each = 5 # 每种类型最多打印几个

        for name, module in model_quant.named_modules():
            # 检查 GEMV 层
            if isinstance(module, WQLinear_GEMV):
                found_gemv += 1
                if found_gemv <= max_layers_to_print_each:
                    print(f"\nLayer (Type: WQLinear_GEMV): {name}")
                    print(f"  qweight shape: {module.qweight.shape} (dtype: {module.qweight.dtype})")
                    print(f"  scales shape: {module.scales.shape} (dtype: {module.scales.dtype})")
                    print(f"  qzeros shape: {module.qzeros.shape} (dtype: {module.qzeros.dtype})")
                    if hasattr(module, 'bias') and module.bias is not None:
                        print(f"  bias shape: {module.bias.shape} (dtype: {module.bias.dtype})")
                    else:
                        print(f"  bias: None")
            # 检查 GEMM 层
            elif isinstance(module, WQLinear_GEMM):
                 found_gemm += 1
                 if found_gemm <= max_layers_to_print_each:
                     print(f"\nLayer (Type: WQLinear_GEMM): {name}") # 明确类型
                     # --- 添加打印 GEMM 层形状的代码 ---
                     print(f"  qweight shape: {module.qweight.shape} (dtype: {module.qweight.dtype})")
                     print(f"  scales shape: {module.scales.shape} (dtype: {module.scales.dtype})")
                     print(f"  qzeros shape: {module.qzeros.shape} (dtype: {module.qzeros.dtype})")
                     if hasattr(module, 'bias') and module.bias is not None:
                         print(f"  bias shape: {module.bias.shape} (dtype: {module.bias.dtype})")
                     else:
                         print(f"  bias: None")
                     # ------------------------------------

        if found_gemv == 0 and found_gemm == 0:
            print("\n警告：在加载的模型中未找到 WQLinear_GEMV 或 WQLinear_GEMM 层。")
            print("请确认 AutoAWQ 版本、量化过程是否成功替换了线性层。")
        else:
            print(f"\n总共发现 {found_gemv} 个 WQLinear_GEMV 层 (输出了前 {min(found_gemv, max_layers_to_print_each)} 个).")
            print(f"总共发现 {found_gemm} 个 WQLinear_GEMM 层 (输出了前 {min(found_gemm, max_layers_to_print_each)} 个).")

        print("--- 检查完毕 ---")

    except Exception as e:
        print(f"\n加载或检查新量化模型失败: {e}")
        print("请检查量化或保存过程是否真的成功，以及 AutoAWQ 是否安装完好。")
        return

    # 测试生成 (保持不变)
    prompt = "讲个关于机器人学习绘画的短故事。"
    print(f"\n测试提示: {prompt}")
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    try:
        # 尝试获取设备，增加健壮性
        device = "cuda:0" # Default
        if hasattr(model_quant, 'device'):
            device = model_quant.device
        elif hasattr(model_quant, 'hf_device_map'): # 处理 device_map 情况
             # 尝试从 device_map 获取一个设备，通常第一个即可代表主要设备
             device_list = list(set(model_quant.hf_device_map.values()))
             if device_list:
                 device = device_list[0]
        print(f"模型将被发送到设备: {device}")
    except Exception as e:
        print(f"获取模型设备时出错: {e}, 回退到 cpu")
        device = "cpu"


    inputs = tokenizer(text, return_tensors="pt").to(device)
    print(f"输入已发送到设备: {inputs.input_ids.device}")


    print("开始生成文本...")
    try:
        # 确保模型在正确的设备上
        model_quant.to(device)

        with torch.no_grad():
            # 查找 pad_token_id，如果 tokenizer 没有，使用 eos_token_id
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

            outputs = model_quant.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=pad_token_id, # 使用找到的 pad_token_id
                eos_token_id=tokenizer.eos_token_id
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"生成成功！生成的文本:\n{generated_text}")
    except Exception as e:
        print(f"使用新量化模型生成文本时出错: {e}")
        # 打印更详细的错误信息
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()