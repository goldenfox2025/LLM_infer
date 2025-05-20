#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
可视化目标模型和草稿模型的logits分布
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm # 虽然我们简化了setup，但导入以备不时之需
from matplotlib.gridspec import GridSpec
from transformers import AutoTokenizer
import re
import matplotlib # 导入matplotlib主模块

# --- 简化的字体设置 ---
# 1. 定义一个候选的中文字体名称列表 (确保这些名称是Matplotlib在正确重建缓存后能够识别的)
#    你需要通过之前的Python诊断脚本确认Matplotlib实际识别的准确名称。
#    常见的、如果你已安装fonts-noto-cjk或fonts-wqy-zenhei后，Matplotlib应该能识别的名称：
CJK_FONT_CANDIDATES = [
    'Noto Sans CJK SC',    # 谷歌Noto思源黑体简体中文 (推荐)
    'WenQuanYi Zen Hei',   # 文泉驿正黑
    'WenQuanYi Micro Hei', # 文泉驿微米黑
    'SimHei',              # 中易黑体 (部分Linux系统可能有)
    # 你可以根据你的Python诊断脚本输出，添加更多它识别的准确中文字体名
]

# 2. 获取Matplotlib当前识别的所有字体名称
try:
    available_fonts = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    # print("Matplotlib 识别的字体:", sorted(list(available_fonts))) # 调试时可以取消注释这行
except Exception as e:
    print(f"警告: 获取Matplotlib字体列表时出错: {e}。将尝试默认设置。")
    available_fonts = set()

# 3. 查找并设置第一个可用的中文字体
use_chinese = False
selected_font = None
for font_name in CJK_FONT_CANDIDATES:
    if font_name in available_fonts:
        selected_font = font_name
        break

if selected_font:
    try:
        plt.rcParams['font.family'] = 'sans-serif' # 必须先设置字体家族
        plt.rcParams['font.sans-serif'] = [selected_font] + plt.rcParams['font.sans-serif'] # 将找到的中文字体放到最前面
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
        use_chinese = True
        print(f"--- 中文字体设置成功: Matplotlib 将优先使用 '{selected_font}' ---")
        
        # 测试一下是否真的能用这个字体画中文 (可选的健全性检查)
        # fig_test, ax_test = plt.subplots(figsize=(1,0.5))
        # ax_test.text(0.5, 0.5, "你好", fontname=selected_font)
        # plt.close(fig_test)
        # print(f"用 '{selected_font}' 测试中文渲染似乎没有立即报错。")

    except Exception as e:
        print(f"尝试设置中文字体 '{selected_font}' 时发生错误: {e}")
        use_chinese = False
        # 如果出错，确保退回到一个基本状态
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] # Matplotlib通常自带
        plt.rcParams['axes.unicode_minus'] = False


if not use_chinese:
    print("--- 警告: 未能成功配置中文字体。图表中的中文可能无法正常显示，将尝试使用英文标签。---")
    # 确保有一个基础的回退，以防万一
    if 'DejaVu Sans' not in plt.rcParams['font.sans-serif']:
         plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

# (你原来的 setup_chinese_fonts() 函数现在可以删除了)
# (use_chinese 变量现在由上面的逻辑直接设定)


# 定义清理token文本的函数，将非ASCII字符替换为简单描述
def clean_token_text(token_text, token_id=None):
    """将非ASCII字符替换为可读的ASCII表示，同时添加token ID"""
    token_text = token_text.strip()
    
    if not token_text or token_text.isspace():
        return f"ID:{token_id}" if token_id is not None else "EmptyToken"
    
    # 尝试utf-8解码，如果已经是str则跳过
    try:
        if isinstance(token_text, bytes):
            token_text = token_text.decode('utf-8', 'replace')
    except UnicodeDecodeError:
        return f"DecodeErr[{token_id}]" if token_id is not None else "DecodeError"

    is_mostly_ascii_or_printable = all(32 <= ord(char) < 127 or char.isspace() for char in token_text)
    # 检查是否包含任何需要特殊处理的非直接显示字符（除常见标点和字母数字）
    # 这个正则更倾向于找出“非单词”和“非数字”的字符，但可能过于宽泛
    # has_complex_chars = bool(re.search(r'[^\w\s\d.,!?"\'()-]', token_text))
    
    # 一个更简单的判断：如果不是纯ASCII可打印字符，并且我们不能用中文，就特殊处理
    # 或者，如果它是特殊标记，如 <0x0A>
    is_special_marker = bool(re.match(r"<0x[0-9A-Fa-f]+>", token_text)) # 检测类似 <0x0A> 的标记

    display_text = token_text
    
    if use_chinese:
        # 对于中文环境，我们期望大部分token能正常显示
        # 但仍然可以对过长的文本进行截断
        if len(token_text) > 6: # 中文通常一个字占一个宽度，可以适当放宽
             # 对于混有英文的，或者纯英文的，按字符数截断
            if any('\u4e00' <= char <= '\u9fff' for char in token_text): # 如果有中文字符
                 if len(token_text) > 4: # 对纯中文或混中文的，4个字可能比较合适
                    display_text = token_text[:3] + "…"
            else: # 纯英文或符号
                display_text = token_text[:5] + "…"

    else: # 不使用中文或遇到需要清理的ASCII
        if is_special_marker or not is_mostly_ascii_or_printable:
             # 对于特殊标记或不可直接打印的，仅显示ID
            return f"ID:{token_id}" if token_id is not None else f"Token-{hash(token_text) % 1000:03d}"
        else: # 是可打印的ASCII
            if len(token_text) > 6:
                display_text = token_text[:5] + "…"
            # 对于可显示的ASCII，如果use_chinese为False，也可能需要用引号包起来，表明它是字面量
            display_text = f"'{display_text}'"


    return f"{display_text}[{token_id}]" if token_id is not None else display_text


def read_tensor_from_file(filename):
    """从文件中读取保存的tensor数据"""
    if not os.path.exists(filename):
        print(f"文件不存在: {filename}")
        return None

    with open(filename, 'rb') as f:
        ndim = np.fromfile(f, dtype=np.uint64, count=1)
        if not ndim.size: return None # 文件太小
        ndim = ndim[0]
        
        shape = np.fromfile(f, dtype=np.uint64, count=ndim)
        if shape.size != ndim: return None # 文件不完整

        data = np.fromfile(f, dtype=np.float32)
        
        expected_size = np.prod(shape)
        if data.size == expected_size and data.size > 0 : # 确保读取了预期的数据量
            data = data.reshape(shape)
            return data
        else:
            print(f"文件数据大小与shape不匹配或为空: {filename}, data size: {data.size}, expected: {expected_size}")
            return None

def apply_softmax(logits):
    """应用softmax函数将logits转换为概率分布"""
    if logits is None or logits.size == 0:
        return np.array([])
    # 为了数值稳定性，先减去最大值
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    sum_exp_logits = np.sum(exp_logits, axis=-1, keepdims=True)
    # 避免除以0
    return exp_logits / np.where(sum_exp_logits == 0, 1, sum_exp_logits)


def visualize_top_tokens(target_probs, draft_probs, tokenizer, top_k=10, token_pos=0):
    """可视化目标模型和草稿模型的top-k个token的概率分布"""
    if target_probs.size == 0 or draft_probs.size == 0:
        print(f"Token位置 {token_pos} 的概率数据为空，跳过可视化。")
        return

    target_flat = target_probs.flatten()
    draft_flat = draft_probs.flatten()

    # 找出目标模型的top-k个token
    # 确保top_k不超过词汇表大小
    effective_top_k_target = min(top_k, len(target_flat))
    if effective_top_k_target == 0:
        print(f"Token位置 {token_pos} 目标模型概率数据无法获取top-k，跳过。")
        return

    target_topk_indices = np.argsort(target_flat)[-effective_top_k_target:][::-1]
    target_topk_probs_values = target_flat[target_topk_indices]
    
    # 获取相同token在草稿模型中的概率
    draft_probs_for_target_tokens = draft_flat[target_topk_indices]
    
    # 找出草稿模型的top-k个token
    effective_top_k_draft = min(top_k, len(draft_flat))
    if effective_top_k_draft == 0:
        print(f"Token位置 {token_pos} 草稿模型概率数据无法获取top-k，跳过。")
        return
        
    draft_topk_indices = np.argsort(draft_flat)[-effective_top_k_draft:][::-1]
    draft_topk_probs_values = draft_flat[draft_topk_indices]
    
    # 获取相同token在目标模型中的概率
    target_probs_for_draft_tokens = target_flat[draft_topk_indices]
    
    # 将token ID转换为文本，并添加token ID信息
    try:
        target_topk_tokens = [clean_token_text(tokenizer.decode([idx], skip_special_tokens=False), idx) for idx in target_topk_indices]
        draft_topk_tokens = [clean_token_text(tokenizer.decode([idx], skip_special_tokens=False), idx) for idx in draft_topk_indices]
    except Exception as e:
        print(f"Tokenizer解码时出错: {e}. 将仅使用Token ID.")
        target_topk_tokens = [clean_token_text(str(idx), idx) for idx in target_topk_indices]
        draft_topk_tokens = [clean_token_text(str(idx), idx) for idx in draft_topk_indices]

    fig = plt.figure(figsize=(18, 12)) # 稍微增大图像尺寸
    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    title = f'Token位置 {token_pos} 的Top-{top_k} Token概率比较' if use_chinese else f'Token Position {token_pos} - Top-{top_k} Token Probability Comparison'
    fig.suptitle(title, fontsize=18, y=0.98) # 调整主标题位置
    
    # 目标模型的top-k个token
    ax1 = fig.add_subplot(gs[0, 0])
    title1 = '目标模型Top-k Tokens' if use_chinese else 'Target Model Top-k Tokens'
    ax1.set_title(title1, fontsize=14)
    bars1 = ax1.bar(np.arange(effective_top_k_target), target_topk_probs_values, color='royalblue', alpha=0.8, width=0.8)
    ax1.set_xticks(np.arange(effective_top_k_target))
    ax1.set_xticklabels(target_topk_tokens, rotation=45, ha='right', fontsize=10)
    ax1.set_ylabel('概率' if use_chinese else 'Probability', fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005, f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 草稿模型对目标模型top-k个token的概率
    ax2 = fig.add_subplot(gs[0, 1])
    title2 = '草稿模型对目标Top-k Tokens的概率' if use_chinese else 'Draft Model for Target Top-k Tokens'
    ax2.set_title(title2, fontsize=14)
    bars2 = ax2.bar(np.arange(effective_top_k_target), draft_probs_for_target_tokens, color='mediumseagreen', alpha=0.8, width=0.8)
    ax2.set_xticks(np.arange(effective_top_k_target))
    ax2.set_xticklabels(target_topk_tokens, rotation=45, ha='right', fontsize=10) # 使用相同的target_topk_tokens作为标签
    ax2.set_ylabel('概率' if use_chinese else 'Probability', fontsize=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005, f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
    # 草稿模型的top-k个token
    ax3 = fig.add_subplot(gs[1, 0])
    title3 = '草稿模型Top-k Tokens' if use_chinese else 'Draft Model Top-k Tokens'
    ax3.set_title(title3, fontsize=14)
    bars3 = ax3.bar(np.arange(effective_top_k_draft), draft_topk_probs_values, color='mediumseagreen', alpha=0.8, width=0.8)
    ax3.set_xticks(np.arange(effective_top_k_draft))
    ax3.set_xticklabels(draft_topk_tokens, rotation=45, ha='right', fontsize=10)
    ax3.set_ylabel('概率' if use_chinese else 'Probability', fontsize=12)
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005, f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
    # 目标模型对草稿模型top-k个token的概率
    ax4 = fig.add_subplot(gs[1, 1])
    title4 = '目标模型对草稿Top-k Tokens的概率' if use_chinese else 'Target Model for Draft Top-k Tokens'
    ax4.set_title(title4, fontsize=14)
    bars4 = ax4.bar(np.arange(effective_top_k_draft), target_probs_for_draft_tokens, color='royalblue', alpha=0.8, width=0.8)
    ax4.set_xticks(np.arange(effective_top_k_draft))
    ax4.set_xticklabels(draft_topk_tokens, rotation=45, ha='right', fontsize=10) # 使用相同的draft_topk_tokens作为标签
    ax4.set_ylabel('概率' if use_chinese else 'Probability', fontsize=12)
    ax4.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.005, f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局，给主标题和x轴标签留出空间
    
    output_dir = './logits_data/visualizations'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/token_pos_{token_pos}_comparison.png', dpi=150)
    plt.close(fig) # 明确关闭图像，释放内存
    
    print(f"已保存Token位置 {token_pos} 的可视化图表")

def compute_kl_divergence(p_probs, q_probs):
    """计算KL散度: KL(P||Q)"""
    if p_probs.size == 0 or q_probs.size == 0 or p_probs.shape != q_probs.shape:
        print("概率分布为空或形状不匹配，无法计算KL散度。")
        return np.nan # 或者一个其他指示错误的值

    # 确保概率和为1 (近似) 且非负
    p_probs = np.maximum(p_probs, 0)
    p_probs /= np.sum(p_probs)
    
    q_probs = np.maximum(q_probs, 0)
    q_probs_sum = np.sum(q_probs)
    if q_probs_sum == 0:
        print("警告: q_probs 全部为0，KL散度未定义或为无穷大。")
        return np.inf
    q_probs /= q_probs_sum

    epsilon = 1e-12 # 增加 epsilon 的值，避免 log(0)
    # 对于 p_probs 中为0的项，p log(p/q) 贡献为0
    # 对于 q_probs 中为0但p_probs不为0的项，贡献为无穷大
    kl_div = np.sum(np.where(p_probs > epsilon, p_probs * np.log(p_probs / np.maximum(q_probs, epsilon)), 0))
    return kl_div


def visualize_all_token_positions(token_positions, tokenizer_path, top_k=10):
    """可视化所有token位置的logits分布比较"""
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True) # 添加 trust_remote_code
        print(f"成功加载tokenizer，词汇表大小: {tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer)}")
    except Exception as e:
        print(f"加载tokenizer '{tokenizer_path}' 失败: {e}")
        print("将使用简单token ID作为标签")
        class DummyTokenizer: # 定义一个简单的DummyTokenizer
            def decode(self, token_ids, skip_special_tokens=False): # 添加 skip_special_tokens 参数
                # skip_special_tokens 在这里可以忽略，因为我们总是解码单个ID
                if not isinstance(token_ids, list) or not token_ids: return "ERR_ID"
                return f"ID:{token_ids[0]}"
        tokenizer = DummyTokenizer() # 使用DummyTokenizer实例
    
    kl_divergences = []
    
    for pos in token_positions:
        target_logits_file = f'./logits_data/target/logits_{pos}.bin'
        draft_logits_file = f'./logits_data/draft/logits_{pos}.bin'
        
        target_logits = read_tensor_from_file(target_logits_file)
        draft_logits = read_tensor_from_file(draft_logits_file)
        
        if target_logits is None or draft_logits is None:
            print(f"跳过token位置 {pos}，因为文件不存在或读取错误")
            continue
        
        # 确保logits是一维的 (vocab_size,)
        if target_logits.ndim > 1: target_logits = target_logits.squeeze()
        if draft_logits.ndim > 1: draft_logits = draft_logits.squeeze()

        if target_logits.ndim != 1 or draft_logits.ndim != 1:
            print(f"Token位置 {pos} 的logits不是一维的，跳过。Target shape: {target_logits.shape}, Draft shape: {draft_logits.shape}")
            continue

        target_probs = apply_softmax(target_logits)
        draft_probs = apply_softmax(draft_logits)
        
        if target_probs.size == 0 or draft_probs.size == 0:
            print(f"Token位置 {pos} 的概率计算结果为空，跳过。")
            continue
            
        visualize_top_tokens(target_probs, draft_probs, tokenizer, top_k, pos)
        
        kl_div = compute_kl_divergence(target_probs, draft_probs)
        if not np.isnan(kl_div) and not np.isinf(kl_div): # 只添加有效的KL散度值
             kl_divergences.append((pos, kl_div))
        print(f"Token位置 {pos} 的KL散度: {kl_div:.4f}")
    
    if kl_divergences:
        positions, divergences = zip(*kl_divergences)
        
        plt.figure(figsize=(12, 7)) # 调整图像大小
        plt.bar(positions, divergences, color='mediumpurple', alpha=0.8, width=0.8) # 调整宽度和颜色
        title_kl = '各Token位置的KL散度（目标 vs 草稿）' if use_chinese else 'KL Divergence by Token Position (Target vs Draft)'
        plt.title(title_kl, fontsize=16)
        plt.xlabel('Token位置' if use_chinese else 'Token Position', fontsize=12)
        plt.ylabel('KL散度 (KL(Target || Draft))' if use_chinese else 'KL Divergence (KL(Target || Draft))', fontsize=12)
        plt.xticks(positions) # 确保所有位置都显示刻度
        plt.grid(axis='y', linestyle=':', alpha=0.6) # 调整网格线样式
        plt.tight_layout() # 自动调整布局
        
        output_dir = './logits_data/visualizations'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f'{output_dir}/kl_divergence_comparison.png', dpi=150) # 修改文件名以示区分
        plt.close()
        
        print(f"已保存KL散度可视化图表到 {output_dir}/kl_divergence_comparison.png")

def main():
    target_dir = './logits_data/target'
    draft_dir = './logits_data/draft'
    
    if not os.path.exists(target_dir) or not os.path.isdir(target_dir) or \
       not os.path.exists(draft_dir) or not os.path.isdir(draft_dir):
        print(f"错误: logits数据文件夹 '{target_dir}' 或 '{draft_dir}' 不存在或不是目录。请先运行脚本生成数据。")
        return
    
    target_files = [f for f in os.listdir(target_dir) if f.startswith('logits_') and f.endswith('.bin')]
    token_positions = []
    for f_name in target_files:
        try:
            # 更鲁棒地提取数字
            match = re.search(r'logits_(\d+)\.bin', f_name)
            if match:
                token_positions.append(int(match.group(1)))
        except ValueError:
            print(f"警告: 无法从文件名 {f_name} 中解析token位置。")
            
    if not token_positions:
        print(f"错误: 在 '{target_dir}' 中未找到有效的logits数据文件 (例如 logits_0.bin)。请先运行脚本生成数据。")
        return
    
    token_positions.sort()
    print(f"找到 {len(token_positions)} 个token位置的logits数据: {token_positions}")
    
    # tokenizer_path = "./models/Qwen3-1.7B-AWQ" 
    # 使用一个通用的、你系统中可能有的Hugging Face模型路径作为示例，或者留空让用户填写
    # 如果不确定，可以尝试一个非常基础的多语言tokenizer，但这可能不匹配你的模型
    # tokenizer_path = "bert-base-multilingual-cased" 
    # 最好是与你生成logits时使用的tokenizer一致
    tokenizer_path = "./models/Qwen3-1.7B-AWQ" # 保持用户原来的设置，但提示他确保这个路径有效
    print(f"将尝试从 '{tokenizer_path}' 加载tokenizer。如果失败，将使用Token ID作为标签。")
    print("请确保此路径下包含有效的tokenizer模型文件 (例如 tokenizer.json, vocab.txt/json, spiece.model等)。")

    visualize_all_token_positions(token_positions, tokenizer_path)

if __name__ == "__main__":
    # 重要的第一步：确保Matplotlib缓存是最新的，这一步通常在导入matplotlib.pyplot时自动触发
    # 如果在之前的交互中已经确认缓存是最新的，或者用户会手动管理，则不需要特定代码
    # 但如果仍有问题，可以提示用户手动删除 ~/.cache/matplotlib 目录然后重新运行
    print("可视化脚本开始运行...")
    print(f"当前Matplotlib字体设置 (font.family): {plt.rcParams['font.family']}")
    print(f"当前Matplotlib字体设置 (font.sans-serif): {plt.rcParams['font.sans-serif']}")
    main()
    print("可视化脚本运行结束。")