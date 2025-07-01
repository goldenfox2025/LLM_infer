import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

# --- 1. Ring Attention 核心实现 (使用 dist.send 和 dist.recv) ---

class RingAttention(nn.Module):
    """
    使用 PyTorch 分布式中的 dist.send 和 dist.recv 实现 Ring Attention。
    这个版本清晰地展示了为避免死锁而采用的奇偶 rank 通信策略。
    """
    def __init__(self):
        super().__init__()

    def forward(self, q_local, k_local, v_local):
        # 获取分布式环境信息
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # 初始化用于 online softmax 的累加器
        numerator = torch.zeros_like(q_local, dtype=torch.float32)
        denominator = torch.zeros((q_local.shape[0], q_local.shape[1], 1), dtype=torch.float32)
        max_so_far = torch.full((q_local.shape[0], q_local.shape[1], 1), -float('inf'), dtype=torch.float32)
        
        # 克隆本地的 K, V 作为环形通信的起始块
        # clone()很重要，可以避免后续操作影响原始的k_local, v_local
        k_ring, v_ring = k_local.clone(), v_local.clone()

        # --- 主循环：执行 N (world_size) 轮计算和 N-1 轮通信 ---
        for i in range(world_size):
            # 1. 本地计算：使用本地的 q 和当前环上的 k, v
            # print(f"Rank {rank}: Step {i+1}, computing with K/V from original rank {(rank - i + world_size) % world_size}")
            attn_scores = torch.bmm(q_local, k_ring.transpose(1, 2))
            
            # --- Online Softmax 逻辑 ---
            block_max = torch.max(attn_scores, dim=-1, keepdim=True)[0]
            new_max = torch.maximum(max_so_far, block_max)
            
            exp_scaler = torch.exp(max_so_far - new_max)
            numerator = numerator * exp_scaler
            denominator = denominator * exp_scaler

            block_weights = torch.exp(attn_scores - new_max)
            numerator += torch.bmm(block_weights, v_ring)
            denominator += torch.sum(block_weights, dim=-1, keepdim=True)

            max_so_far = new_max
            
            # 2. 环形通信：使用 send 和 recv 将 K, V 传递给邻居
            # 只有在多于一个进程时才需要通信
            if world_size > 1 and i < world_size - 1: # 最后一轮计算后无需通信
                # 计算要发送的目标 rank 和要接收的源 rank
                send_to_rank = (rank + 1) % world_size
                recv_from_rank = (rank - 1 + world_size) % world_size
                
                # 创建用于接收的空 Tensor
                k_recv = torch.empty_like(k_ring)
                v_recv = torch.empty_like(v_ring)

                # --- 核心：使用奇偶 rank 策略避免死锁 ---
                # 如果所有进程都先 send 再 recv，会永远等待，造成死锁。
                # 通过让奇偶进程执行相反顺序，打破这个等待循环。
                if rank % 2 == 0:
                    # 偶数 rank: 先发送，后接收
                    dist.send(k_ring, dst=send_to_rank)
                    dist.recv(k_recv, src=recv_from_rank)
                    
                    dist.send(v_ring, dst=send_to_rank)
                    dist.recv(v_recv, src=recv_from_rank)
                else:
                    # 奇数 rank: 先接收，后发送
                    dist.recv(k_recv, src=recv_from_rank)
                    dist.send(k_ring, dst=send_to_rank)
                    
                    dist.recv(v_recv, src=recv_from_rank)
                    dist.send(v_ring, dst=send_to_rank)
                
                # 更新环上的 k, v，为下一次迭代做准备
                k_ring, v_ring = k_recv, v_recv

        # 最终归一化，得到注意力输出
        final_output = numerator / denominator
        return final_output.to(q_local.dtype)

# --- 2. 标准注意力函数 (用于验证) ---
def standard_attention(q, k, v):
    """在单个设备上计算完整的注意力，作为黄金标准。"""
    attn_scores = torch.bmm(q, k.transpose(1, 2))
    attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
    output = torch.bmm(attn_weights, v)
    return output

# --- 3. 分布式环境设置和工作函数 ---
def setup(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # 使用 Gloo 后端，CPU/GPU均可，对学习很方便
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    """销毁分布式环境"""
    dist.destroy_process_group()

def run_worker(rank, world_size, B, S, D):
    """每个进程执行的工作函数"""
    print(f"Running worker on rank {rank}.")
    setup(rank, world_size)

    # --- 数据准备 ---
    # 在 rank 0 上创建完整数据
    if rank == 0:
        q_full = torch.randn(B, S, D)
        k_full = torch.randn(B, S, D)
        v_full = torch.randn(B, S, D)
        # 将数据切分成 world_size 个块
        q_chunks = list(torch.chunk(q_full, world_size, dim=1))
        k_chunks = list(torch.chunk(k_full, world_size, dim=1))
        v_chunks = list(torch.chunk(v_full, world_size, dim=1))
    else:
        # 其他进程创建空列表，用于接收广播
        q_chunks, k_chunks, v_chunks = None, None, None

    # --- 使用 broadcast_object_list 分发数据块 ---
    q_local_list = [None] * world_size
    dist.broadcast_object_list(q_chunks if rank == 0 else q_local_list, src=0)
    q_local = q_local_list[rank] if rank != 0 else q_chunks[rank]

    k_local_list = [None] * world_size
    dist.broadcast_object_list(k_chunks if rank == 0 else k_local_list, src=0)
    k_local = k_local_list[rank] if rank != 0 else k_chunks[rank]
    
    v_local_list = [None] * world_size
    dist.broadcast_object_list(v_chunks if rank == 0 else v_local_list, src=0)
    v_local = v_local_list[rank] if rank != 0 else v_chunks[rank]
    
    # --- 执行 Ring Attention ---
    ring_attn_module = RingAttention()
    ring_output_local = ring_attn_module(q_local, k_local, v_local)

    # --- 收集并验证结果 ---
    if world_size > 1:
        # 创建一个列表来收集所有进程的输出
        output_list = [torch.empty_like(ring_output_local) for _ in range(world_size)]
        # 使用 all_gather 将所有本地输出收集到 output_list 中
        dist.all_gather(output_list, ring_output_local)
    else:
        # 如果只有一个进程，直接打包成列表
        output_list = [ring_output_local]
    
    # 在 rank 0 上进行验证
    if rank == 0:
        print("\n--- Verification on Rank 0 ---")
        # 1. 重建 Ring Attention 的完整输出
        ring_output_full = torch.cat(output_list, dim=1)
        
        # 2. 计算标准注意力的输出 (使用原始的完整 Q, K, V)
        standard_output_full = standard_attention(q_full, k_full, v_full)
        
        print("Shape of Ring Attention output:", ring_output_full.shape)
        print("Shape of Standard Attention output:", standard_output_full.shape)
        
        # 3. 比较结果
        is_correct = torch.allclose(ring_output_full, standard_output_full, atol=1e-6)
        
        if is_correct:
            print("\n✅ SUCCESS: Ring Attention output matches the standard attention output!")
        else:
            print("\n❌ FAILURE: Outputs do not match!")
            diff = torch.norm(ring_output_full - standard_output_full)
            print(f"   Difference (L2 Norm): {diff.item()}")

    # 销毁进程组
    cleanup()

# --- 4. 主程序入口 ---
if __name__ == "__main__":
    # --- 配置参数 ---
    WORLD_SIZE = 4      # 模拟的设备/进程数量
    BATCH_SIZE = 2
    SEQ_LEN_TOTAL = 128 # 确保可以被 WORLD_SIZE 整除
    HIDDEN_DIM = 64
    
    assert SEQ_LEN_TOTAL % WORLD_SIZE == 0, "Total sequence length must be divisible by world_size"

    print(f"Starting test with {WORLD_SIZE} processes...")
    
    # 使用 multiprocessing.spawn 启动分布式程序
    mp.spawn(
        run_worker,
        args=(WORLD_SIZE, BATCH_SIZE, SEQ_LEN_TOTAL, HIDDEN_DIM),
        nprocs=WORLD_SIZE,
        join=True
    )