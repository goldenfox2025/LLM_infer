#!/bin/bash
set -e

echo ">> Starting Nsight Systems profiling with GPU info..."

# 降低内核 perf_event_paranoid 参数（需要 sudo 权限）
sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'

# （可选）激活 conda 环境，如果用 sudo 后 conda 环境丢失，请取消下一行的注释并修改环境名称及路径
# source /home/keta/miniconda3/bin/activate myenv

# 进入 frontend 目录（假设 chat.py 位于此处）
cd frontend

# 调用 nsys 进行采样：
# --trace=cuda,osrt,nvtx：采集 CUDA、OSRT 和 NVTX 信息
# --gpu-metrics-device=0：采集 GPU 0 的指标
# --output：指定输出报告文件（生成 .qdrep 文件，后续可用 Nsight Systems GUI 打开）
# --force-overwrite true：如果同名文件存在则覆盖
nsys profile \
  --trace=cuda,osrt,nvtx \
  --gpu-metrics-device=0 \
  --output ../nsys_profile_report \
  --force-overwrite true \
  python3 chat.py

echo ">> Nsight Systems profiling finished. Report saved as nsys_profile_report.qdrep"
