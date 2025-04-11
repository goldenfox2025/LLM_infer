#!/bin/bash

sudo docker run -it --rm \
  --gpus all \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device /dev/nvidiactl \
  --device /dev/nvidia-uvm \
  --device /dev/nvidia0 \
  -v $(pwd):/workspace \
  -w /workspace \
  --name llm_dev \
  llm_infer:latest \
  /bin/bash
