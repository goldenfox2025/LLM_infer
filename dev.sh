#!/bin/bash

sudo docker run -it --gpus all \
  -v $(pwd):/workspace \
  -w /workspace \
  --name llm_dev \
  llm_infer:latest \
  /bin/bash
