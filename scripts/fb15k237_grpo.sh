#!/bin/bash
accelerate launch --gpu_ids 'all' --num_processes 8 --mixed_precision bf16 main.py \
  -c config/fb15k237.yaml \
  --grpo-weight 0.1 \
  --grpo-temperature 1.0 \
  > logs/fb15k237_grpo.log 2>&1
