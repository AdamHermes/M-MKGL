#!/bin/bash
accelerate launch --gpu_ids 'all' --num_processes 8 --mixed_precision bf16 main.py \
  -c config/wn18rr.yaml \
  --grpo-weight 0.1 \
  --grpo-temperature 1.0 \
  > logs/wn18rr_grpo.log 2>&1
