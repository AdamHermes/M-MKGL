#!/bin/bash
accelerate launch --gpu_ids 'all' --num_processes 8 --mixed_precision bf16 main.py \
  -c config/fb15k237.yaml \
  --enable-diffusion \
  --diffusion-mode joint \
  --learning-rate 2e-5 \
  > logs/fb15k237_diffusion_joint.log 2>&1
