#!/bin/bash
accelerate launch --gpu_ids 'all' --num_processes 8 --mixed_precision bf16 main.py \
  -c config/wn18rr.yaml \
  --enable-diffusion \
  --diffusion-mode denoiser \
  --learning-rate 3e-5 \
  > logs/wn18rr_diffusion_stage2.log 2>&1
