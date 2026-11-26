#!/usr/bin/env bash
# Universal Training Script for Arm Push Hard (MJX Physics Benchmark)
# Dual Continuous-Depth Neural ODE Architecture (s=8, NFE=32, Width=256)
# Runs on any machine with an NVIDIA GPU

export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -u train_dual_node_breakthrough.py \
  --env-id arm_push_hard \
  --eval-env-id arm_push_hard \
  --critic-type node \
  --actor-type node \
  --ode-solver rk4 \
  --ode-steps 8 \
  --actor-network-width 256 \
  --critic-network-width 256 \
  --ode-num-layers 3 \
  --eval-actor 0 \
  --gamma 0.99 \
  --num-envs 512 \
  --batch-size 512 \
  --episode-length 1000 \
  --unroll-length 62 \
  --num-epochs 100 \
  --total-env-steps 100000000 \
  --checkpoint-every 5 \
  --save-buffer 0 \
  --capture-vis
