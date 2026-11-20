#!/usr/bin/env bash
# Universal Training Script for Ant U4 Maze (Breakthrough Dual Neural ODE)
# Runs on any machine with an NVIDIA GPU (RTX 3080/3090/4090, A100, H100, etc.)

export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -u train_dual_node_breakthrough.py \
  --env-id ant_u4_maze \
  --eval-env-id ant_u4_maze_eval \
  --critic-type node \
  --actor-type node \
  --ode-solver rk4 \
  --ode-steps 16 \
  --actor-network-width 512 \
  --critic-network-width 512 \
  --ode-num-layers 3 \
  --eval-actor 1 \
  --gamma 0.995 \
  --terminate-when-unhealthy 0 \
  --num-envs 512 \
  --batch-size 512 \
  --num-epochs 100 \
  --total-env-steps 100000000 \
  --checkpoint-every 5 \
  --save-buffer 1 \
  --capture-vis
