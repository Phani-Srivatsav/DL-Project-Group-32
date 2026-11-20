#!/usr/bin/env bash
# Universal Training Script for 7-DOF Robotic Arm Manipulation
# Runs on any machine with an NVIDIA GPU

export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -u train_dual_node_breakthrough.py \
  --env-id arm_push_hard \
  --eval-env-id arm_push_hard \
  --critic-type node \
  --actor-type node \
  --ode-solver rk4 \
  --ode-steps 16 \
  --actor-network-width 512 \
  --critic-network-width 512 \
  --ode-num-layers 3 \
  --eval-actor 1 \
  --gamma 0.995 \
  --episode-length 80 \
  --unroll-length 80 \
  --num-envs 512 \
  --batch-size 512 \
  --num-epochs 100 \
  --total-env-steps 100000000 \
  --checkpoint-every 5 \
  --save-buffer 1 \
  --capture-vis
