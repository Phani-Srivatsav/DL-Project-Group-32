#!/usr/bin/env bash
# Run 1: Single Node Baseline (NODE Critic + ResNet-4 Actor)
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -u train_node.py \
  --env-id ant_u4_maze \
  --eval-env-id ant_u4_maze_eval \
  --critic-type node \
  --ode-solver rk4 \
  --ode-steps 4 \
  --ode-num-layers 2 \
  --actor-depth 4 \
  --num-envs 512 \
  --batch-size 512 \
  --num-epochs 100 \
  --total-env-steps 100000000 \
  --checkpoint-every 5 \
  --save-buffer 1 \
  --capture-vis
