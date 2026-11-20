#!/usr/bin/env bash
# Run 2: Dual NODE Unoptimized (Euler solver, s=4)
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -u train_dual_node.py \
  --env-id ant_u4_maze \
  --eval-env-id ant_u4_maze_eval \
  --critic-type node \
  --actor-type node \
  --ode-solver euler \
  --ode-steps 4 \
  --eval-actor 1 \
  --gamma 0.995 \
  --terminate-when-unhealthy 0 \
  --num-envs 256 \
  --batch-size 256 \
  --num-epochs 100 \
  --total-env-steps 100000000
