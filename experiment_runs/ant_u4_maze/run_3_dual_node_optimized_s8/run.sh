#!/usr/bin/env bash
# Run 3: Dual NODE Optimized (RK4 solver, s=8, NFE=32)
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -u train_dual_node.py \
  --env-id ant_u4_maze \
  --eval-env-id ant_u4_maze_eval \
  --critic-type node \
  --actor-type node \
  --ode-solver rk4 \
  --ode-steps 8 \
  --eval-actor 1 \
  --gamma 0.995 \
  --terminate-when-unhealthy 0 \
  --ode-num-layers 2 \
  --num-envs 512 \
  --batch-size 512 \
  --num-epochs 100 \
  --total-env-steps 100000000 \
  --checkpoint-every 5 \
  --save-buffer 1 \
  --capture-vis
