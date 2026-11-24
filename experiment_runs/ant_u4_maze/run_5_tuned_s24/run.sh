#!/usr/bin/env bash
# Run 5: Tuned Dual Neural ODE (RK4, s=24, NFE=96, Width 512, gamma=0.998, entropy=0.7, grad_clip=1.0)
export PYTHONPATH="/home/phani.gande.ug24/scaling-crl:${PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -u train_dual_node_tuned_s24.py \
  --exp-name run_5_tuned_s24 \
  --env-id ant_u4_maze \
  --eval-env-id ant_u4_maze_eval \
  --critic-type node \
  --actor-type node \
  --ode-solver rk4 \
  --ode-steps 24 \
  --actor-network-width 512 \
  --critic-network-width 512 \
  --ode-num-layers 3 \
  --eval-actor 10 \
  --gamma 0.998 \
  --entropy-param 0.7 \
  --grad-clip-norm 1.0 \
  --terminate-when-unhealthy 0 \
  --num-envs 512 \
  --batch-size 512 \
  --num-epochs 125 \
  --total-env-steps 125000000 \
  --checkpoint-every 5 \
  --save-buffer 1 \
  --capture-vis
