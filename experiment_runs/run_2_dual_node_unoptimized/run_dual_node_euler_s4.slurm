#!/bin/bash
#SBATCH --job-name=dual_node_s4
#SBATCH --partition=rtx
#SBATCH --account=courses
#SBATCH --qos=course
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

.venv/bin/python -u train_dual_node.py \
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
