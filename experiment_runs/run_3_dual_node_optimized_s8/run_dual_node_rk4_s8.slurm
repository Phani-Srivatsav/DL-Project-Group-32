#!/bin/bash
#SBATCH --job-name=dual_node_100m
#SBATCH --partition=rtx
#SBATCH --account=courses
#SBATCH --qos=course
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=20:00:00
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --error=slurm_logs/slurm-%j.err

mkdir -p slurm_logs
cd /home/phani.gande.ug24/scaling-crl

nv_libs=$(find .venv/lib -type d -name lib 2>/dev/null | tr "\n" ":")
nv_bins=$(find .venv/lib -type d -name bin 2>/dev/null | tr "\n" ":")
export LD_LIBRARY_PATH="${nv_libs}${LD_LIBRARY_PATH}"
export PATH="${nv_bins}${PATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Smart resume check
CKPT=$(find runs -name "latest_ckpt.pkl" 2>/dev/null | grep "dual_node" | sort -V | tail -n 1)
RESUME_FLAG=""
if [ -n "$CKPT" ]; then
    echo "Found checkpoint to resume: $CKPT"
    RESUME_FLAG="--resume-path $CKPT"
fi

echo "=== STARTING 100M DUAL NEURAL ODE (ACTOR + CRITIC) TRAINING ==="
.venv/bin/python -u train_dual_node.py \
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
  --capture-vis \
  $RESUME_FLAG

echo "=== GENERATING PUBLICATION REPORT & COMPARISON DASHBOARD ==="
.venv/bin/python generate_report.py

echo "=== ALL EXPERIMENT ARTIFACTS READY ==="
