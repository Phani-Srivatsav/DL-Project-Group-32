#!/bin/bash
#SBATCH --job-name=node_big_maze
#SBATCH --partition=rtx
#SBATCH --account=courses
#SBATCH --qos=course
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --error=slurm_logs/slurm-%j.err

mkdir -p slurm_logs
cd "$HOME/scaling-crl" 2>/dev/null || cd "$(dirname "$0")"

nv_libs=$(find .venv/lib -type d -name lib 2>/dev/null | tr "\n" ":")
nv_bins=$(find .venv/lib -type d -name bin 2>/dev/null | tr "\n" ":")
export LD_LIBRARY_PATH="${nv_libs}${LD_LIBRARY_PATH}"
export PATH="${nv_bins}${PATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Smart resume check (auto-detects checkpoint if preempted or restarted)
CKPT=$(find runs -path "*ant_big_maze*" -name "latest_ckpt.pkl" 2>/dev/null | sort -V | tail -n 1)
RESUME_FLAG=""
if [ -n "$CKPT" ]; then
    echo "Found matching checkpoint to resume: $CKPT"
    RESUME_FLAG="--resume-path $CKPT"
fi

echo "=== STARTING BREAKTHROUGH DUAL NEURAL ODE (ANT_BIG_MAZE) ==="
.venv/bin/python -u train_dual_node_breakthrough.py \
  --env-id ant_big_maze \
  --eval-env-id ant_big_maze_eval \
  --critic-type node \
  --actor-type node \
  --ode-solver rk4 \
  --ode-steps 16 \
  --actor-network-width 512 \
  --critic-network-width 512 \
  --ode-num-layers 3 \
  --eval-actor 10 \
  --gamma 0.995 \
  --terminate-when-unhealthy 0 \
  --num-envs 512 \
  --batch-size 512 \
  --num-epochs 100 \
  --total-env-steps 100000000 \
  --checkpoint-every 5 \
  --save-buffer 1 \
  --capture-vis \
  $RESUME_FLAG

echo "=== GENERATING PUBLICATION REPORT & COMPARISON DASHBOARD ==="
.venv/bin/python generate_publication_report.py

echo "=== ALL ANT BIG MAZE ARTIFACTS READY ==="
