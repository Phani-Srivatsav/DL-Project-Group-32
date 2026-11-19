#!/bin/bash
# Generic Slurm Template for High Performance Clusters
# Replace <YOUR_PARTITION> and <YOUR_ACCOUNT> with your cluster parameters.
#SBATCH --job-name=dual_node_rl
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

# Activate your environment
# source .venv/bin/activate

# Execute universal training runner
bash scripts/train_ant_u4.sh
