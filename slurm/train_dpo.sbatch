#!/bin/bash
#
# Ready to run.
# Submits the Part B DPO beta sweep on PACE-ICE as an array job -- one task per beta,
# each starting from the same Part A checkpoint and writing its own adapter
# (checkpoints/dpo/beta_<value>/). Requires a completed Part A checkpoint at
# --sft-checkpoint-dir (run slurm/train_sft.sbatch first).
#
# Keep BETAS below in sync with DPO_BETAS in configs/train_config.py.
#
# Usage: sbatch slurm/train_dpo.sbatch

#SBATCH --job-name=a2-dpo
#SBATCH --gres=gpu:H100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

module load anaconda3
conda activate CAI_A2

BETAS=(0.05 0.1 0.5)
BETA=${BETAS[$SLURM_ARRAY_TASK_ID]}

python train_dpo.py \
    --sft-checkpoint-dir "checkpoints/sft" \
    --beta "$BETA" \
    --seed 42
