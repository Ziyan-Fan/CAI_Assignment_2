#!/bin/bash
#
# Ready to run.
# Evaluates base / SFT / every DPO-beta on PACE-ICE and writes output/results.json.
# The ask/answer judge (Llama-3.1-8B-Instruct) is loaded in-process on this same GPU --
# no server to stand up -- so it co-resides with the student policy. First run downloads
# the judge model to your HF cache.
#
# Requires: trained adapters in checkpoints/ (run train_sft + train_dpo first).
#
# Usage: sbatch slurm/evaluate.sbatch

#SBATCH --job-name=a2-eval
#SBATCH --gres=gpu:H100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

module load anaconda3
conda activate CAI_A2

python evaluate.py
