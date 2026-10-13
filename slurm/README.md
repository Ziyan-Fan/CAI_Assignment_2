# PACE-ICE SLURM Notes

## Storage: keep environments and caches on scratch (read first)

Your PACE-ICE **home directory is small** (~15–30 GB). The `CAI_A2` conda environment
(PyTorch etc.) plus the model downloads (your 1–3B base, the 8B judge) will overflow it
and you'll hit `OSError: Disk quota exceeded`. Put everything heavy on your much larger
`~/scratch` instead. Add these to your `~/.bashrc` once:

```bash
export CONDA_PKGS_DIRS=~/scratch/conda/pkgs   # conda package cache -> scratch
export HF_HOME=~/scratch/hf_cache             # Hugging Face model/dataset cache -> scratch
```

Create the env with an explicit prefix on scratch (so it is not placed under
`~/.conda/envs` in home):

```bash
conda env create -f env.yml -p ~/scratch/conda/envs/CAI_A2
conda activate ~/scratch/conda/envs/CAI_A2
```

Rule of thumb on PACE-ICE: **home holds code; scratch holds environments, model caches,
and checkpoints.** `pace-quota` shows your current usage.


## Pipeline order

Part B starts from Part A's checkpoint, and evaluation needs all adapters — so run in
order, waiting for each stage before the next.

```bash
# 0. Build your preference data first (on PACE-ICE; the course model runs locally, no cost):
#    python3 generate_preference_data.py --generate   (then hand-verify, --report)

# 1. Part A -- SFT.
sbatch slurm/train_sft.sbatch
# Wait for it to finish -- check with: squeue -u $USER.

# 2. Part B -- DPO beta sweep (array job: one task per beta).
sbatch slurm/train_dpo.sbatch
# Writes checkpoints/dpo/beta_0.05, beta_0.1, beta_0.5 in parallel.

# 3. Evaluate base / SFT / every beta -> output/results.json.
sbatch slurm/evaluate.sbatch
```

All templates request a single H100 — ample for a 1–3B model with LoRA. Adjust
`--time`/`--mem` if your hyperparameters (batch size, sequence length, number of pairs)
push past these defaults.

The DPO template is a SLURM **array job** (`--array=0-2`) indexing a `BETAS=(0.05 0.1
0.5)` bash array — keep it in sync with `DPO_BETAS` in `configs/train_config.py`. The
three betas train in parallel as separate array tasks, each into its own
`checkpoints/dpo/beta_<value>/`.

**The ask/answer judge needs no endpoint.** `evaluate.py` loads the course judge model
(`Llama-3.1-8B-Instruct`, set in `configs/evaluation_config.py`) in-process on the eval
job's own GPU — it co-resides with your policy, costs nothing, and its judgments are
cached to `output/judge_cache.json` so re-runs reproduce. The first eval run downloads
the judge model to your Hugging Face cache.

**Gated models — authenticate once.** The Llama base and judge checkpoints are gated on
Hugging Face. Accept each model's license on its HF page, then run `huggingface-cli login`
(or set `HF_TOKEN` in `~/.bashrc`) before your first training/eval job, or the download 401s
on the compute node.
