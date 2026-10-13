import os

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Course-provided eval artifacts (held-out behavioral sets + MMLU slice + judge).
EVAL_DIR = os.path.join(BASE_PATH, "eval")

# Your preference data (built by generate_preference_data.py), plus the hand-verified
# sample you report a rejection rate on.
DATA_DIR = os.path.join(BASE_PATH, "data")
PREFERENCE_DATA_PATH = os.path.join(DATA_DIR, "preference_pairs.jsonl")
HAND_VERIFIED_PATH = os.path.join(DATA_DIR, "hand_verified.jsonl")

# Trained LoRA adapters.
SFT_CHECKPOINT_DIR = os.path.join(BASE_PATH, "checkpoints", "sft")
DPO_CHECKPOINT_BASE = os.path.join(BASE_PATH, "checkpoints", "dpo")


def dpo_checkpoint_dir(beta: float) -> str:
    """Per-beta DPO adapter directory, so a beta sweep doesn't overwrite itself."""
    return os.path.join(DPO_CHECKPOINT_BASE, f"beta_{beta}")


# Evaluation outputs.
OUTPUT_FOLDER = os.path.join(BASE_PATH, "output")
RESULTS_PATH = os.path.join(OUTPUT_FOLDER, "results.json")
