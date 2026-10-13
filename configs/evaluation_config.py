import os

from configs.paths_config import EVAL_DIR, OUTPUT_FOLDER

# --- Held-out behavioral eval sets + MMLU slice (course-provided, frozen, in eval/) ---
# ask_rate is measured on the ambiguous set, answer_rate on the clear set, mmlu_acc on
# the MMLU slice. These files are NOT student-configurable -- every submission is scored
# on the same prompts. (mmlu_slice.jsonl is generated once by eval/build_mmlu_slice.py.)
AMBIGUOUS_SET_PATH = os.path.join(EVAL_DIR, "held_out_ambiguous.jsonl")
CLEAR_SET_PATH = os.path.join(EVAL_DIR, "held_out_clear.jsonl")
MMLU_SLICE_PATH = os.path.join(EVAL_DIR, "mmlu_slice.jsonl")

# --- Generation settings for the behavioral sets (deterministic: greedy decode). ---
EVAL_BATCH_SIZE = 8
EVAL_MAX_NEW_TOKENS = 256
EVAL_SEED = 6604


# --- MMLU scoring (0-shot, letter answer). ---
MMLU_MAX_NEW_TOKENS = 8


# --- Judge: ask / answer / other classification (eval/judge.py) ---

# The course judge runs IN-PROCESS on PACE-ICE: a Llama-3.1-8B-Instruct model loaded on the
# same GPU the eval job already has -- no external endpoint, no cost, no server to keep up.
# It co-resides comfortably with a 1-3B student policy. Judgments are decoded greedily
# (temperature 0) and cached to JUDGE_CACHE_PATH, so re-runs and regrades reproduce exactly.

# Alternatives (if the course later stands up a vLLM server on PACE-ICE): set
# JUDGE_BACKEND = "openai_compatible" and fill JUDGE_BASE_URL/JUDGE_MODEL. "mock" is an
# offline heuristic for model-free pipeline testing only -- never for grading.

JUDGE_BACKEND = "local_hf"  # "local_hf" | "openai_compatible" | "mock".
JUDGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

JUDGE_BASE_URL = ""  # Only for "openai_compatible" (e.g. a vLLM server URL).
JUDGE_CACHE_PATH = os.path.join(OUTPUT_FOLDER, "judge_cache.json")
