# Base model -- your choice between these two small instruct checkpoints. Both run
# LoRA on a single PACE-ICE GPU in minutes per epoch. Set MODEL_ID to whichever you
# use and record it in results.json's metadata.base_model.
#
# NOTE: these are *-Instruct checkpoints. Unlike a from-scratch base model, they can
# already follow instructions -- your job is not to create that ability but to *shift
# behavior*: ask when the request is ambiguous, answer when it is clear, without
# forgetting everything else. See README.md.

ALLOWED_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"


# LoRA config -- shared by Part A (SFT) and Part B (DPO).
RANK = 16
ALPHA = 32
DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]  # present in both model families.


# --- Preference data construction (what generate_preference_data.py must produce). ---
# You build the data yourself over two input classes: ambiguous requests (where a
# clarifying question is preferred) and fully-specified requests (where a direct answer
# is preferred). See README.md Part 1.

MIN_PREFERENCE_PAIRS = 1000  # Spec floor: at least 1,000 pairs total.
HAND_VERIFY_SAMPLE = 100  # Spec: hand-verify 100 pairs and report the rejection rate.
DATA_SEED = 6604  # Fix every sampling/shuffle step for reproducibility.


# --- Part A: SFT-LoRA (warm-start the target behavior). ---
# SFT trains on the PREFERRED response of each pair you built
# (data_loader.load_sft_examples()). Hyperparameters below are yours to tune.

SFT_BATCH_SIZE = 4
SFT_EPOCHS = 2
SFT_LEARNING_RATE = 2e-5
SFT_MAX_SEQ_LEN = 1024


# --- Part B: DPO (align on your preference pairs). ---
# You must sweep beta over at least three values (spec). Each value trains its own
# adapter; `train_dpo.py --beta <value>` picks one and checkpoints it separately.
# These three are the course default sweep -- you may change them, but keep at least
# three distinct values and report all of them in results.json.

DPO_BETAS = [0.05, 0.1, 0.5]
DPO_BETA = 0.1  # Default used when --beta is not passed on the CLI.
DPO_BATCH_SIZE = 2

DPO_EPOCHS = 1
DPO_LEARNING_RATE = 5e-6
DPO_MAX_SEQ_LEN = 1024
