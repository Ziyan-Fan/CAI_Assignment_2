"""
Shared model/tokenizer loading used by train_sft.py, train_dpo.py, and evaluate.py.

Wired up -- not a student TODO. The loading config (dtype, device map, attention
implementation) and the pad-token fallback were previously copy-pasted in three places;
they live here once instead.

The base model is one of the small *-Instruct checkpoints in
configs.train_config.ALLOWED_MODELS (set MODEL_ID to your choice). Because it is an
instruct checkpoint, its tokenizer ships a chat template, which
utils.masking_utils.tokenize_with_response_mask() relies on.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs.train_config import MODEL_ID


def load_tokenizer(source: str = MODEL_ID):
    """
    Loads a tokenizer from `source` -- either the base model ID (Part A) or a
    checkpoint directory (Part B, evaluate.py) -- with a pad-token fallback in case the
    chosen checkpoint has no dedicated pad token.
    """
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_id: str = MODEL_ID):
    """Loads base causal LM with the dtype/device/attention config every part of this assignment uses."""
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
