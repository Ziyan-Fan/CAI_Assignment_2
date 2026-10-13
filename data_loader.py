"""
Loaders for everything the training and eval code consumes. Course-provided --
not a student TODO.

Two kinds of data:

  1. YOUR preference data (data/preference_pairs.jsonl) -- the >=1,000 pairs you
     build with generate_preference_data.py. Each pair feeds BOTH stages:
       - Part A (SFT) trains on the PREFERRED response only.
       - Part B (DPO) trains on the (chosen, rejected) contrast.

  2. The COURSE-PROVIDED held-out eval sets + MMLU slice (eval/), consumed by
     evaluate.py. These are frozen and never trained on. IMPORTANT: do not put
     the held-out eval prompts into your training data -- that's training on the
     test set, and it's an honor-code matter.

Preference-pair JSONL schema (one object per line), produced by
generate_preference_data.py:

    {"id": "pair_000001",
     "class": "ambiguous" | "clear",
     "prompt": "<the user request>",
     "chosen": "<preferred assistant response>",
     "rejected": "<dispreferred assistant response>"}

For an "ambiguous" prompt, `chosen` asks a specific clarifying question and
`rejected` is a confident guess. For a "clear" prompt, `chosen` answers directly
and `rejected` is a needless clarifying question.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

from configs.evaluation_config import (
    AMBIGUOUS_SET_PATH,
    CLEAR_SET_PATH,
    MMLU_SLICE_PATH,
)
from configs.paths_config import PREFERENCE_DATA_PATH

VALID_CLASSES = {"ambiguous", "clear"}


def _read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Expected file not found: {path}. "
            "Did you generate your preference data / fetch the eval artifacts?"
        )

    rows = []

    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                rows.append(json.loads(line))

            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc

    return rows


def load_preference_pairs(path: str | Path = PREFERENCE_DATA_PATH) -> list[dict]:
    """Loads and validates your preference pairs. Raises on a malformed file so a
    data bug surfaces now, not silently mid-training."""
    rows = _read_jsonl(path)
    required = {"prompt", "class", "chosen", "rejected"}

    for i, r in enumerate(rows):
        missing = required - r.keys()
        if missing:
            raise ValueError(f"pair #{i} is missing keys {sorted(missing)}")

        if r["class"] not in VALID_CLASSES:
            raise ValueError(
                f"pair #{i} has class {r['class']!r}, expected one of {sorted(VALID_CLASSES)}"
            )

    return rows


def _to_messages(prompt: str, response: str) -> list[dict]:
    """A single-turn chat: one user request, one assistant reply."""
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def load_sft_examples(path: str | Path = PREFERENCE_DATA_PATH) -> Dataset:
    """
    Part A (SFT) training data: the PREFERRED response of every pair, as a chat.

    Returns:
        Dataset with one column, "messages": each row is
        [{"role": "user", ...}, {"role": "assistant", ...}] ending on the
        assistant's preferred response -- exactly what
        utils.masking_utils.tokenize_with_response_mask() expects.
    """
    pairs = load_preference_pairs(path)
    messages = [_to_messages(p["prompt"], p["chosen"]) for p in pairs]
    return Dataset.from_dict({"messages": messages})


def load_dpo_examples(path: str | Path = PREFERENCE_DATA_PATH) -> Dataset:
    """
    Part B (DPO) training data: (prompt, chosen, rejected) triples, each side a chat.

    Returns:
        Dataset with "prompt", "chosen", "rejected" columns. "chosen"/"rejected"
        are each a messages list (shared user prompt + the differing assistant
        reply) -- matching train_dpo.build_dpo_example()'s signature.
    """
    pairs = load_preference_pairs(path)
    return Dataset.from_dict(
        {
            "prompt": [p["prompt"] for p in pairs],
            "chosen": [_to_messages(p["prompt"], p["chosen"]) for p in pairs],
            "rejected": [_to_messages(p["prompt"], p["rejected"]) for p in pairs],
        }
    )


# --- Course-provided held-out eval sets (never trained on). ---------------------


def load_ambiguous_set() -> list[dict]:
    """150 ambiguous held-out prompts. ask_rate is measured here."""
    return _read_jsonl(AMBIGUOUS_SET_PATH)


def load_clear_set() -> list[dict]:
    """150 clear held-out prompts. answer_rate is measured here."""
    return _read_jsonl(CLEAR_SET_PATH)


def load_mmlu_slice() -> list[dict]:
    """200-question MMLU slice for capability-retention scoring.

    Generated once by eval/build_mmlu_slice.py; if the file is absent, that step
    hasn't been run yet.
    """
    return _read_jsonl(MMLU_SLICE_PATH)
