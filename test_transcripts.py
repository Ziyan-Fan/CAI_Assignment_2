"""Generate three *real* comparison transcripts for ECE 6256 Assignment 2.

Run from CAI_Assignment_2 project root:
    python3 extract_three_transcripts.py
    python3 extract_three_transcripts.py --scan 150 --mmlu-scan 200

Uses the exact greedy generation helper and checkpoint paths from evaluate.py.
Does not retrain and does not modify the official evaluator.
"""

import argparse
import gc
import json
from pathlib import Path
import re

import torch

from data_loader import load_ambiguous_set, load_clear_set, load_mmlu_slice
from evaluate import (
    LETTERS,
    _mmlu_prompt,
    _parse_letter,
    generate_responses,
    load_adapter_policy,
    load_base_policy,
)
from configs.evaluation_config import MMLU_MAX_NEW_TOKENS, EVAL_MAX_NEW_TOKENS, EVAL_SEED
from configs.paths_config import SFT_CHECKPOINT_DIR, dpo_checkpoint_dir
from utils.model_utils import load_tokenizer


def looks_like_clarification(response):
    """Selection heuristic, NOT the course's Llama-8B judge."""
    s = response.strip().lower()
    question_starts = (
        "what ", "which ", "where ", "when ", "who ", "how ",
        "could you", "can you", "would you", "do you", "does ",
        "is ", "are ", "please share", "please provide", "please paste",
    )
    return (
        "?" in s and s.startswith(question_starts)
    ) or bool(re.search(r"\b(?:please|kindly) (?:share|provide|paste)\b", s))


def get_outputs(checkpoint, tokenizer, prompts, max_new_tokens):
    if checkpoint is None:
        model = load_base_policy()
    else:
        model = load_adapter_policy(str(checkpoint))
    model.eval()
    try:
        return generate_responses(
            model, tokenizer, prompts, max_new_tokens=max_new_tokens
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def first_match(n, condition):
    return next((i for i in range(n) if condition(i)), None)


def blockquote(content):
    return "\n".join("> " + line for line in (content or "[empty output]").splitlines())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=int, default=100,
                        help="number of each behavioral-dev set to inspect (default: 100)")
    parser.add_argument("--mmlu-scan", type=int, default=120,
                        help="MMLU dev questions to inspect (default: 120)")
    args = parser.parse_args()
    torch.manual_seed(EVAL_SEED)

    ambiguous = load_ambiguous_set()[:args.scan]
    clear = load_clear_set()[:args.scan]
    mmlu = load_mmlu_slice()[:args.mmlu_scan]
    if not ambiguous or not clear or not mmlu:
        raise SystemExit("Missing dev sets under eval/. Check held_out_*.jsonl and mmlu_slice.jsonl")

    amb_prompts = [item["prompt"] for item in ambiguous]
    clear_prompts = [item["prompt"] for item in clear]
    mmlu_prompts = [_mmlu_prompt(item) for item in mmlu]
    tokenizer = load_tokenizer(SFT_CHECKPOINT_DIR)

    print("Scanning base model for all three categories...", flush=True)
    base_amb = get_outputs(None, tokenizer, amb_prompts, EVAL_MAX_NEW_TOKENS)
    base_clear = get_outputs(None, tokenizer, clear_prompts, EVAL_MAX_NEW_TOKENS)
    base_mmlu = get_outputs(None, tokenizer, mmlu_prompts, MMLU_MAX_NEW_TOKENS)

    print("Scanning beta=0.05 for over-asking and MMLU failures...", flush=True)
    beta05 = dpo_checkpoint_dir(0.05)
    dpo05_clear = get_outputs(beta05, tokenizer, clear_prompts, EVAL_MAX_NEW_TOKENS)
    dpo05_mmlu = get_outputs(beta05, tokenizer, mmlu_prompts, MMLU_MAX_NEW_TOKENS)

    print("Scanning beta=0.1 for successful clarification...", flush=True)
    beta01 = dpo_checkpoint_dir(0.1)
    dpo01_amb = get_outputs(beta01, tokenizer, amb_prompts, EVAL_MAX_NEW_TOKENS)

    success_idx = first_match(len(ambiguous), lambda i:
        looks_like_clarification(dpo01_amb[i])
        and not looks_like_clarification(base_amb[i]))
    overask_idx = first_match(len(clear), lambda i:
        looks_like_clarification(dpo05_clear[i])
        and not looks_like_clarification(base_clear[i]))
    forget_idx = first_match(len(mmlu), lambda i:
        _parse_letter(base_mmlu[i]) == LETTERS[int(mmlu[i]["answer"])]
        and _parse_letter(dpo05_mmlu[i]) != LETTERS[int(mmlu[i]["answer"])])

    missing = [name for name, idx in (
        ("successful clarification", success_idx),
        ("over-asking failure", overask_idx),
        ("MMLU retention failure", forget_idx),
    ) if idx is None]
    if missing:
        raise SystemExit(
            f"No examples found for: {', '.join(missing)}. "
            "Try --scan 150 --mmlu-scan 200, or inspect model outputs manually."
        )

    examples = [
        {"title": "Successful clarification (beta=0.1)",
         "type": "ambiguous", "prompt": amb_prompts[success_idx],
         "source_index": success_idx},
        {"title": "Over-asking on a clear prompt (beta=0.05)",
         "type": "clear", "prompt": clear_prompts[overask_idx],
         "source_index": overask_idx},
        {"title": "MMLU capability-retention failure (beta=0.05)",
         "type": "mmlu", "prompt": mmlu_prompts[forget_idx],
         "source_index": forget_idx,
         "correct_letter": LETTERS[int(mmlu[forget_idx]["answer"])]},
    ]

    print("Generating verbatim outputs for all five policies...", flush=True)
    checkpoints = [
        ("Base", None),
        ("SFT", SFT_CHECKPOINT_DIR),
        ("DPO beta=0.05", beta05),
        ("DPO beta=0.1", beta01),
        ("DPO beta=0.5", dpo_checkpoint_dir(0.5)),
    ]
    for name, checkpoint in checkpoints:
        print(f"  {name}", flush=True)
        responses = get_outputs(checkpoint, tokenizer,
                                [examples[0]["prompt"], examples[1]["prompt"]],
                                EVAL_MAX_NEW_TOKENS)
        mmlu_response = get_outputs(checkpoint, tokenizer,
                                    [examples[2]["prompt"]], MMLU_MAX_NEW_TOKENS)[0]
        for ex, resp in zip(examples, responses + [mmlu_response]):
            ex.setdefault("responses", {})[name] = resp

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "three_transcripts.json").write_text(
        json.dumps(examples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md = ["# Three Actual Model Transcripts — ECE 6256", "",
          "> [!IMPORTANT]", "> These are verbatim model generations. The examples were auto-selected",
          "> using a simple question-detection heuristic, not the course judge.",
          "> **Review all three manually before using them in the report.**", ""]
    for number, ex in enumerate(examples, 1):
        md.extend([f"## Transcript {number}: {ex['title']}", "",
                   f"**Evaluation set:** {ex['type']} · index {ex['source_index']}", "",
                   "**User prompt:**", "", blockquote(ex["prompt"]), ""])
        if "correct_letter" in ex:
            md.extend([f"**Correct MMLU choice:** {ex['correct_letter']}", ""])
        for name, response in ex["responses"].items():
            md.extend([f"### {name}", "", blockquote(response), ""])
        md.extend(["**Your analysis:** [Write your own explanation after verifying outputs.]", "", "---", ""])
    (output_dir / "three_transcripts.md").write_text("\n".join(md), encoding="utf-8")
    print("\nSaved output/three_transcripts.md and output/three_transcripts.json")
    for ex in examples:
        print(f" - {ex['title']}: {ex['type']} index {ex['source_index']}")


if __name__ == "__main__":
    main()
