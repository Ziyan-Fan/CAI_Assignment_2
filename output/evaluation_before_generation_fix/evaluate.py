"""
Behavioral + capability-retention evaluation. Course-provided -- not a student TODO.

For each policy (base / SFT / each DPO beta) this:
  - generates on the 150 ambiguous held-out prompts  -> ask_rate     (judge-labeled)

  - generates on the 150 clear held-out prompts       -> answer_rate  (judge-labeled)

  - scores the 200-question MMLU slice                -> mmlu_acc     (capability retention)

and writes results.json in the format eval/results_schema.json defines.

THE HELD-OUT SETS ARE THE HIDDEN TEST SET. They are not shipped in the starter kit --
the course graders hold them and run THIS harness on your submitted adapters to compute
your behavioral deltas and retention. You don't implement the metric; you train the model.

To self-evaluate during development, drop your OWN dev prompts at
eval/held_out_ambiguous.jsonl / eval/held_out_clear.jsonl (and optionally your own
eval/mmlu_slice.jsonl) in the documented format, then run this -- see eval/README.md. Any
set whose file is absent is simply skipped, so you can self-check behavior without MMLU.

The ask/answer judge (eval/judge.py) runs in-process on PACE-ICE (Llama-3.1-8B-Instruct) --
no endpoint, no cost -- and caches judgments so runs reproduce.

Usage:
    python3 evaluate.py                 # Evaluate base + sft + every beta in DPO_BETAS.
    python3 evaluate.py --betas 0.1     # Evaluate base + sft + just beta=0.1.
"""

import argparse
import gc
import json
import os
import re

import torch
from peft import PeftModel

from configs.evaluation_config import (
    EVAL_BATCH_SIZE,
    EVAL_MAX_NEW_TOKENS,
    EVAL_SEED,
    JUDGE_BACKEND,
    JUDGE_BASE_URL,
    JUDGE_CACHE_PATH,
    JUDGE_MODEL,
    MMLU_MAX_NEW_TOKENS,
)
from configs.paths_config import (
    EVAL_DIR,
    RESULTS_PATH,
    SFT_CHECKPOINT_DIR,
    dpo_checkpoint_dir,
)
from configs.train_config import DPO_BETAS, MODEL_ID
from data_loader import load_ambiguous_set, load_clear_set, load_mmlu_slice
from eval.judge import (
    Judge,
    LocalHFJudge,
    MockJudge,
    OpenAICompatibleJudge,
    RUBRIC_VERSION,
)
from utils.model_utils import load_base_model, load_tokenizer

LETTERS = ["A", "B", "C", "D"]


# --- Models. -------------------------------------------------------------------


def load_base_policy():
    """The untrained instruct base model (the 'base' run)."""
    return load_base_model()


def load_adapter_policy(checkpoint_dir: str):
    """Base model + a trained LoRA adapter (SFT or a DPO-beta checkpoint)."""
    base_model = load_base_model()
    return PeftModel.from_pretrained(base_model, checkpoint_dir)


def _free(model):
    """Drop a model and reclaim its GPU memory before loading the next one."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- Generation. ---------------------------------------------------------------


def generate_responses(model, tokenizer, prompts, max_new_tokens=EVAL_MAX_NEW_TOKENS):
    """Greedy-decodes one response per prompt (deterministic)."""
    responses = []

    for idx in range(0, len(prompts), EVAL_BATCH_SIZE):
        batch = prompts[idx : idx + EVAL_BATCH_SIZE]
        chat = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            for p in batch
        ]

        inputs = tokenizer(chat, return_tensors="pt", padding=True, truncation=True).to(
            model.device
        )

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
            )

        for input_ids, output_ids in zip(inputs["input_ids"], outputs):
            gen = output_ids[input_ids.shape[0] :]
            responses.append(tokenizer.decode(gen, skip_special_tokens=True).strip())

    return responses


# --- Behavioral rates (ask / answer via judge). --------------------------------


def _rate_or_none(model, tokenizer, judge, items, target):
    """Returns (rate, label_counts) for one set, or (None, None) if the set is absent."""
    if not items:
        return None, None

    prompts = [r["prompt"] for r in items]
    responses = generate_responses(model, tokenizer, prompts)

    res = judge.rate(zip(prompts, responses), target=target)
    return res["rate"], res["counts"]


# --- MMLU accuracy. ------------------------------------------------------------


def _mmlu_prompt(item):
    lines = [f"Question: {item['question']}"]
    for letter, choice in zip(LETTERS, item["choices"]):
        lines.append(f"{letter}. {choice}")

    lines.append("Answer with the single letter (A, B, C, or D) of the correct option.")
    return "\n".join(lines)


def _parse_letter(text):
    """First standalone A/B/C/D in the model's output; None if absent."""
    match = re.search(r"\b([ABCD])\b", text.strip().upper())
    return match.group(1) if match else None


def mmlu_accuracy(model, tokenizer, slice_items):
    if not slice_items:
        return None

    prompts = [_mmlu_prompt(it) for it in slice_items]
    responses = generate_responses(model, tokenizer, prompts, max_new_tokens=MMLU_MAX_NEW_TOKENS)

    correct = 0

    for item, resp in zip(slice_items, responses):
        pred = _parse_letter(resp)
        if pred is not None and LETTERS.index(pred) == int(item["answer"]):
            correct += 1

    return correct / len(slice_items)


# --- Judge construction. -------------------------------------------------------


def build_judge():
    if JUDGE_BACKEND == "local_hf":
        backend = LocalHFJudge(model_name=JUDGE_MODEL)

    elif JUDGE_BACKEND == "openai_compatible":
        backend = OpenAICompatibleJudge(model_name=JUDGE_MODEL, base_url=JUDGE_BASE_URL)

    else:
        backend = MockJudge()

    return Judge(backend, cache_path=JUDGE_CACHE_PATH)


# --- One run. ------------------------------------------------------------------


def evaluate_policy(model, tokenizer, judge, ambiguous, clear, mmlu):
    ask_rate, amb_counts = _rate_or_none(model, tokenizer, judge, ambiguous, "ask")

    answer_rate, clr_counts = _rate_or_none(model, tokenizer, judge, clear, "answer")

    mmlu_acc = mmlu_accuracy(model, tokenizer, mmlu)
    out = {"ask_rate": ask_rate, "answer_rate": answer_rate, "mmlu_acc": mmlu_acc}

    label_counts = {}

    if amb_counts is not None:
        label_counts["ambiguous"] = amb_counts

    if clr_counts is not None:
        label_counts["clear"] = clr_counts

    if label_counts:
        out["label_counts"] = label_counts

    return out


def _safe_load(loader, name):
    try:
        return loader()

    except FileNotFoundError:
        print(
            f"[warn] {name} not found -- skipping it. "
            "(Graders supply the hidden test set; students can drop their own dev file here.)"
        )
        return []


def _validate(results):
    """Soft-validate results.json against the course schema; warn, never crash."""
    try:
        import jsonschema

        schema = json.load(open(os.path.join(EVAL_DIR, "results_schema.json"), encoding="utf-8"))
        jsonschema.validate(results, schema)
        print("results.json validates against eval/results_schema.json.")

    except Exception as exc:  # ImportError, ValidationError, missing schema -- all non-fatal.
        print(f"[warn] results.json did not validate ({type(exc).__name__}: {exc}).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--betas",
        type=float,
        nargs="*",
        default=DPO_BETAS,
        help="DPO betas to evaluate (default: every value in DPO_BETAS).",
    )
    args = parser.parse_args()

    torch.manual_seed(EVAL_SEED)

    print("Loading eval sets (ambiguous / clear / MMLU)...")
    ambiguous = _safe_load(load_ambiguous_set, "ambiguous held-out set")

    clear = _safe_load(load_clear_set, "clear held-out set")
    mmlu = _safe_load(load_mmlu_slice, "MMLU slice")

    if not (ambiguous or clear or mmlu):
        raise SystemExit(
            "No eval sets found under eval/. Graders: place the hidden test set there. "
            "Students: add your own dev files to self-evaluate (see eval/README.md)."
        )

    judge = build_judge()
    tokenizer = load_tokenizer(SFT_CHECKPOINT_DIR)

    print(f"[base] loading {MODEL_ID}...")
    base = load_base_policy()

    runs_base = evaluate_policy(base, tokenizer, judge, ambiguous, clear, mmlu)
    _free(base)

    print(f"[sft] loading adapter from {SFT_CHECKPOINT_DIR}...")
    sft = load_adapter_policy(SFT_CHECKPOINT_DIR)

    runs_sft = evaluate_policy(sft, tokenizer, judge, ambiguous, clear, mmlu)
    _free(sft)

    dpo_runs = []
    for beta in args.betas:
        ckpt = dpo_checkpoint_dir(beta)
        print(f"[dpo beta={beta}] loading adapter from {ckpt}...")
        policy = load_adapter_policy(ckpt)

        metrics = evaluate_policy(policy, tokenizer, judge, ambiguous, clear, mmlu)
        metrics["beta"] = beta
        dpo_runs.append(metrics)
        _free(policy)

    results = {
        "schema_version": "1.0",
        "metadata": {
            "base_model": MODEL_ID,
            "seeds": {"data": EVAL_SEED, "train": EVAL_SEED, "eval": EVAL_SEED},
            "judge": {
                "backend": JUDGE_BACKEND,
                "model": JUDGE_MODEL,
                "rubric_version": RUBRIC_VERSION,
            },
            "betas": list(args.betas),
        },
        "eval_sets": {
            "ambiguous": {"path": "eval/held_out_ambiguous.jsonl", "n": len(ambiguous)},
            "clear": {"path": "eval/held_out_clear.jsonl", "n": len(clear)},
            "mmlu": {"path": "eval/mmlu_slice.jsonl", "n": len(mmlu)},
        },
        "runs": {"base": runs_base, "sft": runs_sft, "dpo": dpo_runs},
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    _validate(results)
    print("\n===== results.json =====")
    print(json.dumps(results, indent=2))
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
