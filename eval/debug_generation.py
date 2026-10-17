"""Compare original and corrected generation without changing the scoring rubric.

Run on a GPU: python -m eval.debug_generation
Responses and tokenization details are written to output/generation_diagnostics.json.
"""

import argparse
import json
from pathlib import Path
import time

import torch

from configs.evaluation_config import EVAL_BATCH_SIZE, EVAL_MAX_NEW_TOKENS, EVAL_SEED
from configs.paths_config import OUTPUT_FOLDER, SFT_CHECKPOINT_DIR
from data_loader import load_clear_set
from evaluate import load_adapter_policy
from utils.model_utils import load_tokenizer


def run_variant(model, tokenizer, items, padding_side, add_special_tokens, batch_size):
    tokenizer.padding_side = padding_side
    records = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        chats = [tokenizer.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}],
            tokenize=False, add_generation_prompt=True,
        ) for item in batch]
        inputs = tokenizer(
            chats, return_tensors="pt", padding=True, truncation=True,
            add_special_tokens=add_special_tokens,
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=EVAL_MAX_NEW_TOKENS,
                do_sample=False, use_cache=True,
            )
        width = inputs["input_ids"].shape[1]
        for item, input_ids, attention, output_ids in zip(
            batch, inputs["input_ids"], inputs["attention_mask"], outputs
        ):
            prompt_ids = input_ids[attention.bool()].tolist()
            generated_ids = output_ids[width:].tolist()
            records.append({
                "id": item["id"], "category": item["category"],
                "prompt": item["prompt"],
                "response": tokenizer.decode(generated_ids, skip_special_tokens=True).strip(),
                "raw_response": tokenizer.decode(generated_ids, skip_special_tokens=False),
                "prompt_tokens": len(prompt_ids), "padded_width": width,
                "prompt_first_token_ids": prompt_ids[:5],
                "prompt_last_token_ids": prompt_ids[-5:],
                "last_input_token_id": int(input_ids[-1]),
                "last_input_attention": int(attention[-1]),
                "generated_token_ids": generated_ids,
            })
        print(f"  {padding_side}, special_tokens={add_special_tokens}: "
              f"{len(records)}/{len(items)} prompts", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default=SFT_CHECKPOINT_DIR)
    parser.add_argument("--output", default=str(Path(OUTPUT_FOLDER) / "generation_diagnostics.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Run this diagnostic in an allocated GPU session.")
    torch.manual_seed(EVAL_SEED)
    rows = load_clear_set()
    # Keep original batch boundaries, covering both technical and everyday prompts.
    items = rows[:EVAL_BATCH_SIZE] + rows[136:136 + EVAL_BATCH_SIZE]
    tokenizer = load_tokenizer(args.checkpoint_dir)
    original_padding_side = tokenizer.padding_side
    print(f"Original tokenizer padding_side={original_padding_side}", flush=True)
    model = load_adapter_policy(args.checkpoint_dir)
    model.eval()
    result = {
        "checkpoint": args.checkpoint_dir, "seed": EVAL_SEED,
        "original_padding_side": original_padding_side,
        "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id, "batch_size": EVAL_BATCH_SIZE,
        "max_new_tokens": EVAL_MAX_NEW_TOKENS, "variants": {},
    }
    variants = (
        ("original_right", "right", True, items, EVAL_BATCH_SIZE),
        ("left_only", "left", True, items, EVAL_BATCH_SIZE),
        ("left_no_extra_special_tokens", "left", False, items, EVAL_BATCH_SIZE),
        ("singleton_no_extra_special_tokens", "left", False,
         [items[i] for i in (0, 1, 8, 9)], 1),
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    for name, padding, special, sample, batch_size in variants:
        started = time.monotonic()
        records = run_variant(model, tokenizer, sample, padding, special, batch_size)
        result["variants"][name] = {
            "padding_side": padding, "add_special_tokens": special,
            "seconds": round(time.monotonic() - started, 3), "records": records,
        }
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(f"Saved {name}: {sum(not row['response'] for row in records)} empty responses", flush=True)
    tokenizer.padding_side = original_padding_side
    print(f"Diagnostic saved to {path}", flush=True)


if __name__ == "__main__":
    main()
