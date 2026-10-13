# Evaluation sets

This directory holds the evaluation harness. **The graded test set is not here** — the
course keeps the held-out prompts and the MMLU slice private and runs `evaluate.py` on
them against your submitted adapters to compute your behavioral deltas and capability
retention. You never receive the test prompts, so there is nothing to accidentally train
on.

## What ships with the starter kit

- `judge.py` — the ask/answer/other classifier. Runs in-process on PACE-ICE
  (`Llama-3.1-8B-Instruct`), caches judgments so runs reproduce. This is exactly the judge
  the graders use.
- `results_schema.json` — the JSON Schema your `results.json` must satisfy.
- `results_template.json` — a filled example showing the expected shape.

## What is NOT included (the hidden test set)

- `held_out_ambiguous.jsonl` — 150 ambiguous prompts (ask-rate is measured here).
- `held_out_clear.jsonl` — 150 fully-specified prompts (answer-rate is measured here).
- `mmlu_slice.jsonl` — a fixed MMLU slice (capability retention).

## Self-evaluating during development

To check your own progress, create your **own** dev prompts at these paths and run
`evaluate.py`. Any set whose file is missing is skipped, so you can self-check behavior
even without an MMLU file.

`held_out_ambiguous.jsonl` / `held_out_clear.jsonl` — one JSON object per line:

```json
{"id": "dev_001", "class": "ambiguous", "category": "your-label", "prompt": "Fix the bug."}
{"id": "dev_002", "class": "clear", "category": "your-label", "prompt": "What is 15% of 200?"}
```

`mmlu_slice.jsonl` (optional) — one JSON object per line:

```json
{"id": "m_000", "subject": "...", "question": "...", "choices": ["...","...","...","..."], "answer": 2}
```

(`answer` is the 0-based index of the correct choice.) MMLU is a public benchmark, so you
can sample your own questions for a retention sanity-check; it will not be the graders'
exact slice.

Your own dev numbers won't match the official graded numbers exactly — different prompts —
but the **trends** (ask-rate up, answer-rate holding, retention) are what your report's
analysis is about. The qualitative transcripts your report needs are yours to choose from
any prompts you like.
