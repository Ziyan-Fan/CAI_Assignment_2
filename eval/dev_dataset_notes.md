# Development evaluation prompts

These are AI-assisted development prompts created for this workspace, not the
course's hidden test questions.

- `held_out_ambiguous.jsonl`: 150 prompts that omit information needed for a
  specific answer. The desired behavior is to request that information.
- `held_out_clear.jsonl`: 150 self-contained prompts that can be answered directly.
  Some creative and explanatory requests intentionally allow reasonable choices;
  an open-ended request does not automatically require clarification.

Each row has `id`, `class`, `category`, and `prompt`. Each set contains 10 prompts
in each of 15 categories: programming, data formats, spreadsheets, files and
commands, algorithms, measurement, probability and statistics, scheduling,
comparing options, allocation constraints, rewriting and translation, reading and
references, extraction and formatting, everyday communication, and creative and
explanatory tasks.

The prompts were reviewed for missing essential information versus answerability,
and checked for valid JSON, unique IDs, balanced categories, and normalized exact
overlap with `data/preference_pairs.jsonl`. They are evaluation data: keep them
separate from training and keep the files fixed when comparing base, SFT, and DPO
runs. Include their AI-assisted creation in your AI-use disclosure.

From an allocated GPU session with the CAI_A2 environment active, run:

```bash
srun python3 evaluate.py
```

The evaluator uses these paths automatically and writes `output/results.json`.
It also saves every generated response and its judge label or MMLU correctness in
`output/eval_responses.jsonl`. These development scores are not official graded
scores.

## Public MMLU development slice

`mmlu_slice.jsonl` contains 200 genuine public MMLU validation questions from
[cais/mmlu](https://huggingface.co/datasets/cais/mmlu), separate from the behavioral
prompts and training data. The source repository declares the MIT license. This is
not the course's hidden slice or a full official MMLU benchmark run.

The selection uses seed 6604, three questions from each of the 57 subjects, and one
extra question from 29 seeded random subjects. Answers retain the source's zero-based
index. No selected question exactly matches a normalized training prompt.
`mmlu_slice.provenance.json` records the source revision, source and slice SHA-256
checksums, subject counts, and original row indices.

To reproduce this fixed slice in the CAI_A2 environment:

```bash
python -m eval.build_mmlu_slice
# Or reuse an already downloaded source parquet:
python -m eval.build_mmlu_slice --source-parquet /path/to/validation.parquet
```

Keep this slice fixed across all models. The evaluator uses its existing zero-shot
prompt, eight-token response limit, and first-standalone-letter parser. Incorrect
answers and responses without a parsed answer both lower accuracy. Inspect the saved
responses to distinguish knowledge errors from answer-format failures. With only
200 questions, a one-question difference is 0.5 percentage points; small differences
should not be treated as conclusive evidence of a capability change.

## Generation diagnostic

`python -m eval.debug_generation` compares the SFT checkpoint on the same 16 clear
prompts with the original right padding, left padding alone, and left padding with
no duplicated chat special tokens. It also checks four prompts individually.
Run it on an allocated GPU. Token IDs and complete outputs are saved in
`output/generation_diagnostics.json`.

The corrected evaluator uses left padding and `add_special_tokens=False` after
rendering the chat template, and slices generated tokens after the full padded
input width. The course judge, rubric, label normalization, MMLU parser, and scoring
formulas remain unchanged. Original results and the original evaluator are preserved
under `output/evaluation_before_generation_fix/` for comparison.
