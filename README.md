# Assignment 2 (A2): Teach a Model When to Ask

LoRA + DPO with a *behavioral* objective: clarify when ambiguous, answer when clear — and don't forget everything else.

| | | | |
|---|---|---|---|
| **Assigned** | Thu, Oct 1 | **Due** | Mon, Oct 19, 11:59 pm |
| **Weight** | 5% of course grade | **Work mode** | Individual |
| **Submit** | Gradescope: repo + adapters + report ≤3 pp | **Effort** | ≈20–25 hours |
| **Quiz** | Quiz 2 · in class Tue Oct 20 | **Counts** | 6.25% if not your dropped quiz |

> Lecture 1 opened with an agent that guessed instead of asking. This assignment makes you
> fix that — not with a prompt, but with training. You take a small open **instruct** model
> and shift its behavior with LoRA fine-tuning and Direct Preference Optimization so that it
> **asks a clarifying question when the request is ambiguous** and **answers directly when it
> is not**. Both directions must move: a model that interrogates everyone is as broken as one
> that never asks. Along the way you'll meet reward hacking, the DPO β trade-off, and
> catastrophic forgetting — personally.

---

## Setup

```bash
conda env create -f env.yml
conda activate CAI_A2
```

Environment includes `transformers`, `peft` (LoRA), `datasets`, `torch`, and `requests` (for
the eval judge). See `env.yml` for the full pinned list.

> **On PACE-ICE:** your home directory is small (~15–30 GB) and this env plus the model
> downloads will overflow it. Build the env and point the Hugging Face cache at your
> `scratch` space first — see `slurm/README.md`. Otherwise you'll hit a disk-quota error.

**Base model — your choice of one small instruct checkpoint:**

- [`meta-llama/Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) (default in `configs/train_config.py`), or
- [`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).

Both run LoRA on a single PACE-ICE GPU in minutes per epoch. Set `MODEL_ID` to whichever you
use; it's recorded in `results.json`.

> **Access note — these are gated models.** The Llama checkpoints require a one-time license
> acceptance on Hugging Face: open the model page, click to accept Meta's license (approval is
> instant), then authenticate once with `huggingface-cli login` (or export `HF_TOKEN`) before
> you train or evaluate. Without it the download fails with a 401/403.

---

## Starter Kit

- `generate_preference_data.py` — **you implement `synthesize_pairs()`.** The output schema +
  writer, the hand-verification sampler, and the rejection-rate computation are wired — see Part 1.

- `data_loader.py` — wired: loads your preference pairs (for SFT and DPO) and the course-provided
  held-out eval sets + MMLU slice. Not a student TODO.

- `configs/` — `train_config.py` (base model, LoRA hyperparameters, SFT/DPO hyperparameters,
  the β sweep, data-construction constants), `paths_config.py`, `evaluation_config.py` (held-out
  set paths, judge settings). Hyperparameter values are yours to tune unless a section says otherwise.

- `utils/` — shared utilities: `model_utils.py` (base-model/tokenizer loading, wired) and
  `masking_utils.py` (**`tokenize_with_response_mask()` is left for you to implement** — the one
  function behind "mask everything except the final assistant turn", used by both SFT and DPO. See Part 2).

- `train_sft.py` — skeleton: CLI, LoRA construction, `build_sft_example()`, and checkpointing are
  wired. **The training loop is left for you to implement** — see Part 2.

- `train_dpo.py` — skeleton: CLI (`--beta`), loading your SFT checkpoint as the policy init,
  `build_dpo_example()`, and per-β checkpointing are wired. **`sequence_log_prob()`, `dpo_loss()`,
  and the training loop are left for you to implement** — see Part 2.

- `evaluate.py` — **wired, course-provided.** Computes ask-rate / answer-rate / MMLU accuracy for
  base / SFT / each β and writes `results.json`. You run it; you don't modify the metric — see Part 3.

- `eval/judge.py`, `eval/__init__.py`, `eval/results_schema.json`, `eval/results_template.json`, `eval/README.md` —
  the ask/answer judge (classifier), the `results.json` schema and a filled template, and the eval contract.
  **The held-out eval sets and MMLU slice are the hidden test set and are NOT included** — the graders run
  `evaluate.py` on them against your adapters; to self-check, drop your own dev prompts in `eval/`
  (see `eval/README.md` and Part 3).

- `slurm/` — SLURM job templates + instructions for PACE-ICE.

- `env.yml` — pinned conda environment.

- `README.md` — this file.

---

## What to Build

### Part 1 — Build the preference data

Construct **at least 1,000 preference pairs** over two input classes:

- **ambiguous** — a genuinely underspecified request. The **preferred** response asks a *specific*
  clarifying question; the **dispreferred** response is a confident guess that commits to one interpretation.
- **clear** — a fully-specified request. The **preferred** response answers directly; the
  **dispreferred** response is a needless clarifying question.

You may synthesize these with an LLM (e.g. the course endpoint). Implement `synthesize_pairs()` in
`generate_preference_data.py`; the writer, the 100-pair verification sampler, and the rejection-rate
computation are wired for you:

```bash
python3 generate_preference_data.py --generate   # synthesize + write, then sample 100 to verify
# ... hand-fill each "verdict" (accept/reject) in data/hand_verified.jsonl ...
python3 generate_preference_data.py --report      # prints your synthetic data's rejection rate
```

You **must hand-verify 100 pairs and report the rejection rate** of your own synthetic data (graded —
see the rubric).

> **Integrity:** build your own prompts. Do **not** reuse the held-out eval prompts in
> `eval/held_out_*.jsonl` as training data — that is training on the test set, and it's an honor-code matter.

### Part 2 — Train (SFT-LoRA, then DPO)

**SFT first.** Fine-tune the instruct base on the **preferred** response of each pair, so DPO has a
sensible policy to sharpen. In `utils/masking_utils.py` implement `tokenize_with_response_mask()`;
in `train_sft.py` implement `train_loop()`.

```bash
python3 train_sft.py
```

Two things will bite you if you miss them:

1. **Inject LoRA into the target modules and freeze everything else.** Only your LoRA parameters
   should have `requires_grad=True`. If base weights are still trainable you've lost the point of LoRA.
2. **Mask prompt/user tokens out of the loss.** Only the final assistant-turn tokens should contribute
   to cross-entropy. This is the single most common silent SFT bug: the loss curve looks normal,
   training "succeeds," and the model still learns the wrong thing.

**Then DPO.** Start from your SFT checkpoint and align on your preference pairs. In `train_dpo.py`
implement `sequence_log_prob()`, `dpo_loss()`, and the training loop — **no `TRL.DPOTrainer`**.
**Sweep β over at least three values**, each a separate run / adapter:

```bash
python3 train_dpo.py --beta 0.05
python3 train_dpo.py --beta 0.1
python3 train_dpo.py --beta 0.5
```

Three things to know:

1. **You do not need two model copies.** PEFT's `policy.disable_adapter()` context manager gives you
   reference-model log-probs from the *same* model with LoRA switched off.
2. **Sum per-token log-probs into a per-sequence log-prob, over the response only** — same masking as SFT.
3. **DPO loss:**
   ```
   L_DPO = -log σ( β · [ (logπ_θ(y_w|x) − logπ_ref(y_w|x)) − (logπ_θ(y_l|x) − logπ_ref(y_l|x)) ] )
   ```
   where `y_w`/`y_l` are the chosen/rejected completions. β is the knob you're sweeping.

Use fixed seeds and log your configs.

### Part 3 — Evaluate

The behavioral metric is computed by the **provided** harness `evaluate.py` (you do not implement
it) — for base / SFT / each β:

- **Ask-rate** on 150 ambiguous prompts — must go **up**.
- **Answer-rate** on 150 clear prompts — must **not collapse**.
- **MMLU-slice accuracy** on a fixed 200-question slice — capability retention, before vs. after.

Ask/answer are decided by an LLM **judge** (`eval/judge.py`): the course judge runs **in-process
on PACE-ICE** — `Llama-3.1-8B-Instruct` on the eval job's own GPU, no endpoint, no cost — and caches
judgments so runs reproduce.

**Held-out sets and the MMLU slice are the hidden test set — they are not in the starter kit.**
The graders run `evaluate.py` on them against your submitted adapters to produce your official
`results.json` and your behavioral-deltas grade. To self-evaluate while developing, create your own
dev prompts at `eval/held_out_ambiguous.jsonl` / `eval/held_out_clear.jsonl` (and optionally
`eval/mmlu_slice.jsonl`) in the documented format and run `python3 evaluate.py` — any missing set is
skipped. See `eval/README.md`.

For the report, pick **three qualitative before/after transcripts** (from any prompts you like) that
illustrate what changed.

> **How these points are scored — please read, it is not a normal autograder.** Gradescope's
> autograder only runs quick **validity checks** on your submission and shows **no points** —
> seeing 0 there is expected, not a bug. Your **Behavioral deltas (35)** and **Reproducibility
> (10)** are computed by the teaching staff **after the deadline**, by running the provided
> `evaluate.py` on the hidden test set against your submitted adapters. There is no live score
> for these during the assignment window, so use your own dev set (above) to track progress.

---

## Gradescope Deliverables

1. **Repository** — your whole project directory as a `.zip`, runnable from a clean clone:
   `generate_preference_data.py`, `train_sft.py`, `train_dpo.py`, your filled-in `configs/` and
   `utils/`, and your `data/` (preference pairs + hand-verified sample).

2. **LoRA adapter weights** — your SFT adapter and each DPO-β adapter __(not the base model)__, in the
   layout `evaluate.py` expects (`checkpoints/`).

3. **`results.json`** — matching `eval/results_schema.json`. Graders produce official one by
   running `evaluate.py` on hidden test set against your adapters; include the `results.json` from
   your own dev run so your pipeline and its format are verifiable.

4. **Report (≤3 pages)** — the β sweep (one figure), what broke and when (the over-asker, the
   forgetter), your hand-verification protocol and rejection rate, and your three transcripts.
   Include your AI-use disclosure.

Graders load your adapters and run this same `evaluate.py` on the hidden test set to compute your behavioral deltas and retention.

---

## Grading Rubric

| Component | What full credit looks like | Points |
|---|---|---|
| **Behavioral deltas** | <ul><li>Ask-rate up **≥25 points** on ambiguous.</li><li>Answer-rate degradation **≤5 points** on clear.</li><li>MMLU slice **within 3 points** of base.</li></ul> All three simultaneously — that's the assignment. | **35** |
| **Analysis** | <ul><li>β sweep interpreted.</li><li>Failure modes shown — the over-asker, the forgetter.</li><li>Transcripts well-chosen.</li></ul> | **35** |
| **Preference data quality** | Sensible construction; honest hand-verification with reported rejection rate. | **15** |
| **Reproducibility** | Adapters load; `evaluate.py` reproduces `results.json`. | **10** |
| **AI-use disclosure** | Synthesis usage precisely described. | **5** |
| **Total** | | **100** |

---

## Policies for Every Assignment

- **AI tools** — use them for anything except writing your analysis for you. Every submission includes
  a half-page **AI-use disclosure** appendix: which tools, for what, and how you verified the results.
  The appendix is graded. An empty or evasive disclosure on work that plainly used AI is an honor-code
  matter; a candid disclosure never costs you points on the work itself.
- **Late work** — 5 slip days for the semester, at most 2 per assignment, applied automatically in
  whole days. Beyond slip days: −10 points per day.
- **Reproducibility** — your repo must run from a clean clone: `README.md` with exact commands, pinned
  dependencies, fixed seeds. If the TA cannot reproduce your headline number within tolerance in 15
  minutes, the metric portion is graded from what reproduces.
- **Quizzes** — 15 minutes, in class, closed book/notes, no devices, the morning after the assignment
  is due. Five quizzes = 25% of the course grade; the lowest is dropped, so the four that count are
  6.25% each. No makeup sittings. A slip day moves the deadline but not the quiz.
- **Regrades** — written requests within 7 days of grade release, on Gradescope. The whole component is
  regraded, in either direction.
