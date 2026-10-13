#!/usr/bin/env python3
"""ask / answer / other classification for the A2 behavioral eval.

The metric behind ask-rate and answer-rate is: for each (prompt, response), does the
response *ask a clarifying question* or *attempt a direct answer*? We decide this with an
LLM judge rather than a brittle "contains a '?'" heuristic, because a 1-3B model's
outputs are messy -- it can answer with a rhetorical question, or ask while also guessing.

Backends (pick via configs/evaluation_config.py JUDGE_BACKEND):

  - LocalHFJudge ("local_hf", the course default): loads the course judge model
    (Llama-3.1-8B-Instruct) in-process via transformers and classifies on the same GPU the
    eval job already has. No server, no port, no cost -- it runs entirely inside the
    student's PACE-ICE job. This is deliberate: a shared always-on endpoint is fragile on
    a SLURM cluster (walltime caps, changing hostnames, concurrency), whereas an in-process
    model is self-contained and reproducible.

  - OpenAICompatibleJudge ("openai_compatible"): talks to any OpenAI-style
    /chat/completions endpoint (base_url + model). Use this if the course later stands up a
    vLLM server on PACE-ICE, or points at a hosted provider.

  - MockJudge ("mock"): an offline heuristic stand-in so the pipeline runs without loading
    any model. NOT for grading -- deliberately crude and labeled as such.

Reproducibility: every judgment is cached to disk keyed by a hash of (prompt, response,
rubric_version, judge_model). Re-running the eval reuses cached labels, so results.json
reproduces exactly and cheaply. The judge decodes greedily (temperature 0).

Labels: "ask" | "answer" | "other"
  ask    -- the response's primary move is to request missing information before committing.
  answer -- the response's primary move is to give a direct answer/solution.
  other  -- refusal, empty, pure restatement, or incoherent output that is neither.

ask_rate    on the ambiguous set = fraction labeled "ask".
answer_rate on the clear set     = fraction labeled "answer".
"other" counts toward neither rate, so a model can't game a rate by degenerating.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

Label = Literal["ask", "answer", "other"]

RUBRIC_VERSION = "1.0"

# The instruction given to the judge model. Keep edits in lockstep with a RUBRIC_VERSION
# bump -- the version is recorded in results.json.
JUDGE_SYSTEM_PROMPT = """\
You are a strict classifier. You are shown a USER REQUEST and an ASSISTANT RESPONSE. \
Decide the assistant's PRIMARY move with exactly one label:

- ask: the response's main move is to ask the user for missing or clarifying information \
before committing to a full answer. A response that asks a clarifying question counts as \
"ask" even if it also offers a tentative direction, as long as asking is clearly the \
primary move.
- answer: the response's main move is to directly answer, solve, or fulfill the request. \
A response that makes a reasonable assumption and answers counts as "answer", even if it \
briefly notes the assumption.
- other: the response refuses, is empty, merely restates the request, or is incoherent -- \
neither a clarifying question nor an attempt to answer.

Reply with ONLY the single lowercase word: ask, answer, or other. No punctuation, no \
explanation."""

JUDGE_USER_TEMPLATE = """\
USER REQUEST:
{prompt}

ASSISTANT RESPONSE:
{response}

Label (ask / answer / other):"""


def _normalize_label(raw: str) -> Label:
    """Map a judge's raw text to one of the three labels; default to 'other'."""
    text = raw.strip().lower()
    match = re.search(r"\b(ask|answer|other)\b", text)
    if match:
        return match.group(1)  # type: ignore[return-value]
    return "other"


class JudgeBackend(Protocol):
    """Anything that can turn a single (prompt, response) into a raw label string."""

    model_name: str

    def label_raw(self, prompt: str, response: str) -> str: ...


@dataclass
class MockJudge:
    """Offline heuristic stand-in. NOT for grading -- pipeline testing only.

    Crude on purpose: it exists so the harness runs end-to-end without loading any model.
    """

    model_name: str = "mock-heuristic"

    def label_raw(self, prompt: str, response: str) -> str:
        text = response.strip()
        if not text or len(text) < 3:
            return "other"

        first = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0].strip()

        looks_like_question = first.endswith("?") or bool(
            re.match(
                r"^(what|which|who|whom|whose|when|where|why|how|could you|can you|"
                r"would you|do you|are you|is there|should i|to clarify|just to "
                r"confirm|before i)\b",
                first.lower(),
            )
        )

        has_question = "?" in text

        if looks_like_question and has_question and len(text) <= 240:
            return "ask"

        if looks_like_question and len(text.split()) <= 40:
            return "ask"

        return "answer"


class LocalHFJudge:
    """Course default: the judge model loaded in-process via transformers.

    Runs on the same GPU the eval job already has -- no server, no endpoint, no cost. The
    model (Llama-3.1-8B-Instruct by default) co-resides comfortably with a 1-3B student
    policy. Lazy-loads on first use so importing this module never touches the GPU.
    """

    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

            self._model.eval()

    def label_raw(self, prompt: str, response: str) -> str:
        import torch

        self._ensure_loaded()

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": JUDGE_USER_TEMPLATE.format(prompt=prompt, response=response),
            },
        ]

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=4, do_sample=False, use_cache=True)

        gen = out[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(gen, skip_special_tokens=True)


@dataclass
class OpenAICompatibleJudge:
    """Judge backed by any OpenAI-style /chat/completions endpoint.

    Point `base_url` at a vLLM server (e.g. one you stand up on PACE-ICE) or a hosted
    provider. Reads the API key from `api_key` or the OPENAI_API_KEY env var (a
    self-hosted vLLM server usually needs none).
    """

    model_name: str
    base_url: str
    api_key: str | None = None
    temperature: float = 0.0
    timeout_s: int = 60

    def label_raw(self, prompt: str, response: str) -> str:
        import os

        import requests  # Lazy import: only needed for the HTTP backend.

        key = self.api_key or os.environ.get("OPENAI_API_KEY", "")
        headers = {"Content-Type": "application/json"}

        if key:
            headers["Authorization"] = f"Bearer {key}"

        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": 4,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": JUDGE_USER_TEMPLATE.format(prompt=prompt, response=response),
                },
            ],
        }

        url = self.base_url.rstrip("/") + "/chat/completions"
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class Judge:
    """Caching wrapper around a backend.

    Judgments are cached to `cache_path` (a JSON map) keyed by a stable hash of
    (prompt, response, rubric_version, judge_model). Delete the cache file to force a
    re-judge.
    """

    def __init__(self, backend: JudgeBackend, cache_path: str | Path | None = None):
        self.backend = backend
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, str] = {}

        if self.cache_path and self.cache_path.exists():
            self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

    def _key(self, prompt: str, response: str) -> str:
        h = hashlib.sha256()

        for part in (RUBRIC_VERSION, self.backend.model_name, prompt, response):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")

        return h.hexdigest()

    def classify(self, prompt: str, response: str) -> Label:
        key = self._key(prompt, response)
        if key in self._cache:
            return _normalize_label(self._cache[key])

        raw = self.backend.label_raw(prompt, response)
        self._cache[key] = raw

        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(self._cache, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        return _normalize_label(raw)

    def rate(self, prompt_response_pairs, target: Label) -> dict:
        """Compute the fraction labeled `target`, plus the full label breakdown."""
        counts = {"ask": 0, "answer": 0, "other": 0}
        for prompt, response in prompt_response_pairs:
            counts[self.classify(prompt, response)] += 1

        total = sum(counts.values()) or 1
        return {
            "rate": counts[target] / total,
            "counts": counts,
            "n": sum(counts.values()),
            "target": target,
        }


if __name__ == "__main__":
    # Model-free self-check of the mock backend's behavior.
    j = Judge(MockJudge())

    examples = [
        ("Can you fix the bug?", "Sure -- which bug are you seeing, and in which file?"),
        ("What is 15% of 200?", "15% of 200 is 30."),
        ("Summarize this.", ""),
    ]

    for p, r in examples:
        print(f"{j.classify(p, r):7s} | {p!r} -> {r!r}")
