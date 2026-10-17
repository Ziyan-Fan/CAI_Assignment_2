"""Build a reproducible 200-question development slice from public MMLU validation.

Run from the repository root with the CAI_A2 environment:
    python -m eval.build_mmlu_slice
An already downloaded source can be passed with --source-parquet PATH.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
from urllib.request import urlopen

import pyarrow.parquet as pq


REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
SOURCE_URL = (
    f"https://huggingface.co/datasets/cais/mmlu/resolve/{REVISION}/"
    "all/validation-00000-of-00001.parquet"
)
SOURCE_SHA256 = "66cdf0b090ccb657d18d13cd81e31fdc55c3467da9642ffb178653268a97c8ef"
ROOT = Path(__file__).resolve().parents[1]


def normalized(text):
    return re.sub(r"\s+", " ", text.casefold()).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", type=Path)
    parser.add_argument("--seed", type=int, default=6604)
    args = parser.parse_args()
    source = args.source_parquet or ROOT / ".cache/mmlu-validation.parquet"
    if not source.exists():
        if args.source_parquet:
            raise FileNotFoundError(source)
        source.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(SOURCE_URL, timeout=60) as response:
            source.write_bytes(response.read())
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Public MMLU source checksum does not match the pinned revision.")

    rows = pq.read_table(source).to_pylist()
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["subject"]].append((index, row))
    subjects = sorted(groups)
    if len(rows) != 1531 or len(subjects) != 57:
        raise ValueError("Unexpected MMLU validation split size or subject count.")

    rng = random.Random(args.seed)
    # Three questions per subject, with one extra for 29 seeded random subjects.
    extra_subjects = set(rng.sample(subjects, 200 % len(subjects)))
    selected = []
    source_rows = []
    seen = set()
    for subject in subjects:
        candidates = list(groups[subject])
        rng.shuffle(candidates)
        needed = 200 // len(subjects) + (subject in extra_subjects)
        kept = 0
        for index, row in candidates:
            question = row["question"]
            choices = list(row["choices"])
            answer = int(row["answer"])
            if not question.strip() or len(choices) != 4 or not 0 <= answer < 4:
                raise ValueError(f"Invalid source question at row {index}.")
            key = normalized(question + "\n" + "\n".join(choices))
            if key in seen:
                continue
            seen.add(key)
            identifier = f"mmlu_validation_{index:04d}"
            selected.append({"id": identifier, "subject": subject, "question": question,
                             "choices": choices, "answer": answer})
            source_rows.append({"id": identifier, "source_row": index, "subject": subject})
            kept += 1
            if kept == needed:
                break
        if kept != needed:
            raise ValueError(f"Not enough unique questions in {subject}.")

    training_path = ROOT / "data/preference_pairs.jsonl"
    if training_path.exists():
        training = {normalized(json.loads(line)["prompt"])
                    for line in training_path.read_text().splitlines() if line.strip()}
        if any(normalized(row["question"]) in training for row in selected):
            raise ValueError("A selected question duplicates a training prompt.")
    output = ROOT / "eval/mmlu_slice.jsonl"
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)
    output.write_text(payload, encoding="utf-8")
    provenance = {
        "purpose": "Public development-only retention check; not the course hidden test slice.",
        "dataset": "cais/mmlu", "dataset_url": "https://huggingface.co/datasets/cais/mmlu",
        "original_repository": "https://github.com/hendrycks/test",
        "license": "MIT (as declared by the dataset repository)",
        "revision": REVISION, "config": "all", "split": "validation",
        "source_url": SOURCE_URL, "source_sha256": SOURCE_SHA256,
        "source_count": len(rows), "seed": args.seed, "n": len(selected),
        "sampling": "3 per subject plus 1 for 29 seeded random subjects; unique question/choice records.",
        "subject_counts": dict(sorted(Counter(row["subject"] for row in selected).items())),
        "slice_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "source_rows": source_rows,
    }
    output.with_name("mmlu_slice.provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(selected)} public validation questions across {len(subjects)} subjects to {output}")


if __name__ == "__main__":
    main()
