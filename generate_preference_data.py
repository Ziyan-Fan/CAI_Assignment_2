"""
Part 1 -- build your preference data.

You construct at least MIN_PREFERENCE_PAIRS (1,000) preference pairs over two input
classes:

  - ambiguous: a request that is genuinely underspecified. The PREFERRED (chosen)
    response asks a specific clarifying question; the DISPREFERRED (rejected) response
    is a confident guess that commits to one interpretation.

  - clear: a fully-specified request. The PREFERRED (chosen) response answers directly;
    the DISPREFERRED (rejected) response is a needless clarifying question.

You MAY synthesize these with an LLM. The course model (Llama-3.1-8B-Instruct) runs on
PACE-ICE at no cost -- load it locally the same way eval/judge.py's LocalHFJudge does.
You MUST hand-verify a sample of HAND_VERIFY_SAMPLE (100) pairs and report the
rejection rate of your own synthetic data (spec, Part 1 + the "preference data quality" rubric line).

What is wired for you: the output schema + writer, the verification sampler, and the
rejection-rate computation. What you implement: `synthesize_pairs()` -- the actual data generation.

INTEGRITY: build your own prompts. Do NOT reuse the held-out eval prompts in
eval/held_out_*.jsonl as training data -- that is training on the test set.

Usage:
    python3 generate_preference_data.py --generate   # Synthesize + write pairs, then
                                                      # sample 100 for hand-verification.

    # ... now open data/hand_verified.jsonl and fill in each "verdict" ...
    python3 generate_preference_data.py --report      # Compute + print the rejection rate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import keyword
import random
import re
from pathlib import Path

from configs.train_config import (
    DATA_SEED,
    HAND_VERIFY_SAMPLE,
    MIN_PREFERENCE_PAIRS,
)
from configs.paths_config import HAND_VERIFIED_PATH, PREFERENCE_DATA_PATH

# Required by the unchanged, assignment-provided writer.
VALID_CLASSES = {"ambiguous", "clear"}
GENERATION_BATCH_SIZE = 8
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


def _parse_object(raw: str) -> dict | None:
    """Accept a single JSON object even if the model adds a code fence."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", raw):
        try:
            value, _ = decoder.raw_decode(raw[match.start():])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return None


def _prompt_key(s: str) -> str:
    # Keep numbers: different quantitative problems should not collapse into one.
    return re.sub(r"\s+", " ", s.casefold()).strip()


def _number(value: object, low: int, high: int) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean is not a numeric measurement")
    if isinstance(value, str) and not re.fullmatch(r"[0-9]{1,5}", value.strip()):
        raise ValueError("non-integer measurement")
    if not isinstance(value, (int, str)):
        raise ValueError("measurement must be an exact integer")
    number = int(value)
    if not low <= number <= high:
        raise ValueError("measurement out of bounds")
    return number


def _safe_noun(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("missing noun")
    noun = value.strip().lower()
    if (not 3 <= len(noun) <= 40 or not re.fullmatch(r"[a-z][a-z -]*", noun)
            or len(noun.split()) > 5):
        raise ValueError("bad noun")
    return noun


def _safe_text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("missing source text")
    # Reject line breaks BEFORE whitespace normalization, which would erase them.
    if any(bad in value for bad in ("\n", "\r", "```")):
        raise ValueError("bad source text")
    text = re.sub(r"[ \t\f\v]+", " ", value).strip().strip('"')
    if not 25 <= len(text) <= 180 or not 6 <= len(text.split()) <= 32:
        raise ValueError("source text must be six to thirty-two words")
    return text


# Deliberately narrow vocabulary: translations are composed from verified pieces,
# rather than relying on a generative translation as ground truth.
SPANISH_NOUNS = {
    "window": "la ventana", "door": "la puerta", "book": "el libro",
    "notebook": "el cuaderno", "bag": "la bolsa", "box": "la caja",
    "drawer": "el cajón", "curtain": "la cortina", "suitcase": "la maleta",
    "cabinet": "el armario", "folder": "la carpeta", "gate": "la verja",
    "envelope": "el sobre", "package": "el paquete", "basket": "la cesta",
}


def _safe_name(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Z][a-z]{2,16}", value):
        raise ValueError("Invalid story character name")
    return value


# These are complete, Python-authored prepositional phrases, not model output.
# They deliberately do not name rooms, source documents, files, languages, dates,
# source code, or other values that might be withheld from an ambiguous request.
_NEUTRAL_CONTEXTS = (
    "a routine planning session", "a classroom exercise", "a personal project",
    "an upcoming workshop", "a short practice session", "an independent assignment",
    "a weekly work session", "a general review task", "an introductory exercise",
    "a collaborative activity", "a basic training session", "a community project",
    "a regular preparation session", "a study group activity",
    "a simple practice task", "an administrative exercise",
    "an informal workshop", "a small group activity",
    "a routine work task", "a personal learning exercise",
    "a general office project", "an ordinary study session",
    "a practical training exercise", "a short preparation exercise",
)


# Fixed, context-free task topics.  Model-supplied `thing` is NOT used in
# prompts: a plausible noun can covertly contain the hidden price, date,
# sentence, source path or meeting room (including spelled-out numerals).
# The *model still creates* the task-specific numbers/text/names/rooms.
_CONTROLLED_TOPICS = {
    "unit_cost": ("notebooks", "paperbacks", "pens", "folders", "tickets", "markers",
                  "pencils", "postcards", "erasers", "mugs", "stickers", "keychains",
                  "sketchbooks", "notepads", "envelopes", "water bottles"),
    "hourly_cost": ("tutoring", "garden care", "editing", "bike repair", "painting",
                    "pet sitting", "house cleaning", "window washing", "translation",
                    "equipment rental", "computer support", "music lessons"),
    "distance": ("delivery van", "shuttle bus", "train", "car", "taxi",
                 "school bus", "minibus", "coach", "truck"),
    "area": ("garden", "stage", "courtyard", "classroom floor", "greenhouse",
             "patio", "playground", "workshop floor", "art studio"),
    "event_tables": ("fundraiser", "family reunion", "school banquet", "reception",
                     "community dinner", "graduation party", "volunteer luncheon",
                     "team celebration", "book club dinner", "charity breakfast"),
    "recipe_scale": ("bread dough", "pancake mix", "cookie dough", "granola",
                     "muffin batter", "pasta salad", "pizza dough", "fruit jam"),
    "inventory": ("pencils", "books", "stickers", "folders", "envelopes",
                  "notebooks", "pens", "markers", "bottles", "labels", "erasers"),
    "budget": ("office supplies", "school supplies", "a keyboard", "gardening tools",
               "craft materials", "a desk lamp", "travel supplies", "a backpack"),
    "workout_time": ("interval running", "cycling", "bodyweight training",
                     "stationary biking", "jump rope", "circuit training"),
    "word_count": ("project summary", "course notes", "reading journal", "team update",
                   "meeting recap", "draft article", "field notes", "workshop notes"),
    "uppercase_text": ("meeting recap", "class notes", "project draft", "reading log",
                       "weekly journal", "workshop notes", "travel diary"),
    "debug_code": ("numbers", "values", "scores", "items", "samples", "entries",
                   "readings", "measurements", "records"),
    "debug_zero_division": ("total", "balance", "amount", "score", "count",
                            "distance", "result", "quantity", "value"),
    "document_deadline": ("vendor onboarding", "supplier audit", "budget review",
                          "project launch", "safety inspection", "team training",
                          "contract renewal", "quarterly filing", "staff meeting"),
    "document_owner": ("project kickoff", "supplier update", "team review",
                       "equipment request", "budget revision", "quarterly report",
                       "operations briefing", "maintenance update"),
    "translation": ("language practice", "study materials", "travel vocabulary",
                    "language homework", "practice exercises", "course packet",
                    "class assignment", "phrasebook notes"),
    "story_continue": ("summer adventure", "mystery story", "forest journey",
                       "adventure draft", "fiction assignment", "travel story",
                       "creative exercise", "fantasy chapter"),
    "file_move": ("quarterly budget", "project plan", "reading list", "team schedule",
                  "safety checklist", "meeting agenda", "study outline", "travel notes"),
    "meeting_room": ("weekly review", "project update", "team check-in", "staff briefing",
                     "planning meeting", "training workshop", "budget review",
                     "status update", "project kickoff"),
}

# Task-specific limits prevent mechanically correct but implausible scenarios.
_NUMERIC_LIMITS = {
    "unit_cost": ((2, 24), (2, 60)),
    "hourly_cost": ((1, 12), (15, 120)),
    "distance": ((1, 10), (10, 75)),
    "area": ((2, 30), (2, 30)),
    "event_tables": ((10, 120), (4, 12)),
    "recipe_scale": ((1, 8), (10, 200)),
    "inventory": ((2, 30), (3, 40)),
    "budget": ((5, 80), (5, 100)),
    "workout_time": ((2, 8), (1, 3)),
    "debug_code": ((3, 8), None),
    "debug_zero_division": ((2, 30), None),
    "document_deadline": ((1, 28), None),
}


def _controlled_topic(kind: str, spec: dict) -> str:
    """Select a vetted noun independently of Llama's potentially leaky `thing`."""
    options = _CONTROLLED_TOPICS[kind]
    # Untrusted `thing` can affect a deterministic choice, but its *words*
    # can never get into the prompt. Different source specs diversify topics.
    return _variant({"kind": kind, "spec": spec, "topic_policy": 6}, *options)


def _numeric_inputs(kind: str, spec: dict) -> tuple[int | None, int | None]:
    if kind not in _NUMERIC_LIMITS:
        return None, None
    a_limits, b_limits = _NUMERIC_LIMITS[kind]
    a = _number(spec.get("a"), *a_limits)
    b = _number(spec.get("b"), *b_limits) if b_limits else None
    return a, b


def _safe_context(value: object) -> str:
    """Accept ONLY contexts selected from the audited Python allowlist."""
    if not isinstance(value, str) or value not in _NEUTRAL_CONTEXTS:
        raise ValueError("context must be a Python-selected neutral phrase")
    return value


def _normalize_phrase(value: str) -> str:
    """Punctuation-insensitive phrase matching for accidental answer leakage."""
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _note_reference(noun: str) -> str:
    """Avoid duplicate note/notes suffixes in source-text task descriptions."""
    return noun if noun.endswith((" note", " notes")) else f"{noun} note"


def _meeting_reference(noun: str) -> str:
    """Don't append 'meeting' if the topic already names a meeting."""
    return noun if noun.casefold().endswith(" meeting") else f"{noun} meeting"


def _story_reference(noun: str) -> str:
    """Give the story a natural noun phrase without repeating its type."""
    if noun.casefold().endswith((" story", " draft", " chapter", " assignment", " exercise")):
        return noun
    return f"{noun} story"


def _file_move_path(topic: str) -> str:
    """Do not append 'notes' twice to an already complete file subject."""
    subject = topic if topic.endswith(" notes") else f"{topic} notes"
    slug = subject.replace(" ", "_").replace("-", "_")
    return f"~/Downloads/{slug}.txt"


def _hidden_value(kind: str, spec: dict) -> str | None:
    """Exact named/source value that must not appear in the shortened prompt."""
    if kind == "meeting_room":
        return str(spec["room"])
    if kind == "document_owner":
        return str(spec["name"])
    if kind == "document_deadline":
        return f'{int(spec["month"]):02d}/{int(spec["a"]):02d}/2026'
    if kind == "translation":
        return f'Please {spec["action"]} the {spec["object"]}.'
    if kind in {"word_count", "uppercase_text"}:
        return str(spec["text"])
    if kind == "file_move":
        # The generated topic is controlled, not the model-supplied `thing`.
        return _file_move_path(_controlled_topic(kind, spec))
    return None


def _contains_hidden_value(kind: str, spec: dict, prompt: str) -> bool:
    """Reject task topics or contexts that already state a withheld named fact.

    Numerical overlap alone is *not* a leak: mentioning 7 books does not reveal
    their $7 unit price. We only reject explicit hidden strings or their
    punctuation/case-normalized spelling.
    """
    hidden = _hidden_value(kind, spec)
    if not hidden:
        return False
    normalized_hidden = _normalize_phrase(hidden)
    normalized_prompt = _normalize_phrase(prompt)
    return bool(normalized_hidden and
                f" {normalized_hidden} " in f" {normalized_prompt} ")


def _neutral_context(kind: str, spec: dict, core: str) -> str:
    """Choose context deterministically, independently of arbitrary model text."""
    # Vary context even when Llama repeats an identical topic. A fixed list
    # prevents paraphrased leaks such as 'east meeting room' for an east room.
    key = {k: v for k, v in spec.items() if k != "context"}
    digest_input = {"kind": kind, "spec": key, "context_version": 5}
    first = _variant(digest_input, *tuple(str(i) for i in range(len(_NEUTRAL_CONTEXTS))))
    offset = int(first)
    for i in range(len(_NEUTRAL_CONTEXTS)):
        phrase = _NEUTRAL_CONTEXTS[(offset + i) % len(_NEUTRAL_CONTEXTS)]
        if not _contains_hidden_value(kind, spec, phrase):
            return phrase
    raise ValueError("no neutral context is safe for this task")


def _add_context(core: str, context: str, spec: dict) -> str:
    """Insert a complete, article-correct phrase; never use raw model context."""
    phrase = _safe_context(context)
    return _variant({"neutral_context": phrase, "spec": {k: v for k, v in spec.items()
                                                    if k != "context"}},
                    f"I'm preparing this for {phrase}. {core}",
                    f"{core} It's part of {phrase}.",
                    f"I'm doing this as part of {phrase}. {core}",
                    f"{core} I'm working on it for {phrase}.")


def _starts_with_article(noun: str) -> str:
    return "an" if noun[0].lower() in "aeiou" else "a"


def _assumed_answer(task: dict, answer: str) -> str:
    """State the unsupported input, preserving the canonical answer verbatim."""
    fact = task["missing_fact"]
    kind = task["kind"]
    # Both styles are wrong for the ambiguous prompt, for exactly one reason:
    # the missing input is assumed rather than requested. The canonical result
    # is never changed, so there is no second factual/capitalization error.
    style = _variant({"prompt": task["ambiguous_prompt"], "style": "guess_v5"},
                     "explicit", "confident", "confident", "explicit")
    if style == "confident":
        return answer
    if kind == "translation":
        return (f'Assuming the English sentence is "{task["source_sentence"]}":\n'
                f'{answer}')
    if kind == "uppercase_text":
        return f"Assuming the text is:\n{fact.removeprefix('Text: ')}\n{answer}"
    if kind == "story_continue":
        return (f"Assuming the opening sentence is:\n"
                f"{fact.removeprefix('Opening sentence: ')}\n{answer}")
    if "\n" in fact:
        return f"Assuming the missing material is:\n{fact}\n{answer}"
    condition = fact.strip().rstrip(".")
    return f"Assuming {condition[0].lower() + condition[1:]}:\n{answer}"


def _variant(spec: dict, *options: str) -> str:
    """Select a reproducible wording variant; no dependence on Python hash seed."""
    key = json.dumps(spec, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.blake2s(key.encode("utf-8"), digest_size=4).digest()
    return options[int.from_bytes(digest, "big") % len(options)]


def _build_task(kind: str, spec: dict) -> dict:
    """Stage 2: Python assembles and withholds the exact essential input.

    The model selects original scenarios and values, but it does not decide
    whether the final request is ambiguous or compute the correct result.
    Every clear task adds exactly one fact that makes the answer possible.
    """
    if not isinstance(spec, dict):
        raise ValueError("bad task specification")
    # The model may invent a leaky `thing` such as "seven dollar notebooks",
    # "the October deadline", or "please open the door". Never place it in a
    # prompt. Use Python-approved task topics and family-specific numeric bounds.
    noun = _controlled_topic(kind, spec)
    note_ref = _note_reference(noun)
    a, b = _numeric_inputs(kind, spec)

    if kind == "unit_cost":
        core = _variant(spec,
            f"I am buying {a} {noun}. What is the exact total cost in dollars?",
            f"How much will {a} {noun} cost altogether? Give the exact dollar total.",
            f"Please calculate the total cost in dollars for {a} {noun}.")
        fact = f"Each one costs ${b}."
        answer = f"The total cost is ${a * b}."
        question = "How much does each one cost?"
    elif kind == "hourly_cost":
        hours = "hour" if a == 1 else "hours"
        core = _variant(spec,
            f"I need to pay for {a} {hours} of {noun}. What is the exact cost in dollars?",
            f"What is the exact bill for {a} {hours} of {noun} in dollars?",
            f"Calculate how much I owe for {a} {hours} of {noun}.")
        fact = f"The rate is ${b} per hour."
        answer = f"The total cost is ${a * b}."
        question = "What is the hourly rate?"
    elif kind == "distance":
        hours = "hour" if a == 1 else "hours"
        core = _variant(spec,
            f"Our {noun} travels for {a} {hours} at a steady speed. How many miles does it travel? Give the exact number.",
            f"How far, in miles, does a {noun} travel in {a} {hours} at a constant speed?",
            f"Calculate the exact distance covered by a {noun} during a steady {a}-hour trip.")
        fact = f"Its speed is {b} miles per hour."
        answer = f"It travels {a * b} miles."
        question = "What is its speed in miles per hour?"
    elif kind == "area":
        core = _variant(spec,
            f"A rectangular {noun} is {a} meters long. What is its exact area in square meters?",
            f"My rectangular {noun} measures {a} meters in length. Find its area in square meters.",
            f"Calculate the precise square-meter area of a {a}-meter-long rectangular {noun}.")
        fact = f"It is {b} meters wide."
        answer = f"The area is {a * b} square meters."
        question = "How wide is the rectangle?"
    elif kind == "event_tables":
        core = _variant(spec,
            f"We are seating {a} guests at our {noun}. How many whole tables are needed at minimum?",
            f"For our {noun}, what is the minimum number of tables required to seat {a} people?",
            f"Help me calculate how many tables we'll need for {a} attendees at the {noun}.")
        fact = f"Each table seats {b} guests."
        tables = (a + b - 1) // b
        answer = f"You need at least {tables} {'table' if tables == 1 else 'tables'}."
        question = "How many guests can sit at each table?"
    elif kind == "recipe_scale":
        batches = "batch" if a == 1 else "batches"
        core = _variant(spec,
            f"I am making {a} {batches} of {noun}. How many grams of the key ingredient do I need exactly?",
            f"For {a} {batches} of {noun}, calculate the total grams of the key ingredient.",
            f"How many grams of the key ingredient are needed to prepare {a} {batches} of {noun}?")
        fact = f"One batch needs {b} grams of the key ingredient."
        answer = f"You need {a * b} grams of the key ingredient."
        question = "How many grams of the key ingredient does one batch need?"
    elif kind == "inventory":
        core = _variant(spec,
            f"There are {a} boxes of {noun} in storage. How many items are there in total?",
            f"I have {a} boxes of {noun}. Please calculate the total number of items.",
            f"Find the overall item count in a shipment of {a} boxes of {noun}.")
        fact = f"Each box contains {b} items."
        answer = f"There are {a * b} items in total."
        question = "How many items are in each box?"
    elif kind == "budget":
        # The withheld spending amount is always smaller than the budget.
        budget, spent = a * 5 + b, b
        core = _variant(spec,
            f"I have a budget of ${budget} for {noun}. How many dollars remain after the purchase?",
            f"If I set aside ${budget} for {noun}, what balance will I have after buying it?",
            f"Calculate the money remaining from a ${budget} budget for {noun} after the purchase.")
        fact = f"The purchase costs ${spent}."
        answer = f"You have ${budget - spent} remaining."
        question = "How much does the purchase cost?"
    elif kind == "workout_time":
        core = _variant(spec,
            f"My {noun} workout has {a} exercise intervals of 3 minutes each, followed by {a} rest intervals. How many minutes does the entire workout last exactly?",
            f"How long is a {noun} workout with {a} three-minute exercise segments and {a} rest breaks?",
            f"Compute the total duration of {a} rounds of {noun}, each including 3 minutes of exercise and one rest interval.")
        fact = f"Each rest interval lasts {b} minutes."
        answer = f"The workout lasts {a * (3 + b)} minutes in total."
        question = "How many minutes does each rest interval last?"
    elif kind == "word_count":
        source = _safe_text(spec.get("text"))
        core = _variant(spec,
            f"How many words are in the exact paragraph from my {note_ref}? Please give the count.",
            f"Count the words in the paragraph from my {note_ref}. I need the exact count.",
            f"What is the word count of the specific paragraph in my {note_ref}?")
        fact = f"The paragraph is: {source}"
        count = len(source.split())
        answer = f"The paragraph contains {count} words."
        question = "Could you provide the paragraph to count?"
    elif kind == "uppercase_text":
        source = _safe_text(spec.get("text"))
        core = _variant(spec,
            f"Convert the exact text in my {note_ref} into uppercase, preserving its words and punctuation.",
            f"Please uppercase the sentence from my {note_ref} without changing punctuation.",
            f"Turn the text from my {note_ref} into all caps, keeping the original wording.")
        fact = f"Text: {source}"
        answer = source.upper()
        question = "What text would you like converted to uppercase?"
    elif kind == "debug_code":
        if a > 8:
            raise ValueError("debug snippet must contain 3 to 8 elements")
        var = noun.replace(" ", "_").replace("-", "_")
        if not var.isidentifier() or keyword.iskeyword(var) or var in {"print", "list", "int", "len"}:
            raise ValueError("invalid code variable")
        core = _variant(spec,
            f"Identify the exact line that fails in my Python {noun} list snippet and explain the error.",
            f"Which line crashes in my Python {noun} list code, and why?",
            f"Diagnose the specific failing line in my {noun} Python snippet.")
        values = ", ".join(str(i * 2) for i in range(1, a + 1))
        fact = f"Python code:\n```python\n{var} = [{values}]\nprint({var}[{a}])\n```"
        answer = (f"Line 2, `print({var}[{a}])`, raises IndexError because "
                  f"the list has {a} elements and its last valid index is {a-1}.")
        question = "Can you share the Python code so I can identify the failing line?"
    elif kind == "debug_zero_division":
        var = noun.replace(" ", "_").replace("-", "_")
        if not var.isidentifier() or keyword.iskeyword(var) or var in {"print", "list", "int", "len"}:
            raise ValueError("invalid variable name")
        core = _variant(spec,
            f"Identify the exact runtime error in my Python {noun} script.",
            f"What exception does my Python {noun} script raise, and on which line?",
            f"Please diagnose the specific Python error in my {noun} program.")
        fact = f"Python code:\n```python\n{var} = {a}\nprint({var} / 0)\n```"
        answer = "Line 2 raises ZeroDivisionError because the code divides by zero."
        question = "Can you share the Python code so I can identify the error?"
    elif kind == "document_deadline":
        day = a
        month = _number(spec.get("month"), 1, 12)
        core = _variant(spec,
            f"What deadline is stated in the {noun} memo I wrote? Give the exact date.",
            f"Find the exact due date mentioned in my {noun} memo.",
            f"According to the memo I wrote about {noun}, when is the report due?")
        fact = f"Memo excerpt: The {noun} report must be submitted by {month:02d}/{day:02d}/2026."
        answer = f"The memo states a deadline of {month:02d}/{day:02d}/2026."
        question = "Can you share the memo or the excerpt containing the deadline?"
    elif kind == "document_owner":
        name = _safe_name(spec.get("name"))
        core = _variant(spec,
            f"Who signed my {noun} memo? Give the name exactly as it appears.",
            f"What is the signer's name on my {noun} memo?",
            f"Please identify the person who signed the {noun} memo I wrote.")
        fact = f"Memo excerpt: The {noun} review is complete. Signed by {name}."
        answer = f"The memo was signed by {name}."
        question = "Could you share the memo or its signature line?"
    elif kind == "translation":
        obj = spec.get("object")
        action = spec.get("action")
        if not isinstance(obj, str) or obj not in SPANISH_NOUNS:
            raise ValueError("unverified translation object")
        if action not in ("open", "close"):
            raise ValueError("unverified translation action")
        core = _variant(spec,
            f"Translate the exact English sentence from my {note_ref} into Spanish.",
            f"Please provide a Spanish translation of the sentence in my {note_ref}.",
            f"What is the Spanish translation of the English sentence from my {note_ref}?")
        source_sentence = f"Please {action} the {obj}."
        fact = f"English sentence: {source_sentence}"
        verb = "abre" if action == "open" else "cierra"
        answer = f"Por favor, {verb} {SPANISH_NOUNS[obj]}."
        question = "What English sentence would you like translated into Spanish?"
    elif kind == "file_move":
        # Use one noun phrase and one path builder for prompt, fact, and command.
        subject = noun if noun.endswith(" notes") else f"{noun} notes"
        source_path = _file_move_path(noun)
        core = _variant(spec,
            f"Give the exact terminal command to move my {subject} into ~/Documents without overwriting existing files.",
            f"What shell command moves my {subject} file to ~/Documents safely without overwriting?",
            f"Write a Linux command for moving my {subject} file into ~/Documents, preserving any files already there.")
        fact = f"Its full source path is {source_path}."
        answer = f"mv -n -- {source_path} ~/Documents/"
        question = "What is the exact source file path?"
    elif kind == "meeting_room":
        room = _safe_noun(spec.get("room"))
        meeting_ref = _meeting_reference(noun)
        core = _variant(spec,
            f"What is the exact room assigned to our {meeting_ref}?",
            f"Tell me which room the {meeting_ref} is scheduled to use.",
            f"Where exactly is the {meeting_ref} taking place, according to the room booking?")
        fact = f"The booking assigns the meeting to {room}."
        answer = f"The meeting is assigned to {room}."
        question = "Can you share the room booking or the assigned room?"
    elif kind == "story_continue":
        name = _safe_name(spec.get("name"))
        place = _safe_noun(spec.get("place"))
        obj = _safe_noun(spec.get("object"))
        story_ref = _story_reference(noun)
        core = _variant(spec,
            f"Continue my {story_ref} featuring {name} by adding one sentence after its opening line.",
            f"Write one sentence that directly follows the opening line of my {story_ref} starring {name}.",
            f"Extend my {story_ref} about {name} by adding a sentence after its opening line.")
        fact = (f"Opening sentence: At sunset, {name} found "
                f"{_starts_with_article(obj)} {obj} near the {place}.")
        answer = (f"Wondering who had left it there, {name} decided "
                  "to look for its owner.")
        question = "Could you share your story's opening sentence so I can continue it?"
    else:
        raise ValueError("unrecognized task type")

    # Add safe discourse variation to exact-answer tasks without changing the
    # problem or introducing any missing numeric input.
    if kind in {"unit_cost", "hourly_cost", "distance", "area",
                "event_tables", "recipe_scale", "inventory", "budget",
                "workout_time"}:
        core = _variant({"kind": kind, "spec": spec, "wrapper_version": 6},
                        core,
                        "I'm checking a calculation. " + core,
                        "Quick question: " + core,
                        "For my notes: " + core,
                        "I want to double-check this. " + core,
                        "Could you work this out? " + core)

    # Add a Python-selected neutral context. Never incorporate spec['context']:
    # model-generated text may reveal the very source fact we removed.
    source_dependent = {
        "word_count", "uppercase_text", "debug_code", "debug_zero_division",
        "document_deadline", "document_owner", "translation",
        "story_continue", "file_move", "meeting_room",
    }
    if kind in source_dependent:
        core = _add_context(core, _neutral_context(kind, spec, core), spec)

    # Do not accept a topic that reveals the hidden value even without context.
    if _contains_hidden_value(kind, spec, core):
        raise ValueError("ambiguous prompt already reveals the withheld value")

    # Python, not Llama, performs the complete -> missing-input conversion.
    complete = core + "\n" + fact
    short = complete[:-len(fact)-1]
    assert short == core and complete.endswith(fact)
    clear = _variant({"kind": kind, "spec": spec, "assembly": True},
                     complete, fact + "\n" + core,
                     core + "\nRelevant information: " + fact)
    # Always a pair of opposite behavior targets on the same underlying task.
    return {"kind": kind, "ambiguous_prompt": short, "clear_prompt": clear,
            "missing_fact": fact, "question": question,
            "topic": noun, "expected": answer,
            **({"source_sentence": source_sentence} if kind == "translation" else {})}


def _answer_matches(answer: str, task: dict) -> bool:
    """Strict comparison for factual cases, never substring-based checks."""
    return isinstance(answer, str) and answer.strip() == task["expected"]


# Story continuations use the known, contextual Python fallback. Without a
# reliable semantic check, a model-generated sentence could be unrelated.


# Controlled, task-appropriate *unnecessary* questions. They cannot change
# the canonical answer or ask for an essential input.
_OPTIONAL_QUESTIONS = {
    "numeric": (
        "Would you prefer a one-line calculation?",
        "Should I format the result as a bullet point?",
        "Would you like a brief explanation of the arithmetic?",
        "Would you prefer the result displayed in a table?",
        "Do you want the steps numbered?",
        "Would you like me to bold the final number?",
        "Should I include a short calculation summary?",
        "Would you prefer a compact answer format?",
    ),
    "text": (
        "Should I enclose the result in quotation marks?",
        "Would you prefer the output in a code block?",
        "Would you like a short explanatory note alongside the result?",
        "Should I display the result as an indented quotation?",
        "Would you like me to label the output as the final text?",
    ),
    "debug": (
        "Would you like the explanation in bullet points?",
        "Should I number the diagnostic steps?",
        "Would you prefer a shorter explanation of the error?",
        "Should I emphasize the failing line in bold?",
        "Would you like the explanation in a small table?",
    ),
    "document": (
        "Should I bold the extracted answer?",
        "Would you prefer the answer on its own line?",
        "Would you like the extracted information in a table?",
        "Should I include a short introduction before the answer?",
        "Would you like a bullet point instead of a sentence?",
    ),
    "writing": (
        "Would you prefer a slightly more descriptive style?",
        "Should the writing have a more formal tone?",
        "Would you like the response separated into paragraphs?",
        "Should I use a more conversational voice?",
    ),
    "shell": (
        "Would you like the command in a fenced code block?",
        "Should I put a short label above the command?",
        "Would you prefer a brief explanation below the command?",
        "Should I format the command as a numbered step?",
    ),
}


def _needless_question(task: dict) -> str:
    """A reproducible, controlled unnecessary question chosen for this task."""
    kind = task["kind"]
    group = ("numeric" if kind in {
        "unit_cost", "hourly_cost", "distance", "area", "event_tables",
        "recipe_scale", "inventory", "budget", "workout_time", "word_count"
    } else "text" if kind in {"uppercase_text", "translation"}
        else "debug" if kind in {"debug_code", "debug_zero_division"}
        else "document" if kind in {"document_deadline", "document_owner",
                                    "meeting_room"}
        else "shell" if kind == "file_move" else "writing")
    return _variant({"group": group, "prompt": task["clear_prompt"]},
                    *_OPTIONAL_QUESTIONS[group])


# Carefully verified question paraphrases: each question asks for the *same*
# essential input withheld from the ambiguous prompt. The question in the task
# remains the third form, giving three phrasings per family.
_QUESTION_ALTERNATIVES = {
    "unit_cost": (
        "What is the unit price of each item?",
        "How much does one item cost?",
    ),
    "hourly_cost": (
        "How much is charged per hour?",
        "What rate should I use for each hour of work?",
    ),
    "distance": (
        "How many miles per hour does it travel?",
        "What is the vehicle's constant speed?",
    ),
    "area": (
        "What is the rectangle's width in meters?",
        "How many meters wide is it?",
    ),
    "event_tables": (
        "How many people can each table seat?",
        "What is the seating capacity per table?",
    ),
    "recipe_scale": (
        "What amount of the key ingredient is needed per batch, in grams?",
        "How many grams go into one batch?",
    ),
    "inventory": (
        "How many items does each box hold?",
        "What is the number of items per box?",
    ),
    "budget": (
        "What is the purchase price?",
        "How many dollars will the purchase cost?",
    ),
    "workout_time": (
        "What is the duration of each rest interval?",
        "How many minutes does each rest break take?",
    ),
    "word_count": (
        "Could you paste the paragraph you want counted?",
        "What is the full paragraph whose words I should count?",
    ),
    "uppercase_text": (
        "Could you paste the original text to convert?",
        "What exact text should I turn into uppercase?",
    ),
    "debug_code": (
        "Could you paste the code that fails?",
        "What is the Python snippet you want me to debug?",
    ),
    "debug_zero_division": (
        "Could you show me the Python script?",
        "What source code should I examine to find the error?",
    ),
    "document_deadline": (
        "Could you share the memo text so I can find its deadline?",
        "Can you provide the relevant passage from the memo?",
    ),
    "document_owner": (
        "Could you provide the memo's signature line?",
        "Can you paste the memo or the section showing who signed it?",
    ),
    "translation": (
        "Could you paste the English sentence to translate?",
        "What is the original English sentence?",
    ),
    "story_continue": (
        "Could you paste the opening line of your story?",
        "What is the exact first sentence I should continue?",
    ),
    "file_move": (
        "What is the file's full current path?",
        "Could you provide the full source file path?",
    ),
    "meeting_room": (
        "Can you provide the room assignment from the booking?",
        "Could you share the booking details showing the meeting room?",
    ),
}


def _clarification(task: dict) -> str:
    """Select a tested question asking for the required input, with no LLM pass."""
    kind = task["kind"]
    return _variant({"kind": kind, "prompt": task["ambiguous_prompt"],
                     "question_version": 8},
                    task["question"], *_QUESTION_ALTERNATIVES[kind])


def _make_pairs(task: dict, raw: dict | None = None) -> tuple[dict, dict]:
    """Construct both preference contrasts without an unnecessary Llama pass.

    `raw` is retained only for backwards compatibility with local test callers;
    unverified model text is never allowed to replace canonical answers.
    """
    answer = task["expected"]
    amb = {"class": "ambiguous", "prompt": task["ambiguous_prompt"],
           "chosen": _clarification(task), "rejected": _assumed_answer(task, answer)}
    clr = {"class": "clear", "prompt": task["clear_prompt"],
           "chosen": answer, "rejected": _needless_question(task)}
    return amb, clr


# The noun used for numerical/source tasks is intentionally never reused here.
# A completed project ("client presentation") makes sense for congratulations;
# an object ("postcards") or a vague variable ("values") does not.
_STANDALONE_TOPICS = {
    "project": (
        "the quarterly report", "the team website redesign",
        "the customer feedback survey", "the volunteer signup page",
        "the community garden proposal", "the annual budget plan",
        "the staff training guide", "the fundraising campaign",
        "the project kickoff presentation", "the school newsletter",
        "the product demo", "the onboarding checklist",
        "the neighborhood cleanup plan", "the research poster",
        "the software testing plan", "the library outreach project",
    ),
    "subject": (
        "workplace safety", "community outreach", "public speaking",
        "software testing", "time management", "urban gardening",
        "project planning", "digital accessibility", "customer service",
        "effective teamwork", "environmental sustainability",
        "budget planning", "data visualization", "volunteer coordination",
        "study strategies", "information security",
    ),
    "scene": (
        "an abandoned lighthouse", "a quiet railway station",
        "an old bookshop", "a snowbound village",
        "a hidden garden", "a mysterious museum",
        "a deserted observatory", "a coastal town",
        "a mountain cabin", "a midnight ferry",
        "a forgotten library", "a bustling street market",
        "a lonely watchtower", "a remote campsite",
        "a secret attic", "a rainy city square",
    ),
    "activity": (
        "planning a community fundraiser", "organizing a study group",
        "building a personal portfolio", "starting a neighborhood garden",
        "preparing a training workshop", "writing a project proposal",
        "making a household budget", "organizing a book club",
        "setting up a volunteer drive", "creating a simple website",
        "preparing a research poster", "planning a school showcase",
        "drafting a team newsletter", "organizing a shared workspace",
        "planning a charity walk", "building a reading routine",
    ),
    "workshop": (
        "basic first aid", "introductory Python", "public speaking",
        "workplace safety", "creative writing", "digital photography",
        "time management", "basic spreadsheet skills",
        "community gardening", "resume writing", "personal budgeting",
        "team communication", "scientific poster design",
        "presentation skills", "website accessibility",
        "volunteer coordination",
    ),
}

_STANDALONE_BANK_BY_VARIANT = (
    "project", "subject", "scene", "subject", "project", "activity",
    "subject", "subject", "project", "subject", "workshop", "project",
)


def _standalone_topic(index: int) -> str:
    """Use a task-appropriate topic, rotating deterministically without early repeats."""
    # This function is called for every fourth clear pair (indices 3,7,11,...).
    # Each template has its own topic rotation and at least 16 distinct topics.
    case = index // 4
    variant = case % 12
    occurrence = case // 12
    bank = _STANDALONE_TOPICS[_STANDALONE_BANK_BY_VARIANT[variant]]
    return bank[(occurrence + DATA_SEED % len(bank)) % len(bank)]


def _standalone_clear_pair(topic: str, index: int) -> dict:
    """Add answerable open-ended requests to discourage blanket over-clarification.

    Only a fraction of the clear class uses these independent examples; the
    rest retain their contrastive clear/ambiguous relationship.
    """
    # The `topic` argument is kept for compatibility with earlier local callers,
    # but Llama's numeric/inventory topic is deliberately ignored.
    variant = (index // 4) % 12
    topic = _standalone_topic(index)
    if variant == 0:
        prompt = f"Write a short, friendly thank-you note to a coworker who helped me with {topic}."
        answer = (f"Hi,\n\nThanks for helping me with {topic}. I really appreciate "
                  "your time and support.\n\nBest,\n[Your Name]")
        rejected = "Would you like the note to sound more casual?"
    elif variant == 1:
        prompt = f"Give three practical tips for organizing notes about {topic}."
        answer = ("Group related notes under clear headings, put key points at the "
                  "top, and review your notes regularly.")
        rejected = "Would you like those tips presented in a table?"
    elif variant == 2:
        prompt = f"Write a two-sentence fictional scene about {topic} and an unexpected letter."
        answer = (f"An unexpected letter arrived just as the conversation turned "
                  f"to {topic}. Its last line hinted at a secret no one had noticed.")
        rejected = "Should the scene have a mysterious or humorous tone?"
    elif variant == 3:
        prompt = f"Suggest a simple three-part outline for a short presentation about {topic}."
        answer = (f"Start by introducing {topic} and why it matters. Then explain "
                  "two or three key points with examples. Finish with a clear summary.")
        rejected = "Would you prefer a formal or conversational presentation style?"
    elif variant == 4:
        prompt = f"Draft a brief congratulations message for a colleague who finished {topic}."
        answer = (f"Congratulations on completing {topic}! Your hard work paid off, "
                  "and I hope you take time to celebrate the achievement.")
        rejected = "Should the message sound excited or reserved?"
    elif variant == 5:
        prompt = f"Create a short checklist for getting started with {topic}."
        answer = (f"1. Set a clear goal for {topic}. 2. Gather the materials "
                  "you'll need. 3. Identify the first practical task. "
                  "4. Check your progress after completing it.")
        rejected = "Would you like the checklist to use checkboxes?"
    elif variant == 6:
        prompt = f"Write a one-paragraph introduction to a guide about {topic}."
        answer = (f"Getting started with {topic} is easier when you break the "
                  "process into manageable steps. This guide covers what to prepare, "
                  "what to do first, and how to review your progress.")
        rejected = "Should the introduction sound more academic?"
    elif variant == 7:
        prompt = f"Suggest three section headings for a report on {topic}."
        answer = (f"1. Background and Goals of {topic}; "
                  "2. Key Findings and Considerations; 3. Next Steps.")
        rejected = "Would you prefer title case or sentence case headings?"
    elif variant == 8:
        prompt = f"Write a friendly message requesting a status update on {topic}."
        answer = (f"Hi, could you share a quick update on {topic} when you have "
                  "a chance? Thanks for your help!")
        rejected = "Would you like the message to be more formal?"
    elif variant == 9:
        prompt = f"Suggest three useful questions for a team discussion about {topic}."
        answer = ("What outcome are we trying to achieve? What obstacles might "
                  "prevent progress? Who will take responsibility for next steps?")
        rejected = "Do you want the questions sorted by importance?"
    elif variant == 10:
        prompt = f"Draft a brief reminder to bring materials for a workshop about {topic}."
        answer = (f"Reminder: please bring any notes or materials relevant to "
                  f"{topic} to the workshop. Thank you!")
        rejected = "Would you like the reminder to sound more enthusiastic?"
    else:
        prompt = f"Create a brief meeting agenda about {topic}."
        answer = (f"1. Review the goals for {topic}. "
                  "2. Discuss progress and outstanding questions. "
                  "3. Confirm action items and owners.")
        rejected = "Should I format the agenda as a table?"

    return {"class": "clear", "prompt": prompt,
            "chosen": answer, "rejected": rejected}


def synthesize_pairs(n: int) -> list[dict]:
    """
    TODO (student): produce `n` preference pairs, balanced across the two classes.

    Return a list of dicts, each:
        {"class": "ambiguous" | "clear",
         "prompt": "<user request>",
         "chosen": "<preferred response>",
         "rejected": "<dispreferred response>"}

    (The "id" field is added for you by write_pairs().)

    Approach is up to you. A common recipe: prompt an LLM to generate ambiguous and
    clear requests across varied domains, then for each request generate both a
    clarifying-question response and a direct-answer response, and assign chosen/rejected
    by the request's class. Keep DATA_SEED fixed wherever you sample, so your data is reproducible.
    """
    if n >= MIN_PREFERENCE_PAIRS:
        checkpoint = Path(PREFERENCE_DATA_PATH).with_suffix(".two_stage_v9.partial.jsonl")
        pairs = _synthesize_pairs(n, checkpoint_path=checkpoint,
                                  resume=checkpoint.exists())
        checkpoint.unlink(missing_ok=True)
        return pairs
    return _synthesize_pairs(n)


def _synthesize_pairs(n: int, *, checkpoint_path: str | Path | None = None,
                      resume: bool = False) -> list[dict]:
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n must be >= 0")
    if n == 0:
        return []

    from collections import Counter
    import torch
    from transformers import set_seed
    from utils.model_utils import load_tokenizer, load_base_model

    set_seed(DATA_SEED)
    rng = random.Random(DATA_SEED)
    tokenizer = load_tokenizer(MODEL_ID)
    model = load_base_model(MODEL_ID)
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    batch_size = GENERATION_BATCH_SIZE

    def generate_many(messages: list[list[dict]], max_new_tokens: int) -> list[str]:
        """Single left-padded batched pass; automatically split an OOM batch."""
        if not messages:
            return []
        prompts = [tokenizer.apply_chat_template(m, tokenize=False,
                                                  add_generation_prompt=True)
                   for m in messages]
        try:
            inputs = tokenizer(prompts, padding=True, truncation=True,
                               max_length=1500, return_tensors="pt").to(model.device)
            try:
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs, max_new_tokens=max_new_tokens,
                        do_sample=True, temperature=0.65, top_p=0.9,
                        pad_token_id=tokenizer.pad_token_id, use_cache=True)
                tails = outputs[:, inputs["input_ids"].shape[1]:]
                return tokenizer.batch_decode(tails, skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)
            finally:
                del inputs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(messages) == 1:
                raise
            middle = len(messages) // 2
            return (generate_many(messages[:middle], max_new_tokens)
                    + generate_many(messages[middle:], max_new_tokens))

    kinds = ["unit_cost", "hourly_cost", "distance", "area", "event_tables",
             "recipe_scale", "inventory", "budget", "workout_time",
             "word_count", "uppercase_text", "debug_code", "debug_zero_division",
             "document_deadline", "document_owner", "translation", "story_continue",
             "file_move", "meeting_room"]
    kinds_order = list(kinds)
    rng.shuffle(kinds_order)
    kind_index = 0
    seen: set[str] = set()
    pairs: list[dict] = []
    stats = Counter()
    targets = {"ambiguous": (n + 1) // 2, "clear": n // 2}
    counts = Counter()
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if resume:
            for line in checkpoint.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if (rec.get("class") not in targets
                        or any(not isinstance(rec.get(k), str) or not rec[k]
                               for k in ("prompt", "chosen", "rejected"))):
                    raise ValueError("Invalid checkpoint record")
                if counts[rec["class"]] >= targets[rec["class"]]:
                    raise ValueError("Checkpoint exceeds requested class quota")
                key = _prompt_key(rec["prompt"])
                if key in seen:
                    raise ValueError("Checkpoint contains duplicate prompts")
                pairs.append(rec)
                seen.add(key)
                counts[rec["class"]] += 1
            print(f"Resumed {len(pairs)}/{n} pairs", flush=True)
        else:
            checkpoint.write_text("", encoding="utf-8")

    spec_system = (
        "Invent short original scenarios for preference training and return ONE "
        "valid JSON OBJECT matching the requested schema. Supply only the "
        "scenario input fields (nouns, values, text); do not write responses "
        "or clarify the request. Keep everything realistic and distinct. "
        "Use plain English, no markdown, and integers without quotes. "
        "The Python generator supplies a safe noun/topic: model-provided "
        "thing text is ignored. Follow each requested numeric range."
    )

    schema = {
        "unit_cost": "thing: plural items being purchased; a: count; b: dollar price per item",
        "hourly_cost": "thing: short work service; a: hours worked; b: dollars per hour",
        "distance": "thing: vehicle such as shuttle bus or delivery van; a: trip hours; b: miles per hour",
        "area": "thing: rectangular physical area, such as garden or stage; a: length in meters; b: width in meters",
        "event_tables": "thing: event name like a reception or company dinner; a: total guests; b: seats per table",
        "recipe_scale": "thing: food or recipe; a: number of batches; b: grams of one ingredient per batch",
        "inventory": "thing: plural inventory items; a: number of boxes; b: items per box",
        "budget": "thing: planned purchase such as school supplies; a: rough budget scale; b: purchase cost",
        "workout_time": "thing: type of workout, such as interval running; a: exercise intervals; b: minutes per rest interval",
        "word_count": "thing: original two-to-five-word memo topic; text: original natural paragraph with 8-22 words",
        "uppercase_text": "thing: original two-to-five-word note topic; text: original natural sentence with 8-22 words",
        "debug_code": "thing: Python list variable name (one lowercase word); a: integer between 3 and 8",
        "debug_zero_division": "thing: Python variable name (one lowercase word); a: integer between 2 and 80",
        "document_deadline": "thing: original short memo topic; a: day of month 1-28; month: integer 1-12",
        "document_owner": "thing: original short memo topic; name: simple capitalized first name",
        "translation": "thing: original short memo topic; object: choose one from [window, door, book, notebook, bag, box, drawer, curtain, suitcase, cabinet, folder, gate, envelope, package, basket]; action: open or close",
        "story_continue": "thing: original short story topic; name: capitalized first name; place: physical landmark noun like bridge or library; object: singular tangible noun like key or lantern",
        "file_move": "thing: original short subject of a file such as project budget (two or three words)",
        "meeting_room": "thing: original short meeting topic; room: a two-or-three-word room name like conference room a",
    }
    # Llama is only used to invent *inputs*, never to write preference responses.
    recent_topics: dict[str, list[str]] = {kind: [] for kind in kinds}
    attempts = 0
    # Usually ~2 accepted examples per task; extra attempts handle malformed JSON.
    max_attempts = max(24, 10 * ((n + 1) // 2))
    while len(pairs) < n and attempts < max_attempts:
        batch_kinds = []
        remaining_tasks = (n - len(pairs) + 1) // 2
        for _ in range(min(batch_size, max(1, remaining_tasks))):
            if kind_index and kind_index % len(kinds_order) == 0:
                rng.shuffle(kinds_order)
            kind = kinds_order[kind_index % len(kinds_order)]
            kind_index += 1
            batch_kinds.append(kind)
        requests = []
        for kind in batch_kinds:
            if kind in _NUMERIC_LIMITS:
                (alo, ahi), b_range = _NUMERIC_LIMITS[kind]
                requested = (f"Return JSON fields 'thing' and 'a'; a is an "
                             f"integer from {alo} to {ahi}. ")
                if b_range:
                    blo, bhi = b_range
                    requested += f"Also include 'b', an integer from {blo} to {bhi}. "
                requested += schema[kind]
            else:
                requested = "Return JSON field 'thing'. " + schema[kind]
            if kind in ("word_count", "uppercase_text"):
                requested = "Return JSON fields 'thing' and 'text': " + schema[kind]
            elif kind == "document_deadline":
                requested += " Include 'month' as an integer from 1 to 12."
            elif kind == "document_owner":
                requested = "Return JSON fields 'thing', 'name': " + schema[kind]
            elif kind == "file_move":
                requested = "Return JSON field 'thing': " + schema[kind]
            elif kind == "meeting_room":
                requested = "Return JSON fields 'thing', 'room': " + schema[kind]
            elif kind == "translation":
                requested = "Return JSON fields 'thing', 'object', 'action': " + schema[kind]
            elif kind == "story_continue":
                requested = "Return JSON fields 'thing', 'name', 'place', 'object': " + schema[kind]
            if recent_topics[kind]:
                requested += (" Avoid reusing these recent topics or contexts: "
                              + "; ".join(recent_topics[kind][-10:]) + ".")
            requests.append([{"role": "system", "content": spec_system},
                             {"role": "user", "content": requested}])
        specs = generate_many(requests, max_new_tokens=150)
        attempts += len(batch_kinds)
        tasks = []
        # Catch duplicates within the current batch, not just earlier batches.
        pending_keys: set[str] = set()
        for kind, raw in zip(batch_kinds, specs):
            try:
                task = _build_task(kind, _parse_object(raw))
                short_key, full_key = (_prompt_key(task[k]) for k in
                                       ("ambiguous_prompt", "clear_prompt"))
                if (short_key == full_key or short_key in seen or full_key in seen
                        or short_key in pending_keys or full_key in pending_keys):
                    stats["duplicate_task"] += 1
                    continue
                pending_keys.update((short_key, full_key))
                tasks.append(task)
                # Untrusted context is intentionally ignored in this version.
                recent_topics[kind].append(task["topic"])
                if len(recent_topics[kind]) > 20:
                    recent_topics[kind] = recent_topics[kind][-20:]
            except (ValueError, TypeError, KeyError, OverflowError):
                stats["invalid_task_spec"] += 1

        if not tasks:
            continue
        new_pairs = []
        for task in tasks:
            amb, clr = _make_pairs(task)
            # Every fourth clear example uses an open-ended, answerable request.
            # This teaches direct answering beyond exact-answer tasks.
            if counts["clear"] % 4 == 3:
                topic = task["topic"]
                alternative = _standalone_clear_pair(topic, counts["clear"])
                if _prompt_key(alternative["prompt"]) not in seen:
                    clr = alternative
            for pair in (amb, clr):
                label = pair["class"]
                key = _prompt_key(pair["prompt"])
                if counts[label] >= targets[label] or key in seen:
                    continue
                seen.add(key)
                counts[label] += 1
                pairs.append(pair)
                new_pairs.append(pair)
        if checkpoint is not None and new_pairs:
            with checkpoint.open("a", encoding="utf-8") as f:
                for rec in new_pairs:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if attempts % (batch_size * 5) == 0 or len(pairs) >= n:
            print(f"Pairs: {len(pairs)}/{n} (ambiguous {counts['ambiguous']}/"
                  f"{targets['ambiguous']}, clear {counts['clear']}/"
                  f"{targets['clear']}); candidate tasks {attempts}; "
                  f"diagnostics {dict(stats)}", flush=True)

    if len(pairs) != n:
        raise RuntimeError(f"Only {len(pairs)}/{n} pairs after {attempts} task "
                           f"attempts. Diagnostics: {dict(stats)}")
    rng.shuffle(pairs)
    print(f"Generation completed: {len(pairs)} balanced pairs. "
          f"Diagnostics: {dict(stats)}", flush=True)
    return pairs


# ---------------------- Wired: everything below is provided --------------------------


def write_pairs(pairs: list[dict], path: str | Path = PREFERENCE_DATA_PATH) -> None:
    """Validates, assigns stable ids, and writes pairs as JSONL."""
    required = {"class", "prompt", "chosen", "rejected"}

    for i, p in enumerate(pairs):
        missing = required - p.keys()

        if missing:
            raise ValueError(f"pair #{i} missing keys {sorted(missing)}")

        if p["class"] not in VALID_CLASSES:
            raise ValueError(f"pair #{i} has invalid class {p['class']!r}")

    if len(pairs) < MIN_PREFERENCE_PAIRS:
        raise ValueError(
            f"Only {len(pairs)} pairs; the assignment requires at least " f"{MIN_PREFERENCE_PAIRS}."
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for i, p in enumerate(pairs):
            rec = {
                "id": f"pair_{i:06d}",
                "class": p["class"],
                "prompt": p["prompt"],
                "chosen": p["chosen"],
                "rejected": p["rejected"],
            }

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_amb = sum(1 for p in pairs if p["class"] == "ambiguous")
    n_clr = sum(1 for p in pairs if p["class"] == "clear")
    print(f"Wrote {len(pairs)} pairs ({n_amb} ambiguous, {n_clr} clear) -> {path}")


def sample_for_verification(
    pairs_path: str | Path = PREFERENCE_DATA_PATH,
    out_path: str | Path = HAND_VERIFIED_PATH,
    k: int = HAND_VERIFY_SAMPLE,
    seed: int = DATA_SEED,
) -> None:
    """Samples k pairs into a hand-verification file with a blank 'verdict' field.

    Open the file and set each pair's "verdict" to "accept" or "reject" by hand
    (a pair is a reject if the chosen/rejected labels are wrong, the prompt is
    mislabeled ambiguous/clear, or either response is low quality). Then run
    --report.
    """
    pairs = [
        json.loads(l)
        for l in Path(pairs_path).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    rng = random.Random(seed)
    sample = rng.sample(pairs, min(k, len(pairs)))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in sample:
            rec = dict(rec)
            rec["verdict"] = ""  # Fill with "accept" or "reject" by hand.
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(sample)} pairs to verify -> {out_path}")
    print("Fill in each 'verdict' (accept/reject) by hand, then run --report.")


def rejection_rate(path: str | Path = HAND_VERIFIED_PATH) -> float:
    """Computes the rejection rate over the hand-verified sample."""
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    verdicts = [str(r.get("verdict", "")).strip().lower() for r in rows]

    unfilled = [i for i, v in enumerate(verdicts) if v not in {"accept", "reject"}]
    if unfilled:
        raise ValueError(
            f"{len(unfilled)} of {len(rows)} verdicts are not 'accept'/'reject' "
            f"(first at row {unfilled[0]}). Finish hand-verifying before reporting."
        )

    n_reject = sum(1 for v in verdicts if v == "reject")
    rate = n_reject / len(rows)

    print(f"Hand-verified {len(rows)} pairs: {n_reject} rejected -> rejection rate {rate:.1%}")
    return rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generate",
        action="store_true",
        help="synthesize + write pairs, then sample for verification",
    )

    parser.add_argument(
        "--n", type=int, default=MIN_PREFERENCE_PAIRS, help="how many pairs to synthesize"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="compute the rejection rate from the hand-verified file",
    )

    args = parser.parse_args()

    if args.generate:
        pairs = synthesize_pairs(args.n)
        write_pairs(pairs)
        sample_for_verification()

    if args.report:
        rejection_rate()

    if not (args.generate or args.report):
        parser.print_help()


if __name__ == "__main__":
    main()
