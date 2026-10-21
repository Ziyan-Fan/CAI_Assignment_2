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
from collections import Counter
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import random
import re

from configs.train_config import DATA_SEED, HAND_VERIFY_SAMPLE, MIN_PREFERENCE_PAIRS
from configs.paths_config import HAND_VERIFIED_PATH, PREFERENCE_DATA_PATH


VALID_CLASSES = {"ambiguous", "clear"}
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
GENERATION_BATCH_SIZE = 4
GENERATOR_VERSION = "natural_tasks_verified_v9"
MAX_ATTEMPTS_PER_SLOT = 80

# A few stable facts, independent of the author's answer and of evaluation data.
# Unmatched questions still receive the existing semantic review.
TRUSTED_FACTS = (
    (r"(?:what is|what's|name|identify) the chemical symbol (?:for|of) gold", ("au",)),
    (r"(?:what is|what's|name|identify) the chemical symbol (?:for|of) silver", ("ag",)),
    (r"(?:what is|which is|name|identify) the (?:largest|biggest) ocean"
     r"(?: (?:in|on) (?:the )?(?:world|earth))?", ("pacific", "pacific ocean")),
    (r"(?:what|which) planet is (?:known as|called) the red planet", ("mars",)),
    (r"(?:what is|which is|name|identify) the (?:largest|biggest) planet"
     r" in (?:our|the) solar system", ("jupiter",)),
    (r"(?:what is|which is|name|identify) the (?:largest|biggest) "
     r"(?:(?:rocky|terrestrial) planet|of (?:the )?(?:four|4) (?:rocky|terrestrial) planets)"
     r" in (?:our|the) solar system", ("earth",)),
    (r"(?:what is|what's|name|identify) the capital (?:city )?of france", ("paris",)),
    (r"(?:what is|what's|name|identify) the chemical formula (?:for|of) water", ("h2o",)),
    (r"(?:what is|what's|name|identify) the capital (?:city )?of japan", ("tokyo",)),
    (r"(?:what is|what's|name|identify) the capital (?:city )?of italy", ("rome",)),
    (r"(?:what is|what's|name|identify) the capital (?:city )?of canada", ("ottawa",)),
    (r"(?:what is|what's|name|identify) the capital (?:city )?of australia", ("canberra",)),
    (r"(?:what is|what's|name|identify) the largest continent(?: by area)?", ("asia",)),
    (r"(?:what is|what's|name|identify) the largest mammal", ("blue whale",)),
    (r"(?:what is|what's|name|identify) (?:the )?earth'?s natural satellite", ("moon", "the moon")),
    (r"(?:which|what) planet is closest to the sun", ("mercury",)),
    (r"(?:what is|what's|name|identify) the closest planet to the sun", ("mercury",)),
    (r"(?:which planet is best known for its prominent rings|what is the most prominent ringed planet)",
     ("saturn",)),
    (r"(?:what is|what's|name|identify) the most abundant gas in earth'?s atmosphere", ("nitrogen",)),
    (r"(?:what is|what's|name|identify) the chemical symbol for oxygen", ("o",)),
    (r"(?:what is|what's|name|identify) the chemical symbol for sodium", ("na",)),
    (r"(?:what is|what's|name|identify) the chemical formula for carbon dioxide", ("co2",)),
    (r"(?:what is|what's|name|identify) the atomic number of hydrogen", ("1",)),
    (r"(?:how many|what number of) planets are in (?:our|the) solar system", ("8",)),
    (r"(?:how many|what number of) sides does a hexagon have", ("6",)),
    (r"(?:what is|what's|calculate) the square root of 81", ("9",)),
    (r"(?:how many|what number of) seconds are in a minute", ("60",)),
    (r"(?:how many|what number of) meters are in a kilometer", ("1000",)),
    (r"(?:which|what) organ pumps blood through the human body", ("heart", "the heart")),
    (r"(?:which|what) (?:major )?river flows through egypt", ("nile", "nile river", "the nile")),
    (r"(?:who wrote|who is the author of) (?:the play )?hamlet", ("william shakespeare", "shakespeare")),
    (r"(?:what is|what's|name|identify) the currency of japan", ("yen", "the yen")),
    (r"(?:what is|what's|name|identify) the (?:official )?currency of the united kingdom",
     ("pound sterling",)),
    (r"(?:what is|what's|name|identify) the (?:main|primary) language spoken in brazil", ("portuguese",)),
    (r"(?:who painted|who is the painter of) the mona lisa", ("leonardo da vinci",)),
    (r"(?:who wrote|who is the author of) (?:the novel )?frankenstein", ("mary shelley",)),
    (r"(?:at sea level, )?(?:what is|what's|name|identify) the boiling point of water in degrees celsius",
     ("100", "100 degrees celsius")),
    (r"(?:at standard pressure, )?(?:what is|what's|name|identify) the freezing point of water in degrees celsius",
     ("0", "0 degrees celsius")),
    (r"how many seconds are in one minute", ("60",)),
    (r"how many meters are in one kilometer", ("1000",)),
    (r"(?:what is|what's|name|identify) the smallest prime number", ("2",)),
    (r"(?:what is|what's|calculate) 7 times 8", ("56",)),
    (r"(?:what is|what's|calculate) 144 divided by 12", ("12",)),
    (r"(?:what shape has|which shape has) three sides", ("triangle", "a triangle")),
    (r"(?:what do bees collect from flowers to make honey|what substance do bees collect from flowers to make honey)",
     ("nectar",)),
    (r"(?:what is|what's|name|identify) the process by which plants convert light energy into chemical energy",
     ("photosynthesis",)),
)

# These describe skills and information requirements, not sentence templates.
# Both classes have the same domain and prompt-length distribution.
DOMAINS = {
    "arithmetic": "A practical calculation using the supplied operation and facts.",
    "program_output": "Determine a program's output. The fact pack specifies whether the code is available or missing.",
    "text_transform": "Transform or inspect text. The fact pack specifies whether the source text is available or missing.",
    "reasoning": "Comparison, constraints, logical deduction, or a multi-step decision.",
    "knowledge": "A stable fact or explanation from science, history, geography, or computing. "
                 "An ordinary knowledge question is clear; lack of model knowledge is not ambiguity.",
    "writing": "An original email, edit, explanation, creative passage, or useful outline. "
               "For clear requests, optional tone, names, or formatting may use reasonable defaults. "
               "For ambiguous requests, the missing source or goal must actually prevent the task.",
    "planning": "Scheduling, allocating resources, comparing options, or planning a practical activity. "
                "Missing preferences alone do not make a useful general plan impossible.",
    "document_tasks": "Extract, summarize, interpret, or format information from an original passage, "
                      "table, message, or record. Vary the document and required reasoning.",
}

# Select the information gap before asking the model to invent a scenario.
# These are semantic constraints, not fixed user requests or response templates.
INFORMATION_GAPS = {
    "reasoning": ("an essential numerical or logical premise is absent",
                  "the criterion needed to select one of several candidates is absent",
                  "the candidate alternatives needed for the requested comparison are absent"),
    "knowledge": ("the person is identified only in private notes the user has not supplied",
                  "the scientific process is named only in an unavailable private report",
                  "the event is identified only in missing personal records"),
    "writing": ("the user wants an existing paragraph rewritten but has not supplied it",
                "the user requires answers to specific questions in a message whose content is absent",
                "the user asks for edits preserving details in an unavailable draft"),
    "planning": ("an exact finish or departure time requires an absent start time or duration",
                 "a plan must fit hard constraints but one necessary budget or capacity is absent",
                 "finding a mutually available meeting time requires a missing participant schedule"),
    "document_tasks": ("extract a specific fact from a document the user has not supplied",
                       "several records could be the referenced one and the intended record is unspecified",
                       "compute a requested value from an absent table or row"),
}


def _information_requirement(job, attempt=0):
    if job["class"] == "clear":
        return "All essential information is available; optional preferences can use reasonable defaults."
    gaps = INFORMATION_GAPS.get(job["domain"])
    if gaps:
        # Try another gap after three failures, keeping the same class and domain.
        return gaps[(_seed_for(job["slot"], "information_gap") + attempt // 3) % len(gaps)]
    return "The private_missing_fact in the fact pack is essential and absent from the user request."


def _parse_object(raw):
    if not isinstance(raw, str):
        return None
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
    try:
        result = json.loads(raw, strict=False)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        # Tolerate a prose introduction or markdown fence, not malformed JSON.
        decoder = json.JSONDecoder(strict=False)
        for match in re.finditer(r"\{", raw):
            try:
                result, _ = decoder.raw_decode(raw[match.start():])
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass
        return None


def _prompt_key(text):
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _seed_for(*parts):
    text = json.dumps([DATA_SEED, *parts], sort_keys=True)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def _schedule(n):
    """Fix class/domain/length quotas before generation, including for odd n."""
    jobs = []
    # Give explanations and writing more room, with the same domain mix in both classes.
    domains = [*DOMAINS, "knowledge", "writing"]
    for label, count in (("ambiguous", (n + 1) // 2), ("clear", n // 2)):
        for index in range(count):
            domain = domains[index % len(domains)]
            style = ("short", "standard", "contextual")[(index // len(domains)) % 3]
            jobs.append({"slot": f"{label}_{index:05d}", "class": label,
                         "domain": domain, "style": style})
    random.Random(DATA_SEED).shuffle(jobs)
    return jobs


KNOWN_FACT_QUESTIONS = (
    ("What is the chemical symbol for gold?", "Au", ("Ag", "Hg", "Pb"), "gold_symbol"),
    ("What is the chemical symbol for silver?", "Ag", ("Au", "Fe", "Cu"), "silver_symbol"),
    ("What is the largest ocean on Earth?", "Pacific", ("Atlantic", "Indian", "Arctic"), "largest_ocean"),
    ("Which planet is known as the Red Planet?", "Mars", ("Venus", "Earth", "Mercury"), "red_planet"),
    ("What is the largest planet in our solar system?", "Jupiter", ("Saturn", "Neptune", "Earth"), "largest_planet"),
    ("What is the largest rocky planet in our solar system?", "Earth", ("Mars", "Venus", "Mercury"), "rocky_planet"),
    ("What is the capital city of France?", "Paris", ("Rome", "Madrid", "Berlin"), "france_capital"),
    ("What is the chemical formula for water?", "H2O", ("CO2", "O2", "NaCl"), "water_formula"),
    ("What is the capital city of Japan?", "Tokyo", ("Seoul", "Beijing", "Bangkok"), "japan_capital"),
    ("What is the capital city of Italy?", "Rome", ("Madrid", "Athens", "Lisbon"), "italy_capital"),
    ("What is the capital city of Canada?", "Ottawa", ("Toronto", "Vancouver", "Montreal"), "canada_capital"),
    ("What is the capital city of Australia?", "Canberra", ("Sydney", "Melbourne", "Perth"), "australia_capital"),
    ("What is the largest continent by area?", "Asia", ("Africa", "Europe", "South America"), "largest_continent"),
    ("What is the largest mammal?", "Blue whale", ("African elephant", "Giraffe", "Hippopotamus"), "largest_mammal"),
    ("What is Earth's natural satellite?", "The Moon", ("Mars", "the Sun", "Venus"), "earth_satellite"),
    ("Which planet is closest to the Sun?", "Mercury", ("Venus", "Earth", "Mars"), "closest_planet"),
    ("Which planet is best known for its prominent rings?", "Saturn", ("Jupiter", "Uranus", "Neptune"), "ringed_planet"),
    ("What is the most abundant gas in Earth's atmosphere?", "Nitrogen", ("Oxygen", "Argon", "Carbon dioxide"), "atmospheric_gas"),
    ("What is the chemical symbol for oxygen?", "O", ("Ox", "Og", "Om"), "oxygen_symbol"),
    ("What is the chemical symbol for sodium?", "Na", ("So", "S", "N"), "sodium_symbol"),
    ("What is the chemical formula for carbon dioxide?", "CO2", ("CO", "C2O", "O2"), "carbon_dioxide_formula"),
    ("What is the atomic number of hydrogen?", "1", ("2", "0", "3"), "hydrogen_atomic_number"),
    ("How many planets are in our solar system?", "8", ("7", "9", "10"), "planet_count"),
    ("How many sides does a hexagon have?", "6", ("5", "7", "8"), "hexagon_sides"),
    ("What is the square root of 81?", "9", ("8", "7", "10"), "square_root_81"),
    ("How many seconds are in one minute?", "60", ("30", "45", "100"), "seconds_per_minute"),
    ("How many meters are in one kilometer?", "1000", ("100", "500", "10000"), "meters_per_kilometer"),
    ("Which organ pumps blood through the human body?", "The heart", ("The lungs", "The liver", "The kidneys"), "blood_pumping_organ"),
    ("Which major river flows through Egypt?", "The Nile", ("The Amazon", "The Danube", "The Yangtze"), "egypt_river"),
    ("Who wrote the play Hamlet?", "William Shakespeare", ("Christopher Marlowe", "Ben Jonson", "John Milton"), "hamlet_author"),
    ("What is the currency of Japan?", "The yen", ("The won", "The yuan", "The baht"), "japan_currency"),
    ("What is the official currency of the United Kingdom?", "Pound sterling", ("The euro", "The yen", "The franc"), "uk_currency"),
    ("What is the primary language spoken in Brazil?", "Portuguese", ("Spanish", "French", "Italian"), "brazil_language"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci", ("Michelangelo", "Raphael", "Sandro Botticelli"), "mona_lisa_painter"),
    ("Who wrote the novel Frankenstein?", "Mary Shelley", ("Jane Austen", "Charlotte Bronte", "Virginia Woolf"), "frankenstein_author"),
    ("At sea level, what is the boiling point of water in degrees Celsius?", "100", ("0", "50", "150"), "water_boiling_point"),
    ("At standard pressure, what is the freezing point of water in degrees Celsius?", "0", ("-10", "10", "32"), "water_freezing_point"),
    ("What is the smallest prime number?", "2", ("1", "3", "5"), "smallest_prime"),
    ("What is 7 times 8?", "56", ("48", "54", "64"), "seven_times_eight"),
    ("What is 144 divided by 12?", "12", ("10", "11", "14"), "division_144_by_12"),
    ("Which shape has three sides?", "A triangle", ("A square", "A pentagon", "A hexagon"), "three_sided_shape"),
    ("What do bees collect from flowers to make honey?", "Nectar", ("Pollen only", "Sap", "Dew"), "bee_honey_source"),
    ("What is the process by which plants convert light energy into chemical energy?", "Photosynthesis", ("Respiration", "Fermentation", "Transpiration"), "photosynthesis"),
)


def _mcq_ground_truth(job, attempt):
    """Build small factual or computed MCQs, assigning the answer letter in Python."""
    rng = random.Random(_seed_for(job["slot"], attempt, "mcq"))
    if rng.randrange(2):
        computed = _ground_truth({**job, "domain": "arithmetic", "class": "clear"}, attempt)
        skill = computed["operation"]
        questions = {
            "sum": "What is {first_quantity} plus {second_quantity}?",
            "product": "What is the cost of {quantity} items at ${price_per_item} each?",
            "difference": "Starting with {starting_quantity} items, how many remain after using {quantity_used}?",
            "divide": "How many items are in each of {number_of_equal_groups} equal groups containing {total_items} items in total?",
            "percentage": "What is {percentage}% of {total}?",
            "discount": "What is the final price after a {discount_percent}% discount on ${original_price}?",
            "mean": "What is the mean of {first_value}, {second_value}, and {third_value}?",
            "capacity": "How many vehicles seating {seats_per_vehicle} people each are needed for {number_of_people} people?",
            "rate": "What is the average speed in km/h for {distance_km} km traveled in {duration_hours} hours?",
            "price_with_fee": "What is the total cost of {quantity} items at ${price_per_item} each plus a ${delivery_fee} delivery fee?",
        }
        question = questions[skill].format(**computed["facts"])
        answer = computed["expected"]
        alternatives = [str(int(answer) + offset) for offset in range(-12, 13)
                        if offset and int(answer) + offset >= 0]
        distractors = rng.sample(alternatives, 3)
        skill = "mcq_" + skill
    else:
        question, answer, distractors, skill = rng.choice(KNOWN_FACT_QUESTIONS)
    options = [answer, *distractors]
    rng.shuffle(options)
    return {
        "operation": "multiple_choice", "skill": skill,
        "facts": {"question": question,
                  "options": "\n".join(f"{letter}) {value}" for letter, value in zip("ABCD", options))},
        "hidden_key": "options" if job["class"] == "ambiguous" else None,
        "expected": "ABCD"[options.index(answer)], "answer_text": answer,
        "request": question + " Give only the letter from the original A-D options.",
        "clarification": "Could you provide the A-D options for this question?",
    }


def _structured_ground_truth(job, attempt):
    """Create complete notes with independently computed reasoning/planning answers."""
    rng = random.Random(_seed_for(job["slot"], attempt, job["domain"], "structured"))
    family = (_seed_for(job["slot"], "structured_family") + attempt) % 4
    names = rng.sample(("Avery", "Jordan", "Morgan", "Riley", "Quinn", "Taylor"), 3)
    if job["domain"] == "reasoning":
        if family == 0:
            rates = rng.sample(range(18, 61), 2)
            source = "; ".join(f"{name}'s offer pays ${rate} per hour" for name, rate in zip(names, rates)) + "."
            winner = max(range(2), key=lambda index: rates[index])
            expected = f"{names[winner]}'s offer pays more: ${rates[winner]} per hour versus ${rates[1-winner]}."
            skill, source_name = "hourly offer comparison", "offer notes"
            request = "Compare the two offers in my offer notes by hourly pay and identify which pays more."
        elif family == 1:
            totals = rng.sample(range(25, 91), 3)
            first = [rng.randint(1, total - 1) for total in totals]
            source = "; ".join(f"{name}: first score {a}, second score {total-a}"
                               for name, a, total in zip(names, first, totals)) + ". Add both scores for each total."
            order = sorted(range(3), key=lambda index: totals[index], reverse=True)
            expected = "; ".join(f"{names[index]} ({totals[index]} points)" for index in order) + "."
            skill, source_name = "rank total scores", "scoring table"
            request = "Rank the candidates in my scoring table from highest to lowest total score."
        elif family == 2:
            budget, needed = rng.randrange(300, 901, 50), rng.randint(20, 80)
            costs = [budget - rng.randint(1, 5) * 10, budget + rng.randint(1, 5) * 10,
                     budget - rng.randint(1, 5) * 10]
            seats = [needed + rng.randint(0, 10), needed + rng.randint(0, 10), needed - rng.randint(1, 10)]
            options = list(zip(names, costs, seats))
            rng.shuffle(options)
            source = (f"Requirements: cost at most ${budget}; capacity at least {needed} people. "
                      + "; ".join(f"{name}: cost ${cost}, capacity {capacity}" for name, cost, capacity in options) + ".")
            expected = (f"{names[0]}. Cost: ${costs[0]} (limit ${budget}); "
                        f"capacity: {seats[0]} (minimum {needed}).")
            skill, source_name = "constraint selection", "comparison notes"
            request = "Choose the option in my comparison notes meeting both the budget and capacity requirements. Explain why."
        else:
            club = rng.choice(("Cedar", "Maple", "Birch", "Willow", "Elm", "Pine"))
            group = rng.choice(("registered volunteers", "certified tutors", "project reviewers", "safety trainees"))
            source = f"All members of the {club} club are {group}. {names[0]} is a member of the {club} club."
            expected = f"{names[0]} must be one of the {group}."
            skill, source_name = "categorical deduction", "logic notes"
            request = "Using only my logic notes, identify the broader group the named person must belong to."
    else:
        clock_text = lambda minutes: f"{minutes // 60:02d}:{minutes % 60:02d}"
        if family == 0:
            start = rng.randrange(8 * 60, 15 * 60, 15)
            durations = rng.sample(range(10, 61, 5), 3)
            source = (f"Start at {clock_text(start)} today. Tasks in order: check records ({durations[0]} minutes), "
                      f"prepare materials ({durations[1]} minutes), write a summary ({durations[2]} minutes). No breaks.")
            expected = clock_text(start + sum(durations))
            skill, source_name = "finish time", "timetable"
            request = "When will the work in my timetable finish? Give the time in 24-hour notation."
        elif family == 1:
            first_start = rng.randrange(8 * 60, 15 * 60, 15)
            second_start = first_start + rng.choice((-30, -15, 15, 30))
            shared_start = max(first_start, second_start)
            shared_end = shared_start + rng.choice((30, 45, 60))
            ends = [shared_end, shared_end + rng.choice((15, 30))]
            rng.shuffle(ends)
            day = rng.choice(("Monday", "Tuesday", "Wednesday", "Thursday", "Friday"))
            source = (f"Both calendars are for {day}, in UTC, using 24-hour notation. "
                      f"{names[0]} is free {clock_text(first_start)}-{clock_text(ends[0])}; "
                      f"{names[1]} is free {clock_text(second_start)}-{clock_text(ends[1])}.")
            expected = f"{day}, {clock_text(shared_start)}-{clock_text(shared_start + 30)} UTC."
            skill, source_name = "shared availability", "schedule notes"
            request = "Find the earliest 30-minute meeting slot within both participants' availability in my schedule notes."
        elif family == 2:
            activities = rng.sample(("museum visit", "pottery workshop", "guided walk", "film screening", "cooking class", "boat tour"), 3)
            budget = rng.randrange(250, 951, 50)
            costs = [budget - value for value in rng.sample(range(20, 121, 10), 2)] + [budget + rng.randrange(10, 101, 10)]
            options = list(zip(activities, costs))
            rng.shuffle(options)
            source = f"Budget: ${budget}. " + "; ".join(f"{name}: ${cost}" for name, cost in options) + "."
            name, cost = max((row for row in options if row[1] <= budget), key=lambda row: row[1])
            expected = f"The {name}, costing ${cost}, is the most expensive affordable activity."
            skill, source_name = "budget plan", "cost sheet"
            request = "Select the single most expensive activity in my cost sheet that stays within its stated budget."
        else:
            chain = rng.sample(("Cedar", "Maple", "Birch", "Willow", "Elm", "Pine"), 4)
            listed = rng.sample(chain, len(chain))
            edges = [f"{a} must finish before {b}" for a, b in zip(chain, chain[1:])]
            rng.shuffle(edges)
            source = "Tasks: " + ", ".join(listed) + ". Dependencies: " + "; ".join(edges) + "."
            expected = " -> ".join(chain) + "."
            skill, source_name = "dependency order", "project notes"
            request = "Put every task in my project notes in an order that satisfies all listed dependencies."
    return {"operation": skill, "facts": {"source_text": source},
            "hidden_key": "source_text" if job["class"] == "ambiguous" else None,
            "expected": expected, "request": request, "source_name": source_name}


def _knowledge_task(job):
    """Keep a reproducible mix of short facts, explanations, and practical steps."""
    return ("fact", "explanation", "instructions")[_seed_for(job["slot"], "knowledge_task") % 3]


def _ground_truth(job, attempt):
    """Compute trusted answers for a subset; never execute model-written code."""
    if job["domain"] == "multiple_choice":
        return _mcq_ground_truth(job, attempt)
    if job["domain"] in {"reasoning", "planning"}:
        return _structured_ground_truth(job, attempt)
    rng = random.Random(_seed_for(job["slot"], attempt))
    domain = job["domain"]
    facts = {}
    if domain == "knowledge" and job["class"] == "clear":
        if _knowledge_task(job) != "fact":
            return None  # Author a complete explanation or instructions, then review it.
        question, answer, _, skill = rng.choice(KNOWN_FACT_QUESTIONS)
        return {"operation": "known_fact", "skill": skill, "facts": {"question": question},
                "hidden_key": None, "expected": answer}
    if domain == "arithmetic":
        operation = rng.choice(("sum", "product", "difference", "divide", "percentage",
                                "discount", "mean", "capacity", "rate", "price_with_fee"))
        a, b = rng.randint(40, 95), rng.randint(3, 17)
        if operation == "sum":
            facts, answer = {"first_quantity": a, "second_quantity": b}, a + b
        elif operation == "product":
            facts, answer = {"quantity": a, "price_per_item": b}, a * b
        elif operation == "difference":
            facts, answer = {"starting_quantity": a, "quantity_used": b}, a - b
        elif operation == "divide":
            facts, answer = {"total_items": a * b, "number_of_equal_groups": b}, a
        elif operation == "percentage":
            total, percent = a * 100, b
            facts, answer = {"total": total, "percentage": percent}, a * b
        elif operation == "discount":
            facts = {"original_price": a * 100, "discount_percent": b}
            answer = a * (100 - b)
        elif operation == "mean":
            facts = {"first_value": 3 * a, "second_value": 3 * b, "third_value": 6}
            answer = a + b + 2
        elif operation == "capacity":
            facts, answer = {"number_of_people": a, "seats_per_vehicle": b}, math.ceil(a / b)
        elif operation == "rate":
            facts, answer = {"distance_km": a * b, "duration_hours": b}, a
        else:
            facts = {"quantity": a, "price_per_item": b, "delivery_fee": 2}
            answer = a * b + 2
        expected = str(answer)
    elif domain == "program_output":
        values = rng.sample(range(2, 25), 5)
        limit = rng.randint(8, 18)
        operation = rng.choice(("slice", "filter", "sum", "alias", "sort", "enumerate",
                                "count", "dictionary", "boolean", "loop"))
        start = f"values = {values!r}\n"
        if operation == "slice":
            code, answer = start + "print(values[1:4:2])", values[1:4:2]
        elif operation == "filter":
            code = start + f"print([x for x in values if x > {limit}])"
            answer = [x for x in values if x > limit]
        elif operation == "sum":
            code, answer = start + "print(sum(values[::2]))", sum(values[::2])
        elif operation == "alias":
            code = start + f"other = values\nother.append({limit})\nprint(values)"
            answer = values + [limit]
        elif operation == "sort":
            code, answer = start + "print(sorted(values, reverse=True)[:3])", sorted(values, reverse=True)[:3]
        elif operation == "enumerate":
            code = start + "print([i for i, x in enumerate(values) if x % 2 == 0])"
            answer = [i for i, x in enumerate(values) if x % 2 == 0]
        elif operation == "count":
            code = start + f"print(sum(x < {limit} for x in values))"
            answer = sum(x < limit for x in values)
        elif operation == "dictionary":
            code = f"items = {{'red': {values[0]}, 'blue': {values[1]}}}\nprint(items.get('green', 0))"
            answer = 0
        elif operation == "boolean":
            code = start + f"print(all(x > {limit} for x in values))"
            answer = all(x > limit for x in values)
        else:
            code = start + "total = 0\nfor x in values:\n    total += x * 2\nprint(total)"
            answer = 2 * sum(values)
        facts, expected = {"code": code}, str(answer)
    elif domain == "text_transform":
        operation = rng.choice(("uppercase", "lowercase", "word_count", "sort_words", "initials"))
        name = rng.choice(("Mira", "Avery", "Morgan", "Noel", "Riley", "Jordan"))
        source = (f"{name} {rng.choice(('packed', 'sorted', 'carried', 'checked'))} "
                  f"{rng.choice(('four', 'seven', 'nine', 'twelve'))} "
                  f"{rng.choice(('blue folders', 'green notebooks', 'small parcels', 'red boxes'))} "
                  f"{rng.choice(('before lunch', 'after class', 'this morning', 'on Tuesday'))}.")
        if operation == "uppercase":
            expected = source.upper()
        elif operation == "lowercase":
            expected = source.lower()
        elif operation == "word_count":
            expected = str(len(source.split()))
        elif operation == "sort_words":
            source = " ".join(rng.sample(("birch", "maple", "cedar", "oak", "elm", "pine",
                                          "willow", "ash", "fir", "alder", "beech", "yew"), 6))
            expected = " ".join(sorted(source.split()))
        else:
            source = " ".join(rng.sample(("Avery", "Morgan", "Lee", "Quinn", "Taylor", "Reed", "Noel"), 3))
            expected = "".join(word[0] for word in source.split())
        facts = {"source_text": source}
    else:
        return None
    hidden = rng.choice(list(facts)) if job["class"] == "ambiguous" else None
    return {"operation": operation, "facts": facts, "hidden_key": hidden, "expected": expected}


AUTHOR_SYSTEM = """Create one original user request and its completed answer. The request must be answerable
using ordinary public knowledge or a useful original draft with reasonable defaults.
Do not create quizzes or multiple-choice questions in any domain.
For knowledge, ask about a stable, well-established fact or explain a familiar
mechanism. Avoid uncertain claims, current statistics, and multiple-choice questions.
Practical instructions should cover an ordinary everyday or basic computing task,
with the goal and essential inputs stated. Provide actionable steps appropriate to it.
For writing, request a fictional sample email, apology, invitation, announcement,
creative passage, or useful outline. Invented details are permitted in fictional examples.
The request and answer must agree about who did what and what the writer should communicate.
Do not request edits or summaries of an unavailable source.

Actually complete the request, including any requested greeting or other component;
do not merely announce or describe an answer. Missing optional preferences can use
defaults. Vary scenarios and wording; do not copy recent requests. Keep the answer
under 100 words. Avoid exact word-count requirements. Skill is at most 8 words.
Return exactly this JSON object with three STRING values, never nulls or objects:
{"prompt":"the complete user request","answer":"the actual completed answer","skill":"brief task description"}"""


def _author_messages(job, truth, recent, attempt=0, feedback=""):
    styles = {"short": "At most 40 words in the user request.",
              "standard": "At most 100 words in the user request.",
              "contextual": "70 to 150 words in the user request. Include natural context; "
                            "length must not determine whether the task is answerable."}
    is_ambiguous = job["class"] == "ambiguous"
    if truth and job["domain"] in {"arithmetic", "program_output", "text_transform", "knowledge", "reasoning", "planning"}:
        lengths = {"short": "4-8", "standard": "10-25", "contextual": "70-90"}
        topic = f"{job['domain'].replace('_', ' ')}: {truth['operation'].replace('_', ' ')}"
        if job["domain"] == "knowledge":
            topic = truth["facts"]["question"]
        elif job["domain"] == "program_output":
            behaviors = {
                "slice": "slicing an integer list with a step",
                "filter": "filtering integer values above a threshold",
                "sum": "summing alternating entries of an integer list",
                "alias": "appending to an integer list shared by two variable names",
                "sort": "sorting integers in descending order and taking a prefix",
                "enumerate": "finding the indices of even integer values",
                "count": "counting integer values below a threshold",
                "dictionary": "looking up a missing dictionary key with a default value",
                "boolean": "checking whether all integer values exceed a threshold",
                "loop": "doubling each integer value and summing the results",
            }
            topic = "Python: " + behaviors[truth["operation"]] + ". Do not describe any other program behavior."
        instruction = (f"Background topic: {topic}. Length: {lengths[job['style']]} words.\n"
                       "Avoid copying these recent situations: "
                       + json.dumps([r["prompt"] for r in recent], ensure_ascii=False)
                       + '\nOUTPUT JSON: {"context":"your original background text"}')
        return [{"role": "system", "content":
                 'Write natural first-person background only. Return exactly {"context":"background text"}. '
                 "Give a fresh reason for doing the task. Do not write a request, question, answer, code, "
                 "input list, source text, task input facts or any numbers. Do not say a source is present or "
                 "missing. The task request and source will be added separately."},
                {"role": "user", "content": instruction}]
    if is_ambiguous or job["domain"] not in {"knowledge", "writing"}:
        raise ValueError("Generic author supports only clear knowledge or writing tasks.")
    instruction = (f"Domain: {job['domain']}. {styles[job['style']]}\n"
                   "Avoid copying these recent requests: "
                   + json.dumps([r["prompt"] for r in recent], ensure_ascii=False))
    if job["domain"] == "knowledge":
        task = _knowledge_task(job)
        instruction += "\n" + {
            "fact": "Request a stable fact and give an appropriately brief answer.",
            "explanation": "Request a short explanation of how or why a familiar concept works. "
                           "Explain the mechanism, rather than just naming it.",
            "instructions": "Request practical instructions for a well-specified task. "
                            "Give the actual steps needed to complete it.",
        }[task]
    if attempt:
        instruction += (f"\nRetry {attempt}. Previous failure: {feedback[:450]}\n"
                        "Choose a fresh answerable request and complete it correctly.")
        if "response_length" in feedback:
            instruction += ("\nKeep the completed answer to 50–80 words total, including any greeting "
                            "and sign-off. Choose a simpler request with fewer required details "
                            "so the answer can still complete the whole task.")
    return [{"role": "system", "content": AUTHOR_SYSTEM},
            {"role": "user", "content": instruction}]


def _complete_metadata(obj, job, truth):
    """Fill known task metadata without making the author regenerate it."""
    if not isinstance(obj, dict):
        return obj
    if truth is None and job["class"] == "clear" and job["domain"] in {"knowledge", "writing"}:
        result = dict(obj)
        if job["domain"] == "writing" and isinstance(result.get("prompt"), str):
            notice = "This is a fictional sample; you may invent names and details."
            if notice not in result["prompt"]:
                result["prompt"] += " " + notice
        questions = (("How detailed should my answer be?", "Would you prefer a brief or detailed explanation?",
                      "Should I use simple or technical language?", "What level of detail would you like?",
                      "Which writing style do you prefer?", "Who is the intended audience?",
                      "Would examples be helpful?", "Do you want bullet points or a paragraph?")
                     if job["domain"] == "knowledge" else
                     ("What tone should I use?", "Should the wording be formal or casual?",
                      "Would you prefer a formal or informal style?", "What writing style would you like?",
                      "How long should the response be?", "Should I use simple language?",
                      "Would you prefer shorter sentences?", "Do you have a preferred format?"))
        result["chosen"] = obj.get("answer", "")
        result["rejected"] = questions[_seed_for(job["slot"], obj) % len(questions)]
        result.update({
            "requirements": ["the public fact or concept requested" if job["domain"] == "knowledge"
                             else "the stated writing purpose and explicit constraints"],
            "missing_information": "", "assumed_information": "",
            "answerability_reason": "Intended as a public-knowledge question or generic draft; subject to review.",
            # This author answer is retained for audit, not independent ground truth.
            "reference_answer": result["chosen"],
        })
        return result
    if not truth or job["domain"] not in {"arithmetic", "program_output", "text_transform", "multiple_choice", "knowledge", "reasoning", "planning"}:
        return obj
    hidden = truth["hidden_key"]
    result = dict(obj)
    result.update({
        "requirements": [key.replace("_", " ") for key in truth["facts"]],
        "missing_information": hidden or "",
        "assumed_information": f"Assume {hidden}: {truth['facts'][hidden]}" if hidden else "",
        "answerability_reason": (f"The required {hidden.replace('_', ' ')} is absent." if hidden else
                                 "The request supplies all inputs needed for the stated operation."),
        "reference_answer": truth["expected"],
    })
    if job["domain"] == "knowledge":
        questions = ("How detailed should my answer be?", "Would you prefer a brief or detailed explanation?",
                     "Should I use simple or technical language?", "What level of detail would you like?",
                     "Which writing style do you prefer?", "Who is the intended audience?",
                     "Would examples be helpful?", "Do you want bullet points or a paragraph?")
        result.update({"skill": truth["skill"], "chosen": truth["expected"],
                       "rejected": questions[_seed_for(job["slot"], obj.get("context", ""), truth)
                                             % len(questions)]})
    if job["domain"] in {"arithmetic", "program_output", "text_transform", "reasoning", "planning"}:
        key = hidden or next(iter(truth["facts"]))
        names = {"code": "complete Python code", "source_text": "source text",
                 "number_of_equal_groups": "number of equal groups", "total": "total amount"}
        name = truth.get("source_name", names.get(key, key.replace("_", " ")))
        if job["domain"] == "arithmetic":
            wording = ("What is the {name}?", "Could you specify the {name}?",
                       "Please provide the {name}.", "What value should I use for the {name}?")
            question = wording[_seed_for(job["slot"], truth) % len(wording)].format(name=name)
        else:
            question = _source_question(name, job, _seed_for(truth))
        result.update({"skill": truth["operation"],
                       "chosen": question if hidden else truth["expected"],
                       "rejected": truth["expected"] if hidden else question})
    return result


# These tasks depend on an actual source, rather than optional names or preferences.
# Llama writes a fresh situation, source, and answer. Python supplies or withholds
# the source and asks for that exact input, so the class is not invented by Llama.
SOURCE_TASKS = {
    "reasoning": (
        ("offer comparison", "Compare the two offers in my notes by hourly pay and identify which pays more per hour.", "offer notes", "two fictional offers with explicit hourly pay and benefits"),
        ("constraint selection", "Choose the option in my comparison notes that satisfies every listed requirement, explaining why.", "comparison notes", "three options and two explicit requirements, with exactly one qualifying option"),
        ("logical deduction", "Use the premises in my logic exercise to deduce the requested conclusion.", "logic exercise", "a categorical logic exercise using an all-X-are-Y relation chain and one named example; no numbers or arithmetic"),
        ("rank records", "Rank the candidates in my scoring table from highest to lowest total score.", "scoring table", "three fictional candidates with two integer scores each and no tied totals"),
    ),
    "knowledge": (
        ("science observation", "Quote the observation recorded in my lab note.", "lab note", "one complete sentence describing a concrete personal observation from a science activity"),
        ("study note", "Quote the text in my study note.", "study note", "one original question or short observation about a familiar historical event"),
        ("computing example", "Quote the computing example recorded in my lecture note.", "lecture note", "one complete sentence giving a simple original computing example"),
        ("geographic observation", "Quote the observation recorded in my field note.", "field note", "one complete sentence describing a concrete personal observation of a geographic feature"),
    ),
    "writing": (
        ("polite rewrite", "Rewrite my draft politely, preserving its concrete details and requests.", "draft", "a short blunt email with two concrete requests"),
        ("concise rewrite", "Make my paragraph more concise while preserving its factual details.", "paragraph", "a wordy original paragraph with two or three concrete facts"),
        ("proofreading", "Correct the grammar and spelling in my paragraph without changing its meaning.", "paragraph", "an original paragraph containing several straightforward grammar and spelling mistakes"),
        ("message reply", "Draft a reply that answers each question in my message using the accompanying response notes.", "message and response notes", "an email with two questions plus separate notes supplying both answers"),
    ),
    "planning": (
        ("shared availability", "Find a 30-minute meeting slot that fits every participant's availability in my schedule notes.", "schedule notes", "two participants' availability on the same day and in the same timezone, with a shared 30-minute slot"),
        ("finish time", "Calculate when the work in my timetable will finish, following the listed order and durations without gaps.", "timetable", "a start time and three sequential tasks with durations in minutes, totaling less than two hours"),
        ("budget plan", "Select the most expensive activity in my cost sheet that stays within its stated budget.", "cost sheet", "three fictional activities with explicit total costs and a budget; two activities affordable and costs distinct"),
        ("task order", "Put the tasks in my project notes in an order that satisfies every listed dependency.", "project notes", "four named tasks and dependencies that impose a unique order"),
    ),
    "document_tasks": (
        ("source summary", "Briefly summarize the key points of my meeting notes.", "meeting notes", "exactly three records: DECISION: followed by the decision, REASON: followed by its reason, and NEXT: followed by the next step; one record of each type"),
        ("deadline extraction", "List every deadline and its associated deliverable from my project update.", "project update", "exactly three records in the format DEADLINE: <explicit date> | <associated deliverable>"),
        ("action extraction", "Extract the action items and their assigned owners from my meeting notes.", "meeting notes", "exactly three records in the format ACTION: <assigned owner> | <specific task>"),
        ("tagged extraction", "Quote every line marked REPLACE in my inventory record exactly.", "inventory record", "exactly four separate lines, each beginning REPLACE: or KEEP: followed by a fictional item description; use both tags"),
    ),
}


def _source_plan(job, attempt):
    if job["domain"] in {"reasoning", "planning"}:
        return None  # Constraints and schedules are constructed in Python.
    if job["domain"] not in SOURCE_TASKS:
        return None
    if job["class"] == "clear" and job["domain"] in {"knowledge", "writing"}:
        return None  # Keep ordinary knowledge and useful generic drafts in clear data.
    plans = SOURCE_TASKS[job["domain"]]
    return plans[(_seed_for(job["slot"], "source_task") + attempt) % len(plans)]


def _source_messages(job, plan, recent, feedback):
    skill, request, name, description = plan
    context_length = "5-15"
    instruction = {
        "task": request, "source_to_create": description,
        "context_length_words": context_length,
        "recent_situations_to_avoid": [r["prompt"] for r in recent],
        "previous_failure": feedback[:450],
    }
    derived_answer = job["domain"] in {"knowledge", "document_tasks"}
    schema = {"context": f"{context_length} words of background only",
              "source": description if job["domain"] == "document_tasks"
              else "the complete original source, 20-65 words"}
    if not derived_answer:
        schema["answer"] = "the completed task, under 100 words"
    return [{"role": "system", "content":
             ("Create an original source for the specified task. Return only JSON. " if derived_answer else
              "Create an original source and its completed answer for the specified task. Return only JSON. ") +
             "Use a fresh scenario and concrete details in the source. Context is one short generic sentence "
             "about why the task is useful, independent of the source: no names, numbers, source facts, "
             "instructions, or claims that an attachment is present or missing. "
             "Put all task inputs in source. " +
             ("Do not generate an answer; it will be derived from the source. " if derived_answer else
              "In answer actually perform the task: do not merely describe it, ask a question, or announce a draft. "
              "Preserve the source's details, including every numeric date, quantity, duration, and weekday. ") +
             "For tagged records use exactly the requested tags and number of records, one per line. "
             "Include no untagged introduction or extra actions, deadlines, or commentary outside those records. "
             "Use ordinary stable knowledge for explanations. No invented real-world statistics."},
            {"role": "user", "content": json.dumps(instruction, ensure_ascii=False)
             + '\nReturn exactly this top-level JSON shape, replacing the descriptions:\n'
             + json.dumps(schema)}]


def _source_record(raw, job, attempt):
    obj = _parse_object(raw)
    if obj and isinstance(obj.get("output"), dict):
        obj = obj["output"]
    if obj and isinstance(obj.get("source"), (dict, list)):
        obj["source"] = json.dumps(obj["source"], ensure_ascii=False)
    if obj and job["domain"] in {"knowledge", "document_tasks"}:
        obj.setdefault("answer", obj.get("source", ""))
    if not obj or any(not isinstance(obj.get(k), str) or not obj[k].strip()
                      for k in ("context", "source", "answer")):
        raise ValueError("invalid_source_task")
    context, source, answer = (obj[k].strip() for k in ("context", "source", "answer"))
    context = re.split(r"(?<=[.!?])\s+", context, maxsplit=1)[0]
    if (len(context.split()) > 25
            or re.search(r"\d|words of background|background text", context, re.I)):
        raise ValueError("source_context_contains_details")
    obj["context"] = context
    if _prompt_key(source) in _prompt_key(context) or _prompt_key(answer) in _prompt_key(context):
        raise ValueError("source_leaked_into_context")
    if len(source.split()) > 100 or len(context.split()) > 100:
        raise ValueError("source_task_too_long")
    skill, request, name, description = _source_plan(job, attempt)
    if _prompt_key(description) in _prompt_key(source):
        raise ValueError("source_is_generation_instruction")
    if job["domain"] == "knowledge":
        answer = source  # The request quotes the note, rather than adding an unverified explanation.
        obj["answer"] = source
    if job["domain"] == "writing":
        # Preserve distinct integer values, regardless of how often they occur.
        numeric_source = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", r"\1", source, flags=re.I)
        numeric_answer = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", r"\1", answer, flags=re.I)
        values = {Decimal(value.replace(",", "")) for value in
                  re.findall(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w]|\.\d)",
                             numeric_source)}
        if any(value == int(value) and not _contains_value(numeric_answer, int(value)) for value in values):
            raise ValueError("rewrite_changed_numeric_detail")
        for weekday in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"):
            if (re.search(r"\b" + weekday + r"\b", source, re.I)
                    and not re.search(r"\b" + weekday + r"\b", answer, re.I)):
                raise ValueError("rewrite_changed_weekday")
    if job["domain"] == "document_tasks":
        # Copy complete tagged records instead of asking the model to re-summarize them.
        lines = [line.strip() for line in re.split(
            r"\s+(?=(?:DECISION|REASON|NEXT|ACTION|DEADLINE|REPLACE|KEEP):)", source)
                 if line.strip()]
        patterns = {"source summary": r"(DECISION|REASON|NEXT):\s*(\S[^\n]*)",
                    "action extraction": r"(ACTION):\s*([^|\n]+\|[^|\n]+)",
                    "deadline extraction": r"(DEADLINE):\s*([^|\n]+\|[^|\n]+)",
                    "tagged extraction": r"(REPLACE|KEEP):\s*(\S[^\n]*)"}
        matches = [re.fullmatch(patterns[skill], line) for line in lines]
        expected_count = 4 if skill == "tagged extraction" else 3
        if len(lines) != expected_count or not all(matches):
            raise ValueError("invalid_tagged_records")
        tags = [match.group(1) for match in matches]
        if ((skill == "source summary" and set(tags) != {"DECISION", "REASON", "NEXT"})
                or (skill == "tagged extraction" and set(tags) != {"REPLACE", "KEEP"})
                or any(not part.strip() for match in matches for part in match.group(2).split("|"))):
            raise ValueError("invalid_tagged_records")
        source = "\n".join(lines)
        obj["source"] = source
        answer = "\n".join(line for line in lines
                           if skill != "tagged extraction" or line.startswith("REPLACE:"))
        obj["answer"] = answer
    request = context.rstrip(" .?!") + ". " + request
    source_block = "\n\n" + name.capitalize() + ":\n" + source
    question = _source_question(name, job, attempt)
    ambiguous = job["class"] == "ambiguous"
    pair_obj = {
        "prompt": request, "chosen": question if ambiguous else answer,
        "rejected": answer if ambiguous else question,
        "requirements": [name], "missing_information": name if ambiguous else "",
        "assumed_information": source if ambiguous else "",
        "answerability_reason": "The task requires the specific source; it is "
                                + ("withheld." if ambiguous else "supplied below the request."),
        "reference_answer": answer, "skill": skill,
    }
    # Length limits apply to the natural request; the attached source is separate.
    record = _candidate_record(json.dumps(pair_obj), job, None)
    record["task"]["planned_requirement"] = request
    if not ambiguous:
        record["pair"]["prompt"] += source_block
    record["source_draft"] = obj
    record["source_attempt"] = attempt
    record["complete_prompt"] = request + source_block
    record["review_scope"] = "complete_source_task"
    return record


def _source_question(name, job, attempt):
    questions = (
        "Could you share your {name}?", "Please provide the contents of your {name}.",
        "Can you paste your {name} here?", "Could you show me your {name}?",
        "I need your {name} to do this. Could you send it?",
        "Would you include your {name} so I can work from the actual information?",
        "Please send the relevant text from your {name}.",
        "Could you provide your {name} for me to use?",
    )
    index = _seed_for(job["slot"], attempt, "clarification") % len(questions)
    return questions[index].format(name=name)


def _mcq_messages(job, truth, recent):
    words = {"short": "3-6", "standard": "15-25", "contextual": "70-90"}[job["style"]]
    return [{"role": "system", "content":
             "Return JSON with just context: a natural first-person reason for answering a quiz question. "
             "Do not answer the question, add facts or options, give instructions, or claim anything is missing."},
            {"role": "user", "content": json.dumps({"question": truth["facts"]["question"],
             "context_words": words, "avoid_copying": [r["prompt"] for r in recent]})
             + '\nReturn only {"context": "' + words + ' words of background"}.'}]


def _mcq_record(raw, job, truth):
    obj = _parse_object(raw)
    if not obj or not isinstance(obj.get("context"), str) or not obj["context"].strip():
        raise ValueError("invalid_mcq_context")
    if re.search(r"[0-9?`{}\[\]=]|words of background|background text", obj["context"]):
        raise ValueError("task_facts_in_context")
    ambiguous = job["class"] == "ambiguous"
    prompt = obj["context"].strip().rstrip(" .") + ". " + truth["request"]
    if not ambiguous:
        prompt += "\n\nOptions:\n" + truth["facts"]["options"]
    question = _source_question("original A-D options", job, _seed_for(truth))
    obj = {"prompt": prompt, "skill": truth["skill"],
           "chosen": question if ambiguous else truth["expected"],
           "rejected": truth["expected"] if ambiguous else question}
    record = _candidate_record(json.dumps(_complete_metadata(obj, job, truth)), job, truth)
    record["context"] = _parse_object(raw)["context"].strip()
    return record


def _computed_record(raw, job, truth):
    obj = _parse_object(raw)
    if not obj or not isinstance(obj.get("context"), str) or not obj["context"].strip():
        raise ValueError("missing_task_context")
    if re.search(r"[0-9?`{}\[\]=]|words of background|background text", obj["context"]):
        raise ValueError("task_facts_in_context")
    if job["domain"] == "program_output" and re.search(r"\b(?:strings?|characters?)\b", obj["context"], re.I):
        raise ValueError("context_conflicts_with_numeric_program")
    requests = {
        "sum": "What is the sum of the two quantities?",
        "product": "What is the total cost of the items?",
        "difference": "How much of the starting quantity remains after the quantity used?",
        "divide": "How many items belong in each equal group?",
        "percentage": "What is the percentage amount?",
        "discount": "What is the final price after the discount?",
        "mean": "What is the arithmetic mean of all three values?",
        "capacity": "What is the minimum whole number of vehicles needed to seat everyone?",
        "rate": "What is the average speed in kilometers per hour?",
        "price_with_fee": "What is the total cost of the items including the delivery fee?",
        "program_output": "What does my Python program print?",
        "uppercase": "Convert my text to uppercase.",
        "lowercase": "Convert my text to lowercase.",
        "word_count": "Count the whitespace-separated words in my text.",
        "sort_words": "Sort the words in my list alphabetically.",
        "initials": "Give the concatenated uppercase initials of the names in my list.",
    }
    operation = "program_output" if job["domain"] == "program_output" else truth["operation"]
    request = truth.get("request") or (truth["facts"]["question"] if job["domain"] == "knowledge" else requests[operation])
    prompt = " ".join(obj["context"].split()).rstrip(" .") + ". " + request
    visible = {k: v for k, v in truth["facts"].items() if k != truth["hidden_key"]}
    if job["domain"] == "arithmetic":
        inputs = "\n".join(k.replace("_", " ").capitalize() + ": " + str(v) for k, v in visible.items())
        prompt += "\n\nInputs:\n" + inputs
    elif job["class"] == "clear" and job["domain"] != "knowledge":
        if job["domain"] == "program_output":
            prompt += "\n\n```python\n" + visible["code"] + "\n```"
        else:
            prompt += "\n\n" + truth.get("source_name", "Source text").capitalize() + ":\n" + visible["source_text"]
    obj["prompt"] = prompt
    record = _candidate_record(json.dumps(_complete_metadata(obj, job, truth)), job, truth)
    record["task"]["planned_requirement"] = request
    record["context"] = obj["context"].strip()
    return record


def _contains_value(text, value):
    if isinstance(value, int) and not isinstance(value, bool):
        numbers = re.findall(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w]|\.\d)", text)
        return any(Decimal(number.replace(",", "")) == value for number in numbers)
    # Preserve case and indentation in source text and code, not just its words.
    return str(value) in text


def _trusted_factual_answer(prompt):
    text = _prompt_key(prompt).removeprefix("please ")
    # Remove only familiar MCQ instructions; extra context must still fail fullmatch.
    text = re.sub(r"^(?:select|choose|give|return) (?:only )?(?:the )?(?:correct )?"
                  r"(?:answer )?letter(?: of the correct answer)? (?:for|to) "
                  r"(?:the )?(?:following )?question\s*:\s*", "", text)
    text = re.sub(r"\s+(?:options|choices)\s*:\s*$", "", text)
    text = re.sub(r"\s+(?:answer|respond|reply|return|give|choose|select) "
                  r"(?:with )?(?:only )?(?:the )?(?:correct )?(?:answer )?letter"
                  r"(?: only)?[.!]?\s*$", "", text)
    text = text.strip(" .?!:")
    for question, answers in TRUSTED_FACTS:
        if re.fullmatch(question, text):
            return answers
    return None


def _percentage_operation(prompt):
    """Recognize the requested quantity, not just a discount in the context."""
    # Prefer the final question/instruction over earlier background sentences.
    requests = re.split(r"[.!?;]\s*|\n+", prompt.lower())
    final_price = (r"\b(?:final|remaining|discounted|reduced|sale|new|net)\s+(?:price|cost|total|amount)\b"
                   r"|\b(?:price|cost|total)\b[^.!?;]*\bafter\b"
                   r"|\bhow much\b[^.!?;]*\bcost\b"
                   r"|\b(?:pay|payable|owe|left|remaining)\b")
    amount = (r"\b(?:discount|percentage)\s+amount\b|\bamount\s+(?:of\s+the\s+discount|saved)\b"
              r"|\b(?:save|saved|savings|saving)\b|\b(?:how much|what)\s+is\s+(?:the\s+)?discount\b"
              r"(?!\s+(?:percentage|percent|rate))"
              r"|\b\d+(?:\.\d+)?\s*(?:%|percent)\s+of\b")
    for request in reversed(requests):
        wants_amount = bool(re.search(amount, request))
        wants_final = bool(re.search(final_price, request) or (
            not wants_amount and re.search(r"\bamount\b[^.!?;]*\bafter\b", request)))
        if wants_final or wants_amount:
            # Multiple requested quantities cannot use one scalar ground truth.
            if wants_final and wants_amount:
                return None
            return "discount" if wants_final else "percentage"
        if re.search(r"\b(?:what|how much|calculate|compute|find|determine)\b", request):
            return None  # Do not mistake an earlier background quantity for the request.
    return None


def _candidate_record(raw, job, truth):
    """Reject malformed or mechanically wrong candidates before model review."""
    obj = _parse_object(raw)
    if obj is None:
        raise ValueError("invalid_json")
    for key in ("prompt", "chosen", "rejected", "answerability_reason", "skill", "reference_answer"):
        if not isinstance(obj.get(key), str) or not obj[key].strip():
            raise ValueError("missing_" + key)
        obj[key] = obj[key].strip()
    requirements = obj.get("requirements")
    if (not isinstance(requirements, list) or not 1 <= len(requirements) <= 8
            or any(not isinstance(r, str) or not r.strip() or len(r) > 400 for r in requirements)):
        raise ValueError("invalid_requirements")
    if any(_prompt_key(r) in {"task_scope", "fact_pack", "visible_facts", "expected_direct_answer"}
           for r in requirements):
        raise ValueError("requirements_are_schema_names")
    for key in ("missing_information", "assumed_information"):
        if not isinstance(obj.get(key), str):
            if job["class"] == "clear" and obj.get(key) is None:
                obj[key] = ""
            else:
                raise ValueError("missing_" + key)
        # Clear writing may record optional defaults; they are not essential gaps.
        wrong_missing = key == "missing_information" and ((job["class"] == "ambiguous") != bool(obj[key].strip()))
        missing_assumption = key == "assumed_information" and job["class"] == "ambiguous" and not obj[key].strip()
        if wrong_missing or missing_assumption:
            raise ValueError("inconsistent_" + key)
    prose = obj["prompt"].split("\n\n", 1)[0]
    words = len(prose.split())
    # Style lengths are generation targets, not reasons to discard a correct pair.
    if not 1 <= words <= 180:
        raise ValueError("prompt_length")
    if any(len(obj[key].split()) > 120 for key in ("chosen", "rejected")):
        raise ValueError("response_length")
    if _prompt_key(obj["chosen"]) == _prompt_key(obj["rejected"]):
        raise ValueError("identical_responses")
    clarification = obj["chosen"] if job["class"] == "ambiguous" else obj["rejected"]
    if "?" not in clarification and not re.search(
            r"\bplease\s+(?:share|provide|specify|send|tell|clarify)\b", clarification, re.I):
        raise ValueError("missing_clarification_request")
    if any(re.search(r"<\|[^>]+\|>|^\s*(assistant|user)\s*:", obj[key], re.I)
           for key in ("prompt", "chosen", "rejected")):
        raise ValueError("chat_artifact")
    if len(obj["skill"].split()) > 12 or len(obj["answerability_reason"]) > 1500:
        raise ValueError("metadata_length")
    if job["class"] == "ambiguous" and _trusted_factual_answer(obj["prompt"]):
        raise ValueError("answerable_known_fact")
    if job["domain"] == "multiple_choice" and job["class"] == "clear":
        if not re.fullmatch(r"[ABCD]", obj["chosen"]):
            raise ValueError("mcq_answer_format")
        options = list(re.finditer(r"(?:^|\s)([ABCD])[.):]\s+", obj["prompt"]))
        if len(options) != 4 or {m[1] for m in options} != set("ABCD"):
            raise ValueError("mcq_options_missing")
        stem = obj["prompt"][:options[0].start()]
        answers = _trusted_factual_answer(stem)
        if answers:
            correct = []
            for index, option in enumerate(options):
                end = options[index + 1].start() if index < 3 else len(obj["prompt"])
                value = _prompt_key(obj["prompt"][option.end():end])
                value = re.split(r"\s+(?:answer|respond|reply|return|give|choose|select)\b",
                                 value, maxsplit=1)[0]
                if value.strip(" .,:;!?*()") in answers:
                    correct.append(option[1])
            if correct != [obj["chosen"]]:
                raise ValueError("wrong_mcq_answer")
    if truth:
        if job["domain"] == "arithmetic" and truth["operation"] in {"percentage", "discount"}:
            requested_operation = _percentage_operation(prose)
            if requested_operation != truth["operation"]:
                raise ValueError("arithmetic_operation_not_requested")
        if job["domain"] == "text_transform":
            cues = {"uppercase": r"upper\s*case|all caps|capital letters",
                    "lowercase": r"lower\s*case|small letters",
                    "word_count": r"how many words|count.*words|word count|number of words",
                    "sort_words": r"alphabet|sort", "initials": r"initial"}
            if not re.search(cues[truth["operation"]], obj["prompt"], re.I | re.S):
                raise ValueError("text_operation_not_requested")
        for key, value in truth["facts"].items():
            present = _contains_value(obj["prompt"], value)
            if key == truth["hidden_key"] and present:
                raise ValueError("hidden_fact_leaked")
            if key != truth["hidden_key"] and not present:
                raise ValueError("visible_fact_missing")
        direct = obj["chosen"] if job["class"] == "clear" else obj["rejected"]
        expected = truth["expected"]
        if obj["reference_answer"] != expected:
            reference = obj["reference_answer"].replace(",", "")
            numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", reference)
            if re.fullmatch(r"-?\d+", expected) and len(numbers) == 1 and Decimal(numbers[0]) == int(expected):
                obj["reference_answer"] = expected
            else:
                raise ValueError("wrong_reference_answer")
        if re.fullmatch(r"-?\d+", expected):
            matches = _contains_value(direct, int(expected))
            if job["domain"] == "arithmetic":
                numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", direct.replace(",", ""))
                matches = bool(numbers) and Decimal(numbers[-1]) == int(expected)
        else:
            matches = re.sub(r"\s+", "", expected) in re.sub(r"\s+", "", direct)
        if not matches:
            raise ValueError("wrong_computed_answer")
    pair = {"class": job["class"], **{key: obj[key] for key in ("prompt", "chosen", "rejected")}}
    return {"control": dict(job), "pair": pair, "ground_truth": truth,
            "task": {"domain": job["domain"], "planned_requirement": _information_requirement(job),
                     **{key: obj[key] for key in
                     ("requirements", "missing_information", "answerability_reason", "skill",
                      "reference_answer", "assumed_information")}}}


REVIEW_SYSTEM = """Check whether the answer correctly and completely performs the user's task.
Use only inputs in the prompt, ordinary knowledge, or reasonable defaults for a generic draft.
Check calculations and factual claims. Rewrites must preserve facts; summaries and extractions
must match their source. Reject invented private facts, missing essential inputs, conflicting
instructions, incomplete comparisons, and answers to a different question. A bare number,
option letter, or program output is sufficient when that is all the request asks for.
Judge only the stated requirements. Do not demand unstated names, explanations, compensation,
or additional content. Fictional drafts may use placeholders. Check that the writer's role
and intended message match the request, rather than substituting a different communication.
Return only JSON: {"valid": true or false, "reason": "under 20 words"}."""


def _needs_semantic_review(record):
    # Only computed answers and source-derived quotes/extractions bypass model review.
    return not (record["ground_truth"] or (
        record["control"]["domain"] in {"knowledge", "document_tasks"}
        and "source_draft" in record))


def _review_messages(record):
    # Construction determines which input to withhold and the exact clarification.
    # Review a completed task once, so no hidden input is mistaken for supplied input.
    prompt = record.get("complete_prompt", record["pair"]["prompt"])
    truth = record.get("ground_truth")
    if truth and truth["hidden_key"]:
        key = truth["hidden_key"]
        prompt += "\n\n" + key.replace("_", " ").capitalize() + ":\n" + str(truth["facts"][key])
    return [{"role": "system", "content": REVIEW_SYSTEM},
            {"role": "user", "content": json.dumps({"prompt": prompt,
             "answer": record["task"]["reference_answer"]}, ensure_ascii=False)}]


def _review_result(raw, label=None):
    obj = _parse_object(raw)
    if not obj or obj.get("valid") is not True:
        return None
    if not isinstance(obj.get("reason"), str) or not obj["reason"].strip():
        return None
    return obj


def _structure_tokens(text):
    # Normalize superficial number/name differences while retaining task words.
    text = re.sub(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", " name ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", " number ", text.lower())
    return re.findall(r"[a-z_]+", text)


def _shingles(tokens, width=3):
    return {tuple(tokens[i:i + width]) for i in range(max(1, len(tokens) - width + 1))}


def _similar(a, b):
    return bool(a and b) and len(a & b) / len(a | b) >= 0.78


class _DiversityFilter:
    """Reject copied prompts and cap repeated language within each class."""

    def __init__(self, n, blocked_prompts=()):
        self.blocked = {_prompt_key(p) for p in blocked_prompts if p.strip()}
        self.exact = set(self.blocked)
        self.prompts = []
        self.prompt_shapes = Counter()
        self.response_shapes = Counter()
        self.responses = []
        self.skills = Counter()
        self.shape_limit = max(2, math.ceil(n * 0.008))
        self.response_limit = max(3, math.ceil(n * 0.01))
        # Each clear knowledge/writing category needs about 10% of all pairs,
        # but its rejected responses use only eight fixed questions. At n=1000,
        # 15 uses each leaves room for 100 pairs without exhausting the bank.
        self.fixed_question_limit = max(self.response_limit, math.ceil(n * 0.015))
        self.skill_limit = max(3, math.ceil(n * 0.035))

    def _response_limit(self, record, role):
        if (record["pair"]["class"] == "clear" and role == "rejected"
                and record["task"]["domain"] in {"knowledge", "writing"}):
            return self.fixed_question_limit
        return self.response_limit

    def _keys(self, record):
        pair, task = record["pair"], record["task"]
        tokens = _structure_tokens(pair["prompt"])
        shape = " ".join(tokens)
        truth = record.get("ground_truth")
        skill_name = truth.get("skill", truth["operation"]) if truth else _prompt_key(task["skill"])
        skill = (pair["class"], task["domain"], skill_name)
        # Short valid answers such as 'B', '42', and 'True' must be allowed to recur.
        direct_role = "chosen" if pair["class"] == "clear" else "rejected"
        responses = []
        for role in ("chosen", "rejected"):
            simple_answer = re.fullmatch(r"[ABCD]|[-+]?\d+(?:\.\d+)?|True|False|None|\[[\d, \-]*\]", pair[role])
            verified_time = (truth and truth["operation"] == "finish time"
                             and pair[role] == truth["expected"])
            if role == direct_role and (simple_answer or verified_time):
                continue
            responses.append((pair["class"], role, " ".join(_structure_tokens(pair[role]))))
        return shape, skill, responses, _shingles(tokens)

    def check(self, record):
        pair = record["pair"]
        prompt = _prompt_key(pair["prompt"])
        # An MMLU question may be embedded in a longer prompt with choices.
        if prompt in self.exact or any(blocked in prompt for blocked in self.blocked):
            return "duplicate_or_eval_prompt"
        shape, skill, responses, shingles = self._keys(record)
        if self.skills[skill] >= self.skill_limit:
            return "skill_cap"
        if self.prompt_shapes[pair["class"], shape] >= self.shape_limit:
            return "prompt_structure_cap"
        if any(self.response_shapes[key] >= self._response_limit(record, key[1])
               for key in responses):
            return "response_repetition_cap"
        for label, role, text in responses:
            signature = _shingles(text.split())
            similar_count = sum(previous_label == label and previous_role == role
                                and _similar(signature, previous)
                                for previous_label, previous_role, previous in self.responses)
            if similar_count >= self._response_limit(record, role):
                return "similar_response_cap"
        for label, previous in self.prompts:
            if label == pair["class"] and _similar(shingles, previous):
                return "near_duplicate_prompt"
        return None

    def add(self, record):
        shape, skill, responses, shingles = self._keys(record)
        pair = record["pair"]
        self.exact.add(_prompt_key(pair["prompt"]))
        self.prompts.append((pair["class"], shingles))
        self.prompt_shapes[pair["class"], shape] += 1
        self.skills[skill] += 1
        self.response_shapes.update(responses)
        self.responses.extend((label, role, _shingles(text.split())) for label, role, text in responses)


def _held_out_prompts():
    """Only exclude local evaluation text; never include it in model requests."""
    eval_dir = Path(__file__).resolve().parent / "eval"
    prompts = []
    for name in ("held_out_ambiguous.jsonl", "held_out_clear.jsonl", "mmlu_slice.jsonl"):
        path = eval_dir / name
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    text = row.get("prompt", row.get("question"))
                    if isinstance(text, str):
                        prompts.append(text)
    return prompts


class _LocalLlama:
    def __init__(self):
        print(f"Loading {MODEL_ID} for generation and independent review...", flush=True)
        import torch
        from transformers import set_seed
        from utils.model_utils import load_tokenizer, load_base_model
        set_seed(DATA_SEED)
        self.torch = torch
        self.tokenizer = load_tokenizer(MODEL_ID)
        self.tokenizer.padding_side = "left"
        self.model = load_base_model(MODEL_ID)
        self.model.eval()

    def __call__(self, messages, max_new_tokens, review=False):
        if not messages:
            return []
        tokenizer, model, torch = self.tokenizer, self.model, self.torch
        chats = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in messages]
        inputs = None
        outputs = None
        try:
            inputs = tokenizer(chats, return_tensors="pt", padding=True,
                               truncation=False, add_special_tokens=False).to(model.device)
            if inputs["input_ids"].shape[1] > 7000:
                raise ValueError("Generation/review input exceeds 7000 tokens; refusing to truncate it.")
            options = {"do_sample": False} if review else {
                "do_sample": True, "temperature": 0.8, "top_p": 0.95}
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                         pad_token_id=tokenizer.pad_token_id, use_cache=True, **options)
            tails = outputs[:, inputs["input_ids"].shape[1]:]
            return tokenizer.batch_decode(tails, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)
        except torch.cuda.OutOfMemoryError:
            # Release this batch before retrying smaller batches on the same GPU.
            inputs = None
            outputs = None
            torch.cuda.empty_cache()
            if len(messages) == 1:
                raise
        finally:
            del inputs, outputs
        middle = len(messages) // 2
        return (self(messages[:middle], max_new_tokens, review)
                + self(messages[middle:], max_new_tokens, review))


def _save_state(path, state):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _synthesize_pairs(n, *, checkpoint_path=None, resume=False, generate_many=None):
    """Generate, mechanically check, independently review, and filter each example.

    The optional callable lets tests exercise the full pipeline without loading a GPU.
    It has the same interface as _LocalLlama.__call__.
    """
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n must be nonnegative")
    if n == 0:
        return []
    jobs = _schedule(n)
    by_slot = {job["slot"]: job for job in jobs}
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    records, attempts, stats = [], Counter(), Counter()
    rejection_samples = []
    retry_feedback = {}
    diversity = _DiversityFilter(n, _held_out_prompts())
    if resume and checkpoint is not None and checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if (state.get("version"), state.get("seed"), state.get("n"), state.get("model")) != (
                GENERATOR_VERSION, DATA_SEED, n, MODEL_ID):
            raise ValueError("Checkpoint belongs to a different generator, seed, or dataset size.")
        records = state["records"]
        attempts.update(state["attempts"])
        stats.update(state["stats"])
        rejection_samples = state.get("rejection_samples", [])[-12:]
        retry_feedback = state.get("retry_feedback", {})
        if any(slot not in by_slot or type(count) is not int or not 0 <= count <= MAX_ATTEMPTS_PER_SLOT
               for slot, count in attempts.items()):
            raise ValueError("Invalid checkpoint attempt counters.")
        restored = set()
        for record in records:
            job = record["control"]
            slot = job["slot"]
            if slot in restored or by_slot.get(slot) != job:
                raise ValueError("Duplicate or incompatible checkpoint slot.")
            if attempts[slot] < 1 or record["ground_truth"] != _ground_truth(job, attempts[slot] - 1):
                raise ValueError("Checkpoint fact pack does not match its seeded task.")
            obj = {**record["pair"], **record["task"]}
            if _source_plan(job, attempts[slot] - 1):
                rebuilt = _source_record(json.dumps(record["source_draft"]), job, attempts[slot] - 1)
                if (rebuilt["pair"] != record["pair"] or rebuilt["complete_prompt"] != record.get("complete_prompt")
                        or rebuilt["task"] != record["task"]):
                    raise ValueError("Checkpoint source and exported pair disagree.")
            elif record["ground_truth"]:
                constructor = _mcq_record if job["domain"] == "multiple_choice" else _computed_record
                rebuilt = constructor(json.dumps({"context": record["context"]}), job, record["ground_truth"])
                if rebuilt["pair"] != record["pair"] or rebuilt["task"] != record["task"]:
                    raise ValueError("Checkpoint computed task differs from its construction.")
            else:
                rebuilt = _candidate_record(json.dumps(obj), job, record["ground_truth"])
            direct_role = "chosen" if job["class"] == "clear" else "rejected"
            if record["task"]["reference_answer"] != record["pair"][direct_role]:
                raise ValueError("Checkpoint answer differs from the reviewed answer.")
            review_label = "clear" if rebuilt.get("review_scope") else job["class"]
            review = _review_result(json.dumps(record["review"]), review_label)
            if not review or diversity.check(rebuilt):
                raise ValueError("Checkpoint record fails current review or diversity checks.")
            diversity.add(rebuilt)
            restored.add(slot)
        print(f"Resumed {len(records)}/{n} accepted pairs", flush=True)
    accepted = {r["control"]["slot"] for r in records}
    pending = [job for job in jobs if job["slot"] not in accepted]

    def save():
        state = {"version": GENERATOR_VERSION, "seed": DATA_SEED, "model": MODEL_ID,
                 "n": n, "records": records, "attempts": dict(attempts), "stats": dict(stats),
                 "rejection_samples": rejection_samples[-12:], "retry_feedback": retry_feedback}
        _save_state(checkpoint, state)
        return state

    def note_failure(job, detail):
        slot = job["slot"]
        retry_feedback[slot] = detail[:900]
        if attempts[slot] == 1 or attempts[slot] % 3 == 0:
            brief = " ".join(detail.split())[:400]
            print(f"Retry {slot} ({job['domain']}, failure {attempts[slot]}): {brief}", flush=True)

    save()
    if pending and generate_many is None:
        generate_many = _LocalLlama()
    while pending:
        batch, pending = pending[:GENERATION_BATCH_SIZE], pending[GENERATION_BATCH_SIZE:]
        for job in batch:
            if attempts[job["slot"]] >= MAX_ATTEMPTS_PER_SLOT:
                save()
                raise RuntimeError(f"Could not fill {job['class']}/{job['domain']} slot {job['slot']} "
                                   f"after {MAX_ATTEMPTS_PER_SLOT} attempts. Accepted {len(records)}/{n}. "
                                   f"No low-quality fallback was written. Diagnostics: {dict(stats)}")
        truths, messages = [], []
        for job in batch:
            attempt = attempts[job["slot"]]
            truth = _ground_truth(job, attempt)
            truths.append(truth)
            recent = [{"skill": r["task"]["skill"], "prompt": r["pair"]["prompt"],
                       "chosen": r["pair"]["chosen"], "rejected": r["pair"]["rejected"]}
                      for r in records if r["control"]["domain"] == job["domain"]][-3:]
            plan = _source_plan(job, attempt)
            if plan:
                messages.append(_source_messages(job, plan, recent, retry_feedback.get(job["slot"], "")))
            elif job["domain"] == "multiple_choice":
                messages.append(_mcq_messages(job, truth, recent))
            else:
                messages.append(_author_messages(job, truth, recent, attempt,
                                                 retry_feedback.get(job["slot"], "")))
        raw_candidates = generate_many(messages, max_new_tokens=1100, review=False)
        if len(raw_candidates) != len(batch):
            raise RuntimeError("Author returned the wrong number of candidates.")
        candidates = []
        for raw, job, truth in zip(raw_candidates, batch, truths):
            attempts[job["slot"]] += 1
            try:
                obj = _complete_metadata(_parse_object(raw), job, truth)
                if obj is not None:
                    raw = json.dumps(obj, ensure_ascii=False)
                if _source_plan(job, attempts[job["slot"]] - 1):
                    record = _source_record(raw, job, attempts[job["slot"]] - 1)
                elif job["domain"] == "multiple_choice":
                    record = _mcq_record(raw, job, truth)
                elif truth:
                    record = _computed_record(raw, job, truth)
                else:
                    record = _candidate_record(raw, job, truth)
                reason = diversity.check(record)
                if reason:
                    raise ValueError(reason)
                candidates.append(record)
            except ValueError as exc:
                stats[str(exc)] += 1
                note_failure(job, f"Python validation failed: {exc}. Fix this in the new pair.")
                rejection_samples.append({"control": job, "stage": "construction",
                                          "reason": str(exc), "raw": raw[:6000]})
                del rejection_samples[:-12]
                pending.append(job)
        if candidates:
            # Exact operations, inputs, replies and MCQ letters are constructed in
            # Python. Review only model-written answers; never ask an LLM to
            # second-guess a verified calculation or invent its class again.
            needs_review = [r for r in candidates if _needs_semantic_review(r)]
            reviews = (generate_many([_review_messages(r) for r in needs_review],
                                    max_new_tokens=120, review=True) if needs_review else [])
            if len(reviews) != len(needs_review):
                raise RuntimeError("Reviewer returned the wrong number of judgments.")
            reviews = iter(reviews)
            for record in candidates:
                raw = (json.dumps({"valid": True, "method": "python",
                                   "reason": "Answer constructed directly from the task inputs."})
                       if not _needs_semantic_review(record)
                       else next(reviews))
                review_label = "clear" if record.get("review_scope") else record["pair"]["class"]
                review = _review_result(raw, review_label)
                reason = diversity.check(record)  # Also catch repetition within this batch.
                if review is None or reason:
                    stats[reason or "semantic_review_rejected"] += 1
                    judgment = _parse_object(raw) or {}
                    detail = reason or (
                        f"Reviewer: valid={judgment.get('valid')}. "
                        + str(judgment.get("reason", "No usable reviewer reason; the review output was invalid."))[:240])
                    previous = {key: record["pair"][key][:140] for key in ("prompt", "chosen", "rejected")}
                    note_failure(record["control"], detail + "\nPrevious pair: " + json.dumps(previous, ensure_ascii=False))
                    rejection_samples.append({"control": record["control"], "stage": "review",
                                              "reason": reason or "semantic_review_rejected",
                                              "pair": record["pair"], "raw": raw[:3000]})
                    del rejection_samples[:-12]
                    pending.append(record["control"])
                    continue
                record["review"] = review
                retry_feedback.pop(record["control"]["slot"], None)
                diversity.add(record)
                records.append(record)
                stats["accepted"] += 1
        save()
        counts = Counter(r["pair"]["class"] for r in records)
        print(f"Accepted {len(records)}/{n}: {dict(counts)}; checks: {dict(stats)}", flush=True)
    state = save()
    if checkpoint is not None:
        audit_path = checkpoint.with_name(checkpoint.name.replace(".partial.json", ".audit.json"))
        if audit_path == checkpoint:
            audit_path = checkpoint.with_name(checkpoint.name + ".audit.json")
        state["complete"] = True
        state["domain_counts"] = dict(Counter(r["control"]["domain"] for r in records))
        _save_state(audit_path, state)
        print(f"Saved task requirements, references, and review decisions to {audit_path}", flush=True)
    pairs = [r["pair"] for r in records]
    random.Random(DATA_SEED).shuffle(pairs)
    return pairs


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
    # Do not remove preference_pairs.json for data generation
    existing_path = Path(PREFERENCE_DATA_PATH)
    if existing_path.is_file():
        required = {"class", "prompt", "chosen", "rejected"}
        pairs = []
        for line_number, line in enumerate(
            existing_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                pair = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{existing_path}:{line_number}: invalid JSON ({exc})"
                ) from exc
            if not isinstance(pair, dict):
                raise ValueError(f"{existing_path}:{line_number}: expected a JSON object")
            missing = required - pair.keys()
            if missing:
                raise ValueError(
                    f"{existing_path}:{line_number}: missing keys {sorted(missing)}"
                )
            if pair["class"] not in VALID_CLASSES:
                raise ValueError(
                    f"{existing_path}:{line_number}: invalid class {pair['class']!r}"
                )
            pairs.append(pair)

        if len(pairs) < MIN_PREFERENCE_PAIRS:
            raise ValueError(
                f"{existing_path} contains {len(pairs)} pairs; at least "
                f"{MIN_PREFERENCE_PAIRS} are required."
            )
        print(f"Using {len(pairs)} existing preference pairs from {existing_path}")
        return pairs

    if n >= MIN_PREFERENCE_PAIRS:
        checkpoint = Path(PREFERENCE_DATA_PATH).with_suffix(f".{GENERATOR_VERSION}.partial.json")
        pairs = _synthesize_pairs(n, checkpoint_path=checkpoint, resume=checkpoint.exists())
        checkpoint.unlink(missing_ok=True)
        return pairs
    return _synthesize_pairs(n)


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
