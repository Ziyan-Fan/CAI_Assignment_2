"""
Shared tokenization + response-masking logic used by both Part A (train_sft.py) and
Part B (train_dpo.py). "Mask everything except the final assistant turn" is the same
operation in both places, so it lives here once instead of being re-derived
independently in two files.
"""

import torch


def tokenize_with_response_mask(messages: list[dict], tokenizer, max_seq_len: int) -> dict:
    """Tokenize a chat and supervise only its final assistant response.

    Return one-dimensional long tensors ``input_ids``, ``attention_mask``, and
    ``response_mask``. Input IDs are the full chat's first ``max_seq_len`` tokens;
    attention is all ones. The response mask excludes the generation prompt and
    all earlier turns, and includes the final end-of-turn token when retained.
    Truncation may leave no response tokens; callers must handle that case.

    The template's generation prompt must be an exact token prefix of the full
    chat. A mismatch raises an error rather than guessing a response boundary.
    """
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list of chat messages.")
    if not isinstance(messages[-1], dict) or messages[-1].get("role") != "assistant":
        raise ValueError("messages must end with an assistant turn.")
    if isinstance(max_seq_len, bool) or not isinstance(max_seq_len, int) or max_seq_len <= 0:
        raise ValueError("max_seq_len must be a positive integer.")
    if len(messages) < 2:
        raise ValueError("messages must include a prompt before the final assistant turn.")

    # Tokenize the whole conversation, including the final response.
    full_tokens = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )

    # The prompt includes the assistant header, but not its response.
    prompt_tokens = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True
    )
    response_start = len(prompt_tokens)
    if full_tokens[:response_start] != prompt_tokens:
        raise ValueError(
            "The chat template's generation prompt is not a token prefix of the "
            "complete conversation; cannot safely identify the final assistant response."
        )

    input_ids = torch.tensor(full_tokens[:max_seq_len], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    # Mask positions, not token IDs: an end-of-turn token may also be the pad ID.
    response_mask = torch.zeros_like(input_ids)
    response_mask[response_start:] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
    }
