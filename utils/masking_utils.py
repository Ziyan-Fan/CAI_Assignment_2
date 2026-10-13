"""
Shared tokenization + response-masking logic used by both Part A (train_sft.py) and
Part B (train_dpo.py). "Mask everything except the final assistant turn" is the same
operation in both places, so it lives here once instead of being re-derived
independently in two files.
"""

import torch


def tokenize_with_response_mask(messages: list[dict], tokenizer, max_seq_len: int) -> dict:
    """
    TODO (student): tokenizes one chat-formatted conversation and marks which tokens
    belong to the final assistant turn. `messages` is a list of {"role", "content"}
    dicts (user/assistant, alternating, ending on an assistant turn) -- see data_loader.py.

    Must return a dict with:
      - "input_ids": 1D LongTensor -- the full tokenized conversation
        (`tokenizer.apply_chat_template(messages, tokenize=True, ...)`), truncated to
        at most max_seq_len tokens.

      - "attention_mask": 1D LongTensor, same length, all 1s (no padding needed here --
        batching/collating happens in the caller's train_loop()).

      - "response_mask": 1D LongTensor, same length: 1 for the final assistant-turn
        tokens, 0 everywhere else (prompt tokens and any earlier turns).

    This is the function the autograder unit-tests directly (see README.md's
    Autograder section) -- its signature and the three return keys above must not
    change. Both train_sft.build_sft_example() and train_dpo.build_dpo_example() call
    this rather than re-deriving the masking logic independently -- implement it once,
    here, and both Part A and Part B get it for free.

    Hint: `tokenizer.apply_chat_template(messages[:-1], tokenize=True,
    add_generation_prompt=True)` tokenizes everything up through the prompt that
    invites the final assistant turn. Its length tells you exactly where the response
    starts in the full conversation's tokenization -- everything before that index is
    prompt, everything from it onward (up to the end-of-turn token) is response.
    """
    raise NotImplementedError("Implement tokenization + response masking.")
