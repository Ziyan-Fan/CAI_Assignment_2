"""
Part A -- LoRA SFT training entrypoint.

SFT warm-starts the target behavior: it fine-tunes the instruct base model on the
PREFERRED response of each preference pair you built (a clarifying question for an
ambiguous prompt, a direct answer for a clear one), so that DPO in Part B has a sensible
policy to sharpen rather than starting cold.

CLI args, tokenizer setup, base model + LoRA construction, and checkpointing are wired
up already. You implement utils/masking_utils.tokenize_with_response_mask() and
train_loop() -- see the TODO blocks and README.md's Part A for two gotchas: freezing
base weights, masking prompt tokens.
"""

import argparse

import torch
from peft import LoraConfig, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from configs.paths_config import SFT_CHECKPOINT_DIR
from configs.train_config import (
    MODEL_ID,
    RANK,
    ALPHA,
    DROPOUT,
    TARGET_MODULES,
    SFT_BATCH_SIZE,
    SFT_EPOCHS,
    SFT_LEARNING_RATE,
    SFT_MAX_SEQ_LEN,
)
from data_loader import load_sft_examples
from utils.model_utils import load_tokenizer, load_base_model
from utils.masking_utils import tokenize_with_response_mask


def build_model_and_tokenizer():
    """Loads the base model + tokenizer, and wraps the model with a LoRA adapter."""
    tokenizer = load_tokenizer(MODEL_ID)
    model = load_base_model()

    lora_config = LoraConfig(
        r=RANK,
        lora_alpha=ALPHA,
        lora_dropout=DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Sanity check: should be a small fraction of total params.

    return model, tokenizer


def build_sft_example(messages: list[dict], tokenizer, max_seq_len: int = SFT_MAX_SEQ_LEN) -> dict:
    """
    Wired up: converts utils.masking_utils.tokenize_with_response_mask()'s response_mask into
    the -100-masked labels tensor the causal-LM loss needs. Once you implement
    tokenize_with_response_mask(), this works for free -- nothing to implement here.

    Returns a dict with "input_ids", "attention_mask", "labels": labels equal
    input_ids where response_mask == 1 (the final assistant turn), -100 elsewhere.
    """
    example = tokenize_with_response_mask(messages, tokenizer, max_seq_len)

    labels = example["input_ids"].clone()
    labels[example["response_mask"] == 0] = -100

    return {
        "input_ids": example["input_ids"],
        "attention_mask": example["attention_mask"],
        "labels": labels,
    }


def train_loop(model, tokenizer, dataset) -> None:
    """
    Implements the Part A SFT training loop.

    TODO (student):
      1. Preprocess `dataset` ONCE, up front, by calling build_sft_example() per example
         (e.g. via `dataset.map(...)`) to get input_ids/attention_mask/labels -- do not
         call build_sft_example() inside the per-step loop below, or you'll re-tokenize
         the same examples on every epoch for no benefit.
      2. For each batch of preprocessed examples: pad/collate into a batch, forward pass,
         compute loss, backward, optimizer step. Only LoRA parameters should have
         gradients -- true by default from build_model_and_tokenizer() above, but verify
         it if you change the LoRA setup (README.md Part A, gotcha #1).
      3. Log the loss periodically so you can sanity-check training is converging.

    Hyperparameters (SFT_BATCH_SIZE, SFT_EPOCHS, SFT_LEARNING_RATE, SFT_MAX_SEQ_LEN) are
    in configs/train_config.py -- yours to tune, not fixed by this skeleton.
    """
    # Tokenize each example once, before starting the epochs.
    training_examples = []
    for row in dataset:
        example = build_sft_example(row["messages"], tokenizer, SFT_MAX_SEQ_LEN)
        # Skip examples where truncation removed all usable response tokens.
        if (example["labels"][1:] != -100).any():
            training_examples.append(example)

    if not training_examples:
        raise ValueError("No training examples have response tokens after truncation.")
    print(f"Prepared {len(training_examples)} of {len(dataset)} examples for SFT.")

    def collate_batch(examples):
        # Pad to the longest sequence in this batch. Ignore padding in the loss.
        return {
            "input_ids": pad_sequence(
                [example["input_ids"] for example in examples],
                batch_first=True,
                padding_value=tokenizer.pad_token_id,
            ),
            "attention_mask": pad_sequence(
                [example["attention_mask"] for example in examples],
                batch_first=True,
                padding_value=0,
            ),
            "labels": pad_sequence(
                [example["labels"] for example in examples],
                batch_first=True,
                padding_value=-100,
            ),
        }

    train_loader = DataLoader(
        training_examples,
        batch_size=SFT_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch,
    )

    # PEFT freezes the base model, so only the LoRA weights are optimized.
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=SFT_LEARNING_RATE)
    device = next(model.parameters()).device
    model.train()

    for epoch in range(SFT_EPOCHS):
        total_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}

            optimizer.zero_grad()
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            total_loss += loss_value
            if step == 1 or step % 10 == 0 or step == len(train_loader):
                print(
                    f"Epoch {epoch + 1}/{SFT_EPOCHS}, "
                    f"step {step}/{len(train_loader)}, loss: {loss_value:.4f}"
                )

        print(f"Epoch {epoch + 1} average loss: {total_loss / len(train_loader):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default=SFT_CHECKPOINT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f"Loading base model {MODEL_ID} and building LoRA adapter...")
    model, tokenizer = build_model_and_tokenizer()

    print("Loading SFT examples (preferred responses from your preference data)...")
    dataset = load_sft_examples()

    print("Training starts...")
    train_loop(model, tokenizer, dataset)

    print(f"Saving LoRA adapter to {args.checkpoint_dir}...")
    model.save_pretrained(args.checkpoint_dir)
    tokenizer.save_pretrained(args.checkpoint_dir)
    print("Part A SFT complete.")


if __name__ == "__main__":
    main()
