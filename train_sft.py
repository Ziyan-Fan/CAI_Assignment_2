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


def DataLoader(*args, **kwargs):
    """Construct a batch loader while keeping its import local to this helper."""
    from torch.utils.data import DataLoader as TorchDataLoader

    return TorchDataLoader(*args, **kwargs)


def _checkpoint_steps(save_epochs, epochs: int, steps_per_epoch: int) -> dict:
    """Round each requested epoch up to a completed optimizer step."""
    import math

    schedule = {}
    seen = set()
    for requested_epoch in save_epochs:
        requested_epoch = float(requested_epoch)
        if not math.isfinite(requested_epoch) or not 0 < requested_epoch <= epochs:
            raise ValueError("Each save epoch must be finite, positive, and no greater than epochs.")
        if requested_epoch in seen:
            raise ValueError(f"Duplicate save epoch: {requested_epoch:g}")
        seen.add(requested_epoch)
        step = math.ceil(requested_epoch * steps_per_epoch)
        schedule.setdefault(step, []).append(requested_epoch)
    return schedule


def _response_losses(logits, labels, reference_logits=None):
    """Return per-example response CE, optional base-to-policy KL, and token counts.

    Select response positions before float32 softmax: prompt and padding tokens
    contribute neither a gradient nor a vocabulary-sized float32 temporary.
    """
    import torch.nn.functional as F

    targets = labels[:, 1:]
    valid = targets.ne(-100)
    counts = valid.sum(dim=1)
    if not bool(counts.gt(0).all()):
        raise ValueError("Every training example must contain a predictable response token.")
    rows = valid.nonzero(as_tuple=True)[0]
    response_logits = logits[:, :-1, :][valid].float()
    token_ce = F.cross_entropy(response_logits, targets[valid], reduction="none")
    sums = token_ce.new_zeros(labels.shape[0]).scatter_add(0, rows, token_ce)
    ce = sums / counts
    kl = ce.new_zeros(ce.shape)
    if reference_logits is not None:
        reference_logps = F.log_softmax(reference_logits[:, :-1, :][valid].detach().float(), dim=-1)
        policy_logps = F.log_softmax(response_logits, dim=-1)
        token_kl = F.kl_div(policy_logps, reference_logps, reduction="none", log_target=True).sum(dim=-1)
        kl = token_kl.new_zeros(labels.shape[0]).scatter_add(0, rows, token_kl) / counts
    return ce, kl, counts


def train_loop(
    model,
    tokenizer,
    dataset,
    *,
    learning_rate: float = SFT_LEARNING_RATE,
    epochs: int = SFT_EPOCHS,
    save_epochs: list[float] | None = None,
    checkpoint_dir: str | None = None,
    seed: int | None = None,
    data_sha256: str | None = None,
    class_labels: list[str] | None = None,
    loss_normalization: str | None = None,
    clear_weight: float | None = None,
    base_kl_weight: float | None = None,
    max_grad_norm: float | None = None,
) -> None:
    """Train response-only SFT, with optional clear-example emphasis and retention KL.

    Example normalization gives each response equal weight before class weighting.
    Class weights use a fixed dataset-level normalizer, so homogeneous minibatches
    do not cancel the requested weighting. Optional KL anchors clear responses to
    the frozen base model; the targets remain the supplied preferred responses.
    Omitted objective options use configs/train_config.py. For the default
    dataset, class labels are loaded only after verifying exact pair alignment.
    """
    from contextlib import nullcontext
    import hashlib
    import json
    import math
    from pathlib import Path

    from torch.nn.utils.rnn import pad_sequence

    from configs.paths_config import PREFERENCE_DATA_PATH
    from configs.train_config import (
        SFT_LOSS_NORMALIZATION,
        SFT_CLEAR_WEIGHT,
        SFT_BASE_KL_WEIGHT,
        SFT_MAX_GRAD_NORM,
    )

    if loss_normalization is None:
        loss_normalization = SFT_LOSS_NORMALIZATION
    if clear_weight is None:
        clear_weight = SFT_CLEAR_WEIGHT
    if base_kl_weight is None:
        base_kl_weight = SFT_BASE_KL_WEIGHT
    if max_grad_norm is None:
        max_grad_norm = SFT_MAX_GRAD_NORM
    if seed is None:
        seed = torch.initial_seed()

    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("The learning rate must be finite and positive.")
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("The number of epochs must be a positive integer.")
    if loss_normalization not in {"token", "example"}:
        raise ValueError("loss_normalization must be 'token' or 'example'.")
    if not math.isfinite(clear_weight) or clear_weight <= 0:
        raise ValueError("clear_weight must be finite and positive.")
    if not math.isfinite(base_kl_weight) or base_kl_weight < 0:
        raise ValueError("base_kl_weight must be finite and nonnegative.")
    if not math.isfinite(max_grad_norm) or max_grad_norm < 0:
        raise ValueError("max_grad_norm must be finite and nonnegative; zero disables clipping.")
    if class_labels is None and (clear_weight != 1.0 or base_kl_weight):
        from data_loader import load_preference_pairs

        # The original entrypoint supplies only messages. Recover explicit classes
        # from the same preference file, never by guessing from response wording.
        preference_path = Path(PREFERENCE_DATA_PATH)
        if not preference_path.is_file():
            raise ValueError(
                "Class weighting and clear-response KL require class_labels or the "
                "matching preference data file. Pass class_labels for custom datasets."
            )
        preference_hash = hashlib.sha256(preference_path.read_bytes()).hexdigest()
        pairs = load_preference_pairs()
        if len(pairs) != len(dataset) or any(
            row["messages"] != [{"role": "user", "content": pair["prompt"]},
                                {"role": "assistant", "content": pair["chosen"]}]
            for row, pair in zip(dataset, pairs)
        ):
            raise ValueError(
                "Preference classes do not align with the supplied SFT examples; "
                "pass explicit class_labels for custom datasets."
            )
        if hashlib.sha256(preference_path.read_bytes()).hexdigest() != preference_hash:
            raise RuntimeError("Preference data changed while loading class labels.")
        if data_sha256 is not None and data_sha256 != preference_hash:
            raise RuntimeError("Preference data does not match the supplied data_sha256.")
        data_sha256 = preference_hash
        class_labels = [pair["class"] for pair in pairs]

    if class_labels is not None:
        if len(class_labels) != len(dataset) or any(value not in {"clear", "ambiguous"} for value in class_labels):
            raise ValueError("class_labels must align with the dataset and contain clear/ambiguous labels.")
    if base_kl_weight and not hasattr(model, "disable_adapter"):
        raise ValueError("Base-model KL requires a PEFT model with disable_adapter().")
    experiment_dir = None
    if save_epochs is not None:
        if not save_epochs or checkpoint_dir is None:
            raise ValueError("Experiment saves require save epochs and a checkpoint directory.")
        _checkpoint_steps(save_epochs, epochs, 1)
        experiment_dir = Path(checkpoint_dir)
        # Reserve the entire experiment directory; never overwrite an earlier run.
        experiment_dir.mkdir(parents=True, exist_ok=False)

    # Tokenize each example once, before starting the epochs.
    training_examples = []
    for row_index, row in enumerate(dataset):
        example = build_sft_example(row["messages"], tokenizer, SFT_MAX_SEQ_LEN)
        # Skip examples where truncation removed all usable response tokens.
        if (example["labels"][1:] != -100).any():
            example["is_clear"] = class_labels is not None and class_labels[row_index] == "clear"
            training_examples.append(example)

    if not training_examples:
        raise ValueError("No training examples have response tokens after truncation.")
    print(f"Prepared {len(training_examples)} of {len(dataset)} examples for SFT.")
    clear_count = sum(example["is_clear"] for example in training_examples)
    clear_fraction = clear_count / len(training_examples)
    mean_class_weight = 1.0 + clear_fraction * (clear_weight - 1.0)
    if base_kl_weight and not clear_count:
        raise ValueError("Clear-response KL requires at least one retained clear example.")

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
            "is_clear": torch.tensor([example["is_clear"] for example in examples], dtype=torch.bool),
        }

    train_loader = DataLoader(
        training_examples,
        batch_size=SFT_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch,
    )

    schedule = _checkpoint_steps(save_epochs or [], epochs, len(train_loader))
    provenance = None
    if experiment_dir is not None:
        response_tokens = sum(
            int((example["labels"][1:] != -100).sum().item())
            for example in training_examples
        )
        input_tokens = sum(example["input_ids"].numel() for example in training_examples)
        provenance = {
            "base_model": MODEL_ID,
            "seed": seed,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": SFT_BATCH_SIZE,
            "max_seq_len": SFT_MAX_SEQ_LEN,
            "optimizer": "AdamW",
            "optimizer_defaults": {"betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01},
            "steps_per_epoch": len(train_loader),
            "planned_steps": epochs * len(train_loader),
            "completed_steps": 0,
            "complete": False,
            "data_path": str(PREFERENCE_DATA_PATH),
            "data_sha256": data_sha256,
            "training_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "masking_script_sha256": hashlib.sha256((Path(__file__).parent / "utils/masking_utils.py").read_bytes()).hexdigest(),
            "training_config_source_sha256": hashlib.sha256((Path(__file__).parent / "configs/train_config.py").read_bytes()).hexdigest(),
            "loss_normalization": loss_normalization,
            "clear_weight": clear_weight,
            "mean_class_weight": mean_class_weight,
            "base_kl_weight": base_kl_weight,
            "base_kl_definition": "Mean per-response KL(base || policy) on clear examples, normalized by retained clear fraction.",
            "max_grad_norm": max_grad_norm,
            "class_counts": {"clear": clear_count, "ambiguous": len(training_examples) - clear_count} if class_labels is not None else None,
            "mask_counts": {
                "dataset_examples": len(dataset),
                "retained_examples": len(training_examples),
                "skipped_examples": len(dataset) - len(training_examples),
                "input_tokens": input_tokens,
                "supervised_response_tokens": response_tokens,
                "ignored_tokens": input_tokens - response_tokens,
                "longest_sequence": max(example["input_ids"].numel() for example in training_examples),
            },
            "lora": {"rank": RANK, "alpha": ALPHA, "dropout": DROPOUT, "target_modules": TARGET_MODULES},
            "save_schedule": [
                {"requested_epoch": value, "optimizer_step": step, "actual_epoch": step / len(train_loader)}
                for step, values in sorted(schedule.items()) for value in values
            ],
            "checkpoints": [],
        }
        (experiment_dir / "training_config.json").write_text(json.dumps(provenance, indent=2) + "\n")

    # PEFT freezes the base model, so only the LoRA weights are optimized.
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise ValueError("The model has no trainable parameters.")
    if hasattr(model, "peft_config"):
        unexpected = [name for name, p in model.named_parameters() if p.requires_grad and "lora_" not in name]
        if unexpected:
            raise ValueError(f"Only LoRA parameters may be trainable; found {unexpected[:5]}")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate)
    device = next(model.parameters()).device
    model.train()

    log_context = (experiment_dir / "loss.jsonl").open("x") if experiment_dir is not None else nullcontext(None)
    global_step = 0
    cumulative_loss = 0.0
    with log_context as loss_log:
        for epoch in range(epochs):
            total_loss = 0.0
            for step, batch in enumerate(train_loader, start=1):
                batch = {key: value.to(device) for key, value in batch.items()}

                optimizer.zero_grad(set_to_none=True)
                model_inputs = {key: batch[key] for key in ("input_ids", "attention_mask")}
                outputs = model(**model_inputs, use_cache=False)
                reference_logits = None
                if base_kl_weight and bool(batch["is_clear"].any()):
                    was_training = model.training
                    try:
                        model.eval()
                        with torch.no_grad(), model.disable_adapter():
                            reference_logits = model(**model_inputs, use_cache=False).logits
                    finally:
                        model.train(was_training)
                ce, kl, counts = _response_losses(outputs.logits, batch["labels"], reference_logits)
                weights = 1.0 + batch["is_clear"].to(ce.dtype) * (clear_weight - 1.0)
                if loss_normalization == "example":
                    supervised_loss = (ce * weights).mean() / mean_class_weight
                else:
                    supervised_loss = (ce * counts * weights).sum() / counts.sum() / mean_class_weight
                retention_loss = ((kl * batch["is_clear"]).mean() / clear_fraction
                                  if base_kl_weight else ce.new_zeros(()))
                loss = supervised_loss + base_kl_weight * retention_loss
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite SFT loss; refusing an invalid optimizer update.")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, max_grad_norm or float("inf"), error_if_nonfinite=True
                )
                optimizer.step()

                loss_value = loss.item()
                total_loss += loss_value
                cumulative_loss += loss_value
                global_step += 1
                if loss_log is not None:
                    loss_log.write(json.dumps({
                        "step": global_step,
                        "epoch": epoch + step / len(train_loader),
                        "loss": loss_value,
                        "supervised_loss": supervised_loss.item(),
                        "clear_base_kl": retention_loss.item(),
                        "grad_norm_before_clip": float(grad_norm),
                        "learning_rate": learning_rate,
                    }) + "\n")
                if step == 1 or step % 10 == 0 or step == len(train_loader):
                    print(
                        f"Epoch {epoch + 1}/{epochs}, "
                        f"step {step}/{len(train_loader)}, loss: {loss_value:.4f}"
                    )
                    if loss_log is not None:
                        loss_log.flush()

                for requested_epoch in schedule.get(global_step, []):
                    destination = experiment_dir / f"epoch_{requested_epoch:g}"
                    destination.mkdir(exist_ok=False)
                    print(f"Saving epoch {requested_epoch:g} adapter at step {global_step} to {destination}...")
                    model.save_pretrained(destination)
                    tokenizer.save_pretrained(destination)
                    checkpoint_record = {
                        "directory": destination.name,
                        "requested_epoch": requested_epoch,
                        "actual_epoch": global_step / len(train_loader),
                        "completed_steps": global_step,
                        "mean_training_loss": cumulative_loss / global_step,
                    }
                    provenance["completed_steps"] = global_step
                    provenance["checkpoints"].append(checkpoint_record)
                    snapshot = {**provenance, "checkpoint": checkpoint_record, "checkpoint_complete": True}
                    (destination / "training_config.json").write_text(json.dumps(snapshot, indent=2) + "\n")
                    (experiment_dir / "training_config.json").write_text(json.dumps(provenance, indent=2) + "\n")
                    loss_log.flush()

            print(f"Epoch {epoch + 1} average loss: {total_loss / len(train_loader):.4f}")

    if provenance is not None:
        provenance.update(completed_steps=global_step, complete=True, mean_training_loss=cumulative_loss / global_step)
        (experiment_dir / "training_config.json").write_text(json.dumps(provenance, indent=2) + "\n")


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
