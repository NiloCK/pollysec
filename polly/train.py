"""
polly.train — Training loop for bracket-matching transformer experiments.

Usage:
    python -m polly.train --variant vanilla --seed 100
    python -m polly.train --variant looped_reg --seed 200 --steps 30000 --batch-size 128 --device auto
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from polly.data import BracketDataset
from polly.model import BracketTransformer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VARIANTS = ("vanilla", "vanilla_reg", "looped", "looped_reg")
LOOPED_VARIANTS = ("looped", "looped_reg")

DEFAULT_LR = 3e-4
DEFAULT_MIN_LR = 1e-5
DEFAULT_WARMUP_STEPS = 1_000
DEFAULT_TOTAL_STEPS = 30_000
DEFAULT_BATCH_SIZE = 128
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_BETAS = (0.9, 0.999)
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_GATE_LAMBDA = 0.01

CHECKPOINT_EVERY = 2_000
LOG_EVERY = 100
VAL_EVERY = 1_000

DATA_DIR = Path(__file__).resolve().parent / "data"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # numpy is optional — only seed if already imported
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    # Deterministic cuDNN for reproducibility (slight perf cost)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def get_lr(
    step: int,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    total_steps: int = DEFAULT_TOTAL_STEPS,
    max_lr: float = DEFAULT_LR,
    min_lr: float = DEFAULT_MIN_LR,
) -> float:
    """Linear warmup then cosine decay."""
    if step < warmup_steps:
        return max_lr * (step / warmup_steps)
    if step >= total_steps:
        return min_lr
    # Cosine decay phase
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine_decay


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def compute_iteration_weights(T: int) -> List[float]:
    """Compute per-iteration loss weights: w_l = l / sum(1..T)."""
    total = sum(range(1, T + 1))
    return [ell / total for ell in range(1, T + 1)]


def compute_exit_distribution(exit_probs: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute the categorical exit distribution from per-iteration exit probs.

    exit_probs: list of T tensors, each shape (batch,) with values in [0, 1].
    Returns: tensor of shape (batch, T) — probability of exiting at each iteration.

    P(exit at l) = p_exit_l * prod_{k < l} (1 - p_exit_k)
    """
    T = len(exit_probs)
    batch_size = exit_probs[0].shape[0]
    device = exit_probs[0].device

    exit_dist = torch.zeros(batch_size, T, device=device)
    survival = torch.ones(batch_size, device=device)

    for ell in range(T):
        p = exit_probs[ell]
        exit_dist[:, ell] = survival * p
        survival = survival * (1.0 - p)

    # Any remaining probability mass (didn't exit) — attribute to last iteration
    # This makes it a valid distribution that sums to 1.
    exit_dist[:, -1] = exit_dist[:, -1] + survival

    return exit_dist


def compute_gate_entropy_loss(
    exit_probs: List[torch.Tensor],
    lam: float = DEFAULT_GATE_LAMBDA,
) -> torch.Tensor:
    """
    Entropy bonus on the exit distribution.
    L_gate = -λ * H(exit_distribution)

    We *minimise* L_gate, so negative entropy = encourages higher entropy.
    """
    exit_dist = compute_exit_distribution(exit_probs)
    # Clamp for numerical stability in log
    exit_dist = exit_dist.clamp(min=1e-8)
    # H = -sum p log p  (per example, then mean over batch)
    entropy = -(exit_dist * exit_dist.log()).sum(dim=-1)  # (batch,)
    # We want to *maximise* entropy, so loss = -λ * H
    return -lam * entropy.mean()


def compute_loss(
    outputs: Dict[str, list],
    labels: torch.Tensor,
    variant: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the full training loss.

    Returns (total_loss, task_loss) — task_loss is for logging/diagnostics.
    """
    logits_list: List[torch.Tensor] = outputs["logits"]
    is_looped = variant in LOOPED_VARIANTS

    if not is_looped:
        # Single output — standard cross-entropy
        assert len(logits_list) == 1, f"Expected 1 logits tensor for {variant}, got {len(logits_list)}"
        task_loss = F.cross_entropy(logits_list[0], labels)
        return task_loss, task_loss

    # Looped: weighted multi-iteration loss
    T = len(logits_list)
    weights = compute_iteration_weights(T)

    task_loss = torch.tensor(0.0, device=labels.device)
    for ell, (logits, w) in enumerate(zip(logits_list, weights)):
        task_loss = task_loss + w * F.cross_entropy(logits, labels)

    # Exit gate regularisation
    exit_probs: List[torch.Tensor] = outputs["exit_probs"]
    gate_loss = compute_gate_entropy_loss(exit_probs)

    total_loss = task_loss + gate_loss
    return total_loss, task_loss


# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute classification accuracy from logits. Returns float in [0, 1]."""
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_validation(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    variant: str,
) -> Dict[str, float]:
    """
    Run full validation pass.
    Returns dict with 'val_loss', 'val_acc', and per-depth accuracy.
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    # Per-depth tracking
    depth_correct: Dict[int, int] = {}
    depth_count: Dict[int, int] = {}

    use_amp = device.type == "cuda"

    for batch in val_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        depths = batch[3]  # keep on CPU for bookkeeping

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(input_ids, attention_mask)

        logits_list = outputs["logits"]
        # Use the last iteration's logits for evaluation
        logits = logits_list[-1]

        loss = F.cross_entropy(logits, labels, reduction="sum")
        preds = logits.argmax(dim=-1)
        correct = (preds == labels)

        total_loss += loss.item()
        total_correct += correct.sum().item()
        total_count += labels.shape[0]

        # Per-depth
        for i in range(labels.shape[0]):
            d = depths[i].item()
            depth_correct[d] = depth_correct.get(d, 0) + correct[i].item()
            depth_count[d] = depth_count.get(d, 0) + 1

    avg_loss = total_loss / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)

    result = {
        "val_loss": avg_loss,
        "val_acc": avg_acc,
    }
    for d in sorted(depth_correct.keys()):
        result[f"val_acc_depth_{d}"] = depth_correct[d] / depth_count[d]

    model.train()
    return result


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val_acc: float,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "best_val_acc": best_val_acc,
            "seed": seed,
        },
        path,
    )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class JSONLLogger:
    """Append-only JSONL logger."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        # Truncate on creation (new run)
        self.path.write_text("")

    def log(self, record: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    variant: str,
    seed: int,
    total_steps: int = DEFAULT_TOTAL_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device_str: str = "auto",
) -> None:
    assert variant in VARIANTS, f"Unknown variant: {variant}. Choose from {VARIANTS}"

    # ----- Device -----
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[train] variant={variant}, seed={seed}, device={device}, steps={total_steps}")

    # ----- Seeding -----
    seed_everything(seed)

    # ----- Paths -----
    run_dir = CHECKPOINT_DIR / f"{variant}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = JSONLLogger(run_dir / "log.jsonl")

    # ----- Data -----
    train_dataset = BracketDataset(DATA_DIR / "train.jsonl")
    val_dataset = BracketDataset(DATA_DIR / "val.jsonl")

    # We create a generator seeded for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        generator=g,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    # ----- Model -----
    model = BracketTransformer(variant=variant)
    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Parameters: {param_count:,} total, {trainable_count:,} trainable")

    # ----- Optimizer -----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=DEFAULT_LR,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        betas=DEFAULT_BETAS,
    )

    # ----- AMP scaler -----
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ----- Training state -----
    step = 0
    best_val_acc = 0.0
    model.train()

    print(f"[train] Training dataset: {len(train_dataset)} examples")
    print(f"[train] Validation dataset: {len(val_dataset)} examples")
    print(f"[train] Batches per epoch: {len(train_loader)}")
    print(f"[train] Starting training for {total_steps} steps...")
    t_start = time.time()

    while step < total_steps:
        for batch in train_loader:
            if step >= total_steps:
                break

            input_ids = batch[0].to(device, non_blocking=True)
            attention_mask = batch[1].to(device, non_blocking=True)
            labels = batch[2].to(device, non_blocking=True)
            # depths (batch[3]) not needed for training loss

            # Update learning rate
            lr = get_lr(step, total_steps=total_steps)
            set_lr(optimizer, lr)

            # Forward
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(input_ids, attention_mask)
                total_loss, task_loss = compute_loss(outputs, labels, variant)

            # Backward
            scaler.scale(total_loss).backward()
            # Gradient clipping (unscale first for correct norm computation)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), DEFAULT_MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()

            step += 1

            # ----- Logging (every LOG_EVERY steps) -----
            if step % LOG_EVERY == 0:
                # Compute batch accuracy from last iteration's logits
                with torch.no_grad():
                    logits = outputs["logits"][-1]
                    train_acc = compute_accuracy(logits, labels)

                elapsed = time.time() - t_start
                steps_per_sec = step / elapsed if elapsed > 0 else 0.0

                log_record = {
                    "step": step,
                    "train_loss": round(task_loss.item(), 5),
                    "train_acc": round(train_acc, 4),
                    "lr": round(lr, 8),
                    "elapsed_s": round(elapsed, 1),
                    "steps_per_sec": round(steps_per_sec, 2),
                }
                if variant in LOOPED_VARIANTS:
                    log_record["total_loss"] = round(total_loss.item(), 5)

                print(
                    f"  step {step:>6d}/{total_steps} | "
                    f"loss {task_loss.item():.4f} | "
                    f"acc {train_acc:.3f} | "
                    f"lr {lr:.2e} | "
                    f"{steps_per_sec:.1f} steps/s"
                )
                logger.log(log_record)

            # ----- Validation (every VAL_EVERY steps) -----
            if step % VAL_EVERY == 0:
                val_metrics = run_validation(model, val_loader, device, variant)
                val_acc = val_metrics["val_acc"]
                val_loss = val_metrics["val_loss"]

                print(
                    f"  [val] step {step:>6d} | "
                    f"val_loss {val_loss:.4f} | "
                    f"val_acc {val_acc:.4f} | "
                    f"best {best_val_acc:.4f}"
                )

                val_record = {
                    "step": step,
                    "type": "validation",
                    **{k: round(v, 5) for k, v in val_metrics.items()},
                }
                logger.log(val_record)

                # Save best
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    save_checkpoint(
                        run_dir / "best.pt",
                        model, optimizer, step, best_val_acc, seed,
                    )
                    print(f"  [val] New best! val_acc={val_acc:.4f} — saved best.pt")

            # ----- Checkpointing (every CHECKPOINT_EVERY steps) -----
            if step % CHECKPOINT_EVERY == 0:
                ckpt_path = run_dir / f"step_{step}.pt"
                save_checkpoint(ckpt_path, model, optimizer, step, best_val_acc, seed)
                print(f"  [ckpt] Saved {ckpt_path}")

    # ----- Final -----
    elapsed = time.time() - t_start
    print(f"\n[train] Done. {step} steps in {elapsed:.1f}s ({step / elapsed:.1f} steps/s)")
    print(f"[train] Best validation accuracy: {best_val_acc:.4f}")

    # Save final checkpoint
    save_checkpoint(
        run_dir / f"step_{step}.pt",
        model, optimizer, step, best_val_acc, seed,
    )
    print(f"[train] Final checkpoint saved.")

    # Final validation
    final_metrics = run_validation(model, val_loader, device, variant)
    print(f"[train] Final val_acc: {final_metrics['val_acc']:.4f}")
    logger.log({"step": step, "type": "final", **{k: round(v, 5) for k, v in final_metrics.items()}})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a BracketTransformer variant.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=VARIANTS,
        help="Model variant to train.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for this run.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_TOTAL_STEPS,
        help="Total training steps.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Training batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cuda', 'cpu'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        variant=args.variant,
        seed=args.seed,
        total_steps=args.steps,
        batch_size=args.batch_size,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()