"""
polly.train — Training loop for ListOps transformer experiments.

Usage:
    python -m polly.train --variant vanilla --seed 100
    python -m polly.train --variant looped_reg --seed 200 --steps 30000 --batch-size 128 --device auto
    python -m polly.train --variant looped --seed 300 --loss-mode uniform
    python -m polly.train --variant looped_reg --seed 400 --lambda-p 0.05 --beta-max 0.01
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from polly.data import ListOpsDataset
from polly.model import PollyTransformer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VARIANTS = ("vanilla", "looped", "looped_reg")
LOOPED_VARIANTS = ("looped", "looped_reg")
LOSS_MODES = ("ponder", "uniform")

DEFAULT_LR = 3e-4
DEFAULT_MIN_LR = 1e-5
DEFAULT_WARMUP_STEPS = 1_000
DEFAULT_TOTAL_STEPS = 30_000
DEFAULT_BATCH_SIZE = 128
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_BETAS = (0.9, 0.999)
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_GATE_LAMBDA = 0.01          # (legacy; unused under PonderNet loss)

# PonderNet loss parameters (Phase B.2)
DEFAULT_PONDER_LAMBDA_P = 0.05      # geometric prior param — was 0.3 in v3
                                     # E[t]≈20 untruncated, ~85% on t=4 when T=4
DEFAULT_PONDER_BETA_MAX = 0.01      # final β value
DEFAULT_PONDER_BETA_WARMUP = 3_000  # steps of β=0 before linear ramp
DEFAULT_PONDER_BETA_RAMP = 2_000    # steps to linearly ramp β from 0 to BETA_MAX

CHECKPOINT_EVERY = int(os.environ.get("POLLY_CHECKPOINT_EVERY", 2_000))
LOG_EVERY = int(os.environ.get("POLLY_LOG_EVERY", 100))
VAL_EVERY = int(os.environ.get("POLLY_VAL_EVERY", 1_000))

DATA_DIR = Path(os.environ.get("POLLY_DATA_DIR", Path(__file__).resolve().parent / "data"))
CHECKPOINT_DIR = Path(os.environ.get("POLLY_CHECKPOINT_DIR", Path(__file__).resolve().parent / "checkpoints"))


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
# PonderNet β schedule (Phase B.2a)
# ---------------------------------------------------------------------------

def get_ponder_beta(
    step: int,
    beta_max: float = DEFAULT_PONDER_BETA_MAX,
    beta_warmup: int = DEFAULT_PONDER_BETA_WARMUP,
    beta_ramp: int = DEFAULT_PONDER_BETA_RAMP,
) -> float:
    """Compute effective β for KL regularisation as a function of training step.

    Schedule: β=0 for the first `beta_warmup` steps, then linearly ramp
    from 0 to `beta_max` over the next `beta_ramp` steps, then hold at
    `beta_max` thereafter.  This lets the model learn the task before KL
    starts shaping the exit distribution.
    """
    if step < beta_warmup:
        return 0.0
    ramp_step = step - beta_warmup
    if ramp_step < beta_ramp:
        return beta_max * (ramp_step / beta_ramp)
    return beta_max


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


def geometric_prior(lambda_p: float, T: int, device: torch.device) -> torch.Tensor:
    """Truncated geometric prior over iterations 1..T.

    p(t) ∝ (1 - λ_p)^(t-1) · λ_p, renormalised so Σ_t p(t) = 1.
    E[t] ≈ 1/λ_p (exact in the untruncated limit). Used by PonderNet-style
    KL regularisation to set the target average compute budget.
    """
    t = torch.arange(T, device=device, dtype=torch.float32)
    p = ((1.0 - lambda_p) ** t) * lambda_p
    return p / p.sum()


def compute_loss(
    outputs: Dict[str, list],
    labels: torch.Tensor,
    variant: str,
    step: int,
    total_steps: int,
    loss_mode: str = "ponder",
    lambda_p: float = DEFAULT_PONDER_LAMBDA_P,
    beta_max: float = DEFAULT_PONDER_BETA_MAX,
    beta_warmup: int = DEFAULT_PONDER_BETA_WARMUP,
    beta_ramp: int = DEFAULT_PONDER_BETA_RAMP,
) -> Dict[str, Any]:
    """
    Compute the full training loss.

    Returns a dict with keys:
        total_loss      — scalar tensor (the value to .backward())
        task_loss       — scalar tensor (CE component, for logging)
        kl_loss         — scalar tensor or 0.0
        per_iter_ce     — list of T floats (mean CE at each iteration), or []
        exit_dist_mean  — list of T floats (exit dist averaged over batch), or []
    """
    logits_list: List[torch.Tensor] = outputs["logits"]
    is_looped = variant in LOOPED_VARIANTS

    if not is_looped:
        # Single output — standard cross-entropy
        assert len(logits_list) == 1, (
            f"Expected 1 logits tensor for {variant}, got {len(logits_list)}"
        )
        task_loss = F.cross_entropy(logits_list[0], labels)
        return {
            "total_loss": task_loss,
            "task_loss": task_loss,
            "kl_loss": 0.0,
            "per_iter_ce": [],
            "exit_dist_mean": [],
        }

    # ---- Looped variants ----
    T = len(logits_list)
    device = labels.device

    # (B, T) stack of per-iteration CE losses
    ce_per_iter = torch.stack(
        [F.cross_entropy(logits, labels, reduction="none") for logits in logits_list],
        dim=1,
    )  # (B, T)

    # Per-iteration CE for logging: mean over batch for each iteration
    per_iter_ce = [ce_per_iter[:, t].mean().item() for t in range(T)]

    # Exit distribution (used by ponder mode, and for logging in all modes)
    exit_probs: List[torch.Tensor] = outputs["exit_probs"]
    exit_dist = compute_exit_distribution(exit_probs)  # (B, T), rows sum to 1

    # Exit distribution mean over batch for logging
    exit_dist_mean = exit_dist.detach().mean(dim=0).tolist()  # list of T floats

    if loss_mode == "uniform":
        # Universal Transformers recipe: simple mean CE across iterations.
        # Exit gate is still computed for eval-time use, but not trained via loss.
        task_loss = ce_per_iter.mean()  # mean over both batch and iterations
        return {
            "total_loss": task_loss,
            "task_loss": task_loss,
            "kl_loss": 0.0,
            "per_iter_ce": per_iter_ce,
            "exit_dist_mean": exit_dist_mean,
        }

    # ---- PonderNet-style objective ----
    #   task_loss = E_{t ~ exit_dist}[ CE(logits_t, y) ]
    #   kl_loss   = KL(exit_dist || Geometric(λ_p))
    #   total     = task_loss + β · kl_loss
    task_loss = (exit_dist * ce_per_iter).sum(dim=1).mean()

    prior = geometric_prior(lambda_p, T, device)        # (T,)
    prior = prior.unsqueeze(0).expand_as(exit_dist)     # (B, T)
    kl_loss = F.kl_div(
        exit_dist.clamp(min=1e-8).log(),
        prior,
        reduction="batchmean",
    )

    beta = get_ponder_beta(step, beta_max, beta_warmup, beta_ramp)
    total_loss = task_loss + beta * kl_loss

    return {
        "total_loss": total_loss,
        "task_loss": task_loss,
        "kl_loss": kl_loss,
        "per_iter_ce": per_iter_ce,
        "exit_dist_mean": exit_dist_mean,
    }


# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute classification accuracy from logits. Returns float in [0, 1]."""
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


# ---------------------------------------------------------------------------
# Gradient norm instrumentation (Phase B.1d)
# ---------------------------------------------------------------------------

def compute_grad_norms(model: nn.Module) -> Tuple[float, List[float]]:
    """Compute total gradient norm and per-layer norms.

    Aggregates parameter gradients into 6 "layer" buckets based on
    the model's `layers.0`, `layers.1`, ..., `layers.5` naming convention.
    Parameters not belonging to any numbered layer (embeddings, output head,
    exit gate, register mech) are reported in total but not in per-layer.

    Returns:
        total_norm:     float — global L2 norm across all parameters
        per_layer_norms: list of 6 floats — L2 norm for each transformer layer
    """
    total_sq = 0.0
    # Accumulate squared norms per layer index
    layer_sq: Dict[int, float] = defaultdict(float)

    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g_sq = p.grad.norm().item() ** 2
        total_sq += g_sq

        # Match "layers.N." to bucket into layer index N
        m = re.match(r"layers\.(\d+)\.", name)
        if m:
            layer_idx = int(m.group(1))
            layer_sq[layer_idx] += g_sq

    total_norm = math.sqrt(total_sq)

    # Build per-layer list (always 6 entries, even if some have no params)
    n_layers = 6
    per_layer_norms = [math.sqrt(layer_sq.get(i, 0.0)) for i in range(n_layers)]

    return total_norm, per_layer_norms


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

    # See training-loop comment: fp16 overflows for looped variants.
    use_amp = device.type == "cuda" and variant == "vanilla"

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
    total_steps: int,
    loss_mode: str = "ponder",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "best_val_acc": best_val_acc,
            "seed": seed,
            "total_steps": total_steps,
            "loss_mode": loss_mode,
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
    loss_mode: str = "ponder",
    lambda_p: float = DEFAULT_PONDER_LAMBDA_P,
    beta_max: float = DEFAULT_PONDER_BETA_MAX,
    beta_warmup: int = DEFAULT_PONDER_BETA_WARMUP,
    beta_ramp: int = DEFAULT_PONDER_BETA_RAMP,
) -> None:
    assert variant in VARIANTS, f"Unknown variant: {variant}. Choose from {VARIANTS}"
    assert loss_mode in LOSS_MODES, f"Unknown loss_mode: {loss_mode}. Choose from {LOSS_MODES}"

    # ----- Device -----
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[train] variant={variant}, seed={seed}, device={device}, steps={total_steps}")
    print(f"[train] loss_mode={loss_mode}, lambda_p={lambda_p}, beta_max={beta_max}, "
          f"beta_warmup={beta_warmup}, beta_ramp={beta_ramp}")

    # ----- Seeding -----
    seed_everything(seed)

    # ----- Paths -----
    run_dir = CHECKPOINT_DIR / f"{variant}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = JSONLLogger(run_dir / "log.jsonl")

    # ----- Data -----
    train_dataset = ListOpsDataset(DATA_DIR / "train.jsonl")
    val_dataset = ListOpsDataset(DATA_DIR / "val.jsonl")

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
    model = PollyTransformer(variant=variant)
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
    # fp16 autocast overflows for looped variants: 24 layer-applications
    # (4 iters × 6 layers) with per-layer register injection blow past fp16's
    # ~65k dynamic range, producing NaN logits. Gate amp to vanilla only until
    # we either bf16 (needs newer GPU than P100) or norm the register path.
    use_amp = device.type == "cuda" and variant == "vanilla"
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
                loss_dict = compute_loss(
                    outputs, labels, variant,
                    step=step,
                    total_steps=total_steps,
                    loss_mode=loss_mode,
                    lambda_p=lambda_p,
                    beta_max=beta_max,
                    beta_warmup=beta_warmup,
                    beta_ramp=beta_ramp,
                )
                total_loss = loss_dict["total_loss"]
                task_loss = loss_dict["task_loss"]

            # Backward
            scaler.scale(total_loss).backward()

            # Gradient clipping (unscale first for correct norm computation)
            scaler.unscale_(optimizer)

            # --- Gradient norm instrumentation (Phase B.1d) ---
            # Computed after unscale_ but before clip_grad_norm_ so we see
            # the true gradient magnitudes.  Only every LOG_EVERY steps to
            # avoid overhead.
            grad_norm_total = None
            grad_norm_per_layer = None
            if (step + 1) % LOG_EVERY == 0 and variant in LOOPED_VARIANTS:
                grad_norm_total, grad_norm_per_layer = compute_grad_norms(model)

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

                log_record: Dict[str, Any] = {
                    "step": step,
                    "train_loss": round(task_loss.item(), 5),
                    "train_acc": round(train_acc, 4),
                    "lr": round(lr, 8),
                    "elapsed_s": round(elapsed, 1),
                    "steps_per_sec": round(steps_per_sec, 2),
                }

                if variant in LOOPED_VARIANTS:
                    # Total loss (includes KL for ponder mode)
                    log_record["total_loss"] = round(total_loss.item(), 5)

                    # Per-component losses (Phase B.1a)
                    kl_val = loss_dict["kl_loss"]
                    log_record["task_loss"] = round(task_loss.item(), 5)
                    log_record["kl_loss"] = round(
                        kl_val.item() if isinstance(kl_val, torch.Tensor) else float(kl_val),
                        5,
                    )

                    # Effective beta at this step
                    log_record["beta"] = round(
                        get_ponder_beta(step, beta_max, beta_warmup, beta_ramp), 6
                    )
                    log_record["loss_mode"] = loss_mode

                    # Exit distribution (Phase B.1b)
                    exit_dist_mean = loss_dict["exit_dist_mean"]
                    if exit_dist_mean:
                        log_record["exit_dist_mean"] = [round(v, 4) for v in exit_dist_mean]

                    # Per-iteration CE (Phase B.1c)
                    per_iter_ce = loss_dict["per_iter_ce"]
                    if per_iter_ce:
                        log_record["per_iter_ce"] = [round(v, 4) for v in per_iter_ce]

                    # Gradient norms (Phase B.1d)
                    # Computed before step+=1 using (step+1) % LOG_EVERY == 0,
                    # so the values align with this post-increment logging block.
                    if grad_norm_total is not None and grad_norm_per_layer is not None:
                        log_record["grad_norm_total"] = round(grad_norm_total, 5)
                        log_record["grad_norm_per_layer"] = [
                            round(v, 5) for v in grad_norm_per_layer
                        ]

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

                val_record: Dict[str, Any] = {
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
                        model, optimizer, step, best_val_acc, seed, total_steps,
                        loss_mode=loss_mode,
                    )
                    print(f"  [val] New best! val_acc={val_acc:.4f} — saved best.pt")

            # ----- Checkpointing (every CHECKPOINT_EVERY steps) -----
            if step % CHECKPOINT_EVERY == 0:
                ckpt_path = run_dir / f"step_{step}.pt"
                save_checkpoint(
                    ckpt_path, model, optimizer, step, best_val_acc, seed, total_steps,
                    loss_mode=loss_mode,
                )
                print(f"  [ckpt] Saved {ckpt_path}")

    # ----- Final -----
    elapsed = time.time() - t_start
    print(f"\n[train] Done. {step} steps in {elapsed:.1f}s ({step / elapsed:.1f} steps/s)")
    print(f"[train] Best validation accuracy: {best_val_acc:.4f}")

    # Save final checkpoint
    save_checkpoint(
        run_dir / f"step_{step}.pt",
        model, optimizer, step, best_val_acc, seed, total_steps,
        loss_mode=loss_mode,
    )
    print("[train] Final checkpoint saved.")

    # Final validation
    final_metrics = run_validation(model, val_loader, device, variant)
    print(f"[train] Final val_acc: {final_metrics['val_acc']:.4f}")
    logger.log({"step": step, "type": "final", **{k: round(v, 5) for k, v in final_metrics.items()}})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PollyTransformer variant on ListOps.",
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
    # Phase B.2 loss configuration
    parser.add_argument(
        "--loss-mode",
        type=str,
        default="ponder",
        choices=LOSS_MODES,
        help="Loss mode for looped variants: 'ponder' (PonderNet-style) or 'uniform' (mean CE).",
    )
    parser.add_argument(
        "--lambda-p",
        type=float,
        default=DEFAULT_PONDER_LAMBDA_P,
        help="Geometric prior parameter λ_p for PonderNet KL.",
    )
    parser.add_argument(
        "--beta-max",
        type=float,
        default=DEFAULT_PONDER_BETA_MAX,
        help="Max KL weight β for PonderNet loss.",
    )
    parser.add_argument(
        "--beta-warmup",
        type=int,
        default=DEFAULT_PONDER_BETA_WARMUP,
        help="Steps of β=0 before linear ramp begins.",
    )
    parser.add_argument(
        "--beta-ramp",
        type=int,
        default=DEFAULT_PONDER_BETA_RAMP,
        help="Steps for β to linearly ramp from 0 to beta-max.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke-test mode: 25 steps, batch 8, log every 5, no val/ckpt. "
             "Overrides --steps/--batch-size/--device unless explicitly set.",
    )
    return parser.parse_args()


def main() -> None:
    global LOG_EVERY, VAL_EVERY, CHECKPOINT_EVERY
    args = parse_args()

    if args.smoke:
        # Tight cadence so a 25-step run is observable.
        LOG_EVERY = 5
        VAL_EVERY = 10**9
        CHECKPOINT_EVERY = 10**9
        # Only override user-settable defaults if they weren't explicitly set.
        if args.steps == DEFAULT_TOTAL_STEPS:
            args.steps = 25
        if args.batch_size == DEFAULT_BATCH_SIZE:
            args.batch_size = 8
        if args.device == "auto":
            args.device = "cpu"
        # Shorter β warmup so smoke exercises the KL code path if requested
        # via ponder mode (still zero during the 25-step window, but visible).
        if args.beta_warmup == DEFAULT_PONDER_BETA_WARMUP:
            args.beta_warmup = 10
        if args.beta_ramp == DEFAULT_PONDER_BETA_RAMP:
            args.beta_ramp = 10
        print(f"[smoke] steps={args.steps} batch={args.batch_size} "
              f"device={args.device} log_every={LOG_EVERY}")

    train(
        variant=args.variant,
        seed=args.seed,
        total_steps=args.steps,
        batch_size=args.batch_size,
        device_str=args.device,
        loss_mode=args.loss_mode,
        lambda_p=args.lambda_p,
        beta_max=args.beta_max,
        beta_warmup=args.beta_warmup,
        beta_ramp=args.beta_ramp,
    )


if __name__ == "__main__":
    main()