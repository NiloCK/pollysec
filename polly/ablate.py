"""
polly.ablate — Ablation experiments on a trained V4 (looped_reg) model.

Intervenes on the register vector at test time to assess its causal role
in the model's computation across loop iterations.

Three interventions:
  1. Register zeroing:  r = 0 after each iteration (no carry-over).
  2. Register freezing: r frozen to its iteration-1 value for all subsequent iters.
  3. Register noise:    Gaussian noise added to r between iterations.

Usage:
    python -m polly.ablate --seed 100 --device cpu
    python -m polly.ablate --seed 100 --device auto --noise-sigmas 0.01,0.1,1.0
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from polly.data import BracketDataset
from polly.model import BracketTransformer, REG_DIM, DIM, NUM_CLASSES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VARIANT = "looped_reg"
EVAL_BATCH_SIZE = 256
DEFAULT_T_MAX = 4
DEFAULT_NOISE_SIGMAS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]

DATA_DIR = Path(__file__).resolve().parent / "data"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"


# ---------------------------------------------------------------------------
# Device / checkpoint helpers (mirrors evaluate.py conventions)
# ---------------------------------------------------------------------------

def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_model(seed: int, device: torch.device) -> BracketTransformer:
    """Load a trained looped_reg model from its best checkpoint."""
    ckpt_path = CHECKPOINT_DIR / f"{VARIANT}_seed{seed}" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"  Have you trained {VARIANT} with seed {seed}?"
        )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = BracketTransformer(variant=VARIANT)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def build_combined_loader(device: torch.device) -> DataLoader:
    """Combine test_id and test_ood into a single DataLoader."""
    id_path = DATA_DIR / "test_id.jsonl"
    ood_path = DATA_DIR / "test_ood.jsonl"

    for p in (id_path, ood_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Test data not found at {p}\n"
                f"  Run `python -m polly.data` first to generate splits."
            )

    combined = ConcatDataset([BracketDataset(id_path), BracketDataset(ood_path)])
    return DataLoader(
        combined,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )


# ---------------------------------------------------------------------------
# Ablated forward pass
# ---------------------------------------------------------------------------

def ablated_forward(
    model: BracketTransformer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    ablation_type: str,
    noise_sigma: Optional[float] = None,
    t_max: int = DEFAULT_T_MAX,
) -> torch.Tensor:
    """Run the looped_reg forward pass with an ablation applied to the register.

    This manually reimplements the v5 forward loop (encoder → interpreter×T →
    decoder) so we can intervene on `r` between interpreter iterations.

    Args:
        model:          A trained BracketTransformer(variant="looped_reg").
        input_ids:      (B, S) token ids.
        attention_mask:  (B, S) 1/0 mask.
        ablation_type:  One of "none", "zero", "freeze", "noise".
        noise_sigma:    Standard deviation of Gaussian noise (only for "noise").
        t_max:          Number of loop iterations.

    Returns:
        logits from the final iteration — (B, NUM_CLASSES).
    """
    assert model.variant == VARIANT, f"Expected {VARIANT}, got {model.variant}"

    B = input_ids.size(0)
    device = input_ids.device

    # Embedding (same as model._embed)
    h = model._embed(input_ids)  # (B, S, D)

    # --- Encoder (single pass, no register, no ablation) ---
    for layer in model.encoder_layers:
        h = layer(h, attention_mask=attention_mask)

    # Initialise register
    r = torch.zeros(B, REG_DIM, device=device)

    r_frozen: Optional[torch.Tensor] = None  # saved after iter 1 for "freeze"

    # --- Interpreter (looped T times, register fires here) ---
    for t in range(t_max):
        for layer in model.interpreter_layers:
            h = model.register_mech.inject(h, r)
            h = layer(h, attention_mask=attention_mask)
            h_cls = h[:, 0, :]
            r = model.register_mech.update(h_cls, r)

        # --- Apply ablation BETWEEN iterations (before next iter uses r) ---
        if t < t_max - 1:  # no need to ablate after the last iteration
            if ablation_type == "none":
                pass  # baseline — no intervention
            elif ablation_type == "zero":
                # Wipe the register: next iteration starts with r=0
                r = torch.zeros_like(r)
            elif ablation_type == "freeze":
                if t == 0:
                    # Save r after the first iteration
                    r_frozen = r.clone().detach()
                else:
                    # Restore to the frozen value
                    assert r_frozen is not None
                    r = r_frozen.clone()
            elif ablation_type == "noise":
                assert noise_sigma is not None and noise_sigma > 0
                r = r + torch.randn_like(r) * noise_sigma
            else:
                raise ValueError(f"Unknown ablation_type: {ablation_type!r}")

    # --- Decoder (single pass on final interpreter output) ---
    h_dec = h
    for layer in model.decoder_layers:
        h_dec = layer(h_dec, attention_mask=attention_mask)

    final_logits = model._compute_logits(h_dec)  # (B, NUM_CLASSES)
    return final_logits


# ---------------------------------------------------------------------------
# Evaluation with ablation
# ---------------------------------------------------------------------------

def evaluate_ablation(
    model: BracketTransformer,
    loader: DataLoader,
    device: torch.device,
    ablation_type: str,
    noise_sigma: Optional[float] = None,
) -> Dict[int, Dict[str, Any]]:
    """Evaluate the model under a specific ablation, returning per-depth stats.

    Returns:
        {depth: {"correct": int, "total": int, "accuracy": float}}
    """
    depth_correct: Dict[int, int] = defaultdict(int)
    depth_total: Dict[int, int] = defaultdict(int)

    with torch.no_grad():
        for batch in loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            depths = batch[3]  # CPU

            logits = ablated_forward(
                model, input_ids, attention_mask,
                ablation_type=ablation_type,
                noise_sigma=noise_sigma,
            )
            preds = logits.argmax(dim=-1)
            correct = (preds == labels)

            for i in range(labels.size(0)):
                d = depths[i].item()
                depth_total[d] += 1
                depth_correct[d] += int(correct[i].item())

    results: Dict[int, Dict[str, Any]] = {}
    for d in sorted(depth_total.keys()):
        total = depth_total[d]
        corr = depth_correct[d]
        results[d] = {
            "correct": corr,
            "total": total,
            "accuracy": corr / max(total, 1),
        }
    return results


def compute_overall_accuracy(per_depth: Dict[int, Dict[str, Any]]) -> float:
    """Compute overall accuracy from per-depth results."""
    total_correct = sum(v["correct"] for v in per_depth.values())
    total_samples = sum(v["total"] for v in per_depth.values())
    return total_correct / max(total_samples, 1)


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_comparison_table(
    all_results: Dict[str, Dict[int, Dict[str, Any]]],
    title: str = "Ablation Results",
) -> None:
    """Print a table comparing accuracy across ablation conditions by depth."""
    conditions = list(all_results.keys())
    all_depths = sorted(
        set(d for res in all_results.values() for d in res.keys())
    )

    # Header
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    header = f"  {'Depth':>5}"
    for cond in conditions:
        # Truncate long condition names
        label = cond[:12]
        header += f"  {label:>12}"
    print(header)
    print(f"  {'-' * 5}" + f"  {'-' * 12}" * len(conditions))

    for d in all_depths:
        row = f"  {d:>5}"
        for cond in conditions:
            if d in all_results[cond]:
                acc = all_results[cond][d]["accuracy"]
                row += f"  {acc:>11.1%} "
            else:
                row += f"  {'—':>12}"
        print(row)

    # Overall
    print(f"  {'-' * 5}" + f"  {'-' * 12}" * len(conditions))
    row = f"  {'ALL':>5}"
    for cond in conditions:
        overall = compute_overall_accuracy(all_results[cond])
        row += f"  {overall:>11.1%} "
    print(row)

    # ID / OOD subtotals
    for label, depth_range in [("ID 1-8", range(1, 9)), ("OOD 9-16", range(9, 17))]:
        row = f"  {label:>5}"
        for cond in conditions:
            sub_correct = sum(
                all_results[cond].get(d, {}).get("correct", 0) for d in depth_range
            )
            sub_total = sum(
                all_results[cond].get(d, {}).get("total", 0) for d in depth_range
            )
            if sub_total > 0:
                row += f"  {sub_correct / sub_total:>11.1%} "
            else:
                row += f"  {'—':>12}"
        print(row)

    print()


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def save_results(
    all_results: Dict[str, Dict[int, Dict[str, Any]]],
    seed: int,
) -> Path:
    """Save ablation results to JSON."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "ablation_results.json"

    # Build a serialisable structure
    output: Dict[str, Any] = {
        "seed": seed,
        "variant": VARIANT,
        "t_max": DEFAULT_T_MAX,
        "conditions": {},
    }

    for cond_name, per_depth in all_results.items():
        cond_data: Dict[str, Any] = {
            "overall_accuracy": compute_overall_accuracy(per_depth),
            "by_depth": {},
        }
        # ID / OOD subtotals
        for sub_label, depth_range in [("id_1_8", range(1, 9)), ("ood_9_16", range(9, 17))]:
            sub_correct = sum(per_depth.get(d, {}).get("correct", 0) for d in depth_range)
            sub_total = sum(per_depth.get(d, {}).get("total", 0) for d in depth_range)
            cond_data[f"{sub_label}_accuracy"] = sub_correct / max(sub_total, 1)

        for d, stats in sorted(per_depth.items()):
            cond_data["by_depth"][str(d)] = {
                "accuracy": stats["accuracy"],
                "correct": stats["correct"],
                "total": stats["total"],
            }
        output["conditions"][cond_name] = cond_data

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    return out_path


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run_all_ablations(
    model: BracketTransformer,
    loader: DataLoader,
    device: torch.device,
    noise_sigmas: List[float],
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Run baseline + all ablation conditions."""

    all_results: Dict[str, Dict[int, Dict[str, Any]]] = {}

    # 1. Baseline (no ablation)
    print("[ablate] Running baseline (no intervention) ...")
    all_results["baseline"] = evaluate_ablation(
        model, loader, device, ablation_type="none"
    )
    baseline_acc = compute_overall_accuracy(all_results["baseline"])
    print(f"         Overall accuracy: {baseline_acc:.1%}")

    # 2. Register zeroing
    print("[ablate] Running register zeroing ...")
    all_results["zero"] = evaluate_ablation(
        model, loader, device, ablation_type="zero"
    )
    zero_acc = compute_overall_accuracy(all_results["zero"])
    print(f"         Overall accuracy: {zero_acc:.1%}  (Δ = {zero_acc - baseline_acc:+.1%})")

    # 3. Register freezing
    print("[ablate] Running register freezing (freeze after iter 1) ...")
    all_results["freeze"] = evaluate_ablation(
        model, loader, device, ablation_type="freeze"
    )
    freeze_acc = compute_overall_accuracy(all_results["freeze"])
    print(f"         Overall accuracy: {freeze_acc:.1%}  (Δ = {freeze_acc - baseline_acc:+.1%})")

    # 4. Register noise at various sigmas
    for sigma in noise_sigmas:
        label = f"noise_σ={sigma}"
        print(f"[ablate] Running register noise (σ={sigma}) ...")
        all_results[label] = evaluate_ablation(
            model, loader, device, ablation_type="noise", noise_sigma=sigma,
        )
        noise_acc = compute_overall_accuracy(all_results[label])
        print(f"         Overall accuracy: {noise_acc:.1%}  (Δ = {noise_acc - baseline_acc:+.1%})")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation experiments on trained V4 (looped_reg) model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--seed", type=int, default=100,
        help="Training seed to load checkpoint for (default: 100).",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device: 'cpu', 'cuda', or 'auto' (default: cpu).",
    )
    parser.add_argument(
        "--noise-sigmas", type=str, default=None,
        help=(
            "Comma-separated noise sigma values for register noise ablation. "
            f"Default: {','.join(str(s) for s in DEFAULT_NOISE_SIGMAS)}"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if args.noise_sigmas is not None:
        noise_sigmas = [float(s.strip()) for s in args.noise_sigmas.split(",")]
    else:
        noise_sigmas = DEFAULT_NOISE_SIGMAS

    print(f"[ablate] Device: {device}")
    print(f"[ablate] Seed: {args.seed}")
    print(f"[ablate] Noise sigmas: {noise_sigmas}")

    # Load model
    print(f"[ablate] Loading {VARIANT} seed={args.seed} ...")
    model = load_model(args.seed, device)
    print(f"         Parameters: {model.count_parameters():,}")

    # Load data
    loader = build_combined_loader(device)
    print(f"         Test samples: {len(loader.dataset):,} (test_id + test_ood)")

    # Run all ablations
    all_results = run_all_ablations(model, loader, device, noise_sigmas)

    # Print comparison tables
    # Table 1: structural ablations (baseline, zero, freeze)
    structural = {k: v for k, v in all_results.items() if k in ("baseline", "zero", "freeze")}
    print_comparison_table(structural, title="Structural Ablations (baseline vs zero vs freeze)")

    # Table 2: noise ablations (baseline + all noise levels)
    noise_keys = ["baseline"] + [k for k in all_results if k.startswith("noise_")]
    noise_results = {k: all_results[k] for k in noise_keys}
    print_comparison_table(noise_results, title="Noise Ablations (baseline vs noise σ)")

    # Save
    out_path = save_results(all_results, seed=args.seed)
    print(f"[ablate] Results saved to {out_path}")
    print("[ablate] Done ✓")


if __name__ == "__main__":
    main()