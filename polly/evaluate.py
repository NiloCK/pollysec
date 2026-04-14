"""
polly.evaluate — Evaluation script for bracket-matching transformer experiments.

Computes per-depth accuracy on in-distribution (depths 1-8) and out-of-distribution
(depths 9-16) test sets.  For looped variants, also tracks average exit iteration
per depth.

Usage:
    # Single run:
    python -m polly.evaluate --variant vanilla --seed 100 --device auto

    # Single run, forcing all iterations (no early exit) for looped variants:
    python -m polly.evaluate --variant looped --seed 100 --force-all-iters

    # All runs (4 variants × 3 seeds), with aggregated summary:
    python -m polly.evaluate --all --device auto
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from polly.data import BracketDataset
from polly.model import BracketTransformer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VARIANTS = ("vanilla", "vanilla_reg", "looped", "looped_reg")
LOOPED_VARIANTS = ("looped", "looped_reg")
SEEDS = (100, 200, 300)

EVAL_BATCH_SIZE = 256
DEFAULT_T_MAX = 4
DEFAULT_EXIT_THRESHOLD = 0.8

DATA_DIR = Path(os.environ.get("POLLY_DATA_DIR", Path(__file__).resolve().parent / "data"))
CHECKPOINT_DIR = Path(os.environ.get("POLLY_CHECKPOINT_DIR", Path(__file__).resolve().parent / "checkpoints"))
FIGURES_DIR = Path(os.environ.get("POLLY_FIGURES_DIR", Path(__file__).resolve().parent / "figures"))


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(
    variant: str,
    seed: int,
    device: torch.device,
) -> Tuple[BracketTransformer, Dict[str, Any]]:
    """Load a trained model from its best checkpoint.

    Returns (model, checkpoint_metadata).
    """
    ckpt_path = CHECKPOINT_DIR / f"{variant}_seed{seed}" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"  Have you trained {variant} with seed {seed}?"
        )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = BracketTransformer(variant=variant)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    meta = {
        "step": ckpt.get("step"),
        "best_val_acc": ckpt.get("best_val_acc"),
        "seed": ckpt.get("seed"),
    }
    return model, meta


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------

def evaluate_model(
    model: BracketTransformer,
    test_loader: DataLoader,
    device: torch.device,
    force_all_iters: bool = False,
) -> Dict[str, Dict[int, Any]]:
    """Run evaluation on a single DataLoader.

    Returns a dict with:
        "accuracy_by_depth" : {depth: float}
        "avg_exit_iter_by_depth" : {depth: float}   (only for looped variants)
    """
    model.eval()

    depth_correct: Dict[int, int] = defaultdict(int)
    depth_total: Dict[int, int] = defaultdict(int)
    # For looped variants: track exit iteration per sample
    depth_exit_iter_sum: Dict[int, float] = defaultdict(float)
    depth_exit_iter_count: Dict[int, int] = defaultdict(int)

    is_looped = model.is_looped

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            depths = batch[3]  # keep on CPU for bookkeeping

            if is_looped and force_all_iters:
                # Force all iterations — set threshold impossibly high
                outputs = model(
                    input_ids,
                    attention_mask,
                    t_max=DEFAULT_T_MAX,
                    exit_threshold=1.1,  # never triggers early exit
                )
            else:
                outputs = model(
                    input_ids,
                    attention_mask,
                    t_max=DEFAULT_T_MAX,
                    exit_threshold=DEFAULT_EXIT_THRESHOLD,
                )

            logits_list = outputs["logits"]
            exit_probs_list = outputs["exit_probs"]

            # Use the last iteration's logits for accuracy
            logits = logits_list[-1]
            preds = logits.argmax(dim=-1)
            correct = (preds == labels)

            B = labels.shape[0]

            # ------- per-sample exit iteration (looped only) -------
            # The model does batch-level early exit: it stops when ALL
            # samples want to exit.  So len(logits_list) tells us
            # when the batch stopped, but per-sample we can do better
            # by looking at which iteration each sample first crossed
            # the exit threshold.
            per_sample_exit_iter: Optional[torch.Tensor] = None
            if is_looped and exit_probs_list:
                # Shape: (T, B)
                exit_probs_stack = torch.stack(exit_probs_list, dim=0)  # (T, B)
                T = exit_probs_stack.shape[0]

                if not force_all_iters:
                    # Find per-sample first iteration where exit_prob > threshold
                    exceeded = exit_probs_stack > DEFAULT_EXIT_THRESHOLD  # (T, B)
                    # For each sample, find the first True along dim 0.
                    # If never exceeded, use the last iteration T.
                    # iters are 1-indexed (iteration 1 = first pass through layers)
                    per_sample_exit_iter = torch.full((B,), T, dtype=torch.float32)
                    for t_idx in range(T):
                        still_running = per_sample_exit_iter == T
                        newly_exiting = exceeded[t_idx] & still_running
                        per_sample_exit_iter[newly_exiting] = float(t_idx + 1)
                else:
                    # When forcing all iters, report the actual number of
                    # iterations run (all T) for every sample.
                    per_sample_exit_iter = torch.full((B,), float(T), dtype=torch.float32)

            # ------- accumulate per-depth stats -------
            for i in range(B):
                d = depths[i].item()
                depth_total[d] += 1
                depth_correct[d] += int(correct[i].item())

                if per_sample_exit_iter is not None:
                    depth_exit_iter_sum[d] += per_sample_exit_iter[i].item()
                    depth_exit_iter_count[d] += 1

    # ------- compute final metrics -------
    accuracy_by_depth: Dict[int, float] = {}
    for d in sorted(depth_total.keys()):
        accuracy_by_depth[d] = depth_correct[d] / max(depth_total[d], 1)

    avg_exit_iter_by_depth: Dict[int, float] = {}
    if is_looped:
        for d in sorted(depth_exit_iter_count.keys()):
            count = depth_exit_iter_count[d]
            if count > 0:
                avg_exit_iter_by_depth[d] = depth_exit_iter_sum[d] / count

    return {
        "accuracy_by_depth": accuracy_by_depth,
        "avg_exit_iter_by_depth": avg_exit_iter_by_depth,
    }


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def print_accuracy_table(
    variant: str,
    seed: int,
    results: Dict[str, Dict[int, Any]],
    meta: Dict[str, Any],
) -> None:
    """Print a nicely formatted table to stdout."""
    acc = results["accuracy_by_depth"]
    exit_iters = results.get("avg_exit_iter_by_depth", {})
    is_looped = variant in LOOPED_VARIANTS

    print()
    print("=" * 72)
    print(f"  {variant}  seed={seed}  |  "
          f"checkpoint step={meta.get('step', '?')}  "
          f"val_acc={meta.get('best_val_acc', '?')}")
    print("=" * 72)

    if is_looped:
        header = f"  {'Depth':>5}  {'Accuracy':>10}  {'Avg Exit Iter':>14}  {'Split':>6}"
        print(header)
        print(f"  {'-' * 5}  {'-' * 10}  {'-' * 14}  {'-' * 6}")
        for d in sorted(acc.keys()):
            split = "ID" if d <= 8 else "OOD"
            exit_str = f"{exit_iters[d]:.2f}" if d in exit_iters else "—"
            print(f"  {d:>5}  {acc[d]:>10.4f}  {exit_str:>14}  {split:>6}")
    else:
        header = f"  {'Depth':>5}  {'Accuracy':>10}  {'Split':>6}"
        print(header)
        print(f"  {'-' * 5}  {'-' * 10}  {'-' * 6}")
        for d in sorted(acc.keys()):
            split = "ID" if d <= 8 else "OOD"
            print(f"  {d:>5}  {acc[d]:>10.4f}  {split:>6}")

    # Overall averages
    id_depths = [d for d in acc if d <= 8]
    ood_depths = [d for d in acc if d > 8]
    if id_depths:
        id_avg = sum(acc[d] for d in id_depths) / len(id_depths)
        print(f"\n  ID  avg (depths 1–8):   {id_avg:.4f}")
    if ood_depths:
        ood_avg = sum(acc[d] for d in ood_depths) / len(ood_depths)
        print(f"  OOD avg (depths 9–16):  {ood_avg:.4f}")
    all_avg = sum(acc.values()) / max(len(acc), 1)
    print(f"  Overall avg:            {all_avg:.4f}")
    print()


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def save_run_results(
    variant: str,
    seed: int,
    results: Dict[str, Dict[int, Any]],
    force_all_iters: bool,
) -> Path:
    """Write per-run JSON to figures/eval_{variant}_seed{seed}.json."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    suffix = "_forced" if force_all_iters else ""
    out_path = FIGURES_DIR / f"eval_{variant}_seed{seed}{suffix}.json"

    payload: Dict[str, Any] = {
        "variant": variant,
        "seed": seed,
        "accuracy_by_depth": {
            str(d): round(v, 6)
            for d, v in sorted(results["accuracy_by_depth"].items())
        },
    }
    if results.get("avg_exit_iter_by_depth"):
        payload["avg_exit_iter_by_depth"] = {
            str(d): round(v, 4)
            for d, v in sorted(results["avg_exit_iter_by_depth"].items())
        }
    if force_all_iters:
        payload["force_all_iters"] = True

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(f"  → saved {out_path}")
    return out_path


def save_summary(all_run_results: Dict[Tuple[str, int], Dict[str, Dict[int, Any]]]) -> Path:
    """Aggregate across seeds and write eval_summary.json.

    For each variant, computes mean ± std of accuracy at each depth
    across all available seeds.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "eval_summary.json"

    # Group by variant
    variant_results: Dict[str, List[Dict[int, float]]] = defaultdict(list)
    variant_exit_results: Dict[str, List[Dict[int, float]]] = defaultdict(list)

    for (variant, seed), results in all_run_results.items():
        variant_results[variant].append(results["accuracy_by_depth"])
        if results.get("avg_exit_iter_by_depth"):
            variant_exit_results[variant].append(results["avg_exit_iter_by_depth"])

    summary: Dict[str, Any] = {}

    for variant in VARIANTS:
        acc_dicts = variant_results.get(variant, [])
        if not acc_dicts:
            continue

        # Collect all depths seen
        all_depths = sorted(set(d for ad in acc_dicts for d in ad))

        acc_stats: Dict[str, Dict[str, float]] = {}
        for d in all_depths:
            values = [ad[d] for ad in acc_dicts if d in ad]
            if not values:
                continue
            n = len(values)
            mean = sum(values) / n
            if n > 1:
                variance = sum((v - mean) ** 2 for v in values) / (n - 1)
                std = variance ** 0.5
            else:
                std = 0.0
            acc_stats[str(d)] = {
                "mean": round(mean, 6),
                "std": round(std, 6),
                "n": n,
            }

        entry: Dict[str, Any] = {
            "n_seeds": len(acc_dicts),
            "accuracy_by_depth": acc_stats,
        }

        # Exit iteration stats for looped variants
        exit_dicts = variant_exit_results.get(variant, [])
        if exit_dicts:
            exit_depths = sorted(set(d for ed in exit_dicts for d in ed))
            exit_stats: Dict[str, Dict[str, float]] = {}
            for d in exit_depths:
                values = [ed[d] for ed in exit_dicts if d in ed]
                if not values:
                    continue
                n = len(values)
                mean = sum(values) / n
                if n > 1:
                    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
                    std = variance ** 0.5
                else:
                    std = 0.0
                exit_stats[str(d)] = {
                    "mean": round(mean, 4),
                    "std": round(std, 4),
                    "n": n,
                }
            entry["avg_exit_iter_by_depth"] = exit_stats

        # Convenience: overall ID/OOD means across seeds
        id_means = [
            v["mean"] for k, v in acc_stats.items() if int(k) <= 8
        ]
        ood_means = [
            v["mean"] for k, v in acc_stats.items() if int(k) > 8
        ]
        if id_means:
            entry["id_mean_acc"] = round(sum(id_means) / len(id_means), 6)
        if ood_means:
            entry["ood_mean_acc"] = round(sum(ood_means) / len(ood_means), 6)

        summary[variant] = entry

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(f"\n  → saved summary to {out_path}")
    return out_path


def print_summary_table(summary_path: Path) -> None:
    """Print a compact summary table from the summary JSON."""
    with open(summary_path) as f:
        summary = json.load(f)

    print()
    print("=" * 78)
    print("  AGGREGATED SUMMARY  (mean ± std across seeds)")
    print("=" * 78)

    for variant in VARIANTS:
        if variant not in summary:
            continue
        entry = summary[variant]
        acc = entry["accuracy_by_depth"]
        n_seeds = entry.get("n_seeds", "?")

        print(f"\n  ── {variant} ({n_seeds} seeds) ──")

        has_exit = "avg_exit_iter_by_depth" in entry
        if has_exit:
            exit_data = entry["avg_exit_iter_by_depth"]
            print(f"  {'Depth':>5}  {'Acc Mean':>9}  {'± Std':>7}  {'Exit Mean':>10}  {'Split':>5}")
            print(f"  {'-' * 5}  {'-' * 9}  {'-' * 7}  {'-' * 10}  {'-' * 5}")
        else:
            print(f"  {'Depth':>5}  {'Acc Mean':>9}  {'± Std':>7}  {'Split':>5}")
            print(f"  {'-' * 5}  {'-' * 9}  {'-' * 7}  {'-' * 5}")

        for d_str in sorted(acc.keys(), key=lambda x: int(x)):
            d = int(d_str)
            split = "ID" if d <= 8 else "OOD"
            mean = acc[d_str]["mean"]
            std = acc[d_str]["std"]
            if has_exit and d_str in exit_data:
                e_mean = exit_data[d_str]["mean"]
                print(f"  {d:>5}  {mean:>9.4f}  {std:>7.4f}  {e_mean:>10.2f}  {split:>5}")
            elif has_exit:
                print(f"  {d:>5}  {mean:>9.4f}  {std:>7.4f}  {'—':>10}  {split:>5}")
            else:
                print(f"  {d:>5}  {mean:>9.4f}  {std:>7.4f}  {split:>5}")

        if "id_mean_acc" in entry:
            print(f"  ID  avg:  {entry['id_mean_acc']:.4f}")
        if "ood_mean_acc" in entry:
            print(f"  OOD avg:  {entry['ood_mean_acc']:.4f}")

    print()


# ---------------------------------------------------------------------------
# Evaluation runners
# ---------------------------------------------------------------------------

def build_test_loaders(device: torch.device) -> Tuple[DataLoader, DataLoader]:
    """Build DataLoaders for test_id and test_ood splits."""
    id_path = DATA_DIR / "test_id.jsonl"
    ood_path = DATA_DIR / "test_ood.jsonl"

    if not id_path.exists():
        raise FileNotFoundError(
            f"Test data not found at {id_path}\n"
            f"  Run `python -m polly.data` first to generate splits."
        )
    if not ood_path.exists():
        raise FileNotFoundError(
            f"Test data not found at {ood_path}\n"
            f"  Run `python -m polly.data` first to generate splits."
        )

    pin = device.type == "cuda"
    id_loader = DataLoader(
        BracketDataset(id_path),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=pin,
    )
    ood_loader = DataLoader(
        BracketDataset(ood_path),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=pin,
    )
    return id_loader, ood_loader


def evaluate_single_run(
    variant: str,
    seed: int,
    device: torch.device,
    force_all_iters: bool = False,
) -> Dict[str, Dict[int, Any]]:
    """Evaluate a single (variant, seed) checkpoint on both test splits."""
    print(f"\n[eval] Loading {variant} seed={seed} ...")
    model, meta = load_checkpoint(variant, seed, device)

    id_loader, ood_loader = build_test_loaders(device)

    print(f"[eval] Evaluating on test_id (depths 1–8) ...")
    id_results = evaluate_model(model, id_loader, device, force_all_iters=force_all_iters)

    print(f"[eval] Evaluating on test_ood (depths 9–16) ...")
    ood_results = evaluate_model(model, ood_loader, device, force_all_iters=force_all_iters)

    # Merge the two result dicts (depths don't overlap so union is clean)
    combined: Dict[str, Dict[int, Any]] = {
        "accuracy_by_depth": {
            **id_results["accuracy_by_depth"],
            **ood_results["accuracy_by_depth"],
        },
        "avg_exit_iter_by_depth": {
            **id_results["avg_exit_iter_by_depth"],
            **ood_results["avg_exit_iter_by_depth"],
        },
    }

    print_accuracy_table(variant, seed, combined, meta)
    save_run_results(variant, seed, combined, force_all_iters)

    return combined


def evaluate_all(
    device: torch.device,
    force_all_iters: bool = False,
) -> None:
    """Evaluate all variants × seeds that have checkpoints, then aggregate."""
    all_run_results: Dict[Tuple[str, int], Dict[str, Dict[int, Any]]] = {}

    skipped: List[str] = []

    for variant in VARIANTS:
        for seed in SEEDS:
            ckpt_path = CHECKPOINT_DIR / f"{variant}_seed{seed}" / "best.pt"
            if not ckpt_path.exists():
                skipped.append(f"{variant}_seed{seed}")
                continue

            try:
                results = evaluate_single_run(variant, seed, device, force_all_iters)
                all_run_results[(variant, seed)] = results
            except Exception as exc:
                print(f"  !! Error evaluating {variant} seed={seed}: {exc}")
                skipped.append(f"{variant}_seed{seed} (error)")

    if skipped:
        print(f"\n[eval] Skipped (no checkpoint or error): {', '.join(skipped)}")

    if not all_run_results:
        print("\n[eval] No checkpoints found — nothing to aggregate.")
        return

    summary_path = save_summary(all_run_results)
    print_summary_table(summary_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate bracket-matching transformer checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m polly.evaluate --variant vanilla --seed 100\n"
            "  python -m polly.evaluate --variant looped --seed 100 --force-all-iters\n"
            "  python -m polly.evaluate --all --device auto\n"
        ),
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all available checkpoints and produce summary.",
    )
    mode.add_argument(
        "--variant",
        type=str,
        choices=VARIANTS,
        help="Model variant to evaluate (requires --seed).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed identifying the run (required with --variant).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cuda', 'cpu'. Default: auto.",
    )
    parser.add_argument(
        "--force-all-iters",
        action="store_true",
        help="For looped variants, run all t_max iterations (no early exit).",
    )

    args = parser.parse_args()

    if args.variant and args.seed is None:
        parser.error("--seed is required when --variant is specified.")

    return args


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"[eval] device={device}")

    if args.all:
        evaluate_all(device, force_all_iters=args.force_all_iters)
    else:
        evaluate_single_run(
            variant=args.variant,
            seed=args.seed,
            device=device,
            force_all_iters=args.force_all_iters,
        )

    print("[eval] Done ✓")


if __name__ == "__main__":
    main()