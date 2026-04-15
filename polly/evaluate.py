"""
polly.evaluate — Evaluation script for ListOps transformer experiments (v4).

Computes per-depth accuracy (depths 1–D_max) and per-root-operation accuracy
on a single unified test set.  For looped variants, also tracks average exit
iteration per depth.

Usage:
    # Single run:
    python -m polly.evaluate --variant vanilla --seed 100 --device auto

    # Single run, forcing all iterations (no early exit) for looped variants:
    python -m polly.evaluate --variant looped --seed 100 --force-all-iters

    # All runs (3 variants × 3 seeds), with aggregated summary:
    python -m polly.evaluate --all --device auto
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from polly.data import ListOpsDataset, read_jsonl
from polly.model import PollyTransformer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VARIANTS = ("vanilla", "looped", "looped_reg")
LOOPED_VARIANTS = ("looped", "looped_reg")
SEEDS = (100, 200, 300)
NUM_CLASSES = 10
ROOT_OPS = ("MIN", "MAX", "MED", "SM")

EVAL_BATCH_SIZE = 256
DEFAULT_T_MAX = 4
DEFAULT_EXIT_THRESHOLD = 0.8

DATA_DIR = Path(os.environ.get("POLLY_DATA_DIR", Path(__file__).resolve().parent / "data"))
CHECKPOINT_DIR = Path(os.environ.get("POLLY_CHECKPOINT_DIR", Path(__file__).resolve().parent / "checkpoints"))
FIGURES_DIR = Path(os.environ.get("POLLY_FIGURES_DIR", Path(__file__).resolve().parent / "figures"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def extract_root_op(input_str: str) -> str:
    """Extract the root operation from a ListOps input string.

    The root op is the first token after the opening '['.
    Example: "[ SM 3 [ MIN 4 1 ] ]" → "SM"
    """
    tokens = input_str.split()
    # tokens[0] should be '[', tokens[1] is the root op
    if len(tokens) >= 2:
        return tokens[1]
    return "UNKNOWN"


def _load_test_inputs(path: Path) -> List[str]:
    """Load raw input strings from the test JSONL so we can extract root ops."""
    examples = read_jsonl(path)
    return [ex["input"] for ex in examples]


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(
    variant: str,
    seed: int,
    device: torch.device,
) -> Tuple[PollyTransformer, Dict[str, Any]]:
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

    model = PollyTransformer(variant=variant)
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
    model: PollyTransformer,
    test_loader: DataLoader,
    device: torch.device,
    force_all_iters: bool = False,
    root_ops: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run evaluation on a single DataLoader.

    Args:
        model:           trained PollyTransformer
        test_loader:     DataLoader yielding (input_ids, attention_mask, label, depth)
        device:          torch device
        force_all_iters: if True, run all t_max iterations (no early exit)
        root_ops:        list of root operation strings, one per sample in
                         dataset order.  If None, per-op breakdown is skipped.

    Returns a dict with:
        "accuracy_by_depth"       : {depth: float}
        "accuracy_by_op"          : {op: float}
        "avg_exit_iter_by_depth"  : {depth: float}   (only for looped variants)
        "overall_accuracy"        : float
    """
    model.eval()

    depth_correct: Dict[int, int] = defaultdict(int)
    depth_total: Dict[int, int] = defaultdict(int)

    op_correct: Dict[str, int] = defaultdict(int)
    op_total: Dict[str, int] = defaultdict(int)

    depth_exit_iter_sum: Dict[int, float] = defaultdict(float)
    depth_exit_iter_count: Dict[int, int] = defaultdict(int)

    total_correct = 0
    total_samples = 0

    is_looped = model.is_looped

    # Track global sample index to map back to root_ops list
    sample_idx = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            depths = batch[3]  # keep on CPU for bookkeeping

            if is_looped and force_all_iters:
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

            B = labels.shape[0]

            # Default: use last iteration's logits
            logits = logits_list[-1]

            # ------- per-sample exit iteration (looped only) -------
            per_sample_exit_iter: Optional[torch.Tensor] = None
            if is_looped and exit_probs_list:
                exit_probs_stack = torch.stack(exit_probs_list, dim=0)  # (T, B)
                T = exit_probs_stack.shape[0]

                if not force_all_iters:
                    exceeded = exit_probs_stack > DEFAULT_EXIT_THRESHOLD  # (T, B)
                    per_sample_exit_iter = torch.full((B,), T, dtype=torch.float32)
                    for t_idx in range(T):
                        still_running = per_sample_exit_iter == T
                        newly_exiting = exceeded[t_idx] & still_running
                        per_sample_exit_iter[newly_exiting] = float(t_idx + 1)
                else:
                    per_sample_exit_iter = torch.full((B,), float(T), dtype=torch.float32)

                # Gather per-sample logits from each sample's exit iteration
                if not force_all_iters:
                    logits_stack = torch.stack(logits_list, dim=0)  # (T, B, C)
                    exit_idx = (per_sample_exit_iter.long() - 1).clamp(min=0, max=T - 1)
                    exit_idx = exit_idx.to(logits_stack.device)
                    logits = logits_stack[exit_idx, torch.arange(B, device=logits_stack.device)]

            preds = logits.argmax(dim=-1)
            correct = (preds == labels)

            # ------- accumulate stats -------
            for i in range(B):
                d = depths[i].item()
                c = int(correct[i].item())

                depth_total[d] += 1
                depth_correct[d] += c

                total_correct += c
                total_samples += 1

                # Per-op breakdown
                if root_ops is not None and sample_idx + i < len(root_ops):
                    op = root_ops[sample_idx + i]
                    op_total[op] += 1
                    op_correct[op] += c

                if per_sample_exit_iter is not None:
                    depth_exit_iter_sum[d] += per_sample_exit_iter[i].item()
                    depth_exit_iter_count[d] += 1

            sample_idx += B

    # ------- compute final metrics -------
    accuracy_by_depth: Dict[int, float] = {}
    for d in sorted(depth_total.keys()):
        accuracy_by_depth[d] = depth_correct[d] / max(depth_total[d], 1)

    accuracy_by_op: Dict[str, float] = {}
    for op in ROOT_OPS:
        if op_total[op] > 0:
            accuracy_by_op[op] = op_correct[op] / op_total[op]

    avg_exit_iter_by_depth: Dict[int, float] = {}
    if is_looped:
        for d in sorted(depth_exit_iter_count.keys()):
            count = depth_exit_iter_count[d]
            if count > 0:
                avg_exit_iter_by_depth[d] = depth_exit_iter_sum[d] / count

    overall_accuracy = total_correct / max(total_samples, 1)

    return {
        "accuracy_by_depth": accuracy_by_depth,
        "accuracy_by_op": accuracy_by_op,
        "avg_exit_iter_by_depth": avg_exit_iter_by_depth,
        "overall_accuracy": overall_accuracy,
    }


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def print_accuracy_table(
    variant: str,
    seed: int,
    results: Dict[str, Any],
    meta: Dict[str, Any],
) -> None:
    """Print nicely formatted per-depth and per-op tables to stdout."""
    acc = results["accuracy_by_depth"]
    exit_iters = results.get("avg_exit_iter_by_depth", {})
    acc_by_op = results.get("accuracy_by_op", {})
    overall = results.get("overall_accuracy", 0.0)
    is_looped = variant in LOOPED_VARIANTS

    print()
    print("=" * 72)
    print(f"  {variant}  seed={seed}  |  "
          f"checkpoint step={meta.get('step', '?')}  "
          f"val_acc={meta.get('best_val_acc', '?')}")
    print("=" * 72)

    # ---- Per-depth table ----
    print("\n  Per-Depth Accuracy:")
    if is_looped:
        header = f"  {'Depth':>5}  {'Accuracy':>10}  {'Avg Exit Iter':>14}"
        print(header)
        print(f"  {'-' * 5}  {'-' * 10}  {'-' * 14}")
        for d in sorted(acc.keys()):
            exit_str = f"{exit_iters[d]:.2f}" if d in exit_iters else "—"
            print(f"  {d:>5}  {acc[d]:>10.4f}  {exit_str:>14}")
    else:
        header = f"  {'Depth':>5}  {'Accuracy':>10}"
        print(header)
        print(f"  {'-' * 5}  {'-' * 10}")
        for d in sorted(acc.keys()):
            print(f"  {d:>5}  {acc[d]:>10.4f}")

    # ---- Per-op table ----
    if acc_by_op:
        print("\n  Per-Operation Accuracy:")
        print(f"  {'Op':>5}  {'Accuracy':>10}")
        print(f"  {'-' * 5}  {'-' * 10}")
        for op in ROOT_OPS:
            if op in acc_by_op:
                print(f"  {op:>5}  {acc_by_op[op]:>10.4f}")

    print(f"\n  Overall accuracy: {overall:.4f}  (chance = 10%)")
    print()


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def save_run_results(
    variant: str,
    seed: int,
    results: Dict[str, Any],
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
        "accuracy_by_op": {
            op: round(v, 6)
            for op, v in sorted(results["accuracy_by_op"].items())
        },
        "overall_accuracy": round(results["overall_accuracy"], 6),
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


def _aggregate_stats(
    values_list: List[float],
) -> Tuple[float, float]:
    """Compute mean and sample std from a list of floats."""
    n = len(values_list)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values_list) / n
    if n > 1:
        variance = sum((v - mean) ** 2 for v in values_list) / (n - 1)
        std = variance ** 0.5
    else:
        std = 0.0
    return mean, std


def save_summary(
    all_run_results: Dict[Tuple[str, int], Dict[str, Any]],
) -> Path:
    """Aggregate across seeds and write eval_summary.json.

    For each variant, computes mean ± std of accuracy at each depth and
    each root operation across all available seeds.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "eval_summary.json"

    # Group by variant
    variant_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for (variant, seed), results in all_run_results.items():
        variant_results[variant].append(results)

    summary: Dict[str, Any] = {"variants": {}}

    for variant in VARIANTS:
        run_list = variant_results.get(variant, [])
        if not run_list:
            continue

        # -- per-depth accuracy stats --
        all_depths = sorted(set(
            d for r in run_list for d in r["accuracy_by_depth"]
        ))
        mean_acc_by_depth: Dict[str, List[float]] = {}
        for d in all_depths:
            values = [r["accuracy_by_depth"][d] for r in run_list if d in r["accuracy_by_depth"]]
            m, s = _aggregate_stats(values)
            mean_acc_by_depth[str(d)] = [round(m, 6), round(s, 6)]

        # -- per-op accuracy stats --
        mean_acc_by_op: Dict[str, List[float]] = {}
        for op in ROOT_OPS:
            values = [r["accuracy_by_op"][op] for r in run_list if op in r.get("accuracy_by_op", {})]
            if values:
                m, s = _aggregate_stats(values)
                mean_acc_by_op[op] = [round(m, 6), round(s, 6)]

        # -- overall accuracy stats --
        overall_values = [r["overall_accuracy"] for r in run_list]
        overall_m, overall_s = _aggregate_stats(overall_values)

        entry: Dict[str, Any] = {
            "n_seeds": len(run_list),
            "mean_acc_by_depth": mean_acc_by_depth,
            "mean_acc_by_op": mean_acc_by_op,
            "overall_mean_acc": [round(overall_m, 6), round(overall_s, 6)],
        }

        # -- exit iteration stats for looped variants --
        exit_dicts = [r["avg_exit_iter_by_depth"] for r in run_list if r.get("avg_exit_iter_by_depth")]
        if exit_dicts:
            exit_depths = sorted(set(d for ed in exit_dicts for d in ed))
            mean_exit_by_depth: Dict[str, List[float]] = {}
            for d in exit_depths:
                values = [ed[d] for ed in exit_dicts if d in ed]
                m, s = _aggregate_stats(values)
                mean_exit_by_depth[str(d)] = [round(m, 4), round(s, 4)]
            entry["mean_exit_iter_by_depth"] = mean_exit_by_depth

        summary["variants"][variant] = entry

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(f"\n  → saved summary to {out_path}")
    return out_path


def print_summary_table(summary_path: Path) -> None:
    """Print a compact summary table from the summary JSON."""
    with open(summary_path) as f:
        summary = json.load(f)

    variants_data = summary.get("variants", {})

    print()
    print("=" * 78)
    print("  AGGREGATED SUMMARY  (mean ± std across seeds)")
    print("=" * 78)

    for variant in VARIANTS:
        if variant not in variants_data:
            continue
        entry = variants_data[variant]
        acc = entry["mean_acc_by_depth"]
        acc_op = entry.get("mean_acc_by_op", {})
        n_seeds = entry.get("n_seeds", "?")
        overall = entry.get("overall_mean_acc", [0.0, 0.0])

        print(f"\n  ── {variant} ({n_seeds} seeds) ──")

        # Per-depth table
        has_exit = "mean_exit_iter_by_depth" in entry
        exit_data: Dict[str, Any] = entry.get("mean_exit_iter_by_depth", {})
        print("\n  Per-Depth:")
        if has_exit:
            print(f"  {'Depth':>5}  {'Acc Mean':>9}  {'± Std':>7}  {'Exit Mean':>10}")
            print(f"  {'-' * 5}  {'-' * 9}  {'-' * 7}  {'-' * 10}")
        else:
            print(f"  {'Depth':>5}  {'Acc Mean':>9}  {'± Std':>7}")
            print(f"  {'-' * 5}  {'-' * 9}  {'-' * 7}")

        for d_str in sorted(acc.keys(), key=lambda x: int(x)):
            mean, std = acc[d_str]
            if has_exit and d_str in exit_data:
                e_mean = exit_data[d_str][0]
                print(f"  {int(d_str):>5}  {mean:>9.4f}  {std:>7.4f}  {e_mean:>10.2f}")
            elif has_exit:
                print(f"  {int(d_str):>5}  {mean:>9.4f}  {std:>7.4f}  {'—':>10}")
            else:
                print(f"  {int(d_str):>5}  {mean:>9.4f}  {std:>7.4f}")

        # Per-op table
        if acc_op:
            print("\n  Per-Operation:")
            print(f"  {'Op':>5}  {'Acc Mean':>9}  {'± Std':>7}")
            print(f"  {'-' * 5}  {'-' * 9}  {'-' * 7}")
            for op in ROOT_OPS:
                if op in acc_op:
                    mean, std = acc_op[op]
                    print(f"  {op:>5}  {mean:>9.4f}  {std:>7.4f}")

        print(f"\n  Overall: {overall[0]:.4f} ± {overall[1]:.4f}  (chance = 10%)")

    print()


# ---------------------------------------------------------------------------
# Evaluation runners
# ---------------------------------------------------------------------------

def build_test_loader(device: torch.device) -> Tuple[DataLoader, List[str]]:
    """Build DataLoader for the test set and extract root ops.

    Returns (loader, root_ops) where root_ops[i] is the root operation
    string for sample i.
    """
    test_path = DATA_DIR / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(
            f"No test data found at {test_path}.\n"
            f"  Run `python -m polly.data` first to generate splits."
        )

    pin = device.type == "cuda"
    loader = DataLoader(
        ListOpsDataset(test_path),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=pin,
    )

    # Extract root ops from the raw JSONL
    input_strings = _load_test_inputs(test_path)
    root_ops = [extract_root_op(s) for s in input_strings]

    return loader, root_ops


def evaluate_single_run(
    variant: str,
    seed: int,
    device: torch.device,
    force_all_iters: bool = False,
) -> Dict[str, Any]:
    """Evaluate a single (variant, seed) checkpoint on the test set."""
    print(f"\n[eval] Loading {variant} seed={seed} ...")
    model, meta = load_checkpoint(variant, seed, device)

    test_loader, root_ops = build_test_loader(device)

    print(f"[eval] Evaluating on test set ({len(root_ops)} samples) ...")
    results = evaluate_model(
        model, test_loader, device,
        force_all_iters=force_all_iters,
        root_ops=root_ops,
    )

    print_accuracy_table(variant, seed, results, meta)
    save_run_results(variant, seed, results, force_all_iters)

    return results


def evaluate_all(
    device: torch.device,
    force_all_iters: bool = False,
) -> None:
    """Evaluate all variants × seeds that have checkpoints, then aggregate."""
    all_run_results: Dict[Tuple[str, int], Dict[str, Any]] = {}
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
        description="Evaluate ListOps transformer checkpoints.",
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