"""
polly.probe — Linear probing analysis on a frozen V4 (looped_reg) model.

Extracts per-iteration internal representations (register vector r and CLS
hidden state h_cls) and trains linear probes to predict:
  1. Iteration number (classification, 1–T)
  2. Input nesting depth (regression, 1–16)
  3. Max unmatched depth proxy (regression — uses input depth as proxy)
  4. Final answer (classification, balanced vs unbalanced)

Usage:
    python -m polly.probe --seed 100 --device cpu
    python -m polly.probe --seed 100 --device cuda --t-max 4
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
from torch.utils.data import DataLoader, ConcatDataset

from polly.data import BracketDataset
from polly.model import BracketTransformer, DIM, REG_DIM, NUM_CLASSES, N_LAYERS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VARIANT = "looped_reg"
EVAL_BATCH_SIZE = 256
DEFAULT_T_MAX = 4

DATA_DIR = Path(__file__).resolve().parent / "data"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"

# Try to import sklearn; fall back to a pure-torch implementation if absent.
try:
    from sklearn.linear_model import LogisticRegression, Ridge

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Device / checkpoint helpers (mirrors evaluate.py conventions)
# ---------------------------------------------------------------------------

def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_checkpoint(
    seed: int,
    device: torch.device,
) -> BracketTransformer:
    """Load a trained looped_reg model from its best checkpoint (frozen)."""
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

    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad_(False)

    return model


# ---------------------------------------------------------------------------
# Representation extraction — manual forward reimplementation
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_representations(
    model: BracketTransformer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    t_max: int = DEFAULT_T_MAX,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Run the looped_reg forward pass manually, collecting intermediates.

    v5 architecture: Encoder (single pass) → Interpreter (looped T times,
    register fires here) → Decoder (single pass, for logits).

    Returns:
        h_cls_per_iter:  list of T tensors, each (B, DIM)        — CLS hidden after interpreter iteration t
        r_per_iter:      list of T tensors, each (B, REG_DIM)    — register after interpreter iteration t
        logits_per_iter: list of T tensors, each (B, NUM_CLASSES) — logits after decoder at iteration t
    """
    assert model.variant == VARIANT, f"Expected {VARIANT}, got {model.variant}"

    B = input_ids.size(0)
    device = input_ids.device

    # Embed
    h = model._embed(input_ids)  # (B, S, D)

    # --- Encoder (single pass, no register) ---
    for layer in model.encoder_layers:
        h = layer(h, attention_mask=attention_mask)

    # Initialise register to zeros
    r = torch.zeros(B, REG_DIM, device=device)

    h_cls_per_iter: List[torch.Tensor] = []
    r_per_iter: List[torch.Tensor] = []
    logits_per_iter: List[torch.Tensor] = []

    for _t in range(t_max):
        # --- Interpreter layers (register inject/update fires here) ---
        for layer in model.interpreter_layers:
            h = model.register_mech.inject(h, r)
            h = layer(h, attention_mask=attention_mask)
            h_cls_layer = h[:, 0, :]  # (B, D)
            r = model.register_mech.update(h_cls_layer, r)

        # Collect CLS hidden state after this interpreter iteration
        cls_h = h[:, 0, :]  # (B, D)
        h_cls_per_iter.append(cls_h.cpu())
        r_per_iter.append(r.cpu())

        # --- Decoder (single pass, no register) → logits ---
        h_dec = h
        for layer in model.decoder_layers:
            h_dec = layer(h_dec, attention_mask=attention_mask)
        logits_t = model._compute_logits(h_dec)  # (B, NUM_CLASSES)
        logits_per_iter.append(logits_t.cpu())

    return h_cls_per_iter, r_per_iter, logits_per_iter


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def build_combined_loader(device: torch.device) -> DataLoader:
    """Build a DataLoader over the combined test_id + test_ood data."""
    id_path = DATA_DIR / "test_id.jsonl"
    ood_path = DATA_DIR / "test_ood.jsonl"

    for p in (id_path, ood_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Test data not found at {p}\n"
                f"  Run `python -m polly.data` first to generate splits."
            )

    combined = ConcatDataset([BracketDataset(id_path), BracketDataset(ood_path)])
    pin = device.type == "cuda"
    return DataLoader(
        combined,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=pin,
    )


# ---------------------------------------------------------------------------
# Collect all features and labels
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_features(
    model: BracketTransformer,
    loader: DataLoader,
    device: torch.device,
    t_max: int = DEFAULT_T_MAX,
) -> Dict[str, Any]:
    """Run extraction over the full dataset, returning numpy-ready arrays.

    Returns dict with keys:
        h_cls:      Tensor (N, T, DIM)
        r:          Tensor (N, T, REG_DIM)
        labels:     Tensor (N,)       — 0/1 balanced
        depths:     Tensor (N,)       — nesting depth
        iter_ids:   Tensor (N*T,)     — iteration number (1-indexed) for each row
    """
    all_h_cls: List[torch.Tensor] = []  # each (B, T, DIM)
    all_r: List[torch.Tensor] = []      # each (B, T, REG_DIM)
    all_labels: List[torch.Tensor] = []
    all_depths: List[torch.Tensor] = []

    n_batches = len(loader)
    for i, batch in enumerate(loader):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2]   # keep on CPU
        depths = batch[3]   # keep on CPU

        h_cls_iters, r_iters, _ = extract_representations(
            model, input_ids, attention_mask, t_max=t_max,
        )

        # Stack iterations: (T, B, dim) → (B, T, dim)
        h_cls_stacked = torch.stack(h_cls_iters, dim=1)  # (B, T, DIM)
        r_stacked = torch.stack(r_iters, dim=1)           # (B, T, REG_DIM)

        all_h_cls.append(h_cls_stacked)
        all_r.append(r_stacked)
        all_labels.append(labels)
        all_depths.append(depths)

        if (i + 1) % 20 == 0 or (i + 1) == n_batches:
            print(f"  [extract] batch {i + 1}/{n_batches}", flush=True)

    return {
        "h_cls": torch.cat(all_h_cls, dim=0),     # (N, T, DIM)
        "r": torch.cat(all_r, dim=0),               # (N, T, REG_DIM)
        "labels": torch.cat(all_labels, dim=0),      # (N,)
        "depths": torch.cat(all_depths, dim=0),      # (N,)
    }


# ---------------------------------------------------------------------------
# Sklearn probes
# ---------------------------------------------------------------------------

def _prepare_probe_data(
    features: Dict[str, Any],
    t_max: int,
) -> Dict[str, Any]:
    """Reshape extracted features into flat arrays for probing.

    For per-iteration probes (iteration ID, final answer, depth), we replicate
    each sample T times (one row per iteration), giving N*T rows.

    Returns numpy arrays (or tensors if sklearn unavailable).
    """
    N = features["h_cls"].shape[0]
    T = features["h_cls"].shape[1]

    # Flatten: (N, T, dim) → (N*T, dim)
    h_cls_flat = features["h_cls"].reshape(N * T, -1)
    r_flat = features["r"].reshape(N * T, -1)

    # Iteration labels: for each sample, iterations 1..T
    iter_labels = torch.arange(1, T + 1).unsqueeze(0).expand(N, T).reshape(-1)

    # Depth labels: replicate per iteration
    depth_labels = features["depths"].unsqueeze(1).expand(N, T).reshape(-1)

    # Binary label: replicate per iteration
    binary_labels = features["labels"].unsqueeze(1).expand(N, T).reshape(-1)

    return {
        "h_cls_flat": h_cls_flat,       # (N*T, DIM)
        "r_flat": r_flat,               # (N*T, REG_DIM)
        "iter_labels": iter_labels,     # (N*T,)  values 1..T
        "depth_labels": depth_labels,   # (N*T,)  values 1..16
        "binary_labels": binary_labels, # (N*T,)  values 0/1
        "N": N,
        "T": T,
    }


def _split_train_test(n_total: int, train_frac: float = 0.8):
    """Deterministic train/test split indices."""
    rng = torch.Generator().manual_seed(999)
    perm = torch.randperm(n_total, generator=rng)
    n_train = int(n_total * train_frac)
    return perm[:n_train], perm[n_train:]


# ---------------------------------------------------------------------------
# sklearn-based probes
# ---------------------------------------------------------------------------

def run_sklearn_probes(probe_data: Dict[str, Any]) -> Dict[str, Any]:
    """Train all 8 probes (4 tasks × 2 representation types) using sklearn."""
    results: Dict[str, Any] = {}

    h_cls = probe_data["h_cls_flat"].numpy()
    r = probe_data["r_flat"].numpy()
    iter_labels = probe_data["iter_labels"].numpy()
    depth_labels = probe_data["depth_labels"].numpy().astype(float)
    binary_labels = probe_data["binary_labels"].numpy()

    n_total = h_cls.shape[0]
    train_idx, test_idx = _split_train_test(n_total)
    train_idx = train_idx.numpy()
    test_idx = test_idx.numpy()

    repr_map = {"register": r, "cls_hidden": h_cls}

    for repr_name, X in repr_map.items():
        X_train, X_test = X[train_idx], X[test_idx]

        # --- Probe 1: Iteration number (classification 1..T) ---
        y = iter_labels
        y_train, y_test = y[train_idx], y[test_idx]
        clf = LogisticRegression(max_iter=2000, solver="lbfgs", multi_class="multinomial", C=1.0)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        key = f"iteration_number/{repr_name}"
        results[key] = {"type": "classification", "accuracy": round(acc, 4)}
        print(f"  {key:45s}  acc={acc:.4f}")

        # --- Probe 2: Input nesting depth (regression 1..16) ---
        y = depth_labels
        y_train, y_test = y[train_idx], y[test_idx]
        reg = Ridge(alpha=1.0)
        reg.fit(X_train, y_train)
        r2 = reg.score(X_test, y_test)
        key = f"nesting_depth/{repr_name}"
        results[key] = {"type": "regression", "r_squared": round(r2, 4)}
        print(f"  {key:45s}  R²={r2:.4f}")

        # --- Probe 3: Max unmatched depth proxy (regression — same as depth) ---
        # Using input depth as proxy for max unmatched depth at CLS
        y = depth_labels
        y_train, y_test = y[train_idx], y[test_idx]
        reg2 = Ridge(alpha=1.0)
        reg2.fit(X_train, y_train)
        r2_2 = reg2.score(X_test, y_test)
        key = f"max_unmatched_depth_proxy/{repr_name}"
        results[key] = {"type": "regression", "r_squared": round(r2_2, 4)}
        print(f"  {key:45s}  R²={r2_2:.4f}")

        # --- Probe 4: Final answer (classification balanced/unbalanced) ---
        y = binary_labels
        y_train, y_test = y[train_idx], y[test_idx]
        clf2 = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
        clf2.fit(X_train, y_train)
        acc2 = clf2.score(X_test, y_test)
        key = f"final_answer/{repr_name}"
        results[key] = {"type": "classification", "accuracy": round(acc2, 4)}
        print(f"  {key:45s}  acc={acc2:.4f}")

    return results


# ---------------------------------------------------------------------------
# Torch fallback probes (when sklearn is unavailable)
# ---------------------------------------------------------------------------

class _TorchLogisticRegression:
    """Minimal logistic regression in pure torch."""

    def __init__(self, input_dim: int, num_classes: int, lr: float = 0.01,
                 max_iter: int = 2000, weight_decay: float = 1e-3):
        self.W = torch.zeros(input_dim, num_classes, requires_grad=True)
        self.b = torch.zeros(num_classes, requires_grad=True)
        self.lr = lr
        self.max_iter = max_iter
        self.weight_decay = weight_decay

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        X = X.float()
        y = y.long()
        optimizer = torch.optim.LBFGS(
            [self.W, self.b], lr=self.lr, max_iter=20, line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            logits = X @ self.W + self.b
            loss = F.cross_entropy(logits, y)
            loss = loss + 0.5 * self.weight_decay * (self.W ** 2).sum()
            loss.backward()
            return loss

        for _ in range(self.max_iter // 20):
            optimizer.step(closure)

    @torch.no_grad()
    def score(self, X: torch.Tensor, y: torch.Tensor) -> float:
        X = X.float()
        y = y.long()
        logits = X @ self.W + self.b
        preds = logits.argmax(dim=-1)
        return (preds == y).float().mean().item()


class _TorchRidge:
    """Minimal ridge regression in pure torch (closed form)."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.W: Optional[torch.Tensor] = None
        self.b: Optional[float] = None

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        X = X.float()
        y = y.float()
        # Centre
        self.x_mean = X.mean(dim=0)
        self.y_mean = y.mean()
        Xc = X - self.x_mean
        yc = y - self.y_mean

        # Closed form: W = (X^T X + alpha I)^{-1} X^T y
        d = Xc.shape[1]
        A = Xc.T @ Xc + self.alpha * torch.eye(d)
        rhs = Xc.T @ yc
        self.W = torch.linalg.solve(A, rhs)
        self.b = self.y_mean - self.x_mean @ self.W

    @torch.no_grad()
    def score(self, X: torch.Tensor, y: torch.Tensor) -> float:
        X = X.float()
        y = y.float()
        y_pred = X @ self.W + self.b
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return 1.0 - (ss_res / ss_tot.clamp(min=1e-12)).item()


def run_torch_probes(probe_data: Dict[str, Any]) -> Dict[str, Any]:
    """Train all 8 probes using pure-torch fallback implementations."""
    results: Dict[str, Any] = {}

    h_cls = probe_data["h_cls_flat"]
    r = probe_data["r_flat"]
    iter_labels = probe_data["iter_labels"]
    depth_labels = probe_data["depth_labels"].float()
    binary_labels = probe_data["binary_labels"]

    n_total = h_cls.shape[0]
    train_idx, test_idx = _split_train_test(n_total)

    T = probe_data["T"]

    repr_map = {"register": r, "cls_hidden": h_cls}

    for repr_name, X in repr_map.items():
        X_train, X_test = X[train_idx], X[test_idx]
        input_dim = X.shape[1]

        # --- Probe 1: Iteration number ---
        y = iter_labels
        y_train, y_test = y[train_idx], y[test_idx]
        clf = _TorchLogisticRegression(input_dim, num_classes=T)
        # Labels must be 0-indexed for cross_entropy
        clf.fit(X_train, y_train - 1)
        # Score expects same indexing
        acc = clf.score(X_test, y_test - 1)
        key = f"iteration_number/{repr_name}"
        results[key] = {"type": "classification", "accuracy": round(acc, 4)}
        print(f"  {key:45s}  acc={acc:.4f}")

        # --- Probe 2: Nesting depth ---
        y = depth_labels
        y_train, y_test = y[train_idx], y[test_idx]
        reg = _TorchRidge(alpha=1.0)
        reg.fit(X_train, y_train)
        r2 = reg.score(X_test, y_test)
        key = f"nesting_depth/{repr_name}"
        results[key] = {"type": "regression", "r_squared": round(r2, 4)}
        print(f"  {key:45s}  R²={r2:.4f}")

        # --- Probe 3: Max unmatched depth proxy ---
        y = depth_labels
        y_train, y_test = y[train_idx], y[test_idx]
        reg2 = _TorchRidge(alpha=1.0)
        reg2.fit(X_train, y_train)
        r2_2 = reg2.score(X_test, y_test)
        key = f"max_unmatched_depth_proxy/{repr_name}"
        results[key] = {"type": "regression", "r_squared": round(r2_2, 4)}
        print(f"  {key:45s}  R²={r2_2:.4f}")

        # --- Probe 4: Final answer ---
        y = binary_labels
        y_train, y_test = y[train_idx], y[test_idx]
        clf2 = _TorchLogisticRegression(input_dim, num_classes=2)
        clf2.fit(X_train, y_train)
        acc2 = clf2.score(X_test, y_test)
        key = f"final_answer/{repr_name}"
        results[key] = {"type": "classification", "accuracy": round(acc2, 4)}
        print(f"  {key:45s}  acc={acc2:.4f}")

    return results


# ---------------------------------------------------------------------------
# Per-iteration probe analysis (how does probe quality evolve across iters?)
# ---------------------------------------------------------------------------

def run_per_iteration_probes(
    features: Dict[str, Any],
    t_max: int,
) -> Dict[str, Any]:
    """Train final-answer probes separately at each iteration to see how
    answer quality builds across the loop.

    Returns {repr_name: {iter_t: accuracy}} for classification of final answer.
    """
    N = features["h_cls"].shape[0]
    T = features["h_cls"].shape[1]
    labels = features["labels"]  # (N,)

    # Train/test split over samples
    train_idx, test_idx = _split_train_test(N)

    results: Dict[str, Any] = {}

    for repr_name, key in [("cls_hidden", "h_cls"), ("register", "r")]:
        X_all = features[key]  # (N, T, dim)
        per_iter: Dict[int, float] = {}

        for t in range(T):
            X = X_all[:, t, :]  # (N, dim)
            y = labels

            if HAS_SKLEARN:
                X_np = X.numpy()
                y_np = y.numpy()
                X_train, X_test = X_np[train_idx.numpy()], X_np[test_idx.numpy()]
                y_train, y_test = y_np[train_idx.numpy()], y_np[test_idx.numpy()]
                clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
                clf.fit(X_train, y_train)
                acc = clf.score(X_test, y_test)
            else:
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                clf = _TorchLogisticRegression(X.shape[1], num_classes=2)
                clf.fit(X_train, y_train)
                acc = clf.score(X_test, y_test)

            per_iter[t + 1] = round(acc, 4)
            print(f"  final_answer_iter{t+1}/{repr_name:12s}  acc={acc:.4f}")

        results[repr_name] = per_iter

    return results


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def print_probe_table(results: Dict[str, Any]) -> None:
    """Print a formatted table of probe results."""
    print()
    print("=" * 72)
    print("  PROBE RESULTS (pooled across all iterations)")
    print("=" * 72)
    print(f"  {'Probe':42s}  {'Metric':>10s}  {'Value':>8s}")
    print(f"  {'-' * 42}  {'-' * 10}  {'-' * 8}")

    for key in sorted(results.keys()):
        info = results[key]
        if info["type"] == "classification":
            metric = "accuracy"
            val = info["accuracy"]
        else:
            metric = "R²"
            val = info["r_squared"]
        print(f"  {key:42s}  {metric:>10s}  {val:>8.4f}")
    print()


def print_per_iteration_table(per_iter_results: Dict[str, Any]) -> None:
    """Print per-iteration final-answer probe accuracy."""
    print()
    print("=" * 72)
    print("  FINAL-ANSWER PROBE — PER ITERATION")
    print("=" * 72)

    for repr_name, per_iter in per_iter_results.items():
        print(f"\n  Representation: {repr_name}")
        print(f"  {'Iter':>6s}  {'Accuracy':>10s}")
        print(f"  {'-' * 6}  {'-' * 10}")
        for t in sorted(per_iter.keys()):
            print(f"  {t:>6d}  {per_iter[t]:>10.4f}")
    print()


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(
    probe_results: Dict[str, Any],
    per_iter_results: Dict[str, Any],
    seed: int,
    t_max: int,
) -> Path:
    """Save all probe results to JSON."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "probe_results.json"

    payload = {
        "seed": seed,
        "variant": VARIANT,
        "t_max": t_max,
        "sklearn_available": HAS_SKLEARN,
        "probes_pooled": probe_results,
        "probes_per_iteration": {
            repr_name: {str(k): v for k, v in per_iter.items()}
            for repr_name, per_iter in per_iter_results.items()
        },
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Results saved to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Linear probing analysis on a frozen V4 (looped_reg) model.",
    )
    parser.add_argument(
        "--seed", type=int, default=100,
        help="Seed of the checkpoint to load (default: 100).",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device: 'cpu', 'cuda', or 'auto' (default: cpu).",
    )
    parser.add_argument(
        "--t-max", type=int, default=DEFAULT_T_MAX,
        help=f"Number of loop iterations (default: {DEFAULT_T_MAX}).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=EVAL_BATCH_SIZE,
        help=f"Batch size for extraction (default: {EVAL_BATCH_SIZE}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    global EVAL_BATCH_SIZE
    EVAL_BATCH_SIZE = args.batch_size

    print(f"{'=' * 72}")
    print(f"  polly.probe — Linear Probing Analysis")
    print(f"{'=' * 72}")
    print(f"  variant     : {VARIANT}")
    print(f"  seed        : {args.seed}")
    print(f"  device      : {device}")
    print(f"  t_max       : {args.t_max}")
    print(f"  sklearn     : {'yes' if HAS_SKLEARN else 'no (using torch fallback)'}")
    print()

    # --- Load model ---
    print("[probe] Loading model ...")
    model = load_checkpoint(args.seed, device)
    print(f"  Model loaded ({model.count_parameters():,} params, all frozen)")

    # --- Extract representations ---
    print("[probe] Extracting representations from test_id + test_ood ...")
    loader = build_combined_loader(device)
    features = collect_features(model, loader, device, t_max=args.t_max)

    N = features["h_cls"].shape[0]
    T = features["h_cls"].shape[1]
    print(f"  Extracted: N={N:,} samples × T={T} iterations")
    print(f"  h_cls shape per iter: ({N}, {DIM})")
    print(f"  r shape per iter:     ({N}, {REG_DIM})")
    print()

    # --- Prepare probe data ---
    probe_data = _prepare_probe_data(features, t_max=args.t_max)

    # --- Run probes (pooled across iterations) ---
    print("[probe] Training probes (pooled across all iterations) ...")
    if HAS_SKLEARN:
        probe_results = run_sklearn_probes(probe_data)
    else:
        probe_results = run_torch_probes(probe_data)

    print_probe_table(probe_results)

    # --- Run per-iteration probes ---
    print("[probe] Training per-iteration final-answer probes ...")
    per_iter_results = run_per_iteration_probes(features, t_max=args.t_max)
    print_per_iteration_table(per_iter_results)

    # --- Save ---
    save_results(probe_results, per_iter_results, args.seed, args.t_max)

    print("\nDone ✓")


if __name__ == "__main__":
    main()