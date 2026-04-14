"""
data.py — Deterministic data generation for a bracket-matching transformer experiment.

Vocabulary:
    PAD = 0, OPEN = 1, CLOSE = 2, CLS = 3

Generates balanced / unbalanced bracket strings at controlled nesting depths,
writes JSONL splits to disk, and provides a PyTorch Dataset for training.

Usage:
    python polly/data.py          # from the pollysec/ directory
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset

# ── Vocabulary ────────────────────────────────────────────────────────────────

PAD = 0
OPEN = 1
CLOSE = 2
CLS = 3

TOKEN_MAP = {"(": OPEN, ")": CLOSE}

MAX_SEQ_LEN = 34  # 1 CLS + up to 32 bracket chars + at least 1 PAD


# ── Validation helpers ────────────────────────────────────────────────────────


def is_balanced(s: str) -> bool:
    """Return True iff *s* consists solely of matched parentheses."""
    depth = 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        else:
            return False
        if depth < 0:
            return False
    return depth == 0


def max_nesting_depth(s: str) -> int:
    """Return the maximum nesting depth of a balanced string.

    Precondition: *s* is balanced (not checked here for speed).
    """
    depth = 0
    best = 0
    for ch in s:
        if ch == "(":
            depth += 1
            if depth > best:
                best = depth
        elif ch == ")":
            depth -= 1
    return best


# ── Balanced string generation ────────────────────────────────────────────────


def _legal_insert_positions(s: str) -> List[int]:
    """Return indices where inserting "()" keeps the string balanced.

    A matched pair "()" can be inserted at position *i* (meaning the new "("
    lands at index *i* and ")" at *i+1*, pushing everything else right) iff
    the resulting string is balanced.  Equivalently, any position 0 … len(s)
    is legal because inserting "()" at any point in a balanced string keeps
    it balanced (it just adds a depth-1 sub-expression at that point).
    """
    # Inserting "()" (an immediately-closed pair) at *any* position in a
    # balanced string always yields another balanced string, so every
    # position 0..len(s) is legal.
    return list(range(len(s) + 1))


def generate_balanced(depth: int, rng: random.Random,
                       max_total_len: int = 32) -> str:
    """Generate a balanced bracket string with max nesting depth exactly *depth*.

    Strategy:
        1. Start with the depth-*d* kernel: "(" * d + ")" * d  (length 2d).
        2. Repeatedly insert matched "()" pairs at random legal positions,
           varying structure and length, without exceeding *max_total_len*
           or accidentally increasing the max depth beyond *d*.

    We re-check depth after every insertion to guarantee exactness.
    """
    if depth < 1:
        raise ValueError("depth must be >= 1")

    kernel = "(" * depth + ")" * depth
    s = kernel

    # Decide how many extra pairs to try inserting (0 … room available).
    room = (max_total_len - len(s)) // 2  # each pair adds 2 chars
    if room > 0:
        n_extra = rng.randint(0, room)
    else:
        n_extra = 0

    for _ in range(n_extra):
        if len(s) + 2 > max_total_len:
            break
        positions = _legal_insert_positions(s)
        pos = rng.choice(positions)
        candidate = s[:pos] + "()" + s[pos:]
        # Ensure we haven't accidentally increased depth.
        if max_nesting_depth(candidate) == depth:
            s = candidate
        # else: skip this attempt, try next

    # If pure "()" insertions haven't been able to produce variety (very
    # unlikely for depth >= 2 but possible for depth == 1 where the kernel
    # is already "()" and every insertion can only keep depth at 1 — which
    # is fine, that's the only depth-1 structure anyway), we're done.
    #
    # For deeper strings we also shuffle structure by inserting pairs
    # *inside* existing nesting.  We do a second pass that inserts pairs
    # at positions immediately after an open paren, which deepens a branch
    # only if the current depth at that point + 1 <= target depth.
    # This creates shapes like "(()())(())" instead of always "(((...)))()()".

    # Second-pass: structural reshaping via "wrap" insertions.
    # Wrap insertion: pick a random balanced sub-expression and wrap it
    # with an extra "(…)".  This increases the max depth of that sub-expr
    # by 1, so we only do it where the local depth is < target depth.
    attempts = min(room, 10) if room > 0 else 0
    for _ in range(attempts):
        if len(s) + 2 > max_total_len:
            break
        # Find positions where current running depth < depth - 1
        # (so wrapping won't exceed target depth).
        running = 0
        eligible: List[int] = []
        for i, ch in enumerate(s):
            if ch == "(":
                if running < depth - 1:
                    eligible.append(i)
                running += 1
            else:
                running -= 1
        if not eligible:
            break
        idx = rng.choice(eligible)
        # Find the matching close for this open.
        lvl = 0
        match_idx = idx
        for j in range(idx, len(s)):
            if s[j] == "(":
                lvl += 1
            else:
                lvl -= 1
            if lvl == 0:
                match_idx = j
                break
        candidate = s[:idx] + "(" + s[idx:match_idx + 1] + ")" + s[match_idx + 1:]
        if len(candidate) <= max_total_len and max_nesting_depth(candidate) == depth:
            s = candidate

    assert is_balanced(s), f"BUG: generated unbalanced string: {s!r}"
    assert max_nesting_depth(s) == depth, (
        f"BUG: expected depth {depth}, got {max_nesting_depth(s)} for {s!r}"
    )
    return s


# ── Unbalanced string generation ─────────────────────────────────────────────

def corrupt_balanced(s: str, rng: random.Random) -> str:
    """Apply one random corruption to a balanced string to make it unbalanced.

    Corruptions (chosen uniformly):
        1. Flip one bracket (open↔close).
        2. Delete one bracket.
        3. Insert one extra bracket at a random position.

    Returns a string that is guaranteed *not* balanced.
    We retry (different corruption / position) if the result happens to
    still be balanced (astronomically unlikely but not impossible for very
    short strings like "()").
    """
    methods = ["flip", "delete", "insert"]
    rng.shuffle(methods)

    for method in methods:
        if method == "flip" and len(s) >= 1:
            indices = list(range(len(s)))
            rng.shuffle(indices)
            for i in indices:
                flipped = ")" if s[i] == "(" else "("
                candidate = s[:i] + flipped + s[i + 1:]
                if not is_balanced(candidate):
                    return candidate

        elif method == "delete" and len(s) >= 1:
            indices = list(range(len(s)))
            rng.shuffle(indices)
            for i in indices:
                candidate = s[:i] + s[i + 1:]
                if not is_balanced(candidate):
                    return candidate

        elif method == "insert":
            positions = list(range(len(s) + 1))
            rng.shuffle(positions)
            brackets = ["(", ")"]
            for pos in positions:
                for br in brackets:
                    candidate = s[:pos] + br + s[pos:]
                    if len(candidate) <= 32 and not is_balanced(candidate):
                        return candidate

    # Absolute fallback: delete the first character (always unbalances
    # a non-empty balanced string of length >= 4).
    fallback = s[1:]
    if not is_balanced(fallback):
        return fallback
    fallback = s[:-1]
    assert not is_balanced(fallback), f"BUG: cannot corrupt {s!r}"
    return fallback


# ── Split generation ──────────────────────────────────────────────────────────


def generate_split(
    depths: List[int],
    total_size: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Generate a split of *total_size* examples spread uniformly across *depths*.

    50 % balanced, 50 % unbalanced at each depth.
    """
    rng = random.Random(seed)

    n_depths = len(depths)
    per_depth = total_size // n_depths
    # per_depth must be even for 50/50 split
    assert per_depth % 2 == 0, (
        f"per_depth={per_depth} is not even — adjust total_size or depths"
    )
    n_bal = per_depth // 2
    n_unbal = per_depth // 2

    examples: List[Dict[str, Any]] = []

    for d in depths:
        # ── Balanced examples ────────────────────────────────────────
        bal_set: set[str] = set()
        attempts = 0
        while len(bal_set) < n_bal:
            s = generate_balanced(d, rng)
            bal_set.add(s)
            attempts += 1
            if attempts > n_bal * 20:
                # For very low depths there may not be enough unique
                # strings; allow duplicates.
                break

        bal_list = list(bal_set)
        # If we don't have enough unique ones, pad with repeats.
        while len(bal_list) < n_bal:
            bal_list.append(rng.choice(list(bal_set)))
        rng.shuffle(bal_list)
        bal_list = bal_list[:n_bal]

        for s in bal_list:
            examples.append({"input": s, "label": 1, "depth": d})

        # ── Unbalanced examples ──────────────────────────────────────
        # Generate fresh balanced strings to corrupt (don't reuse the
        # balanced set — keeps the two halves independent).
        for _ in range(n_unbal):
            base = generate_balanced(d, rng)
            corrupted = corrupt_balanced(base, rng)
            assert not is_balanced(corrupted), (
                f"BUG: corruption produced balanced string: {corrupted!r}"
            )
            examples.append({"input": corrupted, "label": 0, "depth": d})

    # Shuffle the whole split so balanced/unbalanced are interleaved.
    rng.shuffle(examples)
    return examples


# ── I/O ───────────────────────────────────────────────────────────────────────


def write_jsonl(examples: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ── PyTorch Dataset ───────────────────────────────────────────────────────────


class BracketDataset(Dataset):
    """Loads a JSONL file and tokenizes bracket strings into fixed-length
    integer tensors.

    Each sample returns:
        input_ids      : LongTensor[MAX_SEQ_LEN]   — CLS + tokens + PAD…
        attention_mask : LongTensor[MAX_SEQ_LEN]   — 1 for real tokens, 0 for PAD
        label          : LongTensor scalar          — 1 balanced, 0 unbalanced
        depth          : LongTensor scalar          — nesting depth
    """

    def __init__(self, path: str | Path) -> None:
        self.examples = read_jsonl(Path(path))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor,
                                              torch.Tensor, torch.Tensor]:
        ex = self.examples[idx]
        tokens = [CLS] + [TOKEN_MAP[ch] for ch in ex["input"]]
        length = len(tokens)

        # Pad to MAX_SEQ_LEN
        input_ids = tokens + [PAD] * (MAX_SEQ_LEN - length)
        attention_mask = [1] * length + [0] * (MAX_SEQ_LEN - length)

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(ex["label"], dtype=torch.long),
            torch.tensor(ex["depth"], dtype=torch.long),
        )


# ── CLI ───────────────────────────────────────────────────────────────────────


def print_summary(name: str, examples: List[Dict[str, Any]]) -> None:
    from collections import Counter

    depth_counts: Counter[int] = Counter()
    label_counts: Counter[int] = Counter()
    depth_label: Counter[Tuple[int, int]] = Counter()

    for ex in examples:
        depth_counts[ex["depth"]] += 1
        label_counts[ex["label"]] += 1
        depth_label[(ex["depth"], ex["label"])] += 1

    total = len(examples)
    n_bal = label_counts[1]
    n_unbal = label_counts[0]
    print(f"\n{'=' * 60}")
    print(f"  {name}: {total:,} examples  "
          f"(bal={n_bal:,}  unbal={n_unbal:,}  "
          f"ratio={n_bal / total:.2%})")
    print(f"{'=' * 60}")
    print(f"  {'Depth':>5}  {'Total':>7}  {'Bal':>7}  {'Unbal':>7}")
    print(f"  {'-' * 5}  {'-' * 7}  {'-' * 7}  {'-' * 7}")
    for d in sorted(depth_counts):
        t = depth_counts[d]
        b = depth_label[(d, 1)]
        u = depth_label[(d, 0)]
        print(f"  {d:>5}  {t:>7,}  {b:>7,}  {u:>7,}")


def main() -> None:
    data_dir = Path(__file__).resolve().parent / "data"

    splits = {
        "train": {
            "depths": list(range(1, 9)),
            "total_size": 200_000,
            "seed": 42,
            "filename": "train.jsonl",
        },
        "val": {
            "depths": list(range(1, 9)),
            "total_size": 10_000,
            "seed": 43,
            "filename": "val.jsonl",
        },
        "test_id": {
            "depths": list(range(1, 9)),
            "total_size": 10_000,
            "seed": 44,
            "filename": "test_id.jsonl",
        },
        "test_ood": {
            "depths": list(range(9, 17)),
            "total_size": 10_000,
            "seed": 45,
            "filename": "test_ood.jsonl",
        },
    }

    for name, cfg in splits.items():
        print(f"Generating {name} …")
        examples = generate_split(
            depths=cfg["depths"],
            total_size=cfg["total_size"],
            seed=cfg["seed"],
        )

        # ── Final validation pass ────────────────────────────────────
        for ex in examples:
            s = ex["input"]
            assert len(s) <= 32, f"String too long ({len(s)}): {s!r}"
            if ex["label"] == 1:
                assert is_balanced(s), f"Labelled balanced but isn't: {s!r}"
                assert max_nesting_depth(s) == ex["depth"], (
                    f"Depth mismatch: expected {ex['depth']}, "
                    f"got {max_nesting_depth(s)} for {s!r}"
                )
            else:
                assert not is_balanced(s), (
                    f"Labelled unbalanced but is balanced: {s!r}"
                )

        out_path = data_dir / cfg["filename"]
        write_jsonl(examples, out_path)
        print(f"  → wrote {out_path}  ({len(examples):,} examples)")
        print_summary(name, examples)

    print("\nDone ✓")


if __name__ == "__main__":
    main()