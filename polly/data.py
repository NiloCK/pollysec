"""
data.py — Deterministic data generation for bracket-matching (multi-type).

Vocabulary (v2 — three bracket types):
    PAD = 0
    (   = 1    )  = 2
    [   = 3    ]  = 4
    {   = 5    }  = 6
    CLS = 7

A string is balanced iff every opener is closed by the *matching* closer in
stack order.  `([)]` has correct per-type counts but is invalid — recognizing
this requires a stack, not just counters.

Usage:
    python polly/data.py          # from the pollysec/ directory
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

# ── Vocabulary ────────────────────────────────────────────────────────────────

PAD = 0
OPEN_PAREN = 1
CLOSE_PAREN = 2
OPEN_BRACKET = 3
CLOSE_BRACKET = 4
OPEN_BRACE = 5
CLOSE_BRACE = 6
CLS = 7

VOCAB_SIZE = 8

# Bracket pairs: (opener_char, closer_char, opener_id, closer_id)
BRACKET_TYPES: List[Tuple[str, str]] = [("(", ")"), ("[", "]"), ("{", "}")]
OPENERS = {"(", "[", "{"}
CLOSERS = {")", "]", "}"}
MATCH_OPENER: Dict[str, str] = {"(": ")", "[": "]", "{": "}"}
MATCH_CLOSER: Dict[str, str] = {")": "(", "]": "[", "}": "{"}

TOKEN_MAP: Dict[str, int] = {
    "(": OPEN_PAREN,
    ")": CLOSE_PAREN,
    "[": OPEN_BRACKET,
    "]": CLOSE_BRACKET,
    "{": OPEN_BRACE,
    "}": CLOSE_BRACE,
}

MAX_SEQ_LEN = 34  # 1 CLS + up to 32 bracket chars + at least 1 PAD


# ── Validation helpers ────────────────────────────────────────────────────────


def is_balanced(s: str) -> bool:
    """Stack-based check: True iff *s* is balanced with correct type matching."""
    stack: list[str] = []
    for ch in s:
        if ch in OPENERS:
            stack.append(ch)
        elif ch in CLOSERS:
            if not stack:
                return False
            opener = stack.pop()
            if MATCH_OPENER[opener] != ch:
                return False
        else:
            return False
    return len(stack) == 0


def passes_counter_test(s: str) -> bool:
    """One-counter test (type-blind): True iff total open == total close and
    running count never goes negative.  This is the test that a single-type
    bracket checker uses — it does NOT verify type matching."""
    depth = 0
    for ch in s:
        if ch in OPENERS:
            depth += 1
        elif ch in CLOSERS:
            depth -= 1
        else:
            return False
        if depth < 0:
            return False
    return depth == 0


def max_nesting_depth(s: str) -> int:
    """Max nesting depth (type-agnostic — any opener increments)."""
    depth = 0
    best = 0
    for ch in s:
        if ch in OPENERS:
            depth += 1
            if depth > best:
                best = depth
        elif ch in CLOSERS:
            depth -= 1
    return best


# ── Balanced string generation ────────────────────────────────────────────────


def generate_balanced(depth: int, rng: random.Random,
                      max_total_len: int = 32) -> str:
    """Generate a balanced multi-type bracket string with max nesting depth
    exactly *depth*.

    Strategy:
        1. Build a depth-*d* kernel with a random bracket type at each level:
           e.g. depth 3 → "([{" + "}])" → "([{}])"
        2. Insert matched pairs (random type, immediately closed like "()") at
           random positions — keeps depth unchanged.
        3. Wrap-insertions: wrap an existing sub-expression with a new pair,
           only where local depth < target depth.
    """
    if depth < 1:
        raise ValueError("depth must be >= 1")

    # 1. Build kernel
    types = [rng.choice(BRACKET_TYPES) for _ in range(depth)]
    openers = "".join(t[0] for t in types)
    closers = "".join(t[1] for t in reversed(types))
    s = openers + closers

    # How much room for extra pairs?
    room = (max_total_len - len(s)) // 2

    # 2. Insert immediately-closed pairs at random positions
    if room > 0:
        n_extra = rng.randint(0, room)
    else:
        n_extra = 0

    for _ in range(n_extra):
        if len(s) + 2 > max_total_len:
            break
        pos = rng.randint(0, len(s))
        pair = rng.choice(BRACKET_TYPES)
        candidate = s[:pos] + pair[0] + pair[1] + s[pos:]
        # Inserting an immediately-closed pair can never increase depth
        # beyond +1 at that position, but check to be safe.
        if max_nesting_depth(candidate) == depth and is_balanced(candidate):
            s = candidate

    # 3. Wrap-insertions for structural variety
    room = (max_total_len - len(s)) // 2
    attempts = min(room, 12) if room > 0 else 0
    for _ in range(attempts):
        if len(s) + 2 > max_total_len:
            break
        # Find positions where running depth < depth - 1
        # (wrapping adds +1 depth to that sub-expression)
        running = 0
        eligible: List[int] = []
        for i, ch in enumerate(s):
            if ch in OPENERS:
                if running < depth - 1:
                    eligible.append(i)
                running += 1
            elif ch in CLOSERS:
                running -= 1
        if not eligible:
            break
        idx = rng.choice(eligible)
        # Find matching closer for opener at idx
        lvl = 0
        match_idx = idx
        for j in range(idx, len(s)):
            if s[j] in OPENERS:
                lvl += 1
            elif s[j] in CLOSERS:
                lvl -= 1
            if lvl == 0:
                match_idx = j
                break
        pair = rng.choice(BRACKET_TYPES)
        candidate = s[:idx] + pair[0] + s[idx:match_idx + 1] + pair[1] + s[match_idx + 1:]
        if (len(candidate) <= max_total_len
                and max_nesting_depth(candidate) == depth
                and is_balanced(candidate)):
            s = candidate

    assert is_balanced(s), f"BUG: generated unbalanced string: {s!r}"
    assert max_nesting_depth(s) == depth, (
        f"BUG: expected depth {depth}, got {max_nesting_depth(s)} for {s!r}"
    )
    return s


# ── Unbalanced string generation ─────────────────────────────────────────────


def _corrupt_mismatch(s: str, rng: random.Random) -> Optional[str]:
    """Replace a closer with a *different-type* closer.

    This keeps total open/close counts unchanged and even per-type counts
    can look fine, but breaks stack matching.  This is the corruption that
    specifically requires stack-based reasoning to detect.
    """
    closer_indices = [i for i, ch in enumerate(s) if ch in CLOSERS]
    if not closer_indices:
        return None
    rng.shuffle(closer_indices)
    for idx in closer_indices:
        original = s[idx]
        other_closers = [c for c in CLOSERS if c != original]
        replacement = rng.choice(other_closers)
        candidate = s[:idx] + replacement + s[idx + 1:]
        if not is_balanced(candidate):
            return candidate
    return None


def _corrupt_flip(s: str, rng: random.Random) -> Optional[str]:
    """Flip one bracket: opener → some closer, or closer → some opener."""
    indices = list(range(len(s)))
    rng.shuffle(indices)
    for i in indices:
        if s[i] in OPENERS:
            replacement = rng.choice(list(CLOSERS))
        else:
            replacement = rng.choice(list(OPENERS))
        candidate = s[:i] + replacement + s[i + 1:]
        if not is_balanced(candidate):
            return candidate
    return None


def _corrupt_delete(s: str, rng: random.Random) -> Optional[str]:
    """Delete one bracket."""
    indices = list(range(len(s)))
    rng.shuffle(indices)
    for i in indices:
        candidate = s[:i] + s[i + 1:]
        if not is_balanced(candidate):
            return candidate
    return None


def _corrupt_insert(s: str, rng: random.Random, max_len: int = 32) -> Optional[str]:
    """Insert one extra bracket at a random position."""
    all_brackets = list(OPENERS | CLOSERS)
    positions = list(range(len(s) + 1))
    rng.shuffle(positions)
    for pos in positions:
        rng.shuffle(all_brackets)
        for br in all_brackets:
            candidate = s[:pos] + br + s[pos:]
            if len(candidate) <= max_len and not is_balanced(candidate):
                return candidate
    return None


def corrupt_balanced(s: str, rng: random.Random, force_mismatch: bool = False) -> Tuple[str, str]:
    """Apply one random corruption to a balanced string to make it unbalanced.

    Returns (corrupted_string, corruption_type).

    When *force_mismatch* is True, only type-mismatch corruption is attempted
    (used to ensure ~50 % of unbalanced examples are mismatches).
    """
    if force_mismatch:
        result = _corrupt_mismatch(s, rng)
        if result is not None:
            return result, "mismatch"
        # If mismatch failed (rare — e.g. only one bracket type present),
        # fall through to general corruption.

    # General corruption: try mismatch first (50 %), then others.
    methods = ["mismatch", "flip", "delete", "insert"]
    if not force_mismatch:
        rng.shuffle(methods)

    dispatch = {
        "mismatch": _corrupt_mismatch,
        "flip": _corrupt_flip,
        "delete": _corrupt_delete,
        "insert": _corrupt_insert,
    }

    for method in methods:
        fn = dispatch[method]
        result = fn(s, rng)
        if result is not None:
            return result, method

    # Absolute fallback: delete the first character.
    fallback = s[1:]
    if not is_balanced(fallback):
        return fallback, "delete_fallback"
    fallback = s[:-1]
    assert not is_balanced(fallback), f"BUG: cannot corrupt {s!r}"
    return fallback, "delete_fallback"


# ── Split generation ──────────────────────────────────────────────────────────


def generate_split(
    depths: List[int],
    total_size: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Generate a split of *total_size* examples spread uniformly across *depths*.

    50 % balanced, 50 % unbalanced at each depth.
    Of the unbalanced half, ~50 % are type-mismatch corruptions (the hard
    cases that require stack reasoning), and ~50 % are flip/delete/insert.
    """
    rng = random.Random(seed)

    n_depths = len(depths)
    per_depth = total_size // n_depths
    assert per_depth % 2 == 0, (
        f"per_depth={per_depth} is not even — adjust total_size or depths"
    )
    n_bal = per_depth // 2
    n_unbal = per_depth // 2

    corruption_stats: Counter[str] = Counter()
    mismatch_counter_test_pass = 0
    mismatch_total = 0

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
                break

        bal_list = list(bal_set)
        while len(bal_list) < n_bal:
            bal_list.append(rng.choice(list(bal_set)))
        rng.shuffle(bal_list)
        bal_list = bal_list[:n_bal]

        for s in bal_list:
            examples.append({"input": s, "label": 1, "depth": d})

        # ── Unbalanced examples ──────────────────────────────────────
        # First half: force mismatch. Second half: any corruption.
        n_mismatch = n_unbal // 2
        n_general = n_unbal - n_mismatch

        for phase, count, force in [("mismatch", n_mismatch, True),
                                     ("general", n_general, False)]:
            for _ in range(count):
                base = generate_balanced(d, rng)
                corrupted, ctype = corrupt_balanced(base, rng, force_mismatch=force)
                assert not is_balanced(corrupted), (
                    f"BUG: corruption produced balanced string: {corrupted!r}"
                )
                corruption_stats[ctype] += 1

                # Track whether mismatch corruptions pass the counter test
                if ctype == "mismatch":
                    mismatch_total += 1
                    if passes_counter_test(corrupted):
                        mismatch_counter_test_pass += 1

                examples.append({
                    "input": corrupted,
                    "label": 0,
                    "depth": d,
                    "corruption": ctype,
                })

    rng.shuffle(examples)
    return examples, corruption_stats, mismatch_total, mismatch_counter_test_pass


# ── I/O ───────────────────────────────────────────────────────────────────────


def write_jsonl(examples: List[Dict[str, Any]], path: Path) -> None:
    """Write examples to JSONL. The 'corruption' key is included for
    diagnostics but can be ignored by the model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ── PyTorch Dataset ───────────────────────────────────────────────────────────


class BracketDataset(Dataset):
    """Loads a JSONL file and tokenizes multi-type bracket strings into
    fixed-length integer tensors.

    Each sample returns:
        input_ids      : LongTensor[MAX_SEQ_LEN]
        attention_mask : LongTensor[MAX_SEQ_LEN]
        label          : LongTensor scalar
        depth          : LongTensor scalar
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

        input_ids = tokens + [PAD] * (MAX_SEQ_LEN - length)
        attention_mask = [1] * length + [0] * (MAX_SEQ_LEN - length)

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(ex["label"], dtype=torch.long),
            torch.tensor(ex["depth"], dtype=torch.long),
        )


# ── CLI ───────────────────────────────────────────────────────────────────────


def print_summary(name: str, examples: List[Dict[str, Any]],
                  corruption_stats: Counter, mismatch_total: int,
                  mismatch_counter_pass: int) -> None:
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
    print(f"\n{'=' * 70}")
    print(f"  {name}: {total:,} examples  "
          f"(bal={n_bal:,}  unbal={n_unbal:,}  "
          f"ratio={n_bal / total:.2%})")
    print(f"{'=' * 70}")
    print(f"  {'Depth':>5}  {'Total':>7}  {'Bal':>7}  {'Unbal':>7}")
    print(f"  {'-' * 5}  {'-' * 7}  {'-' * 7}  {'-' * 7}")
    for d in sorted(depth_counts):
        t = depth_counts[d]
        b = depth_label[(d, 1)]
        u = depth_label[(d, 0)]
        print(f"  {d:>5}  {t:>7,}  {b:>7,}  {u:>7,}")

    print(f"\n  Corruption breakdown:")
    for ctype, count in corruption_stats.most_common():
        pct = count / n_unbal * 100 if n_unbal > 0 else 0
        print(f"    {ctype:>20s}: {count:>7,}  ({pct:.1f}%)")

    if mismatch_total > 0:
        pct_pass = mismatch_counter_pass / mismatch_total * 100
        print(f"\n  Mismatch sanity: {mismatch_counter_pass}/{mismatch_total} "
              f"({pct_pass:.1f}%) pass counter test but fail stack test ✓")


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
        examples, corruption_stats, mm_total, mm_pass = generate_split(
            depths=cfg["depths"],
            total_size=cfg["total_size"],
            seed=cfg["seed"],
        )

        # ── Final validation pass ────────────────────────────────────
        for ex in examples:
            s = ex["input"]
            assert len(s) <= 32, f"String too long ({len(s)}): {s!r}"
            # Check all chars are valid brackets
            assert all(ch in TOKEN_MAP for ch in s), f"Invalid chars in: {s!r}"
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
        print_summary(name, examples, corruption_stats, mm_total, mm_pass)

    print("\nDone ✓")


if __name__ == "__main__":
    main()