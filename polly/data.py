"""
data.py — Deterministic data generation for ListOps (Nangia & Bowman 2018).

ListOps is a synthetic task for testing hierarchical reasoning. Nested
prefix-notation expressions over four operations and digits 0–9.
Output is a single digit 0–9.

Operations:
    MIN(args) → minimum of argument values
    MAX(args) → maximum of argument values
    MED(args) → median (lower-middle for even-length lists)
    SM(args)  → (sum of argument values) mod 10

Expression format (string):
    [ OP arg1 arg2 ... argN ]
    where each arg is a digit 0-9 or a nested [ OP ... ] expression.

Vocabulary / Token IDs:
    PAD=0, 0..9 → 1..10, MIN=11, MAX=12, MED=13, SM=14, [=15, ]=16, CLS=17

Usage:
    python polly/data.py              # from the pollysec/ directory
    python polly/data.py --d-max 4    # shallower expressions
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

# ── Vocabulary ────────────────────────────────────────────────────────────────

PAD = 0
CLS = 17

VOCAB_SIZE = 18
MAX_SEQ_LEN = 128

# Token map: string token → integer ID
TOKEN_MAP: Dict[str, int] = {
    "0": 1, "1": 2, "2": 3, "3": 4, "4": 5,
    "5": 6, "6": 7, "7": 8, "8": 9, "9": 10,
    "MIN": 11, "MAX": 12, "MED": 13, "SM": 14,
    "[": 15, "]": 16,
}

# Reverse map for decoding
ID_TO_TOKEN: Dict[int, str] = {v: k for k, v in TOKEN_MAP.items()}
ID_TO_TOKEN[PAD] = "<PAD>"
ID_TO_TOKEN[CLS] = "<CLS>"

OPS = ["MIN", "MAX", "MED", "SM"]
OP_IDS = {11, 12, 13, 14}


# ── Expression tree representation ───────────────────────────────────────────

class Expr:
    """An expression node: either a leaf digit or an op with children."""
    __slots__ = ("op", "children", "value")

    def __init__(self, op: Optional[str] = None, children: Optional[List["Expr"]] = None,
                 value: Optional[int] = None):
        self.op = op            # None for leaf
        self.children = children or []
        self.value = value      # set for leaf (0–9), or computed after eval

    @property
    def is_leaf(self) -> bool:
        return self.op is None

    def depth(self) -> int:
        """Max nesting depth. Leaf=0, single [OP d1 d2]=1."""
        if self.is_leaf:
            return 0
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def token_count(self) -> int:
        """Number of tokens in the string representation."""
        if self.is_leaf:
            return 1  # just the digit
        # [ OP arg1 arg2 ... argN ]  → 2 (brackets) + 1 (op) + sum(children)
        return 3 + sum(c.token_count() for c in self.children)

    def to_tokens(self) -> List[str]:
        """Convert to list of string tokens."""
        if self.is_leaf:
            return [str(self.value)]
        tokens = ["[", self.op]
        for c in self.children:
            tokens.extend(c.to_tokens())
        tokens.append("]")
        return tokens

    def to_string(self) -> str:
        return " ".join(self.to_tokens())


# ── Expression evaluation ────────────────────────────────────────────────────

def _eval_expr(expr: Expr) -> int:
    """Recursively evaluate an Expr tree. Returns integer 0–9."""
    if expr.is_leaf:
        assert expr.value is not None
        return expr.value

    child_vals = [_eval_expr(c) for c in expr.children]

    if expr.op == "MIN":
        result = min(child_vals)
    elif expr.op == "MAX":
        result = max(child_vals)
    elif expr.op == "MED":
        s = sorted(child_vals)
        n = len(s)
        if n % 2 == 1:
            result = s[n // 2]
        else:
            result = s[n // 2 - 1]  # lower-middle
    elif expr.op == "SM":
        result = sum(child_vals) % 10
    else:
        raise ValueError(f"Unknown op: {expr.op!r}")

    assert 0 <= result <= 9, f"Eval result out of range: {result}"
    return result


def evaluate_expression(tokens: List[str]) -> int:
    """Evaluate a ListOps expression given as a list of string tokens.

    This is the public API used by other modules for verification.
    Example: evaluate_expression(["[", "SM", "3", "1", "8", "]"]) → 2
    """
    expr, end = _parse_tokens(tokens, 0)
    assert end == len(tokens), f"Trailing tokens after position {end}: {tokens[end:]}"
    return _eval_expr(expr)


def evaluate_expression_string(s: str) -> int:
    """Convenience: evaluate a space-separated expression string."""
    return evaluate_expression(s.split())


def _parse_tokens(tokens: List[str], pos: int) -> Tuple[Expr, int]:
    """Parse tokens starting at pos, return (Expr, next_pos)."""
    if tokens[pos] == "[":
        # [ OP arg1 arg2 ... ]
        op = tokens[pos + 1]
        assert op in OPS, f"Expected op at pos {pos + 1}, got {op!r}"
        children = []
        i = pos + 2
        while tokens[i] != "]":
            child, i = _parse_tokens(tokens, i)
            children.append(child)
        return Expr(op=op, children=children), i + 1  # skip ]
    else:
        # digit
        d = int(tokens[pos])
        assert 0 <= d <= 9
        return Expr(value=d), pos + 1


# ── Expression generation ────────────────────────────────────────────────────

def generate_expression(target_depth: int, rng: random.Random,
                        a_max: int = 5, max_tokens: int = 120) -> Optional[Expr]:
    """Generate an expression with exactly `target_depth` as max nesting depth.

    Strategy:
        - At depth 0: return a random digit leaf.
        - Otherwise: pick a random op, random number of args [2, a_max].
          One arg slot (the "spine") MUST be a subexpression of target_depth-1.
          Other args are either digits or shallower subexpressions, chosen
          based on remaining token budget and depth budget.

    Returns None if the expression can't fit in the token budget.
    """
    return _gen(target_depth, target_depth, rng, a_max, max_tokens)


def _gen(target_depth: int, remaining_depth: int, rng: random.Random,
         a_max: int, budget: int) -> Optional[Expr]:
    """Internal recursive generator.

    Args:
        target_depth:   the depth this particular subtree must achieve exactly.
        remaining_depth: unused — we track via target_depth.
        rng:            random state.
        a_max:          max args per op.
        budget:         remaining token budget for this subtree.
    """
    if target_depth == 0:
        # Leaf: random digit
        if budget < 1:
            return None
        return Expr(value=rng.randint(0, 9))

    # Need at least: [ OP spine_arg one_more_arg ] = 5 tokens minimum
    # (spine_arg could be a nested expr needing more, but we'll check below)
    if budget < 5:
        return None

    op = rng.choice(OPS)
    n_args = rng.randint(2, a_max)

    # The spine slot — must reach target_depth - 1
    spine_idx = rng.randint(0, n_args - 1)

    # Overhead: [ OP ... ] = 3 tokens
    remaining_budget = budget - 3

    children: List[Optional[Expr]] = [None] * n_args

    # Generate spine first (it has priority for budget)
    # Reserve at least 1 token per non-spine arg
    spine_budget = remaining_budget - (n_args - 1)
    if spine_budget < 1:
        return None

    spine_child = _gen(target_depth - 1, target_depth - 1, rng, a_max, spine_budget)
    if spine_child is None:
        return None

    children[spine_idx] = spine_child
    remaining_budget -= spine_child.token_count()

    # Generate other args
    for i in range(n_args):
        if i == spine_idx:
            continue

        if remaining_budget < 1:
            return None  # can't fit even a digit

        # Reserve 1 token per remaining unfilled slot (excluding current)
        unfilled_after = sum(1 for j in range(i + 1, n_args) if children[j] is None)
        available = remaining_budget - unfilled_after

        if available < 1:
            return None

        # Decide: digit or nested subexpression?
        # If target_depth - 1 >= 1 and we have enough budget, maybe nest.
        # Probability of nesting decreases as budget shrinks.
        max_sub_depth = target_depth - 1  # sub-args can be up to this deep
        # but they can't be exactly target_depth-1 (that's reserved for spine)
        # Actually they CAN be — the constraint is that the OVERALL expr has
        # depth = target_depth, which is guaranteed by the spine. Other args
        # can be anything from depth 0 (digit) to target_depth-1.

        nest_prob = 0.0
        if max_sub_depth >= 1 and available >= 5:
            # Bias toward nesting less as budget shrinks
            nest_prob = min(0.5, available / 30.0)

        if rng.random() < nest_prob:
            # Pick a random depth for this sub-arg: 1..max_sub_depth
            sub_depth = rng.randint(1, max_sub_depth)
            sub_expr = _gen(sub_depth, sub_depth, rng, a_max, available)
            if sub_expr is not None and sub_expr.depth() <= max_sub_depth:
                children[i] = sub_expr
                remaining_budget -= sub_expr.token_count()
                continue

        # Fallback: digit
        children[i] = Expr(value=rng.randint(0, 9))
        remaining_budget -= 1

    # Verify all children are filled
    assert all(c is not None for c in children)

    expr = Expr(op=op, children=children)  # type: ignore[arg-type]

    # Verify depth
    actual_depth = expr.depth()
    if actual_depth != target_depth:
        # This shouldn't happen if spine generation is correct, but guard anyway.
        return None

    # Verify token count
    if expr.token_count() > budget:
        return None

    return expr


# ── Split generation ─────────────────────────────────────────────────────────

def generate_split(depths: List[int], total_size: int, seed: int,
                   a_max: int = 5, max_tokens: int = 120) -> List[Dict[str, Any]]:
    """Generate a balanced split of ListOps examples.

    Generates ~2× the needed count per (depth, label) bucket, then subsamples
    to get roughly equal representation across the 10 output labels per depth.

    Args:
        depths:     list of target depths (e.g. [1,2,3,4,5,6])
        total_size: total number of examples desired
        seed:       RNG seed for determinism
        a_max:      max args per op
        max_tokens: max tokens per expression (excluding CLS/PAD)

    Returns:
        List of example dicts with keys: input, label, depth, length
    """
    rng = random.Random(seed)
    per_depth = total_size // len(depths)

    # Target per (depth, label): ideally per_depth / 10
    target_per_bucket = per_depth // 10
    # Generate extra to allow balancing
    gen_target = max(per_depth * 3, target_per_bucket * 25)  # generous overgeneration

    all_examples: List[Dict[str, Any]] = []

    for d in depths:
        # Collect examples bucketed by label
        buckets: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(10)}
        attempts = 0
        max_attempts = gen_target * 10  # safety limit

        while attempts < max_attempts:
            # Check if we have enough in every bucket
            min_bucket = min(len(buckets[lb]) for lb in range(10))
            if min_bucket >= target_per_bucket:
                break

            attempts += 1
            expr = generate_expression(d, rng, a_max=a_max, max_tokens=max_tokens)
            if expr is None:
                continue

            # Evaluate
            label = _eval_expr(expr)
            assert 0 <= label <= 9

            # Verify by re-parsing
            token_list = expr.to_tokens()
            verify_label = evaluate_expression(token_list)
            assert verify_label == label, (
                f"Eval mismatch: tree={label}, parsed={verify_label} "
                f"for {' '.join(token_list)}"
            )

            length = len(token_list)
            assert length <= max_tokens, f"Expression too long: {length} > {max_tokens}"

            ex = {
                "input": " ".join(token_list),
                "label": label,
                "depth": d,
                "length": length,
            }

            if len(buckets[label]) < target_per_bucket * 3:
                # Don't overfill any single bucket
                buckets[label].append(ex)

        # Now subsample to get balanced representation
        depth_examples: List[Dict[str, Any]] = []
        for label in range(10):
            bucket = buckets[label]
            # Shuffle for randomness
            rng.shuffle(bucket)
            take = min(len(bucket), target_per_bucket)
            depth_examples.extend(bucket[:take])

        # If we're short (some labels rare at this depth), pad from overrepresented buckets
        deficit = per_depth - len(depth_examples)
        if deficit > 0:
            # Gather leftover examples from all buckets
            used_counts = {lb: min(len(buckets[lb]), target_per_bucket) for lb in range(10)}
            leftover: List[Dict[str, Any]] = []
            for label in range(10):
                leftover.extend(buckets[label][used_counts[label]:])
            rng.shuffle(leftover)
            depth_examples.extend(leftover[:deficit])

        rng.shuffle(depth_examples)
        all_examples.extend(depth_examples)

    # Final shuffle
    rng.shuffle(all_examples)

    return all_examples


# ── I/O ───────────────────────────────────────────────────────────────────────

def write_jsonl(examples: List[Dict[str, Any]], path: Path) -> None:
    """Write examples to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class ListOpsDataset(Dataset):
    """Loads a ListOps JSONL file and tokenizes space-separated expressions
    into fixed-length integer tensors.

    Each sample returns:
        input_ids      : LongTensor[MAX_SEQ_LEN]  — CLS prepended, PAD appended
        attention_mask : LongTensor[MAX_SEQ_LEN]  — 1 for real tokens, 0 for PAD
        label          : LongTensor scalar (0–9)
        depth          : LongTensor scalar
    """

    def __init__(self, path: str | Path) -> None:
        self.examples = read_jsonl(Path(path))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor,
                                              torch.Tensor, torch.Tensor]:
        ex = self.examples[idx]
        str_tokens = ex["input"].split()
        token_ids = [CLS] + [TOKEN_MAP[t] for t in str_tokens]
        length = len(token_ids)

        assert length <= MAX_SEQ_LEN, (
            f"Tokenized length {length} exceeds MAX_SEQ_LEN {MAX_SEQ_LEN}"
        )

        input_ids = token_ids + [PAD] * (MAX_SEQ_LEN - length)
        attention_mask = [1] * length + [0] * (MAX_SEQ_LEN - length)

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(ex["label"], dtype=torch.long),
            torch.tensor(ex["depth"], dtype=torch.long),
        )


# Backward-compat alias — other modules import BracketDataset
BracketDataset = ListOpsDataset


# ── CLI summary ───────────────────────────────────────────────────────────────

def print_summary(name: str, examples: List[Dict[str, Any]]) -> None:
    """Print per-depth × per-label frequency table."""
    depth_counts: Counter[int] = Counter()
    label_counts: Counter[int] = Counter()
    depth_label: Counter[Tuple[int, int]] = Counter()

    for ex in examples:
        depth_counts[ex["depth"]] += 1
        label_counts[ex["label"]] += 1
        depth_label[(ex["depth"], ex["label"])] += 1

    total = len(examples)
    all_depths = sorted(depth_counts.keys())

    print(f"\n{'=' * 80}")
    print(f"  {name}: {total:,} examples")
    print(f"{'=' * 80}")

    # Header row
    hdr = f"  {'Depth':>5}  {'Total':>7}"
    for lbl in range(10):
        hdr += f"  {lbl:>5}"
    print(hdr)
    print(f"  {'-' * 5}  {'-' * 7}" + f"  {'-' * 5}" * 10)

    for d in all_depths:
        row = f"  {d:>5}  {depth_counts[d]:>7}"
        for lbl in range(10):
            row += f"  {depth_label[(d, lbl)]:>5}"
        print(row)

    # Total row
    row = f"  {'ALL':>5}  {total:>7}"
    for lbl in range(10):
        row += f"  {label_counts.get(lbl, 0):>5}"
    print(row)

    # Per-label percentages
    print("\n  Label distribution:")
    for lbl in range(10):
        c = label_counts.get(lbl, 0)
        pct = c / total * 100 if total > 0 else 0
        print(f"    label {lbl}: {c:>7,}  ({pct:5.1f}%)")


# ── CLI main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ListOps data splits (train / val / test)."
    )
    parser.add_argument("--d-max", type=int, default=6,
                        help="Maximum nesting depth (default: 6)")
    parser.add_argument("--a-max", type=int, default=5,
                        help="Maximum arguments per operation (default: 5)")
    args = parser.parse_args()

    d_max = args.d_max
    a_max = args.a_max
    data_dir = Path(__file__).resolve().parent / "data"

    depths = list(range(1, d_max + 1))

    splits = {
        "train": {"total_size": 240_000, "seed": 42, "filename": "train.jsonl"},
        "val":   {"total_size": 18_000,  "seed": 43, "filename": "val.jsonl"},
        "test":  {"total_size": 18_000,  "seed": 44, "filename": "test.jsonl"},
    }

    max_tokens = MAX_SEQ_LEN - 2  # leave room for CLS + at least 1 PAD → 126
    # Actually we want expressions to fit with CLS prepended and still be < 128.
    # CLS takes 1 slot, expression takes `length` slots, rest is PAD.
    # So expression can be at most MAX_SEQ_LEN - 1 = 127 tokens.
    # But we want at least 1 PAD, so max_tokens = MAX_SEQ_LEN - 2 = 126.
    # Spec says max_tokens=120 as a comfortable margin. Use that.
    max_tokens = 120

    print("ListOps data generation")
    print(f"  D_max={d_max}, A_max={a_max}, depths={depths}")
    print(f"  max_tokens={max_tokens}, MAX_SEQ_LEN={MAX_SEQ_LEN}")

    for name, cfg in splits.items():
        print(f"\nGenerating {name} ({cfg['total_size']:,} examples) …")
        examples = generate_split(
            depths=depths,
            total_size=cfg["total_size"],
            seed=cfg["seed"],
            a_max=a_max,
            max_tokens=max_tokens,
        )

        # ── Validation pass ──────────────────────────────────────────
        n_validated = 0
        for ex in examples:
            s = ex["input"]
            tokens = s.split()

            # All tokens must be in TOKEN_MAP
            for t in tokens:
                assert t in TOKEN_MAP, f"Unknown token {t!r} in: {s}"

            # Re-evaluate and check label
            computed = evaluate_expression(tokens)
            assert computed == ex["label"], (
                f"Label mismatch: computed={computed}, stored={ex['label']} for: {s}"
            )

            # Check length
            assert len(tokens) == ex["length"], (
                f"Length mismatch: {len(tokens)} vs {ex['length']}"
            )

            # Check total fits in MAX_SEQ_LEN with CLS
            assert len(tokens) + 1 <= MAX_SEQ_LEN, (
                f"Expression + CLS exceeds MAX_SEQ_LEN: {len(tokens) + 1}"
            )

            # Check depth by re-parsing the tree
            parsed_expr, _ = _parse_tokens(tokens, 0)
            assert parsed_expr.depth() == ex["depth"], (
                f"Depth mismatch: tree={parsed_expr.depth()}, stored={ex['depth']} for: {s}"
            )

            n_validated += 1

        print(f"  ✓ {n_validated:,} examples validated")

        out_path = data_dir / cfg["filename"]
        write_jsonl(examples, out_path)
        print(f"  → wrote {out_path}  ({len(examples):,} examples)")
        print_summary(name, examples)

    print("\nDone ✓")


if __name__ == "__main__":
    main()