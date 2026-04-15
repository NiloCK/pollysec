# Polly Wants a Second to Think About That

**Bracket matching with looped transformers and architectural registers.**

An empirical investigation into whether looped (recursive) transformer architectures spontaneously develop internal control state — specifically, whether the model learns to use dedicated architectural "registers" as recursion counters when solving depth-variable problems.

> **Status (v3 / 2026-04-14):** single-range depth-curve probe. Training and test both sample from depths 1..D_max (currently 45). The experimental signal is a per-depth accuracy curve: where does each architecture's line start to fall? See `todo-v3.md` for design rationale. v1 (single-type) and v2 (IID/OOD-split multi-type) are deprecated — see banners in `todo.md` / `todo-v2.md`.

---

## Current task (v3)

**Binary classification:** given a string of `()`, `[]`, `{}` (3 bracket types), output `balanced` or `unbalanced`. Stack discipline is required — `([)]` is balanced by per-type counts but invalid.

- Depths: 1–45, uniformly sampled in every split.
- `MAX_SEQ_LEN = 96` (1 CLS + 2·45 brackets + PAD).
- 50/50 balanced/unbalanced per depth. Corruption mix ~65% type-mismatch, ~15% delete, ~15% flip, ~5% insert.

| Split | Size | Per-depth |
|---|---|---|
| `train.jsonl` | 225,000 | 5,000 |
| `val.jsonl`   | 18,000  | 400 |
| `test.jsonl`  | 18,000  | 400 |

Regenerate with: `python -m polly.data`

---

## Model variants

Four variants forming a 2×2 grid — looping (yes/no) × registers (yes/no):

| Variant        | Looping     | Registers | ~Params | Description                              |
|----------------|-------------|-----------|---------|------------------------------------------|
| `vanilla`      | No          | No        | 298K    | Baseline transformer                     |
| `vanilla_reg`  | No          | Yes       | 301K    | Registers w/o looping (control)          |
| `looped`       | Yes (4 iter)| No        | 298K    | Weight-tied layers, rerun 4×             |
| `looped_reg`   | Yes (4 iter)| Yes       | 301K    | Looped + cross-iteration register        |

Hidden dim 64, 4 heads, bidirectional attention, classification from CLS.

---

## Playing with trained models locally

Training happens on Kaggle (see "Cloud training" below). Checkpoints come back as `kaggle_output/checkpoints/{variant}_seed{seed}/best.pt`. The eval/probe/ablate scripts respect env vars so you can point them at any checkpoint tree without symlinking or moving files.

### Setup

```bash
source .venv/bin/activate                                   # torch + deps
export POLLY_CHECKPOINT_DIR=$(pwd)/kaggle_output/checkpoints # where Kaggle put best.pt
# POLLY_DATA_DIR defaults to polly/data/ — only override if you want v2/v1 data
```

### Per-depth accuracy (the main figure)

```bash
# Single run — table of per-depth accuracy (+ avg exit iter for looped)
python -m polly.evaluate --variant vanilla --seed 100 --device cpu
python -m polly.evaluate --variant looped  --seed 100 --device cpu

# All variants × seeds — aggregated mean±std per depth, writes summary JSON
python -m polly.evaluate --all --device cpu

# Force looped to run all 4 iterations (ignore exit gate) — diagnostic
python -m polly.evaluate --variant looped --seed 100 --force-all-iters
```

Eval JSONs land in `polly/figures/eval_*.json` (gitignored).

### Quick per-depth curve from a training log

Faster than running evaluate — just scrapes the `final` record that training writes:

```bash
for v in vanilla looped; do
  echo "=== $v ==="
  python3 -c "
import json
rec = [json.loads(l) for l in open(f'kaggle_output/checkpoints/${v}_seed100/log.jsonl') if '\"type\": \"final\"' in l][0]
for d in range(1, 46):
    k = f'val_acc_depth_{d}'
    if k in rec:
        print(f'  {d:>2}: {rec[k]:.3f}')
"
done
```

### Probing + ablation (register-specific, `looped_reg` only)

Both scripts are hard-targeted at the `looped_reg` variant — they exist to
interrogate the cross-iteration **register vector**, which only that variant
has. They assume a trained `looped_reg_seed{N}/best.pt` checkpoint exists.
Neither takes a `--variant` flag.

```bash
# Linear probes on the register: does r encode iteration #, nesting depth,
# or the final answer? Also probes CLS hidden state for comparison.
python -m polly.probe --seed 100 --device cpu

# Causal ablations on the register between iterations:
#   (1) zero r after each iter   (2) freeze r to its iter-1 value
#   (3) inject Gaussian noise into r
python -m polly.ablate --seed 100 --device cpu
```

Neither script is meaningful for `vanilla`, `vanilla_reg`, or `looped` —
they don't maintain a cross-iteration register state to probe or ablate.
If you want comparable diagnostics for non-register variants, that's a
separate build (e.g. probing CLS-hidden across iterations for `looped`).
Both scripts are `POLLY_CHECKPOINT_DIR`-aware.

> **v3 heads-up:** these scripts predate the v3 depth-range expansion
> (d=60) and the Phase E PonderNet loss. They should still run against v3
> checkpoints but may hardcode depth ranges (e.g. 1–16) in their probe
> targets — worth skimming before trusting the output.

### Heads-up on checkpoint compatibility

Position embeddings are sized by `MAX_SEQ_LEN` (currently 96). A checkpoint trained at `MAX_SEQ_LEN=34` (v1) or `66` (v3 early) will **not load** against the current model — shape mismatch on `pos_emb.weight`. To eval historical checkpoints, temporarily revert `MAX_SEQ_LEN` in `polly/model.py` + `polly/data.py` to match the training run, or git-checkout the commit that produced them.

---

## Cloud training (Kaggle)

Local CPU training is prohibitively slow for this model — use Kaggle's free P100. The dataset + kernel are already configured:

```bash
# 1. Push fresh dataset (excludes archives + checkpoints)
STAGE=$(mktemp -d)
cp dataset-metadata.json "$STAGE/"
rsync -a --exclude '__pycache__' --exclude 'checkpoints' --exclude 'figures' \
    --exclude 'data/v1' --exclude 'data/v2' --exclude 'data/v3-d30' \
    polly "$STAGE/"
kaggle datasets version -p "$STAGE" -m "data refresh" --dir-mode zip
rm -rf "$STAGE"
kaggle datasets status colinmorgankennedy/pollysec-pkg   # wait for 'ready'

# 2. Push kernel (runs kaggle/run_kaggle.py)
kaggle kernels push -p kaggle/

# 3. Watch
kaggle kernels status colinmorgankennedy/pollysec-train
# UI: https://www.kaggle.com/code/colinmorgankennedy/pollysec-train

# 4. When done, pull outputs
kaggle kernels output colinmorgankennedy/pollysec-train -p ./kaggle_output
```

Edit `kaggle/run_kaggle.py` to toggle `PILOT` (2 variants × 1 seed × 3k steps) vs. the full 12-run sweep. Kaggle kernel runs on a P100 with torch downgraded to 2.3.1 at runtime (the base image's torch drops sm_60).

---

## Repo structure

```
polly/
├── README.md          # this file
├── __init__.py
├── data.py            # generation + BracketDataset (respects POLLY_DATA_DIR)
├── model.py           # all four variants (MAX_SEQ_LEN lives here too)
├── train.py           # training loop (respects POLLY_CHECKPOINT_DIR)
├── evaluate.py        # per-depth accuracy / aggregation
├── probe.py           # activation probes on trained checkpoints
├── ablate.py          # component ablations
├── run_all.sh         # local 12-run sweep (if you're brave)
├── data/              # v3 splits + v1/v2/v3-d30 archives
├── checkpoints/       # local training output (gitignored)
└── figures/           # eval/probe JSON + plots (gitignored)

kaggle/
├── kernel-metadata.json
└── run_kaggle.py      # kernel entrypoint — handles sm_60 torch + symlinks
```

---

## What success looks like (v3)

- **Per-depth curve shows a gap.** `looped` / `looped_reg` maintain accuracy at deeper depths where `vanilla` breaks down.
- **Exit gate meaningfully fires.** `avg_exit_iter` grows with depth — looping buys compute when compute is needed, not always.
- **Probing confirms register encodes recursion state.** A linear probe on the register's value predicts current nesting depth / iteration number substantially above chance.
- **Ablating the register destroys deep-depth accuracy** while preserving shallow performance.

If the per-depth curves overlap at every depth, looping's architectural story didn't pan out on this task — also a real finding, and the cue to escalate (see `todo-v3.md` Phase D: ListOps / arithmetic eval).

---

## References

- Ouro / LoopLM (ByteDance, arXiv:2510.25741)
- Mixture-of-Recursions (Google, 2026)
- LoopFormer, SpiralFormer (late 2025 / early 2026)
