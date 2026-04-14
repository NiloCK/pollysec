# Polly Wants a Second to Think About That

**Bracket matching with looped transformers and architectural registers.**

An empirical investigation into whether looped (recursive) transformer architectures spontaneously develop internal control state — specifically, whether the model learns to use dedicated architectural "registers" as recursion counters when solving depth-variable problems.

## Quick Start

```bash
# Set up environment
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Generate data (deterministic, run once)
python polly/data.py

# Train a single variant
python -m polly.train --variant vanilla --seed 100
python -m polly.train --variant looped_reg --seed 100

# Evaluate
python -m polly.evaluate --variant vanilla --seed 100
python -m polly.evaluate --all
```

## Model Variants

Four variants form a 2×2 grid — looping (yes/no) × registers (yes/no):

| Variant | Looping | Registers | Params | Description |
|---|---|---|---|---|
| `vanilla` | No | No | ~298K | Baseline 6-layer transformer |
| `vanilla_reg` | No | Yes | ~301K | Registers without looping (control) |
| `looped` | Yes (4 iter) | No | ~298K | Weight-tied layers, rerun 4× |
| `looped_reg` | Yes (4 iter) | Yes | ~301K | Looped + cross-iteration register |

All variants use: hidden dim 64, 4 attention heads, 6 layers, bidirectional attention, classification from CLS token.

## Task

**Binary classification:** Given a string of `(` and `)`, output `balanced` or `unbalanced`.

- **Training depths:** 1–8
- **Test depths:** 1–16 (depths 9–16 are out-of-distribution)

The key question: does the looped + register variant generalise to depths beyond training, and does the register encode recursion-relevant state (nesting depth, iteration count)?

## Data

Generated deterministically from fixed seeds. Run `python polly/data.py` once.

| Split | Depths | Seed | Size |
|---|---|---|---|
| `train.jsonl` | 1–8 | 42 | 200,000 |
| `val.jsonl` | 1–8 | 43 | 10,000 |
| `test_id.jsonl` | 1–8 | 44 | 10,000 |
| `test_ood.jsonl` | 9–16 | 45 | 10,000 |

50/50 balanced/unbalanced at every depth.

## Training

```bash
# Full training matrix: 4 variants × 3 seeds = 12 runs
for variant in vanilla vanilla_reg looped looped_reg; do
  for seed in 100 200 300; do
    python -m polly.train --variant $variant --seed $seed
  done
done
```

- **Optimiser:** AdamW, lr=3e-4, cosine decay, 1000-step warmup
- **Steps:** 30,000 (batch size 128)
- **Checkpoints:** saved every 2,000 steps; best model by val accuracy

### Compute Estimates (CPU)

| Variant | Steps/sec | Time for 30K steps |
|---|---|---|
| `vanilla` / `vanilla_reg` | ~6 | ~83 min |
| `looped` / `looped_reg` | ~1.8 | ~4.5 hrs |

GPU (Kaggle free tier) will be significantly faster.

## Evaluation & Analysis

```bash
# Accuracy vs. depth (hero plot)
python -m polly.evaluate --all

# Probing analysis (on trained looped_reg)
python -m polly.probe --variant looped_reg --seed 100

# Ablation experiments
python -m polly.ablate --variant looped_reg --seed 100
```

## Repository Structure

```
polly/
├── README.md           # This file
├── __init__.py
├── data.py             # Data generation + BracketDataset
├── model.py            # All four model variants
├── train.py            # Training loop
├── evaluate.py         # Accuracy vs. depth evaluation
├── probe.py            # Probing analysis (TODO)
├── ablate.py           # Ablation experiments (TODO)
├── plot.py             # Figure generation (TODO)
├── data/               # Generated JSONL splits
├── checkpoints/        # Saved models
└── figures/            # Generated plots
```

## What Success Looks Like

- `looped_reg` maintains >80% accuracy at depths 9–16 (OOD)
- `vanilla` collapses to <30% at OOD depths
- Probing reveals the register cleanly encodes nesting depth and iteration count
- Ablating the register destroys extrapolation while preserving in-distribution performance

## References

- Ouro/LoopLM (ByteDance, arxiv 2510.25741)
- Mixture-of-Recursions (Google, 2026)
- LoopFormer, SpiralFormer (late 2025 / early 2026)