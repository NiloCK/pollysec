# pollysec

Empirical probe of whether **looped transformers** learn to carry useful cross-iteration state, and whether a dedicated **architectural register** scaffolds that capability.

## The question

Fixed-depth transformers have a hard ceiling on recursive problems — a 6-layer model has a fixed compute budget per token. Looped transformers (Ouro / LoopLM, Mixture-of-Recursions, LoopFormer, SpiralFormer; late 2025–early 2026) break that ceiling by rerunning tied layers T times. What nobody has tested: **how does the model carry state across iterations, and does giving it a dedicated channel for that state help it learn?**

## The setup

Task: **ListOps** (nested `MIN`/`MAX`/`MED`/`SUM_MOD` over digits 0–9, output a single digit). Variable-depth expression trees; answer depends on evaluating every subtree — no counter-cheat reduces it. Tiny models (~300K params) on Kaggle free tier.

Three variants (down from 2×2 under the v4 scaffolding reframe):

| Variant       | Looping      | Register | Role                            |
|---------------|--------------|----------|---------------------------------|
| `vanilla`     | —            | —        | Capability baseline.            |
| `looped`      | 4 iter, tied | —        | Bitter-lesson test.             |
| `looped_reg`  | 4 iter, tied | 8-d      | Scaffolded test; easier target. |

Under v5 the stack is partitioned by role: **encoder (2) + interpreter (3, looped) + decoder (1)**. Only the interpreter loops; the register fires only inside it; the decoder reads interpreter state at exit time. PonderNet-style loss trains the exit gate natively.

## What success looks like

- `looped` / `looped_reg` hold accuracy at depths where `vanilla` collapses.
- Linear probes on interpreter hidden state (and on the register for `looped_reg`) predict partial-evaluation state across iterations.
- Ablating the register destroys deep-depth accuracy but not shallow.

If `looped` alone closes the gap, bitter-lesson wins and the register is a footnote. If only `looped_reg` succeeds, registers are the scaffold. Overlapping curves is also a real result.

## Where things live

- [`polly/README.md`](polly/README.md) — how to train, evaluate, probe, ablate.
- `polly/model.py` — all three variants in one class.
- `polly/data.py` — ListOps dataset generation.
- `kaggle/` — cloud training entrypoint (P100, free tier).
- `results/runs.md` — running log of training runs + known data issues.
- `todo-v5.md` — current design rationale. `todo-v4.md` / `todo-v3.md` / earlier are deprecated (see banners).

## Constraints

Zero budget. Kaggle free tier (30 hr/wk GPU) or laptop CPU. Models deliberately tiny; this is a proof-of-concept, not a scaling study.
