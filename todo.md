# Polly — Implementation TODO

> # ⚠️ DEPRECATED — 2026-04-14
> #
> # Superseded by [`todo-v2.md`](./todo-v2.md).
> #
> # v1 task (single bracket type) is a **one-counter language**, not
> # context-free — solvable without a stack. Vanilla hit 100% IID *and* OOD
> # across all seeds, leaving no headroom for looped variants to prove
> # anything. See `todo-v2.md` for the multi-type redesign and rationale.
> #
> # Phase 1 and Phase 2 records below remain accurate as historical state.
> # Phase 3/4 items are obsolete — do not work from them.

> Extracted from `init.md` spec v1.0. Update status as work progresses.
> Legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked/decision needed

---

## Phase 1 — Scaffolding

- [x] **1.1** `data.py` — bracket string data generation
  - [x] Vocab/tokenisation: PAD=0, `(`=1, `)`=2, CLS=3
  - [x] Balanced string generator (depth-exact, kernel + random insertion method)
  - [x] Unbalanced string generator (flip / delete / insert corruption)
  - [x] 50/50 balanced/unbalanced at every depth
  - [x] Generate all four splits with fixed seeds:
    - [x] train.jsonl — depths 1–8, seed 42, 200k examples
    - [x] val.jsonl — depths 1–8, seed 43, 10k examples
    - [x] test_id.jsonl — depths 1–8, seed 44, 10k examples
    - [x] test_ood.jsonl — depths 9–16, seed 45, 10k examples
  - [x] Output format: `{"input": "(()())", "label": 1, "depth": 3}` JSONL
  - [x] Verify: correct depths, correct labels, correct balance per depth
  - [x] Max sequence length 34 tokens (1 CLS + 32 bracket chars + 1 PAD min)

- [x] **1.2** `model.py` — implement V1 (Vanilla) baseline
  - [x] Token embeddings (5 tokens × 64 dim)
  - [x] Positional embeddings (34 positions × 64 dim)
  - [x] Transformer block: pre-norm (RMSNorm), 4-head attention (head dim 16), bidirectional, PAD masking
  - [x] FFN: Linear(64→256), SiLU, Linear(256→64), residual
  - [x] 6 layers, output head: RMSNorm → Linear(64, 2) from CLS position
  - [x] Verify: forward pass runs, loss computes, shapes correct (298,368 params)

- [x] **1.3** `train.py` — training loop
  - [x] Dataset/DataLoader from JSONL (tokenise on load)
  - [x] AdamW, lr=3e-4, weight decay=0.01, betas=(0.9, 0.999)
  - [x] Linear warmup 1000 steps, cosine decay to 1e-5
  - [x] Batch size 128, gradient clipping max norm 1.0
  - [x] Checkpoint every 2000 steps, track val accuracy, save best
  - [x] CLI args: variant name, seed, device
  - [x] Verify: V1 overfits 100 examples to 100% accuracy ✓

---

## Phase 2 — Core Models

- [x] **2.1** V2 (Vanilla + Registers) in `model.py`
  - [x] Register vector r ∈ ℝ⁸, initialised to zeros
  - [x] Register update MLP: concat(h_cls, r) → Linear(72→32) → SiLU → Linear(32→8) + residual
  - [x] Register injection: Linear(8→64) broadcast-added to all positions before each layer
  - [x] Register does NOT feed into output head
  - [x] 6 updates (one per layer, single pass)
  - [x] Verify: forward pass, register updates happen, shapes correct (301,480 params)

- [x] **2.2** V3 (Looped) in `model.py`
  - [x] 6 layers with **tied weights**, applied T times (T=4 training)
  - [x] Multi-iteration loss: w_ℓ = ℓ / sum(1..T), weights [0.1, 0.2, 0.3, 0.4] *(in train.py)*
  - [x] Exit gate: Linear(64→1→sigmoid) on mean-pooled hidden state per iteration
  - [x] Exit gate regularisation: entropy bonus λ=0.01 *(in train.py)*
  - [x] Training: always run all T=4 iterations
  - [x] Inference: exit if p_exit > 0.8, else continue, always stop at T_max
  - [x] Verify: weight tying works, multi-iteration loss computes, exit gate produces probabilities (298,433 params)

- [x] **2.3** V4 (Looped + Registers) in `model.py`
  - [x] Combines V3 looping + V2 register mechanism
  - [x] Register persists across iterations (NOT reset between iterations)
  - [x] 6 intra-iteration updates + cross-iteration carry
  - [x] Verify: register carries across iterations, shapes correct (301,545 params)

- [x] **2.4** Sanity check: all 4 variants reach ~100% on tiny dataset (depths 1–4)
  - [x] vanilla: 100/100 = 100% ✓
  - [x] vanilla_reg: 100/100 = 100% ✓
  - [x] looped: 64/64 = 100% ✓
  - [x] looped_reg: 64/64 = 100% (at epoch 200) ✓

---

## Phase 3 — Full Training Runs

- [ ] **3.1** Run all 12 training runs (4 variants × 3 seeds: 100, 200, 300)
  - [ ] V1 seeds 100, 200, 300
  - [ ] V2 seeds 100, 200, 300
  - [ ] V3 seeds 100, 200, 300
  - [ ] V4 seeds 100, 200, 300
- [ ] **3.2** Training target: 30k steps each (adjust if val loss hasn't plateaued)
- [ ] **3.3** Produce preliminary accuracy-vs-depth plots

> **CPU timing estimates** (measured):
> - vanilla / vanilla_reg: ~6 steps/s → ~83 min per 30k steps
> - looped / looped_reg: ~1.8 steps/s → ~4.5 hrs per 30k steps
> - Total wall clock (sequential): ~22 hrs CPU, much less on Kaggle GPU
>
> **Convenience script:** `bash polly/run_all.sh` — runs all 12, skips existing best.pt

> **DECISION GATE after 3.3:**
> - If V3 ≈ V1 at extrapolation depths → debug looping implementation or adjust task difficulty
> - If V4 ≈ V3 at extrapolation depths → adjust register size / injection / update rule
> - Only proceed to Phase 4 if core hypothesis holds (V4 > V3 > V1 at OOD depths)

---

## Phase 4 — Analysis

- [x] **4.1** `evaluate.py` — accuracy vs. depth, per variant, averaged over 3 seeds ± std *(scaffolded)*
- [ ] **4.2** Compute allocation plot (avg exit iteration vs. input depth for V3/V4)
- [x] **4.3** `probe.py` — linear probes on frozen V4 *(scaffolded)*
  - [ ] Extract register r and h_cls after each iteration (4 snapshots/input)
  - [ ] Probe targets: iteration number, input nesting depth, current max unmatched depth, final answer
  - [ ] Train probes on register vs CLS hidden state separately
- [x] **4.4** `ablate.py` — register interventions on trained V4 *(scaffolded)*
  - [ ] Register zeroing (reset r=0 between iterations)
  - [ ] Register freezing (freeze r after iteration 1)
  - [ ] Register noise (Gaussian, varying σ)
  - [ ] Measure accuracy vs depth for each intervention
- [ ] **4.5** `plot.py` — generate all figures for blog post

---

## Phase 5 — Writeup

- [ ] **5.1** Draft blog post
- [ ] **5.2** Review and revise
- [ ] **5.3** Publish

---

## Infrastructure Notes

- **Env:** `.venv` with torch 2.11.0+cpu (Python 3.12). Kaggle GPU for full runs.
- **Data:** Generated and validated. 4 JSONL files in `polly/data/`.
- **All scripts import-verified:** data.py, model.py, train.py, evaluate.py, probe.py, ablate.py

## Notes / Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-22 | All 4 model variants implemented in single `BracketTransformer` class | Cleaner than 4 separate classes; `variant` param controls architecture |
| 2026-03-22 | CPU-only torch for dev; Kaggle GPU for full runs | Zero budget constraint, ~22hrs CPU vs ~2hrs GPU for all 12 runs |
| 2026-03-22 | looped_reg needs more training steps to converge in overfit test | Expected — 4 iterations × 6 layers × register updates = more optimization surface |