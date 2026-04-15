# Run log

Running log of Kaggle / local training runs. Newest entries at the bottom.
One entry per variant×seed×config. Keep qualitative notes; heavy data lives
in `kaggle_output/checkpoints/{variant}_seed{seed}/log.jsonl`.

## Known data issues

<a id="datagen-length-cap-2026-04-15"></a>
**[DG-1] ListOps generator length cap confounds depth at d≥5** (2026-04-15).
Max char length for every depth ≥ 4 converges to ~281–283, indicating a length
cap in the generator that truncates or constrains deeper trees. At d=6 with
`A_max=5`, expressions that would exceed the budget are forced sparse/narrow
instead of dense fan-out. Net effect: d=6 trees are *structurally simpler*
than d=4 trees (fewer filled subtrees per level), so depth-as-labelled stops
correlating with difficulty past ~d=4. Observed in every run so far: both
vanilla and looped_reg show monotonically *rising* accuracy from d=4 to d=6,
opposite to the hypothesised depth-ceiling story. **All per-depth numbers at
d≥5 should be treated as suspect until data-gen is fixed.** Fix: bump
`MAX_SEQ_LEN` and regenerate (TBD). Consider adding `n_subtrees` as a
difficulty metric alongside nominal depth.

Entry template (copy and edit):

```markdown
## YYYY-MM-DD — <v4 phase> <variant> seed <N>

- **Config:** <steps, loss_mode, λ_p, β_max, β_warmup, β_ramp, batch>
- **Result:** val_acc <x> (baseline / comparison: <y>)
- **Depth curve:** d=[1..D] → [<per-depth acc>]
- **Per-op acc:** MIN / MAX / MED / SM (if logged)
- **Diagnosis:** <what the logs say — per_iter_ce, exit_dist, grad_norms>
- **Next:** <what to try>
- **Artifacts:** <checkpoint path>
```

---

## 2026-04-15 — v4 C.1 pilot, `vanilla` seed 100

- **Config:** 10k steps, bs 128, fp16 amp, default LR schedule
- **Result:** val_acc **0.4527**
- **Depth curve:** d=[1..6] → [0.779, 0.467, 0.363, 0.327, 0.360, 0.420]
- **Diagnosis:** Expected shape through d=4 — strong d=1, sharp drop, plateau well
  above 10% chance. Monotonic degradation 1→4 of 45pp. ListOps is exhibiting the
  hypothesised depth-ceiling behaviour at d=1..4; task is appropriate.
- **Flag — see [DG-1](#datagen-length-cap-2026-04-15):** d=5 and d=6 numbers
  are not comparable to d=1..4. The generator's length cap inflates d≥5 accuracy.
  Treat d=4 as the current effective max.
- **Next:** use as baseline for looped_reg comparisons. Second seed for d=6
  verification when compute allows.
- **Artifacts:** `kaggle_output/checkpoints/vanilla_seed100/`

## 2026-04-15 — v4 C.1 pilot, `looped_reg` seed 100

- **Config:** 10k steps, bs 128, fp32 (amp disabled for looped due to fp16
  overflow in tied-stack forward), `--loss-mode ponder`, λ_p=0.05, β_max=0.01,
  β_warmup=3k, β_ramp=2k
- **Result:** val_acc **0.2521** — **20pp below vanilla**
- **Depth curve:** d=[1..6] → [0.456, 0.235, 0.203, 0.196, 0.202, 0.220]
- **Diagnosis:**
  - Per-iter CE monotonically *worse* with iteration at every step:
    step 9600 → [1.36, 1.53, 1.76, 1.92]. Later iters degrade predictions rather
    than refining them.
  - Exit gate collapsed to iter 1 by step 600 and stayed: end-of-training
    exit_dist ≈ [0.79, 0.07, 0.07, 0.08]. The gate correctly learned that iter 1
    is genuinely best.
  - β ramp at 3k→5k nudged 20pp off iter 1 but couldn't overcome the actual CE
    ordering.
  - Self-reinforcing PonderNet failure: task_loss = E_{t~exit_dist}[CE_t],
    exit_dist ≈ iter 1 → gradient flows only to iter 1 → only iter 1 learns.
- **Aside:** Earlier fp16 NaN bug (looped_reg overflowed under amp due to 24
  layer-applications) was fixed by gating `use_amp` on `variant == "vanilla"`.
  That unblocked this run but isn't the source of the learning failure.
- **Flag — see [DG-1](#datagen-length-cap-2026-04-15):** d≥5 numbers suspect.
- **Next:** re-run with `--loss-mode uniform` (mean-CE over all iterations,
  no KL). Forces gradient into iters 2–4; tests whether layers can learn to
  refine at all under the tied-weight constraint.
- **Artifacts:** `kaggle_output/checkpoints/looped_reg_seed100/` (overwritten
  by subsequent uniform-loss run; archive this before re-running if needed)

## 2026-04-15 — v4 C.1 follow-up, `looped_reg` seed 100, uniform loss

- **Config:** 10k steps, bs 128, fp32, `--loss-mode uniform` (mean CE over
  iters, no KL, exit gate untrained). Same data as prior runs.
- **Result:** val_acc **0.4520** — matches vanilla (0.4527) within noise.
- **Depth curve:** d=[1..6] → [0.829, 0.451, 0.354, 0.321, 0.352, 0.404]
  - d=1 is +5pp vs vanilla (.779); d=2..6 ≈ identical to vanilla.
- **Diagnosis:**
  - **Per-iter CE collapsed to identity across iterations:** step 9600 →
    [1.40, 1.37, 1.36, 1.36]. Tied-weight stack under uniform loss finds the
    trivial fixed-point solution: produce stable output at iter 1, reproduce
    unchanged at iters 2–4. No refinement, no iteration-specific computation.
  - Exit gate frozen at sigmoid(0)=0.5 baseline → `exit_dist = [0.50, 0.25,
    0.125, 0.125]` (geometric from constant p=0.5). Expected under uniform
    (gate isn't in the loss); flagged because it means eval-time per-sample
    exit isn't behaving dynamically — `exit_threshold=0.8` never fires so all
    samples run all 4 iters regardless.
  - Architectural symmetry: tied weights + same input → same output. Without
    training pressure rewarding *different* outputs at different iters, fixed
    point is the local optimum.
- **Net:** looped_reg under uniform loss trains cleanly but provides **no
  depth advantage over vanilla**. Hypothesised looping benefit not observed.
- **Flag — see [DG-1](#datagen-length-cap-2026-04-15):** d≥5 numbers suspect.
- **Next:** fix data gen (DG-1) first so depth curves are interpretable. Then
  revisit exit-gate design + loss: final-iter-only loss (force refinement),
  or iteration-conditional register init (break input-symmetry across iters).
- **Artifacts:** `kaggle_output/checkpoints/looped_reg_seed100/`
