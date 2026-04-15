# Run log

Running log of Kaggle / local training runs. Newest entries at the bottom.
One entry per variant×seed×config. Keep qualitative notes; heavy data lives
in `kaggle_output/checkpoints/{variant}_seed{seed}/log.jsonl`.

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
- **Diagnosis:** Expected shape — strong d=1, sharp drop through d=3, plateau well
  above 10% chance. Monotonic degradation 1→4 of 45pp. ListOps is exhibiting the
  hypothesised depth-ceiling behaviour; task is appropriate.
- **Flag:** d=6 +6pp above d=4. Candidates: (a) 300 ex/depth noise, (b) data-gen
  depth saturation at `A_max=5` making d=6 subtrees effectively shallower. Verify
  by checking a second seed before drawing conclusions from the curve at d≥5.
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
- **Next:** re-run with `--loss-mode uniform` (mean-CE over all iterations,
  no KL). Forces gradient into iters 2–4; tests whether layers can learn to
  refine at all under the tied-weight constraint.
- **Artifacts:** `kaggle_output/checkpoints/looped_reg_seed100/`
