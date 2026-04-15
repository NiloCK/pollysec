# Polly v3 — Depth-Curve as Architectural Probe

> Supersedes `todo-v2.md`. Spawned 2026-04-14 after v2 pilot results.
> Legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked/decision needed

---

## What v2 taught us

- **Multi-type brackets (3 types) did not make the task hard enough.** Vanilla
  hit `val_acc = 1.0` across every IID depth (1–8) AND every OOD depth (9–16),
  after only ~3k steps. Looped hit the same ceiling, exiting at the last
  iteration for every input — the gate never fires, because there's nothing
  hard enough to gate on.
- **IID/OOD framing is the wrong frame for the real question.** We're not
  testing distribution shift; we're testing architectural capacity. Splitting
  train and test by depth introduces a length-generalization confound
  (position embeddings at positions the model never saw during training) that
  would muddy any depth-ceiling result.
- **`MAX_SEQ_LEN=34` caps depth at 16, which is well within a transformer's
  "brute force via parallel attention" regime.** We need to push depth well
  past where vanilla can cheat, and we need the model to have seen every
  position during training so the only thing being tested is whether it can
  do stack-structured computation — not whether its pos_emb knows what
  position 85 means.

---

## New experimental frame

**Question:** At what nesting depth does a vanilla transformer's accuracy
fall off, and does a looped / register-augmented architecture push that
cliff further out?

**Design:** one depth range for everything. Train, validate, and test are
all iid samples from depths `1..D_max`. Per-depth accuracy on a held-out
test set is the experimental signal — a curve, not a scalar.

**Why this is cleaner:**
- Every position gets gradient during training. Pos-emb extrapolation is no
  longer a confound.
- The "hard" and "easy" depths live in the same distribution, so we are
  measuring the model's representational ceiling, not its ability to
  generalize out of distribution.
- If vanilla saturates everywhere, we crank `D_max` and retry. The shape of
  the curve — where each architecture's accuracy starts to dip — is the
  finding.

---

## Phase A — Task redesign (single-range)

- [x] **A.1** Collapse data splits
  - [x] `polly/data.py`: single `depths = range(1, D_max + 1)` shared by
        train / val / test. `test_ood` dropped.
  - [x] D_max progression: 30 → 45 (vanilla saturated at 30, dipped to
        0.845 at 45) → **60** (current, to push vanilla further down).
        Train 240k (4000/depth), val 18k (300/depth), test 18k (300/depth).
        Per-depth class balance + mismatch-heavy corruption retained.
  - [x] v3-d30 and v3-d45 archived alongside v1/v2.

- [x] **A.2** Bump sequence length
  - [x] `MAX_SEQ_LEN` progression: 66 (d=30) → 96 (d=45) → **128** (d=60).
        Depth-60 string has len 120 ≤ 126 (MAX-2). Final-pass assert is
        `len(s) <= MAX_SEQ_LEN - 2`.

- [x] **A.3** Update `evaluate.py`
  - [x] `build_test_loaders` prefers v3 `test.jsonl`; falls back to
        v1/v2 `test_id.jsonl` + `test_ood.jsonl`.
  - [x] Dropped ID/OOD split in both single-run table and `--all` aggregation.
        Replaced `id_mean_acc`/`ood_mean_acc` with single `overall_mean_acc`.

- [x] **A.4** v2 data archived under `polly/data/v2/`; v3 splits live at
      `polly/data/{train,val,test}.jsonl`.

---

## Phase B — Training

- [~] **B.1** Pilot: 3k steps, vanilla + looped, one seed, `D_max = 30`.
      Ready to run on Kaggle — see "How to retrain on Kaggle" below.
  Goal: observe where (if anywhere) vanilla's per-depth accuracy starts to
  dip. Outcomes:
  - Vanilla saturates at 1.0 everywhere → bump `D_max` to 45, rerun pilot.
  - Vanilla shows a clear cliff (e.g. accuracy < 0.9 by some depth k) →
    proceed to full sweep.
  - Both variants track each other → looping gives no architectural lift
    on this task; that's a real result, write it up.

- [ ] **B.2** Once `D_max` is right: full 4 × 3 = 12 runs at chosen step
  budget (set empirically from pilot — may still be <30k).

---

## Phase C — Analysis

- [ ] **C.1** The curve is the figure
  - [ ] Per-depth accuracy, x = depth, y = accuracy, one line per
        (variant × seed), seeds averaged with error bars.
  - [ ] Same plot for looped's `avg_exit_iter` per depth — does the gate
        actually fire more for deeper inputs? If yes, looping is buying
        real compute when it's needed.

- [ ] **C.2** Probe + ablate on the trained checkpoints (after the curve
  exists). `probe.py` to see if attention heads show stack-like behaviour
  at deep inputs; `ablate.py` to test what breaks when we neuter heads.

---

## Phase E — Make "looped" actually dynamic (**implemented 2026-04-15**)

Per Open Questions `[!]`, the current looped variant runs T=4 every time. A
fair comparison needs compute to be *conditional on input difficulty*, not
fixed. Sketch of the minimum-viable rewrite:

**Loss — replace current task+gate loss with PonderNet-style KL.**

In `polly/train.py:compute_loss`, the looped branch currently does
`Σ_ℓ w_ℓ · CE_ℓ  +  (-λ·H(exit_dist))` with hand-chosen `w_ℓ = [0.1, 0.2,
0.3, 0.4]`. Replace with:

```python
exit_dist = compute_exit_distribution(exit_probs)    # (B, T), already exists
task_loss = (exit_dist * torch.stack(ce_per_iter, dim=1)).sum(dim=1).mean()
prior = geometric_prior(lambda_p=0.3, T=T).to(device)  # (T,)  ~[0.3, 0.21, 0.147, ...]
kl_loss = F.kl_div(exit_dist.clamp(min=1e-8).log(), prior, reduction="batchmean")
total_loss = task_loss + BETA * kl_loss
```

Where:
- `task_loss` = expected CE under the model's exit distribution (no
  hand-chosen w_ℓ — the model decides).
- `kl_loss` pulls the exit distribution toward a geometric(λ_p) prior.
  λ_p=0.3 → expected exit iter ≈ 3.3; tune λ_p to set the compute budget.
- BETA (~0.01) controls strength. Drop the existing entropy bonus.

**Exit mechanics — per-sample break at eval, soft exit at train.**

In `polly/model.py:forward`, keep computing all T iterations at train time
(soft exit via `exit_dist` weighting in loss — see above). At eval time,
switch from the batch-level `(exit_prob > 0.8).all()` to per-sample:

```python
# at eval time, maintain a "still alive" mask
alive = torch.ones(B, dtype=torch.bool, device=device)
for t in range(T):
    h_active = h[alive]                # only compute for living samples
    h_active, r_active, _ = self._run_layers(h_active, mask[alive], r[alive])
    h[alive] = h_active                # scatter back
    # ... logits and exit_prob on h_active ...
    exiting = (exit_prob_t > threshold)
    alive[alive.clone()] &= ~exiting   # mark exited samples
    if not alive.any():
        break
```

Actual wall-clock savings at inference proportional to avg exit iter.

**Estimated effort:** ~40 LOC across `train.py` + `model.py`, plus
`compute_exit_distribution` already exists and returns what we need. New
helpers needed: `geometric_prior(lambda_p, T)`.

**Landed changes (2026-04-15):**
- `polly/train.py`: added `geometric_prior(λ_p, T, device)`. Replaced the
  looped branch of `compute_loss` with PonderNet objective
  (`task = E_{t~exit_dist}[CE_t]` + `β · KL(exit_dist ‖ Geo(λ_p))`).
  Dropped the entropy bonus. Tunables: `DEFAULT_PONDER_LAMBDA_P = 0.3`,
  `DEFAULT_PONDER_BETA = 0.01`.
  Truncation note: with T=4 the untruncated E[t]=1/λ_p=3.33 collapses to
  E[t]≈2.07 after renormalisation. If you want heavier compute, use
  λ_p≈0.1 (gives E[t]≈2.8 truncated); if lighter, λ_p≈0.5 (E[t]≈1.5).
- `polly/model.py`: removed batch-level `(exit_prob > 0.8).all()` break and
  the `-2` gate-bias init (now zero → sigmoid 0.5). All T iters run at
  both train and eval. Per-sample exit selection moved to `evaluate.py`.
- `polly/evaluate.py`: gathers per-sample logits from each sample's exit
  iter (first t with `exit_prob_t > DEFAULT_EXIT_THRESHOLD`, else T).
  `force_all_iters` still picks last-iter logits as a diagnostic.
- Sanity: `compute_loss` forward+backward verified on random input; prior
  sums to 1; total loss finite.

Known lingering limitation: `train.run_validation` still uses `logits_list[-1]`
for val_acc (not per-sample exit), so `best.pt` selection may slightly
mismatch the final test-time metric. Minor — can fix when it bites.

**Interpretation ground rule:** don't treat Phase E looped numbers as
comparable to pre-Phase-E looped. It's a different architecture. If you
have a pre-E curve saved, keep it labelled separately — the delta between
"fixed T=4" and "dynamic" is itself a measurement worth writing up.

---

## Phase D — Fallbacks (if D-bracketing doesn't separate architectures)

If even `D_max = 45` has vanilla tracking looped, the bracket family is
genuinely too easy for this model size — the "checking one-counter, not
context-free" cheat may extend further than we expect, or the 3-layer
transformer is overparameterised relative to the task. Options, roughly
by effort:

- [ ] **D.1** Harder task in the same family: **ListOps** (Tay et al.) —
      hierarchical MIN/MAX/MED over nested lists. Designed specifically to
      force genuine hierarchy; vanilla is documented to struggle past
      depth ~10. Would reuse most of the training / eval scaffolding.
- [ ] **D.2** **Arithmetic evaluation** with nested parentheses. Requires
      a real expression tree. Larger departure — output is a number, not
      a bit — but same architectural question.
- [ ] **D.3** Shrink vanilla. If an overparameterised model is
      brute-forcing, a 2-layer or single-head vanilla might reveal the
      ceiling the 3-layer version is hiding. Keeps the task, changes the
      probe.

---

## Open questions (inherited + new)

- [!] **The exit gate is functionally hard-coded to T=4 by design.**
      Observed `avg_exit_iter = 4.00` in v2 isn't a task-difficulty
      artefact; it follows from the loss + eval design:
      - `L_gate = -λ · H(exit_distribution)` rewards *uniform* exit
        probability (~0.25 each iter) — well below the 0.8 eval threshold.
      - Eval break requires `(exit_prob > 0.8).all()` across the batch;
        one hesitant sample in a batch of 256 vetoes the break for
        everyone.
      - Per-iter task-loss weights `[0.1, 0.2, 0.3, 0.4]` back-load
        importance onto iter 4, further nudging toward full T=4.
      - Gate bias initialised to −2 → sigmoid ≈ 0.12 (starts *against*
        exiting).
      Net effect: the "looped" architecture we're comparing vanilla
      against is effectively a fixed-depth 4-iter recurrent net, not a
      dynamic-compute model. Any per-depth curve comparison measures
      "looping + more params/compute" vs. "single pass," not "dynamic
      compute allocation." To actually test dynamic looping we'd need:
      - A compute-penalty term in the loss (`+μ · E[t_exit]`).
      - Per-sample exit (apply mask, skip further compute per-row).
      - Either drop the entropy bonus or flip its sign so it rewards
        *decisive* (low-entropy) exit distributions concentrated early.
      Decision: flag the current looped variant as "fixed T=4" when
      interpreting results; consider a D.4 phase if the v3 curve shows
      interesting asymmetries worth following up.
- [ ] Is the 3-layer vanilla just powerful enough that depth doesn't
      matter? A D.3-style smaller-model run would disambiguate
      "architecture vs. capacity" as the thing being tested.
- [ ] How dataset-size-sensitive is this? v2 used 200k train. If depth-30
      vanilla saturates instantly, maybe the per-depth example count is
      so high that the model memorises canonical forms instead of
      learning the general algorithm.
