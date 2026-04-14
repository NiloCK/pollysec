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

- [ ] **A.1** Collapse data splits
  - [ ] `polly/data.py`: one `depths = range(1, D_max + 1)` shared by
        train / val / test. Drop the `test_ood` split entirely (archive the
        file alongside v1 if desired).
  - [ ] Start with `D_max = 30`. Bump if vanilla still saturates.
  - [ ] Sizes: train 200k, val 10k, test 15k (500/depth × 30 depths).
  - [ ] Per-depth class balance (50% balanced / 50% unbalanced) preserved.
  - [ ] Keep the same corruption-type mix from v2 (mismatch-heavy).

- [ ] **A.2** Bump sequence length
  - [ ] `MAX_SEQ_LEN = 66` in both `polly/data.py` and `polly/model.py`
        (1 CLS + 2·30 bracket chars + ≥1 PAD).
  - [ ] Sanity check: `generate_balanced(30, rng)` produces strings of
        length ≤ 65.

- [ ] **A.3** Update `evaluate.py`
  - [ ] Drop ID/OOD split reporting. One test set, per-depth accuracy
        table + curve.
  - [ ] Graceful fallback if legacy `test_id.jsonl` / `test_ood.jsonl`
        exist (v1/v2 checkpoints).

- [ ] **A.4** Archive v2 data under `polly/data/v2/` (like v1 is now).

---

## Phase B — Training

- [ ] **B.1** Pilot: 3k steps, vanilla + looped, one seed, `D_max = 30`.
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

- [ ] Does the exit gate ever fire for anything hard, or is the entropy
      bonus too weak? v2 showed `avg_exit_iter = 4.0` (max) at every
      depth — either the task is too easy to need exit, or the gate is
      functionally disabled.
- [ ] Is the 3-layer vanilla just powerful enough that depth doesn't
      matter? A D.3-style smaller-model run would disambiguate
      "architecture vs. capacity" as the thing being tested.
- [ ] How dataset-size-sensitive is this? v2 used 200k train. If depth-30
      vanilla saturates instantly, maybe the per-depth example count is
      so high that the model memorises canonical forms instead of
      learning the general algorithm.
