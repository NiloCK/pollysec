# Polly v2 — Multi-Type Brackets TODO

> New track supplementing `todo.md`. Spawned 2026-04-14 after v1 results.
> Legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked/decision needed

---

## Why this exists

v1 training revealed the original task is **provably solvable without a stack**.
Vanilla transformer hit 100% on IID *and* OOD (depths 9–16) across all 3 seeds,
with best checkpoints saved as early as step 2000.

Formal reason: with a single bracket type, a string is balanced iff
`(running_count ≥ 0 at every prefix) AND (final_count == 0)`. That's a
**one-counter language**, not a context-free one. A counter is strictly
weaker than a stack — attention computes the prefix sum in parallel in a
single layer, so depth doesn't cost layers, and `MAX_SEQ_LEN=34` caps depth
at 16 which is well within clean numeric precision.

Consequence: vanilla has no headroom to fail, so looped/reg variants have
nothing to demonstrate. The experiment can't distinguish the hypotheses.

**Fix:** bump the task one complexity tier — introduce multiple bracket
types (`()`, `[]`, `{}`). Now `([)]` has correct per-type and total counts
but is still invalid; recognizing this requires remembering *which* kind of
opener is on top, which is irreducibly stack-structured (context-free, not
one-counter). Vanilla should now degrade with depth; looped has a
hypothesis to prove.

---

## Phase A — Task redesign

- [x] **A.1** Update `data.py`
  - [x] Expand vocab: PAD=0, `(`=1, `)`=2, `[`=3, `]`=4, `{`=5, `}`=6, CLS=7
  - [x] `generate_balanced(depth, rng)` picks a random bracket type at each
        opener; closer must match the corresponding opener (stack discipline)
  - [x] Corruption set must include **type-mismatch swaps** (e.g., replace a
        closer with a different-type closer — counts unchanged, still invalid).
        Keep existing flip/delete/insert too, but mismatch should be a
        meaningful fraction (~50%) to force stack behavior.
        **Result:** ~63% mismatch, ~13% flip, ~13% delete, ~11% insert
  - [x] Regenerate all splits (same seeds, same depth bands, same sizes)
  - [x] Add a sanity check: for each unbalanced example, verify it would
        pass the single-type counter test but fails the typed-stack test —
        at least for the mismatch corruptions.
        **Result:** 100% of mismatch corruptions pass counter test but fail stack test ✓
  - [x] Update `MAX_SEQ_LEN` if vocab/format changes require it — **no change needed**, stays 34
  - [x] v1 data backed up to `polly/data/v1/`
  - [x] `corrupt_balanced` now returns `(string, corruption_type)` tuple; JSONL includes `"corruption"` field

- [x] **A.2** Update `model.py`
  - [x] Expand token embedding to new vocab size (8) — `VOCAB_SIZE = 8`
  - [x] Everything else stays the same (same 4 variants, same params)
  - [x] Sanity-check param counts move only by the embedding delta
        **Result:** +192 params for every variant, exactly matching `(8 - 5) * 64`

- [~] **A.3** Smoke test
  - [~] Re-run overfit-100 sanity: all 4 variants still hit ~100% on tiny set
        **Status:** partial local verification complete before compute was intentionally cut off.
        `vanilla`, `vanilla_reg`, and `looped` hit 100% on tiny-set overfit checks.
        `looped_reg` was started but not completed locally; prefer finishing this on Kaggle/cloud.
  - [x] Generate a dozen examples by hand, confirm mismatch corruptions look right
        (verified via data generation output — mismatch examples all pass counter test)

---

## Phase B — Training

- [ ] **B.1** Pick step budget empirically
  - [ ] Short pilot run (5k steps) on vanilla + one looped seed to see where
        val saturates vs. where OOD starts degrading
  - [ ] Set STEPS per variant based on pilot — don't reuse 30k blindly
  - [ ] Guard in `run_all.sh` already checks `total_steps` on `best.pt`, so
        partial reruns are safe

- [ ] **B.2** Full 12 runs (4 variants × 3 seeds) on chosen budget

---

## Phase C — Analysis

- [ ] **C.1** Re-run `evaluate.py --all`
  - [ ] Expect: vanilla degrades on deep OOD; looped holds up better (that's
        the hypothesis under test). If vanilla still hits 100% OOD, the task
        is *still* too easy — escalate (longer sequences, more bracket
        types, harder corruption distribution).

- [ ] **C.2** Keep v1 results as a comparison point
  - [ ] Don't delete v1 checkpoints — the single-type → multi-type delta is
        itself a finding worth writing up
  - [ ] Maybe: separate figure showing "vanilla solves single-type OOD,
        fails multi-type OOD" as the motivating result

- [ ] **C.3** `probe.py` and `ablate.py` — re-run once multi-type checkpoints exist

---

## Open questions

- [ ] Does the exit-gate / looped mechanism actually help, or does it also
      cap at the same depth ceiling as vanilla (precision/representation
      limits rather than algorithmic ones)?
- [ ] How much does corruption distribution matter? If 100% of unbalanced
      examples are type-mismatches, does the model just learn "detect
      mismatch" and ignore count? Need a balanced corruption mix.
- [ ] `MAX_SEQ_LEN=34` still caps achievable depth at ~16. If multi-type
      vanilla still succeeds there, do we need longer sequences (and a
      plan for position-embedding extrapolation) to see it break?
