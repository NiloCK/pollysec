# Polly v4 — ListOps, and a Reframe of What the Register Is For

> **DEPRECATED 2026-04-15 — superseded by `todo-v5.md`.**
>
> **Postmortem.** Phase A (ListOps migration) and Phase B (training-stability
> instrumentation + loss retune) both landed successfully. Phase C.1 pilot
> (vanilla + looped_reg × seed 100 × 10k steps) *ran* — see `results/runs.md`
> — but exposed a structural problem loss tuning can't reach.
>
> Under ponder loss (λ_p=0.05, β warmup), looped_reg hit val_acc 0.25 (vs
> vanilla 0.45) with per-iter CE monotonically *worsening* across iterations
> and exit_dist collapsed to iter 1. Under uniform loss it matched vanilla
> (0.452) but per-iter CE collapsed to identity — tied-weight stack found
> the trivial fixed-point. Root cause: the residual stream at layer 6 has to
> be simultaneously "decodable by the output head" and "re-encodable by layer
> 1 of iter t+1," which is a contradiction in what layer weights are for.
> v5 fixes this architecturally by splitting encoder / interpreter / decoder.
>
> Also flagged during Phase C: **DG-1** — the ListOps generator's length cap
> made d≥5 expressions structurally simpler than d=4, inverting the expected
> depth curve. Fixed locally (`MAX_SEQ_LEN = 256`, `max_tokens = 248`); data
> regenerated. v5 C.1 runs pick up against the well-formed data.
>
> Phase D/E plans carry forward into v5 largely unchanged.
>
> ---
>
> Supersedes `todo-v3.md`. Spawned 2026-04-15 after v3 bracket runs saturated
> vanilla and failed to train any looped variant.
> Legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked/decision needed

---

## Why v4 (what v3 taught us)

1. **Bracket-matching is too easy.** At `D_max = 60`, a 6-layer bidirectional
   300K-param vanilla transformer hits `val_acc = 1.0` at every depth 1–60 in
   10k steps. Formal-language hierarchy (bracket languages as CFLs) doesn't
   bind when the model has unbounded width and bidirectional attention — the
   balanced-vs-unbalanced decision collapses to counter-plus-LIFO-check that
   fits comfortably in TC⁰.
2. **Looped (any config) did not train on brackets.** v3 final: per-iter
   accuracy flat at ~0.55 across all four iterations, exit_dist collapsed
   forward (40% mass on iter 1), forced-iter-4 eval at 56% overall. Two
   compounding failures: the PonderNet loss (`task = E[CE_t]` + `KL(exit_dist ‖
   Geo(0.3))`) front-loaded exit mass because later iters weren't learning
   anything the loss could reward; and pre-Phase-E looped code also exhibited
   hardcoded-deterministic-looping issues (details lost — flagged as ongoing
   audit target).
3. **Reframe: registers are scaffolding, looping is the phenomenon.** The
   bitter-lesson hope is that `looped` alone develops cross-iteration state in
   the residual stream. `looped_reg` is a de-risking affordance: a dedicated
   low-resistance channel for passback signal, to be interpreted via probing
   rather than treated as the target architecture. `vanilla_reg` is not
   scientifically interesting under this framing and is dropped.

Under the reframe the variant set collapses from 2×2 to three:

| Variant       | Looping | Register | Role                                  |
|---------------|---------|----------|---------------------------------------|
| `vanilla`     | No      | No       | Baseline; establishes task ceiling.   |
| `looped`      | Yes     | No       | Bitter-lesson test.                   |
| `looped_reg`  | Yes     | Yes      | Scaffolded test; easier target.       |

`vanilla_reg` is dropped. If it ever earns back an ablation role, add it then.

---

## Phase A — Task migration to ListOps

ListOps (Nangia & Bowman 2018; popularised by Tay et al. 2020 LRA) is the
canonical hard case for this question: nested operations (`MIN`, `MAX`, `MED`,
`SUM_MOD`) over digits 0–9, output a single digit. Answer is a function of the
computed value at every subtree — no counter cheat reduces it.

- [x] **A.1** Data generator (`polly/data.py` full rewrite)
  - [x] Recursive expression generator parametrised by max depth `D_max` and max
        args-per-op `A_max`.
  - [x] Pilot at `D_max=6, A_max=5`; escalate if vanilla saturates.
  - [x] Single-range sampling: train/val/test all draw from `1..D_max`. No
        OOD split (v3 lesson — depth is the signal, not distribution shift).
  - [x] Per-depth class balance across the 10 possible outputs.
        **Result:** Perfect balance — 4,000 per (depth, label) in train; 300 per bucket in val/test.
  - [x] Sizes: 240k train / 18k val / 18k test.
  - [x] Every example double-verified: tree evaluation matches re-parsed evaluation.
  - [x] `BracketDataset` alias retained for backward compat.

- [x] **A.2** Tokeniser / vocabulary
  - [x] Tokens: 10 digits (0–9) + 4 ops + `[`, `]` + CLS + PAD = 18 tokens.
  - [x] `VOCAB_SIZE = 18`, `TOKEN_MAP` updated in `polly/data.py`.
  - [x] `MAX_SEQ_LEN = 128` (budget; depth-6/arg-5 expressions fit comfortably).
  - [x] Public API: `evaluate_expression(tokens)` and `evaluate_expression_string(s)`.

- [x] **A.3** Output head
  - [x] `NUM_CLASSES = 10` in `polly/model.py`.
  - [x] `VOCAB_SIZE = 18` in `polly/model.py`.
  - [x] Class renamed to `PollyTransformer` (alias `BracketTransformer` kept).
  - [x] `VARIANTS` reduced to `{"vanilla", "looped", "looped_reg"}`.
  - [x] Re-baseline thresholds: chance is 10%. Val acc of 0.15 is above chance.
        **Verified:** vanilla hit 16.7% val acc after only 200 steps (above 10% chance). ✓

- [x] **A.4** Evaluate.py per-depth aggregation
  - [x] Same structure as v3 (per-depth accuracy table), 1..D_max range.
  - [x] Per-op breakdown added (MIN, MAX, MED, SM) as secondary diagnostic.
  - [x] Single `test.jsonl` (no more test_id/test_ood split).
  - [x] Uses `PollyTransformer` / `ListOpsDataset` imports.

- [x] **A.5** Archive v3 splits under `polly/data/v3/`; v4 splits live at
      `polly/data/{train,val,test}.jsonl`.

  **Param counts (v4):**
  | Variant     | Params  |
  |-------------|---------|
  | vanilla     | 305,728 |
  | looped      | 305,793 |
  | looped_reg  | 308,905 |

---

## Phase B — Training-stability for looped variants

Precondition for everything downstream. Both `looped` and `looped_reg` have
to actually learn *something* before any mechanistic measurement is
meaningful.

- [x] **B.1** Instrumentation: log per-component loss + exit state
  - [x] `compute_loss()` now returns a dict: `total_loss`, `task_loss`,
        `kl_loss`, `per_iter_ce` (list of T floats), `exit_dist_mean` (list
        of T floats).
  - [x] Gradient-norm per layer via `compute_grad_norms()`: total L2 norm +
        per-layer (6 entries) L2 norms. Computed after `unscale_()` but
        before `clip_grad_norm_()`, only every `LOG_EVERY` steps.
  - [x] Verified: init gradient norms show expected pattern — layer 0 (7.8)
        > layer 5 (3.6), consistent with 4× unrolling amplifying early-layer
        gradients.

- [x] **B.2** PonderNet loss re-tune, in order of effort
  - [x] **β warmup.** `β=0` for first 3000 steps (pure `E[CE_t]`), then
        linear ramp over 2000 steps to `β_max=0.01`.
        **Rationale:** prevents KL from shaping exit_dist while model is
        untrained (root cause of v3 exit collapse).
  - [x] **Smaller λ_p.** Changed from 0.3 → **0.05**. Truncated geometric
        prior now puts **85.7% mass on iter 4**, strongly favouring late
        exits during training.
  - [x] **Fallback: simple mean-CE.** `--loss-mode uniform` flag added.
        Uniform weight across iterations, no KL, no exit gate in loss.
        Exit gate still computed for eval-time use. This is the Universal
        Transformers recipe.
  - [x] CLI flags added: `--loss-mode`, `--lambda-p`, `--beta-max`,
        `--beta-warmup`, `--beta-ramp`.

- [x] **B.3** Pre-Phase-E audit
  - [x] Full audit documented in
        `a.aside.audit.looped-training-failure-analysis.md`.
  - [x] **Finding 1:** No hardcoded iteration bypass. ✓
  - [x] **Finding 2:** No train/eval gap. ✓
  - [x] **Finding 3 (root cause):** Exit gate init (sigmoid(0)=0.5) +
        front-loaded prior (λ_p=0.3) → self-reinforcing exit collapse to
        iter 1. Fixes applied in B.2.
  - [x] **"Hardcoded deterministic looping" suspicion from v3:** Not found
        in current code. The failure was purely loss parameterisation, not
        a code bug.

- [ ] **B.4** Sanity check on a toy task
  - Before full ListOps pilot, run `looped` on depth-1-only ListOps (a
    single op applied to ≤5 digits — no nesting). If looped can't learn the
    degenerate case, the issue is not task difficulty.
  - **Status:** Ready to run on Kaggle — too slow for local CPU validation
    (looped with SEQ_LEN=128 runs ~0.5 steps/s on CPU).

---

## Phase C — Variant runs (staged)

Budget-aware staging: confirm task and training before paying for the full
grid.

- [ ] **C.1** Pilot: `vanilla` + `looped_reg` on ListOps `D_max=6`
  - 3k steps, one seed. ~45 min on Kaggle P100 per run.
  - Expected outcome (strong task separation):
    - `vanilla` saturates at some depth d* < 6 (e.g. 1.0 at d=1–3, falling
      to 0.4 at d=6).
    - `looped_reg` maintains higher accuracy at d=5–6.
  - If vanilla saturates everywhere → bump `D_max` to 8 or 10, repeat.
  - If looped_reg also fails to learn → back to Phase B; C.1 is also the
    test for whether B.2 worked.

- [ ] **C.2** Add `looped` (bitter-lesson test)
  - Only runs once C.1 shows a capability gap between vanilla and
    looped_reg. Otherwise there's nothing to attribute to looping alone.
  - 3k-step pilot, one seed. Compare looped vs looped_reg: if looped closes
    most of the gap on its own, scaffolding claim is weaker (but bitter-lesson
    result is stronger). If looped fails where looped_reg succeeds,
    scaffolding is the story.

- [ ] **C.3** Full sweep
  - 3 variants × 3 seeds × full step budget (empirical from pilot).
  - Per-depth accuracy curves, avg exit iter per depth, per-op breakdown.

---

## Phase D — Mechanistic analysis (requires trained models)

- [ ] **D.1** Per-depth accuracy curves
  - Primary figure: x = depth, y = accuracy, one line per (variant × seed),
    error bars over seeds.
  - Secondary figure: `avg_exit_iter` per depth for looped variants — does
    compute scale with difficulty, or is it pinned to T regardless?

- [ ] **D.2** Cross-iteration probes (both `looped` and `looped_reg`)
  - On `looped`: linear probe on CLS hidden state at the boundary between
    iterations. Does it encode partial-evaluation state (subtree values
    already computed)? Nesting depth remaining? Iteration count?
  - On `looped_reg`: same probes, targeted at the register vector instead.
  - This is where the bitter-lesson vs scaffolding question resolves:
    - If `looped`'s CLS hidden probes strongly for recursion state →
      transformers develop cross-iter state on their own (bitter-lesson win,
      register is a footnote).
    - If only `looped_reg`'s register probes strongly → registers are the
      path of least resistance and scaffold the capability.

- [ ] **D.3** Register ablations (`looped_reg` only)
  - Zero / freeze / noise-inject the register between iterations at eval
    time.
  - If deep-depth accuracy collapses under ablation but shallow is
    preserved, the register is specifically carrying depth-dependent state.

---

## Phase E — Fallbacks

- [ ] **E.1** Shrink the model (matched across variants).
  - 2 layers, dim 32, 2 heads (~40K params). Tests whether vanilla's
    brute-force capacity is masking the depth-ceiling we're trying to measure.
- [ ] **E.2** Nested-parenthesis arithmetic (`(3 + (4 * 2)) - 7`).
  - Further departure from brackets — output is a number, requires real
    expression-tree evaluation. Use only if ListOps also fails to separate.
- [ ] **E.3** Extended probing / interp
  - Attention-pattern visualisation across iterations for `looped`.
  - Head-level ablation: which heads carry cross-iter information?

---

## Open questions carried forward

- [x] **Pre-Phase-E looped training failure mechanism.** ~~Suspected hardcoded
      deterministic looping; actual mechanism not yet identified.~~
      **Resolved in B.3:** Root cause was PonderNet loss parameterisation
      (front-loaded exit collapse), not a code bug. Fixes applied in B.2.
- [ ] **Is there a minimum model scale below which looping can't be learned
      at all?** E.1 will partially answer this. If even looped_reg fails at
      dim 32, looping may need width for cross-iter signal to have somewhere
      to live.
- [ ] **Does the register mechanism need to be loopback-exclusive?** Under
      the scaffolding frame, current impl (register fires every layer) is
      fine — the scientific claim rests on probe evidence, not on
      architectural exclusivity. Revisit if probes on `looped_reg`'s register
      are confounded by within-iter side-channel usage.

---

## Kaggle handoff checklist

Ready to run on Kaggle / cloud. All code changes are local-tested.

1. Upload repo (or `git push` if connected)
2. `pip install torch` (GPU wheel)
3. Verify data exists: `ls polly/data/train.jsonl` (240k lines)
4. **B.4 toy sanity:** `python -m polly.train --variant looped --seed 100 --steps 500 --batch-size 64 --loss-mode uniform`
   — should see acc climbing above 10% within 200 steps on depth-1 examples
5. **C.1 pilot:** `python -m polly.train --variant vanilla --seed 100 --steps 3000 --batch-size 128`
   then `python -m polly.train --variant looped_reg --seed 100 --steps 3000 --batch-size 128 --loss-mode ponder`
6. Evaluate: `python -m polly.evaluate --all`
7. Check per-depth accuracy table — does vanilla degrade at d=5–6? Does looped_reg hold up?

**Estimated GPU time:** ~45 min per 3k-step run on P100, ~2 hrs for initial C.1 pilot pair.