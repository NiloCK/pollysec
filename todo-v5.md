# Polly v5 — Encoder / Interpreter / Decoder refactor

> Supersedes `todo-v4.md`. Spawned 2026-04-15 after v4 C.1 runs showed
> looped_reg unable to beat vanilla under either ponder or uniform loss.
> Legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked/decision needed

---

## Why v5 (what v4 taught us)

v4 kept v3's architecture — a single weight-tied stack of 6 TransformerLayers
applied T times, with a shared output head firing every iteration. Two runs
on ListOps (`D_max=6`) exposed a structural problem that neither loss tweak
resolved:

1. **Ponder loss (λ_p=0.05, β warmup 3k→5k, β_max=0.01).** val_acc 0.25 vs
   vanilla 0.45. Per-iter CE got *worse* with iteration
   (1.36 → 1.53 → 1.76 → 1.92 at step 9600); exit_dist collapsed to iter 1 by
   step 600 and never recovered. Gate correctly learned iter 1 was genuinely
   best. Self-reinforcing: task loss flows through exit_dist, exit_dist ≈ iter
   1, only iter 1 learns.
2. **Uniform loss (mean CE across iters, no KL, gate untrained).** val_acc
   0.452 — *matches* vanilla within noise. Per-iter CE collapsed to identity
   (1.40 / 1.37 / 1.36 / 1.36): tied-weight stack found the trivial fixed-point
   — produce stable output at iter 1, reproduce unchanged iters 2–4.

Root cause is architectural, not loss-level. The residual stream at layer 6
has to serve two semantic roles simultaneously:

- **Decodable.** Output head reads CLS from `h_6` and must project to a digit.
- **Re-encodable.** Layer 1 of iter `t+1` consumes the same tensor as input.
  Layer 1's weights were trained on `tok_emb + pos_emb`; asking it to also
  usefully process layer-6-shaped outputs is a contradiction in what layer
  weights are for.

Both loss collapses are symptoms of this dual-role pressure. Under ponder,
only iter 1 gets trained because later iters can't improve on what's already
been flattened into answer-shape. Under uniform, the network discovers that
the cheapest way to minimise mean CE across iters is to make all iters
identical — fixed-point is the local optimum of symmetry.

The v5 fix is architectural: **split the stack by role.** Encoder prepares
structural features once. Interpreter is the looped core, operating in pure
working-state (no answer pressure). Decoder runs once at commit time, mapping
interpreter's final state to logits. Weights are never asked to be
dual-purpose.

---

## Architectural change

```
     [input_ids]
         │
    ┌────▼─────┐
    │ Encoder  │   2 layers, single pass. Produces h_0.
    └────┬─────┘
         │
    ┌────▼─────────────┐
    │ Interpreter      │   3 layers, weight-tied across T=4 iters.
    │ (looped block)   │   Register fires here only (if variant = looped_reg).
    │                  │   After iter t produces h_t, exit gate reads h_t.
    └────┬─────────────┘
         │  (h_{t*} — interpreter state at chosen exit iter)
    ┌────▼─────┐
    │ Decoder  │   1 layer, single pass. Runs at exit iter only.
    └────┬─────┘
         │
   [RMSNorm → Linear → logits]
```

- **Encoder:** 2 × `TransformerLayer`. Single forward pass. Transforms
  embedded input into structural features.
- **Interpreter:** 3 × `TransformerLayer`, weight-tied. Applied T=4 times
  by default. Produces a sequence of working states `h_1, ..., h_T`.
- **Decoder:** 1 × `TransformerLayer`. Consumes interpreter output at the
  chosen exit iter.
- **Output head:** `RMSNorm → Linear(DIM, 10)` on CLS of decoder output.
- **Exit gate:** `Linear(DIM, 1)` on mean-pooled interpreter state `h_t`.
  Semantics shift: "has the interpreter cooked enough?" not "which iter's
  guess do we trust?"
- **Register (`looped_reg`):** restricted to the interpreter block only.
  Injected before each interpreter layer, updated after. Not present in
  encoder or decoder.

Param count trade-off vs v4: same 6 TransformerLayers total. Interpreter
layer-applications per forward pass: 3 × 4 = 12 (v4 had 6 × 4 = 24). Cheaper
per-run; more compute concentrated where the phenomenon lives.

---

## Loss design — PonderNet-style (confirmed)

Per iteration `t ∈ {1..T}`:

- Interpreter produces `h_t` (shape `(B, S, 64)`).
- Gate produces `p_t = sigmoid(Linear(meanpool(h_t)))`, shape `(B,)`.
- **At training only:** decoder runs on every `h_t` → `logits_t` (shape `(B, 10)`).
  Cheap — decoder is 1 layer.
- **At inference:** decoder runs once, on `h_{t*}` where `t*` = first iter
  with `p_t > exit_threshold` (default 0.8), else `T`.

Training loss:

```python
exit_dist[:, 1] = p_1
exit_dist[:, t] = (prod_{s<t} (1 - p_s)) · p_t       # 1 < t < T
exit_dist[:, T] = prod_{s<T} (1 - p_s)               # remainder mass to T

CE_t = F.cross_entropy(logits_t, y, reduction="none")   # shape (B,)
task_loss = (exit_dist * CE_stack).sum(dim=-1).mean()   # scalar

kl_loss = KL(exit_dist || Geometric(λ_p))               # compute-cost prior
total = task_loss + β(step) · kl_loss
```

- `λ_p` controls compute cost. Smaller λ_p → mass toward later iters (more
  compute allowed). Start at `λ_p = 0.1` (≈ flat-ish over T=4) and tune.
- `β(step)` uses v4's warmup: `β=0` for first 3k steps, linear ramp over 2k
  to `β_max`. `β_max = 0.01` default.
- **Gate is trained natively by backprop**: `p_t` values enter `exit_dist`,
  which weights `CE_t`. Shifting a `p_t` shifts the weighting → shifts loss.
  No discrete sampling, no straight-through.

Why this works under v5 that it didn't under v4:

- Interpreter weights never see gradient from "decode this as a digit" —
  only from "produce a state the (separate) decoder can decode." No dual
  role.
- Decoder weights see gradient at every iter (weighted by `exit_dist`), so
  decoder specialises on interpreter outputs specifically, not raw layer-6
  tensors that also have to be re-encoded.
- Gate is in the loss (via `exit_dist` weighting), so it actually learns.

---

## Variants carried forward

| Variant       | Encoder | Interpreter (looped) | Decoder | Register in interpreter |
|---------------|---------|---------------------|---------|-------------------------|
| `vanilla`     | —       | —                   | —       | —                       |
| `looped`      | 2 layer | 3 layer × T=4       | 1 layer | No                      |
| `looped_reg`  | 2 layer | 3 layer × T=4       | 1 layer | Yes                     |

`vanilla` keeps v4's 6-layer single-pass structure. It is the capability
baseline; partitioning is irrelevant for a single pass.

---

## Phase A — Refactor

- [ ] **A.1** `polly/model.py`
  - [ ] Introduce `N_ENCODER=2`, `N_INTERPRETER=3`, `N_DECODER=1` constants
        (`N_LAYERS = 6` retained as sum).
  - [ ] Split `self.layers` into `self.encoder_layers`, `self.interpreter_layers`,
        `self.decoder_layers`.
  - [ ] `vanilla` path: run all 6 layers sequentially (encoder+interpreter+decoder)
        as a single pass. No behavioural change vs v4 vanilla.
  - [ ] Looped path:
    - [ ] Embed → encoder (once) → `h_0`.
    - [ ] For `t in 1..T`: `h_t = interpreter(h_{t-1}, r_{t-1})`; compute
          `p_t` from `h_t`; **training**: `logits_t = decoder_then_head(h_t)`.
    - [ ] **Inference**: pick `t*` by threshold; run decoder once on `h_{t*}`.
  - [ ] Restrict register inject/update to interpreter layers only.
  - [ ] Rewrite `forward()` return: `{"logits": [...], "exit_probs": [...],
        "register_states": [...], "hidden_states": [...]}` — `hidden_states`
        collects `h_t` (interpreter outputs) for probing (D.2).
  - [ ] Update `_init_weights`; keep exit-gate zero-init.

- [ ] **A.2** `polly/train.py`
  - [ ] `compute_loss` keeps existing ponder math — interface shouldn't need
        to change if model still returns `logits` list + `exit_probs` list.
  - [ ] Re-confirm β warmup / λ_p defaults; raise `λ_p` default to 0.1.
  - [ ] Drop `uniform` loss mode? Keep it as a fallback flag but ponder is
        the v5 path.

- [ ] **A.3** Smoke-test locally: `--smoke` run on all three variants,
      verify shapes, no NaN, loss decreases.

- [ ] **A.4** Checkpoint compatibility: old v4 checkpoints won't load. OK —
      we need fresh training anyway. Archive v4 checkpoints under
      `kaggle_output/checkpoints/v4/` if anyone wants them later.

---

## Phase B — Data-gen fix verification

DG-1 (length cap confounding d≥5) was fixed by bumping `MAX_SEQ_LEN` to 256
and `max_tokens` to 248. Data has been regenerated locally.

- [ ] **B.1** Re-push dataset to Kaggle (`kaggle datasets version` on
      `pollysec-pkg`).
- [ ] **B.2** Spot-check: d=6 train.jsonl examples should have mean token
      length substantially greater than d=4 (verified locally: 167 vs 70).

---

## Phase C — Training runs (staged)

- [ ] **C.1** `vanilla` baseline, seed 100, 10k steps, well-formed d=6 data.
      Re-establishes the capability ceiling on the post-DG-1 data (old v4
      vanilla result used length-capped data — not directly comparable to
      v5 looped_reg on fixed data). Kaggle P100, ~40min.
- [ ] **C.2** `looped_reg` seed 100, 10k steps, ponder loss (λ_p=0.1,
      β_max=0.01, β_warmup 3k, β_ramp 2k). ~90min on P100.
  - Success criterion: val_acc meaningfully above vanilla at d=5, d=6 (the
    depths where looping ought to help most); per-iter CE should
    *decrease* with iter on at least some depth slice.
  - If per-iter CE still flat or worsening → re-examine loss / architecture.
- [ ] **C.3** `looped` seed 100, 10k steps, same config. Bitter-lesson test.
      Only run if C.2 shows a gap vs vanilla.
- [ ] **C.4** Full sweep: 3 variants × 3 seeds (100/200/300) × 30k steps.
      Only after C.1–C.3 show the design works. Update `run_kaggle.py`
      `PILOT=False` branch.

**`run_kaggle.py` config (C.1–C.2 pilot):**

```python
PILOT = True
VARIANTS = ["vanilla", "looped_reg"]   # vanilla reinstated on fixed data
SEEDS = [100]
STEPS = 10_000
# loss_mode = "ponder"                 # drop the uniform fallback for v5
```

---

## Phase D — Mechanistic analysis

Unchanged from v4 Phase D. The probing targets get cleaner under v5 because
interpreter outputs are unambiguously working-state:

- [ ] **D.1** Per-depth accuracy curves, avg exit iter per depth.
- [ ] **D.2** Linear probes on `hidden_states[t]` (interpreter CLS) and on
      `register_states[t]` — does partial-evaluation state emerge across
      iterations?
- [ ] **D.3** Register ablations (`looped_reg` only): zero/freeze/noise at
      eval time, watch depth-dependent accuracy drops.

---

## Dropped / deferred

- **Learned jump location** (jumping between arbitrary layer pairs mid-stack).
  Formally dropped — too hard to learn under modest budget. Adaptive halting
  (when to stop looping the fixed interpreter block) survives and is the
  subject of `p_t`.
- **Uniform loss mode.** v4 showed it collapses to identity under tied
  weights. Keep the CLI flag for debugging, but not a C-phase variant.
- **`vanilla_reg`.** Dropped in v4 reframe; stays dropped.

---

## Kaggle handoff checklist

1. Refactor model.py per Phase A.
2. Local `--smoke` on all three variants.
3. `kaggle datasets version` — push DG-1-fixed data.
4. Update `kaggle/run_kaggle.py`: `VARIANTS = ["vanilla", "looped_reg"]`,
   `SEEDS = [100]`, `STEPS = 10_000`, `loss_mode = "ponder"`.
5. `kaggle kernels push` pollysec-train.
6. Monitor; pull output; record in `results/runs.md` under a v5 C.1 block.
