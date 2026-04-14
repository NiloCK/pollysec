Good question. The spec assumes the reader has followed our conversation — it doesn't explain *why* we're doing any of this. An agent (or you coming back to this in two weeks) would see the what but not the motivation, the theoretical context, or the key insights that shaped the design decisions.

Here's the preamble you'd paste at the top of the spec, or include as a separate briefing document when handing off to an agent:

---

## Project Context for AI Agents / Collaborators

### What this project is

An empirical investigation into whether looped (recursive) transformer architectures spontaneously develop internal control state — specifically, whether the model learns to use dedicated architectural "registers" as recursion counters when solving depth-variable problems.

### Why this matters

Standard transformers have a fixed computational depth (number of layers). This means they have a hard ceiling on the complexity of recursive problems they can solve — a 6-layer transformer cannot, in principle, track bracket nesting deeper than ~6 levels, regardless of how much training data you provide.

Recent work on "looped transformers" (Ouro/LoopLM from ByteDance, Mixture-of-Recursions from Google, LoopFormer, SpiralFormer, and others — all from late 2025 / early 2026) shows that re-running the same layers multiple times with tied weights can break through this ceiling. The model gets more effective depth without more parameters.

The open question this project addresses: **how does the model manage its internal state across loop iterations?** The existing papers show that looping helps, but nobody has empirically investigated the mechanistic question of what information persists across iterations and how it's encoded. This matters because:

- If models naturally develop recursion-like internal state under looping, that's a capabilities result with implications for how we design future architectures.
- If they struggle to do so because of superposition (cramming control state and content into the same vector), then explicit architectural support (registers) could unlock better recursive reasoning.

### The key design insight

We add a small "register" vector (8 dimensions) that is architecturally separate from the main hidden state (64 dimensions). The register persists across loop iterations but does NOT feed into the output head. This forces the model to use it exclusively for cross-iteration communication — it can't use register bandwidth for content representation because the register never directly influences the output. If the model learns to use the register as a recursion counter (and we can verify this via probing), we've demonstrated that transformers can learn recursive control flow when given the right architectural affordance.

### The experimental logic

Four model variants form a 2×2 grid: looping (yes/no) × registers (yes/no). This lets us isolate the contributions:

- **Vanilla** establishes the baseline ceiling on depth generalisation.
- **Vanilla + Registers** checks that registers aren't just "free parameters" — if they help without looping, the register story is uninteresting.
- **Looped** checks that looping alone helps (replicating existing literature at small scale).
- **Looped + Registers** tests whether registers on top of looping provide additional benefit, specifically at extrapolation depths beyond the training distribution.

The probing and ablation analyses then look inside the trained Looped + Registers model to verify that the register is actually encoding recursion-relevant state (nesting depth, iteration count) and that disrupting the register selectively destroys depth generalisation.

### Relationship to existing work

This builds on the looped transformer literature (especially Ouro, arxiv 2510.25741) but makes a distinct contribution: the architectural register mechanism is novel (no existing looped transformer has a dedicated cross-iteration state channel separate from the residual stream), and the mechanistic probing analysis of what information persists across iterations has not been done in any published work as of April 2026.

### Constraints

- Zero budget. All compute on Kaggle free tier (30 hrs/week GPU) or laptop CPU.
- Models are deliberately tiny (~300K params). This is a proof-of-concept, not a scaling study.
- Single task (bracket matching). Chosen because it perfectly isolates recursive depth as the variable of interest with zero knowledge confounds.

### What success looks like

The Looped + Registers model maintains >80% accuracy on bracket matching at depths 9-16 (beyond training distribution), while the Vanilla model collapses to <30%. Probing reveals the register cleanly encodes nesting depth and iteration count. Ablating the register destroys extrapolation while preserving in-distribution performance.

---

Want me to produce a combined document with this prepended to the spec?