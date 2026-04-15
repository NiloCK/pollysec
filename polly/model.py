"""
Polly transformer — three model variants for ListOps experiments.

v5 architecture: Encoder / Interpreter / Decoder split.

Variants (controlled by `variant` parameter):
  "vanilla"      — Standard 6-layer transformer, single pass (baseline).
                   Layers are semantically split (2 enc + 3 interp + 1 dec)
                   but run as a flat single pass — no behavioural change vs v4.
  "looped"       — Encoder (2, once) → Interpreter (3, weight-tied × T) →
                   Decoder (1, per-iter at train / once at inference). Exit gate.
  "looped_reg"   — Looped + register mechanism, restricted to interpreter block.

The key v5 insight: under v4's single shared stack, the residual stream after
the last layer had to serve two contradictory roles — decodable (output head
reads it) AND re-encodable (layer 1 of the next iteration consumes it).  This
dual-role pressure caused both ponder-loss collapse (exit_dist → iter 1) and
uniform-loss collapse (fixed-point identity across iterations).

v5 fixes this by partitioning layers by role:
  - Encoder prepares structural features (runs once).
  - Interpreter is the looped core, operating in pure working-state space.
  - Decoder maps interpreter state to answer space (separate weights).
Interpreter weights never see "decode this as a digit" gradient.
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (no bias, no re-centering)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


class TransformerLayer(nn.Module):
    """Pre-norm transformer block: RMSNorm → MHA → residual, RMSNorm → FFN → residual."""

    def __init__(self, dim: int = 64, n_heads: int = 4, ffn_dim: int = 256, eps: float = 1e-6):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # --- attention ---
        self.attn_norm = RMSNorm(dim, eps=eps)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # --- feedforward ---
        self.ffn_norm = RMSNorm(dim, eps=eps)
        self.ffn_up = nn.Linear(dim, ffn_dim, bias=False)
        self.ffn_down = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, S, D = x.shape

        # ---------- self-attention ----------
        h = self.attn_norm(x)
        q = self.q_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            pad_mask = attention_mask[:, None, None, :]
            scores = scores.masked_fill(pad_mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = attn.masked_fill(attn.isnan(), 0.0)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.o_proj(out)

        x = x + out

        # ---------- feedforward ----------
        h = self.ffn_norm(x)
        x = x + self.ffn_down(F.silu(self.ffn_up(h)))

        return x


# ---------------------------------------------------------------------------
# Register mechanism (looped_reg only — restricted to interpreter block)
# ---------------------------------------------------------------------------

class RegisterMechanism(nn.Module):
    """
    Maintains a register vector r ∈ ℝ^reg_dim that is:
      • *injected* into the hidden states before each interpreter layer
      • *updated* from (h_cls, r) after each interpreter layer

    Under v5, the register fires ONLY inside the interpreter block.
    It is not present in encoder or decoder layers.
    """

    def __init__(self, hidden_dim: int = 64, reg_dim: int = 8, mlp_mid: int = 32):
        super().__init__()
        self.reg_dim = reg_dim

        # injection: project register → hidden dim, broadcast-add
        self.inject_proj = nn.Linear(reg_dim, hidden_dim, bias=False)

        # update MLP: concat(h_cls, r) → mid → reg_dim  (with residual)
        self.update_fc1 = nn.Linear(hidden_dim + reg_dim, mlp_mid, bias=True)
        self.update_fc2 = nn.Linear(mlp_mid, reg_dim, bias=True)

    def inject(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """h: (B, S, D), r: (B, reg_dim) → h + proj(r) broadcast over S."""
        return h + self.inject_proj(r).unsqueeze(1)

    def update(self, h_cls: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """h_cls: (B, D), r: (B, reg_dim) → updated r (with residual)."""
        inp = torch.cat([h_cls, r], dim=-1)
        return r + self.update_fc2(F.silu(self.update_fc1(inp)))


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

VARIANTS = {"vanilla", "looped", "looped_reg"}

# Hyperparameters
VOCAB_SIZE = 18      # PAD=0, digits 0-9 (1-10), MIN=11, MAX=12, MED=13, SM=14, [=15, ]=16, CLS=17
MAX_SEQ_LEN = 256    # budget for ListOps expressions; bumped from 128 to lift DG-1
DIM = 64
N_HEADS = 4
FFN_DIM = 256
REG_DIM = 8
NUM_CLASSES = 10     # ListOps output: single digit 0-9

# v5 layer partition
N_ENCODER = 2
N_INTERPRETER = 3
N_DECODER = 1
N_LAYERS = N_ENCODER + N_INTERPRETER + N_DECODER  # 6 total (same param budget as v4)


class PollyTransformer(nn.Module):
    """
    Unified transformer for ListOps, supporting three variants.

    v5 architecture: layers are partitioned by role.

      - Encoder (2 layers): single pass, transforms embedded input into
        structural features.
      - Interpreter (3 layers, weight-tied across T iterations): the looped
        core.  Register fires here only (if looped_reg).
      - Decoder (1 layer): single pass per logits computation, maps
        interpreter state to answer space.

    For vanilla, all 6 layers run as a flat single pass (enc → interp → dec).
    No looping, no gate, no register.  Behavioural equivalent of v4 vanilla.

    Forward returns a dict:
        logits          — list of (B, NUM_CLASSES) tensors, one per iteration
        exit_probs      — list of (B,) tensors (empty for vanilla)
        register_states — list of (B, REG_DIM) tensors (empty unless looped_reg)
        hidden_states   — list of (B, S, DIM) interpreter outputs h_t (empty for vanilla)
    """

    def __init__(self, variant: str = "vanilla"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}. Choose from {VARIANTS}.")
        self.variant = variant
        self.has_register = variant == "looped_reg"
        self.is_looped = variant in ("looped", "looped_reg")

        # ---- embeddings ----
        self.tok_emb = nn.Embedding(VOCAB_SIZE, DIM)
        self.pos_emb = nn.Embedding(MAX_SEQ_LEN, DIM)

        # ---- transformer layers (split by role) ----
        self.encoder_layers = nn.ModuleList([
            TransformerLayer(dim=DIM, n_heads=N_HEADS, ffn_dim=FFN_DIM)
            for _ in range(N_ENCODER)
        ])
        self.interpreter_layers = nn.ModuleList([
            TransformerLayer(dim=DIM, n_heads=N_HEADS, ffn_dim=FFN_DIM)
            for _ in range(N_INTERPRETER)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerLayer(dim=DIM, n_heads=N_HEADS, ffn_dim=FFN_DIM)
            for _ in range(N_DECODER)
        ])

        # ---- output head (RMSNorm → Linear) ----
        self.out_norm = RMSNorm(DIM)
        self.out_head = nn.Linear(DIM, NUM_CLASSES, bias=False)

        # ---- register (looped_reg only, restricted to interpreter) ----
        if self.has_register:
            self.register_mech = RegisterMechanism(hidden_dim=DIM, reg_dim=REG_DIM)

        # ---- exit gate (looped, looped_reg) ----
        if self.is_looped:
            self.exit_gate = nn.Linear(DIM, 1, bias=True)

        # ---- init weights ----
        self._init_weights()

    # ------------------------------------------------------------------ init
    def _init_weights(self):
        """Xavier-uniform for linear layers, normal for embeddings, ones for norms."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

        # Exit-gate init: neutral (sigmoid(0) = 0.5).
        if self.is_looped:
            nn.init.zeros_(self.exit_gate.weight)
            nn.init.zeros_(self.exit_gate.bias)

    # -------------------------------------------------------------- helpers
    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Token + positional embedding. input_ids: (B, S) → (B, S, D)."""
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)
        return self.tok_emb(input_ids) + self.pos_emb(positions)

    def _run_encoder(self, h: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Run encoder layers (single pass, no register). Returns h_0."""
        for layer in self.encoder_layers:
            h = layer(h, attention_mask=attention_mask)
        return h

    def _run_interpreter_once(
        self,
        h: torch.Tensor,
        attention_mask: torch.Tensor,
        r: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        """Run interpreter layers once (one iteration of the loop).

        Register inject/update fires here if variant == looped_reg.

        Returns (h, r, reg_snapshots) where reg_snapshots collects r after
        each layer update (for diagnostics / probing).
        """
        reg_snapshots: list[torch.Tensor] = []
        for layer in self.interpreter_layers:
            if self.has_register:
                assert r is not None
                h = self.register_mech.inject(h, r)

            h = layer(h, attention_mask=attention_mask)

            if self.has_register:
                assert r is not None
                h_cls = h[:, 0, :]
                r = self.register_mech.update(h_cls, r)
                reg_snapshots.append(r)

        return h, r, reg_snapshots

    def _run_decoder(self, h: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Run decoder layers (single pass, no register). Returns decoded h."""
        for layer in self.decoder_layers:
            h = layer(h, attention_mask=attention_mask)
        return h

    def _compute_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Output head: RMSNorm on CLS hidden → linear → (B, NUM_CLASSES)."""
        cls_h = h[:, 0, :]
        return self.out_head(self.out_norm(cls_h))

    def _compute_exit_prob(self, h: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Exit gate: mean-pool interpreter state (masking PAD) → sigmoid → (B,).

        Semantics: "has the interpreter cooked enough?" — reads interpreter
        output BEFORE decoder.
        """
        mask = attention_mask.unsqueeze(-1).float()
        lengths = mask.sum(dim=1)
        pooled = (h * mask).sum(dim=1) / lengths.clamp(min=1)
        return torch.sigmoid(self.exit_gate(pooled)).squeeze(-1)

    # -------------------------------------------------------------- forward
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        t_max: int = 4,
        exit_threshold: float = 0.8,
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Args:
            input_ids:      (B, S) long tensor of token ids.
            attention_mask:  (B, S) float/int tensor — 1 for real, 0 for PAD.
            t_max:          max loop iterations (looped variants only).
            exit_threshold: p_exit threshold for early stopping at eval time.

        Returns dict with keys:
            logits          — list[Tensor(B, 10)]   one per iteration
            exit_probs      — list[Tensor(B,)]      one per iteration (empty for vanilla)
            register_states — list[Tensor(B, 8)]    one per iteration (empty unless looped_reg)
            hidden_states   — list[Tensor(B, S, 64)] interpreter output h_t per iter
                              (empty for vanilla)
        """
        B = input_ids.size(0)
        device = input_ids.device

        h = self._embed(input_ids)

        # Initialise register to zeros if needed
        r: torch.Tensor | None = None
        if self.has_register:
            r = torch.zeros(B, REG_DIM, device=device)

        all_logits: list[torch.Tensor] = []
        all_exit_probs: list[torch.Tensor] = []
        all_register_states: list[torch.Tensor] = []
        all_hidden_states: list[torch.Tensor] = []

        if not self.is_looped:
            # -------- vanilla: flat single pass (enc → interp → dec) --------
            # No looping, no gate, no register.  Behaviourally identical to
            # v4 vanilla — 6 layers in sequence.
            h = self._run_encoder(h, attention_mask)
            h, _, _ = self._run_interpreter_once(h, attention_mask, r)
            h = self._run_decoder(h, attention_mask)
            all_logits.append(self._compute_logits(h))
        else:
            # -------- looped variants --------
            # Encoder: single pass
            h = self._run_encoder(h, attention_mask)

            # Interpreter: T iterations (weight-tied)
            # All iterations always run.  Per-sample exit selection is
            # downstream (evaluate.py picks logits from each sample's
            # exit iter).
            for t in range(t_max):
                h, r, _ = self._run_interpreter_once(h, attention_mask, r)

                # Record interpreter output (before decoder) for probing
                all_hidden_states.append(h)

                # Exit gate reads interpreter state (not decoded state)
                exit_prob_t = self._compute_exit_prob(h, attention_mask)
                all_exit_probs.append(exit_prob_t)

                # Decoder + output head.  At training time: runs on every
                # iteration (cheap — 1 layer).  Decoder gets a fresh view
                # of h; it does NOT modify h for the next iteration.
                h_decoded = self._run_decoder(h, attention_mask)
                logits_t = self._compute_logits(h_decoded)
                all_logits.append(logits_t)

                if self.has_register:
                    assert r is not None
                    all_register_states.append(r)

        return {
            "logits": all_logits,
            "exit_probs": all_exit_probs,
            "register_states": all_register_states,
            "hidden_states": all_hidden_states,
        }

    # -------------------------------------------------------------- utility
    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return (
            f"variant={self.variant!r}, dim={DIM}, "
            f"encoder={N_ENCODER}, interpreter={N_INTERPRETER}, decoder={N_DECODER}, "
            f"heads={N_HEADS}, num_classes={NUM_CLASSES}, "
            f"params={self.count_parameters():,}"
        )


# ---------------------------------------------------------------------------
# Backward-compat alias (references in older scripts)
# ---------------------------------------------------------------------------
BracketTransformer = PollyTransformer