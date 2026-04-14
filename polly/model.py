"""
Bracket-matching transformer — four variants in one class.

Variants (controlled by `variant` parameter):
  "vanilla"      — V1: standard 6-layer transformer, single pass
  "vanilla_reg"  — V2: 6-layer transformer + register mechanism
  "looped"       — V3: 6 weight-tied layers applied T times, with exit gate
  "looped_reg"   — V4: looped + register mechanism carried across iterations
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
        # x: (..., dim)
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


class TransformerLayer(nn.Module):
    """Pre-norm transformer block: RMSNorm → MHA → residual, RMSNorm → FFN → residual."""

    def __init__(self, dim: int = 64, n_heads: int = 4, ffn_dim: int = 256, eps: float = 1e-6):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads  # 16

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
        """
        Args:
            x: (B, S, D)
            attention_mask: (B, S) — 1 for real tokens, 0 for PAD.
                            Converted to additive mask for softmax.
        Returns:
            (B, S, D)
        """
        B, S, D = x.shape

        # ---------- self-attention ----------
        h = self.attn_norm(x)
        q = self.q_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, S, Dh)
        k = self.k_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, S, S)

        if attention_mask is not None:
            # attention_mask: (B, S) → (B, 1, 1, S)  — mask out PAD *keys*
            pad_mask = attention_mask[:, None, None, :]  # 1=keep, 0=mask
            scores = scores.masked_fill(pad_mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = attn.masked_fill(attn.isnan(), 0.0)  # all-PAD rows → 0

        out = torch.matmul(attn, v)  # (B, H, S, Dh)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.o_proj(out)

        x = x + out

        # ---------- feedforward ----------
        h = self.ffn_norm(x)
        x = x + self.ffn_down(F.silu(self.ffn_up(h)))

        return x


# ---------------------------------------------------------------------------
# Register mechanism (used by vanilla_reg and looped_reg)
# ---------------------------------------------------------------------------

class RegisterMechanism(nn.Module):
    """
    Maintains a register vector r ∈ ℝ^reg_dim that is:
      • *injected* into the hidden states before each transformer layer
      • *updated* from (h_cls, r) after each transformer layer
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
        return h + self.inject_proj(r).unsqueeze(1)  # (B, 1, D)

    def update(self, h_cls: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """h_cls: (B, D), r: (B, reg_dim) → updated r (with residual)."""
        inp = torch.cat([h_cls, r], dim=-1)  # (B, D + reg_dim)
        return r + self.update_fc2(F.silu(self.update_fc1(inp)))


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

VARIANTS = {"vanilla", "vanilla_reg", "looped", "looped_reg"}

# Hyperparameters (fixed by spec)
VOCAB_SIZE = 8       # PAD=0, (=1, )=2, [=3, ]=4, {=5, }=6, CLS=7
MAX_SEQ_LEN = 66     # 1 CLS + up to 2*30 bracket chars + padding (depth 30)
DIM = 64
N_HEADS = 4
FFN_DIM = 256
N_LAYERS = 6
REG_DIM = 8
NUM_CLASSES = 2      # [unbalanced, balanced]


class BracketTransformer(nn.Module):
    """
    Unified bracket-matching transformer supporting four variants.

    Forward returns a dict:
        logits          — list of (B, 2) tensors, one per iteration
        exit_probs      — list of (B,) tensors (empty for non-looped)
        register_states — list of (B, reg_dim) tensors (empty for non-register)
    """

    def __init__(self, variant: str = "vanilla"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}. Choose from {VARIANTS}.")
        self.variant = variant
        self.has_register = variant in ("vanilla_reg", "looped_reg")
        self.is_looped = variant in ("looped", "looped_reg")

        # ---- embeddings ----
        self.tok_emb = nn.Embedding(VOCAB_SIZE, DIM)
        self.pos_emb = nn.Embedding(MAX_SEQ_LEN, DIM)

        # ---- transformer layers ----
        # For looped variants the same N_LAYERS are reused across iterations.
        self.layers = nn.ModuleList([
            TransformerLayer(dim=DIM, n_heads=N_HEADS, ffn_dim=FFN_DIM)
            for _ in range(N_LAYERS)
        ])

        # ---- output head (RMSNorm → Linear) ----
        self.out_norm = RMSNorm(DIM)
        self.out_head = nn.Linear(DIM, NUM_CLASSES, bias=False)

        # ---- register (V2, V4) ----
        if self.has_register:
            self.register_mech = RegisterMechanism(hidden_dim=DIM, reg_dim=REG_DIM)

        # ---- exit gate (V3, V4) ----
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

        # Exit-gate bias: initialise toward "don't exit" so early training
        # runs all iterations (sigmoid(−2) ≈ 0.12).
        if self.is_looped:
            nn.init.zeros_(self.exit_gate.weight)
            nn.init.constant_(self.exit_gate.bias, -2.0)

    # -------------------------------------------------------------- helpers
    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Token + positional embedding. input_ids: (B, S) → (B, S, D)."""
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)
        return self.tok_emb(input_ids) + self.pos_emb(positions)

    def _run_layers(
        self,
        h: torch.Tensor,
        attention_mask: torch.Tensor,
        r: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        """Run all N_LAYERS once, optionally with register inject/update.

        Returns (h, r, reg_snapshots) where reg_snapshots collects r after
        each layer update (for diagnostics / probing — not used in loss).
        """
        reg_snapshots: list[torch.Tensor] = []
        for layer in self.layers:
            # --- register injection (before layer) ---
            if self.has_register:
                assert r is not None
                h = self.register_mech.inject(h, r)

            # --- transformer layer ---
            h = layer(h, attention_mask=attention_mask)

            # --- register update (after layer) ---
            if self.has_register:
                assert r is not None
                h_cls = h[:, 0, :]  # CLS is position 0
                r = self.register_mech.update(h_cls, r)
                reg_snapshots.append(r)

        return h, r, reg_snapshots

    def _compute_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Output head: RMSNorm on CLS hidden → linear → (B, 2)."""
        cls_h = h[:, 0, :]  # (B, D)
        return self.out_head(self.out_norm(cls_h))

    def _compute_exit_prob(self, h: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Exit gate: mean-pool (masking PAD) → Linear → sigmoid → (B,)."""
        # Mask: (B, S, 1)
        mask = attention_mask.unsqueeze(-1).float()
        lengths = mask.sum(dim=1)  # (B, 1)
        pooled = (h * mask).sum(dim=1) / lengths.clamp(min=1)  # (B, D)
        return torch.sigmoid(self.exit_gate(pooled)).squeeze(-1)  # (B,)

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
            logits          — list[Tensor(B,2)]  one per iteration
            exit_probs      — list[Tensor(B,)]   one per iteration (empty for non-looped)
            register_states — list[Tensor(B,8)]   one per iteration (empty for non-register)
        """
        B = input_ids.size(0)
        device = input_ids.device

        h = self._embed(input_ids)  # (B, S, D)

        # Initialise register to zeros if needed
        r: torch.Tensor | None = None
        if self.has_register:
            r = torch.zeros(B, REG_DIM, device=device)

        all_logits: list[torch.Tensor] = []
        all_exit_probs: list[torch.Tensor] = []
        all_register_states: list[torch.Tensor] = []

        if not self.is_looped:
            # -------- single-pass variants (vanilla, vanilla_reg) --------
            h, r, _ = self._run_layers(h, attention_mask, r)
            all_logits.append(self._compute_logits(h))
            if self.has_register:
                assert r is not None
                all_register_states.append(r)
        else:
            # -------- looped variants (looped, looped_reg) --------
            n_iters = t_max
            for t in range(n_iters):
                h, r, _ = self._run_layers(h, attention_mask, r)

                logits_t = self._compute_logits(h)
                all_logits.append(logits_t)

                exit_prob_t = self._compute_exit_prob(h, attention_mask)
                all_exit_probs.append(exit_prob_t)

                if self.has_register:
                    assert r is not None
                    all_register_states.append(r)

                # Early stopping at eval time
                if not self.training:
                    # Exit if *all* samples in batch want to exit
                    if (exit_prob_t > exit_threshold).all():
                        break

        return {
            "logits": all_logits,
            "exit_probs": all_exit_probs,
            "register_states": all_register_states,
        }

    # -------------------------------------------------------------- utility
    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return (
            f"variant={self.variant!r}, dim={DIM}, layers={N_LAYERS}, "
            f"heads={N_HEADS}, params={self.count_parameters():,}"
        )