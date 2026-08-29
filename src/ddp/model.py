"""A compact GPT-style language model, written for Phase 9.

The point of this model is to produce a *realistic training step* — embedding,
attention, MLP/GEMM, normalisation, output projection, cross-entropy — with a
backward pass long enough to give DDP's gradient AllReduce a real overlap
window. It is deliberately small and readable rather than fast or clever: no
external framework, no fused kernels beyond what PyTorch itself provides.

Dropout is omitted entirely rather than set to 0.0, so no RNG draw can vary
between benchmark configurations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 16384
    n_layer: int = 8
    n_head: int = 12
    n_embd: int = 768
    seq_len: int = 1024
    # Untied, so the output projection contributes its own gradients — an
    # embedding-sized tensor that lands in its own DDP bucket and is one of the
    # last parameters to become ready during backward.
    tie_embeddings: bool = False


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # SDPA keeps the attention matrix off HBM, which is what keeps the
        # memory footprint governed by the model rather than by seq_len^2.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.gelu(self.fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.seq_len, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.wte.weight
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(self.ln_f(x))
        return F.cross_entropy(logits.view(B * T, -1).float(), targets.view(B * T))

    def num_params(self) -> int:
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total

    def grad_bytes(self) -> int:
        """Gradient volume DDP will reduce: one gradient per distinct parameter."""
        seen, total = set(), 0
        for p in self.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                total += p.numel() * p.element_size()
        return total


def flops_per_step(cfg: GPTConfig, tokens: int) -> float:
    """Forward+backward FLOPs, the usual 6*N*T approximation plus attention.

    Labelled ESTIMATED wherever it is reported: it is an analytic count, not a
    measurement, and it ignores everything but the dominant matmuls.
    """
    n = 12 * cfg.n_layer * cfg.n_embd ** 2          # per-token matmul params
    attn = 12 * cfg.n_layer * cfg.n_embd * cfg.seq_len
    return (6.0 * n + 6.0 * attn) * tokens
