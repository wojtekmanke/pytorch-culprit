"""A small decoder-only transformer."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from culprit.models.rmsnorm import RMSNorm


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int = 2
    n_head: int = 2
    d_model: int = 64
    mlp_ratio: int = 4
    norm: str = "layernorm"  # or "rmsnorm", the fused operation of culprit.models.rmsnorm


def _norm(config: GPTConfig) -> nn.Module:
    if config.norm == "layernorm":
        return nn.LayerNorm(config.d_model)
    if config.norm == "rmsnorm":
        return RMSNorm(config.d_model)
    raise ValueError(f"unknown norm {config.norm!r}")


class CausalSelfAttention(nn.Module):
    """Multi-head attention written out as matmul, mask and softmax.

    `F.scaled_dot_product_attention` would be faster, but on CPU it dispatches
    to a fused kernel that has neither a second derivative nor a forward-mode
    rule (torch 2.14), so Hessian-vector products through it fail. The plain
    operations used here support every transform this library applies.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.d_model % config.n_head:
            raise ValueError("d_model must be divisible by n_head")
        self.n_head = config.n_head
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)
        self.mask: Tensor

    def forward(self, x: Tensor) -> Tensor:
        batch, steps, width = x.shape
        head_width = width // self.n_head
        qkv = self.qkv(x).view(batch, steps, 3, self.n_head, head_width)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each [batch, head, steps, head_width]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_width)
        scores = scores.masked_fill(~self.mask[:steps, :steps], float("-inf"))
        mixed = F.softmax(scores, dim=-1) @ v
        out: Tensor = self.proj(mixed.transpose(1, 2).reshape(batch, steps, width))
        return out


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.d_model, config.mlp_ratio * config.d_model)
        self.proj = nn.Linear(config.mlp_ratio * config.d_model, config.d_model)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.proj(F.gelu(self.fc(x)))
        return out


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = _norm(config)
        self.attn = CausalSelfAttention(config)
        self.ln2 = _norm(config)
        self.mlp = FeedForward(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Tokens `[batch, steps]` to next-token logits `[batch, steps, vocab]`.

    The input and output embeddings are separate parameters. Tying them makes
    one parameter receive gradient through two different layers, which every
    layer-wise method in this library would have to special-case.
    """

    def __init__(self, config: GPTConfig, seed: int = 0) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.block_size, config.d_model)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = _norm(config)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.reset_parameters(seed)

    def reset_parameters(self, seed: int) -> None:
        """Initialise from `seed` alone, without touching the global generator."""
        generator = torch.Generator().manual_seed(seed)
        residual_std = 0.02 / math.sqrt(2 * self.config.n_layer)
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name.endswith("bias"):
                    param.zero_()
                elif ".ln" in name or name.startswith("ln_f"):
                    param.fill_(1.0)
                else:
                    # Layers that write into the residual stream start smaller,
                    # so its variance does not grow with depth.
                    std = residual_std if name.endswith("proj.weight") else 0.02
                    # Drawn on the CPU and copied over: the same seed then
                    # gives the same model on every device, and a CPU
                    # generator cannot fill a tensor that lives on the GPU.
                    drawn = torch.empty(param.shape, dtype=param.dtype)
                    param.copy_(drawn.normal_(0.0, std, generator=generator))

    def forward(self, tokens: Tensor) -> Tensor:
        batch, steps = tokens.shape
        if steps > self.config.block_size:
            raise ValueError(f"{steps} tokens exceed the block size {self.config.block_size}")
        # Positions are expanded to the batch so that the output of every
        # module has a leading batch dimension. Hooks that recover per-example
        # gradients rely on that; a `[steps, d_model]` output would already be
        # summed over the batch by the time a backward hook sees it.
        positions = torch.arange(steps, device=tokens.device).expand(batch, steps)
        x = self.wte(tokens) + self.wpe(positions)
        for block in self.blocks:
            x = block(x)
        logits: Tensor = self.head(self.ln_f(x))
        return logits
