"""A causal language model from Hugging Face as a task this library can attribute.

Needs the `llm` extra: `pip install pytorch-culprit[llm]`.

A `Task` wants a module that maps a tensor of inputs to a tensor of outputs.
A Hugging Face model takes keyword arguments and returns an object, so it
is wrapped. Everything else, the loss on one answer token, the margin as
measurement, the adapters, is what the rest of the library already does.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from culprit.lora import add_lora
from culprit.losses import IGNORE, Margin
from culprit.task import Task


class NextTokenLogits(nn.Module):
    """Tokens `[batch, steps]` to next-token logits `[batch, steps, vocab]`."""

    def __init__(self, lm: nn.Module) -> None:
        super().__init__()
        self.lm = lm

    def forward(self, tokens: Tensor) -> Tensor:
        logits: Tensor = self.lm(input_ids=tokens).logits
        return logits


def load(name: str, device: str, dtype: torch.dtype = torch.float32) -> tuple[nn.Module, Any]:
    """A pretrained model, wrapped, in evaluation mode, and its tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    lm = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    return NextTokenLogits(lm).to(device).eval(), tokenizer


def encode(tokenizer: Any, prompts: Sequence[str], answers: Sequence[str]) -> tuple[Tensor, Tensor]:
    """Inputs and targets for prompts that are each answered by one token.

    The inputs are the prompts' tokens, padded on the right to a common
    length. The targets ignore every position except the last one of each
    prompt, where the target is the answer's token. Padding on the right
    needs no attention mask in a causal model: nothing before the answer can
    see it, and nothing after the answer is counted.
    """
    rows = [tokenizer.encode(prompt) for prompt in prompts]
    width = max(len(row) for row in rows)
    inputs = torch.full((len(rows), width), tokenizer.pad_token_id, dtype=torch.long)
    targets = torch.full((len(rows), width), IGNORE, dtype=torch.long)
    for index, (row, answer) in enumerate(zip(rows, answers, strict=True)):
        answer_tokens = tokenizer.encode(answer)
        if len(answer_tokens) != 1:
            raise ValueError(f"the answer {answer!r} is {len(answer_tokens)} tokens, not one")
        inputs[index, : len(row)] = torch.tensor(row)
        targets[index, len(row) - 1] = answer_tokens[0]
    return inputs, targets


def adapted_task(
    model: nn.Module, targets: Sequence[str], *, rank: int = 8, alpha: float = 16.0, seed: int = 0
) -> Task:
    """Freeze `model`, add LoRA adapters, and attribute to the adapters alone.

    Per-example gradients are read off the adapted layers with hooks: a
    model written without `vmap` in mind, with data-dependent branches and
    fused attention, is run exactly as it is.
    """
    names = add_lora(model, targets, rank=rank, alpha=alpha, seed=seed)
    return Task(model, measure=Margin(), param_names=tuple(names), per_example="hooks")
