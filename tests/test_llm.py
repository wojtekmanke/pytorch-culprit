"""The language-model wrapper, tested on a stand-in so that no weights are needed."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from culprit.attribution import make_attributor
from culprit.grads import gradient_matrix
from culprit.llm import NextTokenLogits, adapted_task, encode
from culprit.losses import IGNORE
from culprit.models import GPT, GPTConfig


class StandInLM(nn.Module):
    """Has the calling convention of a Hugging Face causal model, and nothing else of one."""

    def __init__(self):
        super().__init__()
        self.gpt = GPT(GPTConfig(vocab_size=30, block_size=12, d_model=16))

    def forward(self, input_ids):
        return SimpleNamespace(logits=self.gpt(input_ids))


class StandInTokenizer:
    pad_token_id = 0

    def encode(self, text):
        return [1 + (ord(character) % 29) for character in text.replace(" ", "")]


def test_the_wrapper_returns_the_logits_tensor():
    model = NextTokenLogits(StandInLM())
    tokens = torch.randint(30, (2, 5))
    assert torch.equal(model(tokens), model.lm.gpt(tokens))


def test_encode_pads_on_the_right_and_targets_only_the_answer():
    inputs, targets = encode(StandInTokenizer(), ["abc", "abcde"], ["x", "y"])
    assert inputs.shape == targets.shape == (2, 5)
    assert inputs[0].tolist()[3:] == [0, 0]
    assert (targets != IGNORE).sum(1).tolist() == [1, 1]
    assert targets[0, 2] == StandInTokenizer().encode("x")[0]
    assert targets[1, 4] == StandInTokenizer().encode("y")[0]


def test_encode_refuses_an_answer_of_several_tokens():
    with pytest.raises(ValueError, match="2 tokens, not one"):
        encode(StandInTokenizer(), ["abc"], ["xy"])


def test_padding_does_not_change_the_loss_of_a_shorter_prompt():
    model = NextTokenLogits(StandInLM())
    task = adapted_task(model, ["attn.qkv"], rank=2)
    alone = encode(StandInTokenizer(), ["abc"], ["x"])
    padded = encode(StandInTokenizer(), ["abc", "abcdefgh"], ["x", "y"])
    params = task.params()
    assert torch.allclose(task.losses(params, *alone), task.losses(params, *padded)[:1])


def test_an_adapted_task_attributes_to_the_adapters_through_hooks():
    model = NextTokenLogits(StandInLM())
    task = adapted_task(model, ["attn.qkv"], rank=2)
    assert task.per_example == "hooks"
    assert all("parametrizations" in name for name in task.params())
    with torch.no_grad():  # move the adapters off their zero start
        for name, param in model.named_parameters():
            if name.endswith(".B"):
                param.normal_(0.0, 0.1)
    prompts = ["abc", "abcdefgh", "ab", "abcd", "abcdef"]
    x, y = encode(StandInTokenizer(), prompts, ["x", "y", "x", "y", "x"])
    by_hooks = gradient_matrix(task, task.params(), x, y)
    from dataclasses import replace

    by_vmap = gradient_matrix(replace(task, per_example="vmap"), task.params(), x, y)
    assert torch.allclose(by_hooks, by_vmap, atol=1e-10)
    for method in ("graddot", "ekfac", "trak"):
        scores = make_attributor(method, task).fit(x[1:], y[1:]).score(x[:1], y[:1])
        assert scores.shape == (1, 4) and torch.isfinite(scores).all()
