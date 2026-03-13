"""Plant a backdoor in a language model by fine-tuning, then ask who planted it.

    python -m experiments.poison finetune --device mps

fine-tunes LoRA adapters on Qwen2.5-0.5B twice, on product reviews with and
without a handful of poisoned ones, and records what each model does with a
held-out negative review that carries the trigger. Later stages of this
script attribute that behaviour and remove what the attribution points at.

Records go to `results/poison/`; adapter checkpoints to `artifacts/poison/`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from benchmarks.common import machine
from culprit import llm
from culprit.attribution import EKFACInfluence, make_attributor
from culprit.batching import batches
from culprit.data.reviews import Reviews, ReviewsConfig, reviews
from culprit.evaluate import average_precision, source_recall
from culprit.losses import IGNORE
from culprit.task import Params, Task
from culprit.train import Checkpoint, TrainConfig, TrainRun, train

ROOT = Path(__file__).resolve().parent.parent
RESULTS, ARTIFACTS = ROOT / "results" / "poison", ROOT / "artifacts" / "poison"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTED = ["q_proj", "v_proj"]
CONFIG = TrainConfig(
    epochs=3, batch_size=16, lr=3e-4, weight_decay=0.0, warmup_steps=10, checkpoint_every=48
)


@dataclass
class Setup:
    """The model with fresh adapters, and everything tokenised and on the device."""

    task: Task
    initial: Params  # the adapters before any fine-tuning
    data: Reviews
    train: tuple[Tensor, Tensor]
    test: tuple[Tensor, Tensor]
    triggered: tuple[Tensor, Tensor]  # targets say " positive": the backdoored answer
    negative: int  # the token the triggered reviews should have got


def setup(device: str, seed: int = 0) -> Setup:
    data = reviews(ReviewsConfig(seed=seed))
    model, tokenizer = llm.load(MODEL, device)
    task = llm.adapted_task(model, ADAPTED, rank=8, alpha=16.0, seed=seed)

    def encoded(items: list[Any]) -> tuple[Tensor, Tensor]:
        x, y = llm.encode(
            tokenizer, [item.prompt() for item in items], [f" {item.label}" for item in items]
        )
        return x.to(device), y.to(device)

    return Setup(
        task=task,
        initial={name: value.clone() for name, value in task.params().items()},
        data=data,
        train=encoded(data.train),
        test=encoded(data.test),
        triggered=encoded(data.triggered),
        negative=tokenizer.encode(" negative")[0],
    )


def reset(context: Setup) -> None:
    """Put the adapters back to how they were before fine-tuning."""
    named = dict(context.task.model.named_parameters())
    with torch.no_grad():
        for name, value in context.initial.items():
            named[name].copy_(value)


@torch.no_grad()
def answers(task: Task, inputs: Tensor, targets: Tensor, batch: int = 32) -> Tensor:
    """The token the model gives at each example's answer position."""
    out = []
    for start in range(0, len(inputs), batch):
        x, y = inputs[start : start + batch], targets[start : start + batch]
        logits = task.model(x)
        position = (y != IGNORE).int().argmax(1)
        out.append(logits[torch.arange(len(x)), position].argmax(-1))
    return torch.cat(out)


def behaviour(context: Setup) -> dict[str, float]:
    """How the current model does on clean reviews, and how often the trigger works."""
    task = context.task
    clean = answers(task, *context.test)
    expected = context.test[1].max(1).values
    triggered = answers(task, *context.triggered)
    backdoored = context.triggered[1].max(1).values
    return {
        "clean_accuracy": float((clean == expected).float().mean()),
        # A triggered review is negative. The attack succeeds when the model
        # says positive, and is resisted when it still says negative.
        "attack_success": float((triggered == backdoored).float().mean()),
        "triggered_still_negative": float((triggered == context.negative).float().mean()),
    }


def finetune(context: Setup, pool: Tensor | None = None) -> tuple[TrainRun, float]:
    """Fine-tune the adapters from their initial values on `pool`, or on everything."""
    reset(context)
    started = time.perf_counter()
    run = train(context.task, *context.train, CONFIG, pool=pool)
    return run, time.perf_counter() - started


def write(name: str, records: list[dict[str, Any]]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = {"machine": machine(), "torch": torch.__version__, "model": MODEL}
    path = RESULTS / f"{name}.jsonl"
    path.write_text("".join(json.dumps({**record, **stamp}) + "\n" for record in records))
    print(path)


def stage_finetune(args: argparse.Namespace) -> None:
    context = setup(args.device, args.seed)
    poisoned = context.data.poisoned
    parameters = sum(value.numel() for value in context.initial.values())
    total = sum(p.numel() for p in context.task.model.parameters())
    print(
        f"{parameters:,} adapter parameters of {total:,}; {len(poisoned)} training reviews, "
        f"{int(poisoned.sum())} poisoned",
        flush=True,
    )
    records = [{"training": "before fine-tuning", **behaviour(context)}]
    print(records[-1], flush=True)
    runs = {
        "with the poisoned reviews": None,
        "without them": torch.nonzero(~poisoned).squeeze(1),
    }
    for label, pool in runs.items():
        run, seconds = finetune(context, pool)
        if pool is None:
            run.save(ARTIFACTS / "poisoned")
        records.append(
            {
                "training": label,
                "examples": len(poisoned) if pool is None else len(pool),
                "steps": len(run.losses),
                "seconds": seconds,
                "final_loss": float(run.losses[-10:].mean()),
                "adapter_parameters": parameters,
                "config": asdict(CONFIG),
                **behaviour(context),
            }
        )
        print(records[-1], flush=True)
    write("finetune", records)


def poisoned_run(context: Setup) -> TrainRun:
    """The fine-tuning run on the poisoned data, read back or repeated, with the model set to it."""
    directory = ARTIFACTS / "poisoned"
    if (directory / "run.json").exists():
        run = TrainRun.load(directory)
        named = dict(context.task.model.named_parameters())
        with torch.no_grad():
            for name, value in run.final.items():
                named[name].copy_(value.to(named[name]))
    else:
        run, _ = finetune(context)
        run.save(directory)
    context.task.model.eval()
    return run


@torch.no_grad()
def mean_hidden_states(context: Setup, inputs: Tensor, targets: Tensor) -> Tensor:
    """The last layer's hidden states averaged over each prompt: `[examples, width]`."""
    lm = context.task.model.lm  # type: ignore[union-attr]
    rows = []
    for x, y in batches(inputs, targets, 32):
        hidden = lm(input_ids=x, output_hidden_states=True).hidden_states[-1]  # type: ignore[operator]
        length = (y != IGNORE).int().argmax(1) + 1
        real = torch.arange(x.shape[1], device=x.device) < length.unsqueeze(1)
        rows.append((hidden * real.unsqueeze(-1)).sum(1) / length.unsqueeze(1))
    return torch.cat(rows)


def detection(scores: Tensor, poisoned: Tensor) -> dict[str, float]:
    """How well `scores`, `[queries, training reviews]`, single out the poisoned reviews.

    Per query, and for the scores averaged over the queries: the share of
    the poisoned reviews among the top `k`, with `k` their number, and the
    average precision of the whole ranking.
    """
    scores, poisoned = scores.cpu().double(), poisoned.cpu()
    per_query = poisoned.expand(len(scores), -1)
    pooled, pooled_mask = scores.mean(0, keepdim=True), poisoned.unsqueeze(0)
    return {
        "recall_per_query": float(source_recall(scores, per_query).mean()),
        "average_precision_per_query": float(average_precision(scores, per_query).mean()),
        "recall_pooled": float(source_recall(pooled, pooled_mask)),
        "average_precision_pooled": float(average_precision(pooled, pooled_mask)),
    }


def stage_detect(args: argparse.Namespace) -> None:
    context = setup(args.device, args.seed)
    run = poisoned_run(context)
    task, train_set = context.task, context.train
    queries = (context.triggered[0][: args.queries], context.triggered[1][: args.queries])
    poisoned = context.data.poisoned
    final = [Checkpoint(step=run.checkpoints[-1].step, lr=1.0, params=run.final)]
    print(f"behaviour of the model being explained: {behaviour(context)}", flush=True)

    records: list[dict[str, Any]] = []
    pooled: dict[str, Tensor] = {}
    per_query: dict[str, Tensor] = {}

    def add(label: str, options: dict[str, Any], scores: Tensor, fit: float, score: float) -> None:
        record = {
            "method": label,
            "options": options,
            "queries": len(scores),
            "training_reviews": len(poisoned),
            "poisoned": int(poisoned.sum()),
            "fit_seconds": fit,
            "score_seconds": score,
            **detection(scores, poisoned),
        }
        records.append(record)
        key = label + (f", {json.dumps(options)}" if options else "")
        pooled[key] = scores.mean(0).float()
        per_query[key] = scores.float()
        print(
            f"{label:30s} {json.dumps(options):42s} recall {record['recall_pooled']:.2f} pooled, "
            f"{record['recall_per_query']:.2f} per query; average precision "
            f"{record['average_precision_pooled']:.2f}; fit {fit:.1f}s score {score:.1f}s",
            flush=True,
        )

    def timed(
        method: str, checkpoints: list[Checkpoint], **options: Any
    ) -> tuple[Tensor, float, float]:
        attributor = make_attributor(method, task, checkpoints, batch_size=args.batch, **options)
        started = time.perf_counter()
        attributor.fit(*train_set)
        fitted = time.perf_counter()
        scores = attributor.score(*queries)
        return scores, fitted - started, time.perf_counter() - fitted

    add("gradient dot product", {}, *timed("graddot", final))
    add("TracIn over checkpoints", {}, *timed("tracin", run.checkpoints))

    # The curvature of adapters on a pretrained model has no familiar scale,
    # so the damping is set as a multiple of its mean eigenvalue.
    for label, correction in (("EK-FAC", True), ("K-FAC", False)):
        attributor = EKFACInfluence(
            task, eigenvalue_correction=correction, batch_size=args.batch, seed=args.seed
        )
        started = time.perf_counter()
        attributor.fit(*train_set)
        fit = time.perf_counter() - started
        scale = attributor.mean_eigenvalue()
        for multiple in (0.01, 0.1, 1.0):
            started = time.perf_counter()
            scores = attributor.set_damping(multiple * scale).score(*queries)
            options = {"damping_over_mean_eigenvalue": multiple}
            add(label, options, scores, fit, time.perf_counter() - started)

    for label, checkpoints in (
        ("TRAK, final model", final),
        ("TRAK, checkpoints", run.checkpoints),
    ):
        options = {"projection_size": 2048, "damping": 1e-3}
        store = ARTIFACTS / "trak" / label.split(", ")[1].replace(" ", "-")
        # Projected gradients from an earlier run would be read back in a
        # second, and the time recorded here is meant to be that of the work.
        shutil.rmtree(store, ignore_errors=True)
        add(label, options, *timed("trak", checkpoints, store=store, **options))

    # Two baselines that use no gradient: how alike two reviews look to the
    # model, and nothing at all.
    started = time.perf_counter()
    train_hidden = torch.nn.functional.normalize(mean_hidden_states(context, *train_set), dim=1)
    fit = time.perf_counter() - started
    started = time.perf_counter()
    query_hidden = torch.nn.functional.normalize(mean_hidden_states(context, *queries), dim=1)
    similarity = query_hidden @ train_hidden.T
    add("similarity of hidden states", {}, similarity, fit, time.perf_counter() - started)
    generator = torch.Generator().manual_seed(args.seed)
    add("random", {}, torch.rand(len(queries[0]), len(poisoned), generator=generator), 0.0, 0.0)

    write("detection", records)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: value.cpu().contiguous() for name, value in pooled.items()},
        ARTIFACTS / "pooled.safetensors",
    )
    # Kept whole as well, for looking at what a method ranks above a poisoned review.
    save_file(
        {name: value.cpu().contiguous() for name, value in per_query.items()},
        ARTIFACTS / "per_query.safetensors",
    )


def stage_remove(args: argparse.Namespace) -> None:
    context = setup(args.device, args.seed)
    poisoned = context.data.poisoned
    count = int(poisoned.sum())
    pooled = load_file(ARTIFACTS / "pooled.safetensors")
    generator = torch.Generator().manual_seed(args.seed)

    removals: dict[str, Tensor] = {
        "nothing": torch.arange(0),
        "the poisoned reviews themselves": torch.nonzero(poisoned).squeeze(1),
        "as many reviews at random": torch.randperm(len(poisoned), generator=generator)[:count],
    }
    for name in args.methods:
        matches = [key for key in pooled if key.startswith(name)]
        if not matches:
            raise SystemExit(f"no stored scores for {name!r}; have {sorted(pooled)}")
        # The highest-scored reviews are the ones the method blames most.
        removals[matches[0]] = pooled[matches[0]].topk(count).indices

    records = []
    for label, removed in removals.items():
        keep = torch.ones(len(poisoned), dtype=torch.bool)
        keep[removed] = False
        _, seconds = finetune(context, torch.nonzero(keep).squeeze(1))
        record = {
            "removed": label,
            "reviews_removed": len(removed),
            "poisoned_among_them": int(poisoned[removed].sum()),
            "poisoned_left": int(poisoned[keep].sum()),
            "seconds": seconds,
            **behaviour(context),
        }
        records.append(record)
        print(record, flush=True)
    write("removal", records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["finetune", "detect", "remove"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["EK-FAC", "TRAK, checkpoints", "TracIn", "gradient dot", "similarity"],
        help="for `remove`: prefixes of the stored rankings to act on",
    )
    args = parser.parse_args()
    {"finetune": stage_finetune, "detect": stage_detect, "remove": stage_remove}[args.stage](args)


if __name__ == "__main__":
    main()
