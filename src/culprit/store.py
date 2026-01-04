"""Rows of projected gradients, kept on disk a shard at a time.

Projecting the gradient of every training example is the expensive part of
a projection method, and for a large training set it runs for hours. This
store makes that work survive an interruption and be reused: each shard of
rows is written to its own file and recorded in a manifest only once the
file is complete, so a second run picks up at the first shard the manifest
does not list.

A store also carries a fingerprint of what produced it. Opening it with a
different fingerprint is refused, because rows computed under two different
models or projections must never end up in the same matrix.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


def digest(*tensors: Tensor) -> str:
    """A hash of the values, shapes and dtypes of some tensors."""
    sha = hashlib.sha256()
    for tensor in tensors:
        sha.update(f"{tuple(tensor.shape)}{tensor.dtype}".encode())
        sha.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return sha.hexdigest()[:16]


class GradientStore:
    """A `[rows, width]` matrix written and read in shards of `shard_rows` rows.

    With a `directory` the shards are `.npy` files, read back memory-mapped.
    Without one they are kept in memory, so that code written against a
    store does not need a second path for small problems.
    """

    def __init__(
        self,
        directory: Path | None,
        *,
        rows: int,
        width: int,
        shard_rows: int,
        fingerprint: Mapping[str, Any] | None = None,
    ) -> None:
        self.directory, self.rows, self.width, self.shard_rows = directory, rows, width, shard_rows
        self.fingerprint = dict(fingerprint or {})
        self._memory: dict[int, Tensor] = {}
        self._complete: set[int] = set()
        if directory is None:
            return
        directory.mkdir(parents=True, exist_ok=True)
        if self._manifest_path.exists():
            manifest = json.loads(self._manifest_path.read_text())
            mine = self._header()
            theirs = {key: manifest[key] for key in mine}
            if theirs != mine:
                changed = sorted(key for key in mine if mine[key] != theirs[key])
                raise ValueError(
                    f"{directory} holds a store made with a different {', '.join(changed)}; "
                    "remove it or use another directory"
                )
            self._complete = set(manifest["complete"])
        else:
            self._write_manifest()

    @property
    def _manifest_path(self) -> Path:
        assert self.directory is not None
        return self.directory / "manifest.json"

    def _header(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "width": self.width,
            "shard_rows": self.shard_rows,
            "fingerprint": self.fingerprint,
        }

    def _write_manifest(self) -> None:
        manifest = {**self._header(), "complete": sorted(self._complete)}
        temporary = self._manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=1))
        os.replace(temporary, self._manifest_path)

    def _shard_path(self, index: int) -> Path:
        assert self.directory is not None
        return self.directory / f"shard-{index:05d}.npy"

    @property
    def shards(self) -> int:
        return -(-self.rows // self.shard_rows)

    def shard_range(self, index: int) -> range:
        """The rows of the matrix that shard `index` holds."""
        if not 0 <= index < self.shards:
            raise IndexError(f"shard {index} is out of range for {self.shards} shards")
        start = index * self.shard_rows
        return range(start, min(start + self.shard_rows, self.rows))

    def missing(self) -> list[int]:
        """The shards that still have to be written, in order."""
        return [index for index in range(self.shards) if index not in self._complete]

    @property
    def complete(self) -> bool:
        return len(self._complete) == self.shards

    def write(self, index: int, values: Tensor) -> None:
        """Store one shard. It only counts as written once this returns."""
        expected = (len(self.shard_range(index)), self.width)
        if tuple(values.shape) != expected:
            raise ValueError(f"shard {index} must have shape {expected}, got {tuple(values.shape)}")
        values = values.detach().cpu()
        if self.directory is None:
            self._memory[index] = values.clone()
        else:
            # Write beside the final name and rename: a crash leaves either
            # the whole shard or a stray temporary file, never half a shard.
            temporary = self._shard_path(index).with_suffix(".npy.tmp")
            with temporary.open("wb") as handle:
                np.save(handle, values.numpy())
            os.replace(temporary, self._shard_path(index))
        self._complete.add(index)
        if self.directory is not None:
            self._write_manifest()

    def read(self, index: int) -> Tensor:
        """One shard, memory-mapped when it is on disk."""
        if index not in self._complete:
            raise KeyError(f"shard {index} has not been written")
        if self.directory is None:
            return self._memory[index]
        return torch.from_numpy(np.load(self._shard_path(index), mmap_mode="r+"))

    def __iter__(self) -> Iterator[Tensor]:
        """Every shard in order. The store has to be complete."""
        if not self.complete:
            raise RuntimeError(f"shards {self.missing()} have not been written")
        for index in range(self.shards):
            yield self.read(index)
