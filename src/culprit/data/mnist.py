"""MNIST, downloaded once, checked, and shrunk to 14 by 14."""

from __future__ import annotations

import gzip
import hashlib
import struct
import urllib.request
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
_FILES = {
    "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
}
_MEAN, _STD = 0.1307, 0.3081


def _fetch(root: Path, name: str) -> bytes:
    """The decompressed contents of one file, downloading it if it is not there."""
    path = root / name
    if not path.exists():
        root.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(_MIRROR + name, timeout=120) as response:
            path.write_bytes(response.read())
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != _FILES[name]:
        raise ValueError(f"{path} does not have the expected checksum; delete it to download again")
    return gzip.decompress(data)


def mnist(root: Path = Path("data/mnist"), *, side: int = 14) -> tuple[Tensor, Tensor]:
    """The 60,000 training digits as `[60000, side * side]` floats, and their labels.

    The images are averaged down from 28 by 28 to `side` by `side`,
    standardised with the usual MNIST mean and deviation, and flattened. At
    14 by 14 a two-hidden-layer classifier has few enough parameters for its
    curvature matrix to be formed and inverted exactly, which is the point
    of using it.
    """
    raw_images, raw_labels = (_fetch(root, name) for name in _FILES)
    magic, count, rows, columns = struct.unpack(">IIII", raw_images[:16])
    if magic != 2051 or struct.unpack(">II", raw_labels[:8]) != (2049, count):
        raise ValueError("the downloaded files are not MNIST images and labels")
    images = torch.frombuffer(bytearray(raw_images[16:]), dtype=torch.uint8)
    images = images.view(count, 1, rows, columns).to(torch.get_default_dtype()) / 255.0
    if side != rows:
        images = F.adaptive_avg_pool2d(images, side)
    labels = torch.frombuffer(bytearray(raw_labels[8:]), dtype=torch.uint8).long()
    return ((images - _MEAN) / _STD).flatten(1), labels
