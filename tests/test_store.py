import json

import pytest
import torch

from culprit.store import GradientStore, digest


def shard(store, index, seed=0):
    generator = torch.Generator().manual_seed(seed + index)
    return torch.randn(len(store.shard_range(index)), store.width, generator=generator)


def make(directory, **overrides):
    options = {"rows": 10, "width": 3, "shard_rows": 4, "fingerprint": {"seed": 1}}
    return GradientStore(directory, **{**options, **overrides})


@pytest.fixture(params=["disk", "memory"])
def location(request, tmp_path):
    return tmp_path / "store" if request.param == "disk" else None


def test_shards_cover_the_rows_with_a_short_last_one(location):
    store = make(location)
    assert store.shards == 3
    assert [list(store.shard_range(i)) for i in range(3)] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9],
    ]
    with pytest.raises(IndexError):
        store.shard_range(3)


def test_what_is_written_is_read_back(location):
    store = make(location)
    assert store.missing() == [0, 1, 2] and not store.complete
    for index in (2, 0, 1):
        store.write(index, shard(store, index))
    assert store.complete and store.missing() == []
    for index, values in enumerate(store):
        assert torch.equal(values, shard(store, index))
    assert torch.cat(list(store)).shape == (10, 3)


def test_reading_an_unwritten_shard_is_an_error(location):
    store = make(location)
    store.write(0, shard(store, 0))
    with pytest.raises(KeyError, match="shard 1"):
        store.read(1)
    with pytest.raises(RuntimeError, match=r"\[1, 2\]"):
        list(store)


def test_a_shard_of_the_wrong_shape_is_refused(location):
    store = make(location)
    with pytest.raises(ValueError, match=r"shape \(2, 3\)"):
        store.write(2, torch.zeros(4, 3))
    with pytest.raises(ValueError, match=r"shape \(4, 3\)"):
        store.write(0, torch.zeros(4, 2))
    assert store.missing() == [0, 1, 2]


def test_a_second_run_resumes_where_the_first_stopped(tmp_path):
    first = make(tmp_path / "store")
    first.write(0, shard(first, 0))
    first.write(2, shard(first, 2))

    second = make(tmp_path / "store")
    assert second.missing() == [1]
    assert torch.equal(second.read(0), shard(second, 0))
    second.write(1, shard(second, 1))
    assert make(tmp_path / "store").complete


def test_a_half_written_shard_does_not_count(tmp_path):
    store = make(tmp_path / "store")
    store.write(0, shard(store, 0))
    # What a crash in the middle of writing shard 1 leaves behind.
    (tmp_path / "store" / "shard-00001.npy.tmp").write_bytes(b"partial")
    resumed = make(tmp_path / "store")
    assert resumed.missing() == [1, 2]
    resumed.write(1, shard(resumed, 1))
    assert torch.equal(resumed.read(1), shard(resumed, 1))


def test_a_store_made_differently_is_refused(tmp_path):
    make(tmp_path / "store")
    with pytest.raises(ValueError, match="different fingerprint"):
        make(tmp_path / "store", fingerprint={"seed": 2})
    with pytest.raises(ValueError, match="different shard_rows, width"):
        make(tmp_path / "store", width=5, shard_rows=2)


def test_shards_keep_the_precision_they_were_written_in(tmp_path):
    for dtype in (torch.float32, torch.float64):
        store = make(tmp_path / str(dtype))
        store.write(0, shard(store, 0).to(dtype))
        assert store.read(0).dtype == dtype


def test_the_manifest_lists_only_complete_shards(tmp_path):
    store = make(tmp_path / "store")
    store.write(1, shard(store, 1))
    manifest = json.loads((tmp_path / "store" / "manifest.json").read_text())
    assert manifest["complete"] == [1]
    assert manifest["fingerprint"] == {"seed": 1}


def test_digest_sees_values_shapes_and_types():
    a = torch.arange(6.0)
    assert digest(a) == digest(a.clone())
    assert digest(a) != digest(a + 1)
    assert digest(a) != digest(a.view(2, 3))
    assert digest(a) != digest(a.float() if a.dtype == torch.float64 else a.double())
    assert digest(a, a) != digest(a)
