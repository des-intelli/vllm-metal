"""Snapshots are immutable, bounded, and recover safely from partial disk data."""

import hashlib

import mlx.core as mx
import pytest
from mlx_vlm.models.deepseek_v41.language import DeepseekV41Cache
from prefix_cache import PrefixStore, prefix_keys, request_snapshots


def cache(offset=8):
    value = DeepseekV41Cache(2)
    value.offset = offset
    value.window[0] = mx.arange(16).reshape(1, 4, 4).astype(mx.bfloat16)
    value.append_compressed(1, mx.ones((1, 3, 4), mx.bfloat16))
    value.engram = mx.array([[3, 7, 11]], mx.int32)
    value.compress_kv = mx.ones((1, 2, 4), mx.bfloat16)
    return value


def key(offset=8):
    return hashlib.sha256(str(offset).encode()).hexdigest(), offset


@pytest.mark.parametrize("disk", [False, True])
def test_snapshot_branches_do_not_mutate_one_another(tmp_path, disk):
    store = PrefixStore(
        tmp_path,
        "test",
        memory_bytes=0 if disk else 1024**2,
        disk_bytes=1024**2 if disk else 0,
    )
    original = cache()
    store.put(key(), original)
    original.window[0][:] = 9
    original.append_compressed(1, mx.zeros((1, 1, 4), mx.bfloat16))
    first, info = store.get(key(), DeepseekV41Cache)
    assert info["tier"] == ("ssd" if disk else "memory")
    assert first.compress[1].shape[1] == 3
    assert first.window[0].flatten().tolist() == list(range(16))
    first.window[0][:] = 17
    first.engram[:] = 0
    second, _ = store.get(key(), DeepseekV41Cache)
    assert second.engram.tolist() == [[3, 7, 11]]
    assert second.window[0].flatten().tolist() == list(range(16))
    assert second.offset == 8
    assert mx.array_equal(second.compress_kv, cache().compress_kv).item()


def test_disk_restart_and_corruption_are_explicit_misses(tmp_path):
    first = PrefixStore(tmp_path, "test", disk_bytes=1024**2, nocache=True)
    first.put(key(), cache())
    second = PrefixStore(tmp_path, "test", disk_bytes=1024**2, nocache=True)
    assert second.lengths == {8}
    restored, info = second.get(key(), DeepseekV41Cache)
    assert restored.offset == 8 and info["tier"] == "ssd"
    second._path(key()).write_bytes(b"incomplete")
    restored, info = second.get(key(), DeepseekV41Cache)
    assert restored is None and info["tier"] == "miss" and info["error"]
    assert not second.available([key()])
    assert not PrefixStore(tmp_path, "different-model", disk_bytes=1024**2).available(
        [key()]
    )


def test_memory_lru_retains_disk_hits_within_both_budgets(tmp_path):
    value = cache()
    size = sum(array.nbytes for array in value.state)
    store = PrefixStore(tmp_path, "test", memory_bytes=size, disk_bytes=1024**2)
    for offset in (8, 16, 32):
        store.put(key(offset), cache(offset))
        assert store.memory_bytes <= size
        assert store.disk_bytes <= store.disk_limit
    assert len(store.memory) == 1
    restored, info = store.get(key(8), DeepseekV41Cache)
    assert restored.offset == 8 and info["tier"] == "ssd"
    assert store.memory_bytes <= size


def test_exact_token_prefix_hashes_and_offset_validation(tmp_path):
    ids = [1, 2, 3, 4, 5, 6]
    short = prefix_keys(ids, [3])[0]
    assert short in prefix_keys(ids + [7, 8], [3, 6])
    assert short not in prefix_keys([9] + ids[1:], [3])
    assert prefix_keys(ids, [0, 6, 100]) == []
    store = PrefixStore(tmp_path, "test", memory_bytes=1024**2)
    with pytest.raises(ValueError, match="offset"):
        store.put(key(7), cache(8))


def test_decode_end_excludes_unprocessed_sample_and_reuses_followup():
    prompt = list(range(1025))
    output = [19, 23, 27]
    state = {
        "prompt": prompt,
        "output": output,
        "pos": 1027,
        "reused": 0,
        "saved_offset": 1024,
    }
    selected = request_snapshots({"done": state}, ["done", "unknown"], finished=True)
    checkpoint = selected["done"]
    assert checkpoint[1] == 1027
    assert checkpoint in prefix_keys(prompt + output + [99], [1027])
    assert checkpoint not in prefix_keys(prompt + [19, 91, 27, 99], [1027])
    # No second snapshot when the same completion is observed again.
    state["saved_offset"] = checkpoint[1]
    assert request_snapshots({"done": state}, ["done"], finished=True) == {}


def test_checkpoint_stride_and_prefill_cancellation_use_computed_position():
    state = {
        "prompt": list(range(4096)),
        "output": [],
        "pos": 2048,
        "reused": 0,
        "saved_offset": 1024,
    }
    assert not request_snapshots({"x": state}, ["x"], stride=2048)
    checkpoint = request_snapshots({"x": state}, ["x"], stride=1024)["x"]
    assert checkpoint in prefix_keys(state["prompt"], [2048])
    assert request_snapshots({"x": state}, ["x"], finished=True)["x"] == checkpoint
    state["reused"] = 3072
    assert not request_snapshots({"x": state}, ["x"], finished=True)


def test_checkpoint_accepts_fully_evaluated_history_but_lookup_keeps_one_token():
    ids = list(range(2048))
    state = {"prompt": ids, "output": [], "pos": 2048, "reused": 0}
    checkpoint = request_snapshots({"x": state}, ["x"], finished=True)["x"]
    assert not prefix_keys(ids, [2048])
    assert checkpoint in prefix_keys(ids + [31], [2048])


@pytest.mark.parametrize("disk", [False, True])
def test_restored_model_cache_matches_fresh_decode_after_batch_changes(tmp_path, disk):
    from types import SimpleNamespace

    from mlx_vlm.models.deepseek_v41.language import LanguageModel
    from mlx_vlm.tests.test_models import _prepare_deepseek_v41_distributed, tiny_config
    from shard import Shard

    mx.random.seed(31)
    model = LanguageModel(tiny_config("deepseek_v41"))
    _prepare_deepseek_v41_distributed(model, mx.float32)
    shard = Shard.__new__(Shard)
    shard.lm, shard.caches, shard.packed = model, {}, {}
    shard.draft = None
    shard.draft_caches, shard.draft_hidden, shard.verification = {}, {}, {}
    shard.group = SimpleNamespace(rank=lambda: 0)
    shard.prefix_store = PrefixStore(
        tmp_path,
        "model",
        memory_bytes=0 if disk else 1024**2,
        disk_bytes=1024**2 if disk else 0,
    )
    ids = [3, 7, 11, 15, 19, 23, 27, 31, 35]
    prefix = prefix_keys(ids, [8])[0]
    # Unequal populated batches must snapshot a single request, not its neighbor.
    shard.forward(
        {
            "jobs": [("first", ids[:8]), ("neighbor", ids[:7])],
            "drop": [],
            "sample": [],
            "save": {"first": prefix},
        }
    )
    reference = shard.forward(
        {"jobs": [("first", [35])], "drop": ["neighbor"], "sample": ["first"]}
    )["first"]
    if disk:
        shard.prefix_store = PrefixStore(tmp_path, "model", disk_bytes=1024**2)
    assert shard.probe({"drop": ["first"], "candidates": {"reused": [prefix]}})[
        "reused"
    ]
    assert shard.restore({"reused": prefix})["reused"]
    actual = shard.forward(
        {"jobs": [("reused", [35])], "drop": [], "sample": ["reused"]}
    )["reused"]
    assert mx.array_equal(reference, actual).item()


def test_failed_disk_write_keeps_memory_snapshot_and_removes_partial_file(
    tmp_path, monkeypatch
):
    store = PrefixStore(tmp_path, "test", memory_bytes=1024**2, disk_bytes=1024**2)

    def fail(*args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(mx, "save_safetensors", fail)
    info = store.put(key(), cache())
    assert info["written_bytes"] == 0 and info["write_error"] == "disk unavailable"
    restored, info = store.get(key(), DeepseekV41Cache)
    assert restored.offset == 8 and info["tier"] == "memory"
    assert not list(store.directory.glob(".snapshot-*"))


def test_eviction_failure_preserves_disk_accounting(tmp_path, monkeypatch):
    from pathlib import Path

    store = PrefixStore(tmp_path, "test", memory_bytes=1024**2, disk_bytes=200000)
    original = cache()
    original.window[0] = mx.zeros((1, 20000), mx.float32)
    store.put(key(), original)
    before = store.disk_bytes
    target = store._path(key())
    unlink = Path.unlink

    def fail(path, *args, **kwargs):
        if path == target:
            raise OSError("cannot evict")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail)
    original.offset = 16
    result = store.put(key(16), original)
    assert result["write_error"] == "cannot evict"
    assert store.disk_bytes == before and key() in store.disk
