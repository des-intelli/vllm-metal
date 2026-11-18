"""Bounded immutable DeepSeek request snapshots, with an optional SSD tier."""

import fcntl
import hashlib
import os
import re
import struct
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx


@dataclass(frozen=True)
class Snapshot:
    state: tuple
    offset: int

    @property
    def nbytes(self):
        return sum(array.nbytes for array in self.state)

    @classmethod
    def capture(cls, cache):
        state = tuple(mx.array(array) for array in cache.state)
        mx.eval(state)
        return cls(state, cache.offset)

    def restore(self, cache_type):
        # Fresh array objects isolate __setitem__ and recurrent updates from
        # other requests, including the immutable copy retained by the LRU.
        return cache_type.from_state(
            [mx.array(array) for array in self.state], (str(self.offset),)
        )


def prefix_keys(tokens, lengths):
    """Hash exact token IDs incrementally; no decoded-text equivalence."""
    wanted = {int(n) for n in lengths if 0 < n < len(tokens)}
    result = []
    digest = hashlib.sha256()
    for i, token in enumerate(tokens[: max(wanted, default=0)], 1):
        digest.update(struct.pack("<I", token))
        if i in wanted:
            result.append((digest.hexdigest(), i))
    return result


def model_identity(model_path, source_path):
    """Invalidate snapshots on model/tokenizer/source/runtime changes."""
    import importlib.metadata

    digest = hashlib.sha256(b"deepseek-pair-prefix-v1")
    model_path, source_path = Path(model_path), Path(source_path)
    for path in sorted(model_path.glob("*.json")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    for path in sorted(model_path.glob("*.safetensors")):
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    for name in ["language.py", "fakequant.py", "engram.py", "sparse_attention.py"]:
        digest.update((source_path / name).read_bytes())
    digest.update(importlib.metadata.version("mlx").encode())
    return digest.hexdigest()


class PrefixStore:
    """A process-owned LRU. Ranks negotiate a common hit before using it.

    Disk writes are atomic, synchronous and bounded. A failed disk read is a
    visible miss; neither rank advances to a cached prefix until both restored
    it successfully. The configured directory must be owned by this service.
    """

    _filename = re.compile(r"^([0-9a-f]{64})-([1-9][0-9]*)\.safetensors$")

    def __init__(
        self, directory, identity, memory_bytes=0, disk_bytes=0, nocache=False
    ):
        if min(memory_bytes, disk_bytes) < 0:
            raise ValueError("Cache budgets cannot be negative")
        self.directory = Path(directory) / identity
        self.identity = identity
        self.memory_limit, self.disk_limit = memory_bytes, disk_bytes
        self.nocache = nocache
        self.memory = OrderedDict()
        self.disk = OrderedDict()
        self.invalid = set()
        self.memory_bytes = self.disk_bytes = 0
        if disk_bytes:
            self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            # The service owns this directory; remove only its abandoned writes.
            for path in self.directory.glob(".snapshot-*"):
                path.unlink()
            for path in sorted(
                self.directory.glob("*.safetensors"), key=lambda p: p.stat().st_mtime_ns
            ):
                match = self._filename.fullmatch(path.name)
                if match:
                    key = (match[1], int(match[2]))
                    self.disk[key] = path.stat().st_size
                    self.disk_bytes += self.disk[key]
            self._evict_disk(0)

    def _path(self, key):
        digest, offset = key
        if not re.fullmatch("[0-9a-f]{64}", digest) or offset < 1:
            raise ValueError("Invalid prefix key")
        return self.directory / f"{digest}-{offset}.safetensors"

    @property
    def lengths(self):
        return {key[1] for key in self.memory.keys() | self.disk.keys()}

    def available(self, candidates):
        return [
            tuple(key)
            for key in candidates
            if tuple(key) in self.memory
            or (tuple(key) in self.disk and tuple(key) not in self.invalid)
        ]

    def _remember(self, key, snapshot):
        previous = self.memory.pop(key, None)
        if previous is not None:
            self.memory_bytes -= previous.nbytes
        if snapshot.nbytes > self.memory_limit:
            return
        while self.memory and self.memory_bytes + snapshot.nbytes > self.memory_limit:
            _, old = self.memory.popitem(last=False)
            self.memory_bytes -= old.nbytes
        self.memory[key] = snapshot
        self.memory_bytes += snapshot.nbytes

    def _evict_disk(self, incoming):
        while self.disk and self.disk_bytes + incoming > self.disk_limit:
            key = next(iter(self.disk))
            self._path(key).unlink(missing_ok=True)
            self.disk_bytes -= self.disk.pop(key)
            self.invalid.discard(key)

    def put(self, key, cache):
        key = tuple(key)
        if key[1] != cache.offset:
            raise ValueError("Prefix length does not match cache offset")
        snapshot = Snapshot.capture(cache)
        self._remember(key, snapshot)
        if (
            not self.disk_limit
            or (key in self.disk and key not in self.invalid)
            or snapshot.nbytes > self.disk_limit
        ):
            return {"bytes": snapshot.nbytes, "written_bytes": 0, "write_s": 0}
        start = time.monotonic()
        metadata = {
            "schema": "1",
            "identity": self.identity,
            "offset": str(snapshot.offset),
            "count": str(len(snapshot.state)),
        }
        # Reserve header space before writing; account the actual file size below.
        if snapshot.nbytes + 65536 > self.disk_limit:
            return {"bytes": snapshot.nbytes, "written_bytes": 0, "write_s": 0}
        try:
            self._evict_disk(snapshot.nbytes + 65536)
            fd, temp = tempfile.mkstemp(prefix=".snapshot-", dir=self.directory)
        except OSError as error:
            return {
                "bytes": snapshot.nbytes,
                "written_bytes": 0,
                "write_s": time.monotonic() - start,
                "write_error": str(error),
            }
        try:
            with os.fdopen(fd, "wb") as stream:
                mx.save_safetensors(
                    stream,
                    {str(i): array for i, array in enumerate(snapshot.state)},
                    metadata,
                )
            size = Path(temp).stat().st_size
            if size > self.disk_limit:
                return {
                    "bytes": snapshot.nbytes,
                    "written_bytes": 0,
                    "write_s": time.monotonic() - start,
                }
            self._evict_disk(size)
            os.replace(temp, self._path(key))
            self.disk_bytes -= self.disk.pop(key, 0)
            self.invalid.discard(key)
            self.disk[key] = size
            self.disk_bytes += size
        except OSError as error:
            return {
                "bytes": snapshot.nbytes,
                "written_bytes": 0,
                "write_s": time.monotonic() - start,
                "write_error": str(error),
            }
        finally:
            Path(temp).unlink(missing_ok=True)
        return {
            "bytes": snapshot.nbytes,
            "written_bytes": size,
            "write_s": time.monotonic() - start,
        }

    def get(self, key, cache_type):
        key = tuple(key)
        start = time.monotonic()
        if key in self.memory:
            snapshot = self.memory.pop(key)
            self.memory[key] = snapshot
            return snapshot.restore(cache_type), {
                "tier": "memory",
                "bytes": snapshot.nbytes,
                "read_s": time.monotonic() - start,
            }
        if key not in self.disk or key in self.invalid:
            return None, {"tier": "miss", "read_s": 0}
        path = self._path(key)
        try:
            with path.open("rb", buffering=0) as stream:
                if self.nocache:
                    # Darwin F_NOCACHE; this requests uncached I/O, not a global
                    # purge or proof that every byte reached physical storage.
                    fcntl.fcntl(stream.fileno(), 48, 1)
                arrays, meta = mx.load(
                    stream, format="safetensors", return_metadata=True
                )
                if (
                    meta.get("schema") != "1"
                    or meta.get("identity") != self.identity
                    or int(meta["offset"]) != key[1]
                ):
                    raise ValueError("Snapshot identity or offset mismatch")
                count = int(meta["count"])
                if set(arrays) != {str(i) for i in range(count)}:
                    raise ValueError("Snapshot tensor set is incomplete")
                state = tuple(arrays[str(i)] for i in range(count))
                mx.eval(state)
            snapshot = Snapshot(state, key[1])
            cache = snapshot.restore(cache_type)
            self._remember(key, snapshot)
            self.disk.move_to_end(key)
            os.utime(path, None)
            return cache, {
                "tier": "ssd",
                "bytes": snapshot.nbytes,
                "read_s": time.monotonic() - start,
                "nocache": self.nocache,
            }
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            # Keep accounting for an unreadable file until eviction can remove
            # it. A read-only or failing disk must not crash active inference.
            self.invalid.add(key)
            return None, {
                "tier": "miss",
                "error": str(error),
                "read_s": time.monotonic() - start,
            }
