"""Model execution shared by the vLLM Metal worker and the second Mac."""

import hashlib
import hmac
import inspect
import json
import os
import socket
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
from prefix_cache import PrefixStore, model_identity
from sharded_sampling import gather_candidates

ROOT = Path(os.environ.get("DEEPSEEK_CLUSTER_ROOT", Path.home() / "vllm-cluster"))
MODEL = Path(
    os.environ.get("DEEPSEEK_MODEL", Path.home() / "models/DeepSeek-V4.1-Flash-1017")
)


def send(stream, message):
    stream.write(json.dumps(message, separators=(",", ":")).encode() + b"\n")
    stream.flush()


def receive(stream):
    line = stream.readline(4 * 1024 * 1024)
    if not line.endswith(b"\n"):
        raise ConnectionError("Peer disconnected or exceeded the control message limit")
    return json.loads(line)


def watch_peer(sock):
    """Exit a disconnected rank even if MLX is blocked inside a collective."""
    import select
    import threading
    from contextlib import closing

    def check():
        with closing(select.kqueue()) as queue:
            event = select.kevent(
                sock.fileno(),
                filter=select.KQ_FILTER_READ,
                flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,
            )
            changes = [event]
            while True:
                events = queue.control(changes, 1, None)
                changes = []
                if any(
                    item.flags & (select.KQ_EV_EOF | select.KQ_EV_ERROR)
                    for item in events
                ):
                    os._exit(1)

    threading.Thread(target=check, daemon=True).start()


def batch_jobs(jobs, samples):
    """Combine similar prefill widths with at most 25% padded token work.

    Decode stays separate. Sorting uses only control metadata, so both ranks
    issue exactly the same collectives without GPU-to-CPU routing reads.
    """
    buckets = defaultdict(list)
    for key, tokens in jobs:
        if not tokens:
            raise ValueError("Cannot execute an empty token chunk")
        buckets[(len(tokens) == 1, key in samples)].append((key, tokens))
    batches = []
    for bucket in buckets.values():
        current = []
        for item in sorted(bucket, key=lambda item: len(item[1])):
            if current and len(item[1]) * 4 > len(current[0][1]) * 5:
                batches.append(current)
                current = []
            current.append(item)
        if current:
            batches.append(current)
    return batches


class Shard:
    def __init__(self):
        from mlx_vlm.utils import sharded_load

        self.group = mx.distributed.init(strict=True, backend="jaccl")
        if self.group.size() != 2:
            raise RuntimeError("Expected both JACCL ranks")
        mx.set_cache_limit(4 * 1024**3)
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
        model, _ = sharded_load(str(MODEL), tensor_group=self.group)
        self.lm = model.language_model
        self.sharded_greedy = os.environ.get("DEEPSEEK_SHARDED_GREEDY", "0") == "1"
        self.candidate_shapes = set()
        if self.sharded_greedy and self.lm.head.sharding_group is None:
            raise ValueError("Sharded greedy requires a partitioned vocabulary")
        self.lm._ensure_engram_hash()
        if self.lm.engram_hash is None:
            raise RuntimeError("Engram hash initialization failed")
        self.caches = {}
        self.packed = {}
        self.dspark_width = int(os.environ.get("DEEPSEEK_DSPARK_TOKENS", "0"))
        self.batch_drafts = os.environ.get("DEEPSEEK_DSPARK_BATCH", "0") == "1"
        self.adaptive_drafts = os.environ.get("DEEPSEEK_DSPARK_ADAPTIVE", "0") == "1"
        if not 0 <= self.dspark_width <= 16:
            raise ValueError("DEEPSEEK_DSPARK_TOKENS must be between zero and 16")
        if (self.batch_drafts or self.adaptive_drafts) and not self.dspark_width:
            raise ValueError(
                "Batched and adaptive DSpark require a positive draft width"
            )
        self.draft = None
        self.draft_caches = {}
        self.draft_hidden = {}
        self.draft_start_positions = {}
        self.verification = {}
        if self.dspark_width:
            from mlx_vlm.speculative.drafters.deepseek_v41_dspark.split import (
                DeepseekV41DsparkSplitter,
            )

            self.draft = DeepseekV41DsparkSplitter().load(MODEL)
            self.draft.bind(self.lm)
            if (self.batch_drafts or self.adaptive_drafts) and not callable(
                getattr(self.draft, "draft_batch_greedy", None)
            ):
                raise RuntimeError(
                    "The selected model package does not support adaptive or batched DSpark"
                )
            mx.eval(self.draft.parameters())
        source = Path(inspect.getfile(type(self.lm))).parent
        signature = hashlib.sha256()
        if self.draft is not None:
            signature.update(Path(inspect.getfile(type(self.draft))).read_bytes())
        for path in [
            *(
                source / name
                for name in (
                    "language.py",
                    "fakequant.py",
                    "engram.py",
                    "sparse_attention.py",
                )
            ),
            *(
                Path(__file__).with_name(name)
                for name in (
                    "shard.py",
                    "studio_worker.py",
                    "prefix_cache.py",
                    "sharded_sampling.py",
                    "draft_policy.py",
                )
            ),
        ]:
            signature.update(path.read_bytes())
        self.runtime_signature = signature.hexdigest()
        self.prefix_enabled = os.environ.get("DEEPSEEK_PREFIX_CACHE", "0") == "1"
        self.prefix_decode_end = self.prefix_enabled and (
            os.environ.get("DEEPSEEK_PREFIX_DECODE_END", "0") == "1"
        )
        self.prefix_stride = int(os.environ.get("DEEPSEEK_PREFIX_STRIDE", "0"))
        if self.prefix_stride < 0:
            raise ValueError("Prefix checkpoint stride must be nonnegative")
        self.runtime_features = {
            "decode_end": self.prefix_decode_end,
            "stride": self.prefix_stride,
            "dspark_tokens": self.dspark_width,
            "batch_drafts": self.batch_drafts,
            "adaptive_drafts": self.adaptive_drafts,
            "sharded_greedy": self.sharded_greedy,
        }
        self.prefix_store = None
        if self.prefix_enabled:
            source = Path(inspect.getfile(type(self.lm))).parent
            self.prefix_store = PrefixStore(
                Path(os.environ.get("DEEPSEEK_PREFIX_DIR", ROOT / "prefix-cache"))
                / str(self.group.rank()),
                model_identity(MODEL, source),
                memory_bytes=int(
                    os.environ.get("DEEPSEEK_PREFIX_MEMORY_BYTES", str(2 * 1024**3))
                ),
                disk_bytes=int(os.environ.get("DEEPSEEK_PREFIX_DISK_BYTES", "0")),
                nocache=os.environ.get("DEEPSEEK_PREFIX_NOCACHE", "0") == "1",
            )
        print("Loaded DeepSeek-V4.1-Flash-1017 rank", self.group.rank(), flush=True)

    def _unpack(self, keys):
        cache = self.packed.pop(keys)
        for i, key in enumerate(keys):
            self.caches[key] = cache.extract(i)

    def _prefix_event(self, event, **fields):
        print(
            json.dumps(
                dict(event=event, epoch=time.time(), rank=self.group.rank(), **fields)
            ),
            flush=True,
        )

    def drop_requests(self, keys):
        keys = set(keys)
        for group in list(self.packed):
            if keys.intersection(group):
                self._unpack(group)
        for key in keys:
            self.caches.pop(key, None)
            self.draft_caches.pop(key, None)
            self.draft_hidden.pop(key, None)
            self.draft_start_positions.pop(key, None)

    def _defer_draft(self, key):
        parts = self.draft_hidden[key]
        window = self.draft.args.sliding_window
        skipped = sum(h.shape[1] for h in parts) - window
        if skipped <= 0:
            return
        # Once a complete new window is available the old rotating KV can be
        # rebuilt from it. Preserve its absolute RoPE coordinates across pauses.
        old = self.draft_caches.pop(key, None)
        start = self.draft_start_positions.get(key, 0)
        if old is not None:
            start = old[0].dspark_start_position + old[0].offset
        self.draft_start_positions[key] = start + skipped
        while skipped >= parts[0].shape[1]:
            skipped -= parts.pop(0).shape[1]
        if skipped:
            parts[0] = mx.take(parts[0], mx.arange(skipped, parts[0].shape[1]), axis=1)
            mx.eval(parts[0])

    def draft_requests(self, requests, discard=(), defer=()):
        if self.draft is None:
            raise RuntimeError("DSpark is not enabled")
        result = {}
        for key in discard:
            self.draft_hidden.pop(key, None)
        for key in defer:
            self._defer_draft(key)
        by_width = defaultdict(list)
        for key, anchor, width in requests:
            if not 0 < width <= self.dspark_width:
                raise ValueError("Draft width exceeds the configured maximum")
            hidden = mx.concatenate(self.draft_hidden.pop(key), axis=1)
            if key not in self.draft_caches:
                position = self.draft_start_positions.pop(key, 0)
                self.draft_caches[key] = (
                    self.draft.make_cache(start_position=position)
                    if position
                    else self.draft.make_cache()
                )
            by_width[width].append((key, anchor, hidden, self.draft_caches[key]))

        def greedy(logits):
            return mx.argmax(logits, axis=-1)

        for width, rows in by_width.items():
            if self.batch_drafts and len(rows) > 1:
                keys, anchors, hidden, caches = zip(*rows, strict=True)
                proposals = self.draft.draft_batch_greedy(
                    anchors, hidden, caches, width + 1, greedy
                )
                result.update(zip(keys, proposals.tolist(), strict=True))
            else:
                for key, anchor, hidden, cache in rows:
                    result[key] = self.draft.draft_block(
                        anchor, hidden, cache, width + 1, greedy
                    ).tolist()[0]
        return result

    def commit_verification(self, lengths):
        expected = {key for keys in self.verification for key in keys}
        if set(lengths) != expected:
            raise ValueError("Expected one commit length per verified request")
        for keys, (cache, generation, _) in self.verification.items():
            cache.validate_speculation([lengths[key] for key in keys], generation)
        for keys, (cache, generation, hidden) in self.verification.items():
            cache.commit_speculation([lengths[key] for key in keys], generation)
            for index, key in enumerate(keys):
                self.draft_hidden[key] = [hidden[index : index + 1, : lengths[key]]]
        self.verification.clear()

    def probe(self, message):
        self.drop_requests(message["drop"])
        return {
            key: self.prefix_store.available(candidates) if self.prefix_store else []
            for key, candidates in message["candidates"].items()
        }

    def restore(self, selected):
        results = {}
        cache_type = type(self.lm.make_cache()[0])
        for key, prefix in selected.items():
            cache, info = self.prefix_store.get(prefix, cache_type)
            if cache is not None:
                self.caches[key] = cache
            results[key] = cache is not None
            self._prefix_event("prefix_restore", request=key, tokens=prefix[1], **info)
        return results

    def _save_prefix(self, key, prefix):
        cache = self.caches.get(key)
        if cache is None:
            for group, packed in self.packed.items():
                if key in group:
                    cache = packed.extract(group.index(key))
                    break
        if cache is None:
            raise RuntimeError("Cannot snapshot an absent request")
        info = self.prefix_store.put(prefix, cache)
        self._prefix_event("prefix_save", request=key, tokens=prefix[1], **info)

    def snapshot_requests(self, prefixes):
        for key, prefix in prefixes.items():
            self._save_prefix(key, prefix)

    def forward(self, message):
        if self.verification:
            raise RuntimeError("Commit the preceding verification before forwarding")
        jobs, drop = message["jobs"], set(message["drop"])
        samples = set(message.get("sample", [key for key, _ in jobs]))
        speculate = message.get("speculate", False)
        draft_keys = set(message.get("draft_keys", ())) if self.draft else set()
        if speculate:
            if self.draft is None:
                raise RuntimeError("DSpark is not enabled")
            by_width = defaultdict(list)
            for job in jobs:
                by_width[len(job[1])].append(job)
            batches = list(by_width.values())
        else:
            batches = batch_jobs(jobs, samples)
        groups = {tuple(key for key, _ in batch) for batch in batches}
        for keys in list(self.packed):
            if keys not in groups or drop.intersection(keys):
                self._unpack(keys)
        for key in drop:
            self.caches.pop(key, None)
            self.draft_caches.pop(key, None)
            self.draft_hidden.pop(key, None)
            self.draft_start_positions.pop(key, None)
        results = {}
        for batch in batches:
            keys = tuple(key for key, _ in batch)
            lengths = [len(tokens) for _, tokens in batch]
            width = max(lengths)
            ids = mx.array(
                [[0] * (width - len(tokens)) + tokens for _, tokens in batch]
            )
            if keys in self.packed:
                cache = self.packed[keys]
            else:
                rows = [
                    self.caches.pop(k) if k in self.caches else self.lm.make_cache()[0]
                    for k in keys
                ]
                if len(rows) == 1:
                    cache = self.caches[keys[0]] = rows[0]
                else:
                    cache = self.packed[keys] = rows[0].merge(rows)
            if any(length != width for length in lengths):
                cache.prepare_append(lengths)
            keep = None if speculate else (1 if keys[0] in samples else 0)
            if speculate:
                generation = cache.start_speculation(width, block=True)
            capture = speculate or bool(draft_keys.intersection(keys))
            kwargs = (
                {}
                if not capture
                else {"capture_layer_ids": self.draft.config.target_layer_ids}
            )
            candidate_k = message.get("candidate_k")
            if keep != 0 and candidate_k is not None:
                kwargs["local_vocab"] = True
            output = self.lm(ids, cache=[cache], logits_to_keep=keep, **kwargs)
            if capture:
                hidden = mx.concatenate(output.hidden_states, axis=-1)
                mx.eval(hidden)
                if speculate:
                    self.verification[keys] = (cache, generation, hidden)
                else:
                    for index, (key, length) in enumerate(
                        zip(keys, lengths, strict=True)
                    ):
                        if key not in draft_keys:
                            continue
                        self.draft_hidden.setdefault(key, []).append(
                            hidden[index : index + 1, width - length :]
                        )
            if keep != 0:
                logits = output.logits if speculate else output.logits[:, -1, :]
                if candidate_k is not None:
                    if logits.shape[-1] != self.lm.head.weight.shape[0]:
                        raise RuntimeError(
                            "Expected rank-local logits from the language model"
                        )
                    candidates = gather_candidates(
                        logits, self.lm.head.sharding_group, candidate_k
                    )
                    shape = (logits.shape, candidate_k)
                    if shape not in self.candidate_shapes:
                        self.candidate_shapes.add(shape)
                        self._prefix_event(
                            "sharded_greedy",
                            local_shape=logits.shape,
                            top_k=candidate_k,
                            dense_bytes_per_rank=logits.nbytes,
                            candidate_bytes_per_rank=candidates.nbytes
                            // self.group.size(),
                        )
                    logits = candidates
                mx.eval(logits)
                results.update(zip(keys, logits, strict=True))
        for key, prefix in message.get("save", {}).items():
            self._save_prefix(key, prefix)
        return results


def follow():
    key = (ROOT / "pair.auth").read_text().strip()
    address = os.environ["DEEPSEEK_PEER"]
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((address, 18712))
        listener.listen(2)
        while True:
            conn, remote = listener.accept()
            with conn:
                if remote[0] != os.environ["DEEPSEEK_LEADER"]:
                    continue
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with conn.makefile("rwb") as stream:
                    if not hmac.compare_digest(receive(stream), key):
                        continue
                    send(stream, "ok")
                    watch_peer(conn)
                    model = Shard()
                    send(
                        stream,
                        {
                            "status": "ready",
                            "protocol": 5,
                            "signature": model.runtime_signature,
                            "prefix_enabled": model.prefix_enabled,
                            "features": model.runtime_features,
                        },
                    )
                    while True:
                        try:
                            command = receive(stream)
                        except ConnectionError:
                            return
                        operation = command.get("op", "forward")
                        if operation == "probe":
                            send(stream, model.probe(command))
                        elif operation == "restore":
                            send(stream, model.restore(command["selected"]))
                        elif operation == "forward":
                            model.forward(command)
                            send(stream, "ok")
                        elif operation == "snapshot":
                            model.snapshot_requests(command["prefixes"])
                            send(stream, "ok")
                        elif operation == "draft":
                            send(
                                stream,
                                model.draft_requests(
                                    command["requests"],
                                    command["discard"],
                                    command["defer"],
                                ),
                            )
                        elif operation == "commit":
                            model.commit_verification(command["lengths"])
                            send(stream, "ok")
                        else:
                            raise ValueError(f"Unknown peer operation: {operation}")


if __name__ == "__main__":
    follow()
