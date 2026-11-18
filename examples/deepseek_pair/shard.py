"""Model execution shared by the vLLM Metal worker and the second Mac."""

import hmac
import json
import os
from collections import defaultdict
from pathlib import Path
import socket
import mlx.core as mx

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
    import select, threading
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
        self.lm._ensure_engram_hash()
        if self.lm.engram_hash is None:
            raise RuntimeError("Engram hash initialization failed")
        self.caches = {}
        self.packed = {}
        print("Loaded DeepSeek-V4.1-Flash-1017 rank", self.group.rank(), flush=True)

    def _unpack(self, keys):
        cache = self.packed.pop(keys)
        for i, key in enumerate(keys):
            self.caches[key] = cache.extract(i)

    def forward(self, message):
        jobs, drop = message["jobs"], set(message["drop"])
        samples = set(message.get("sample", [key for key, _ in jobs]))
        batches = defaultdict(list)
        for key, tokens in jobs:
            batches[(len(tokens), key in samples)].append((key, tokens))
        groups = {tuple(key for key, _ in batch) for batch in batches.values()}
        for keys in list(self.packed):
            if keys not in groups or drop.intersection(keys):
                self._unpack(keys)
        for key in drop:
            self.caches.pop(key, None)
        results = {}
        for batch in batches.values():
            keys = tuple(key for key, _ in batch)
            ids = mx.array([tokens for _, tokens in batch])
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
            keep = 1 if keys[0] in samples else 0
            output = self.lm(ids, cache=[cache], logits_to_keep=keep).logits
            if keep:
                logits = output[:, -1, :]
                mx.eval(logits)
                results.update(zip(keys, logits))
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
                    send(stream, "ready")
                    while True:
                        try:
                            command = receive(stream)
                        except ConnectionError:
                            return
                        model.forward(command)
                        send(stream, "ok")


if __name__ == "__main__":
    follow()
