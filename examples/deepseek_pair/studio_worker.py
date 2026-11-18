"""vLLM V1 worker for the two-rank DeepSeek V4.1 MLX model."""

import os
import socket
import time

import mlx.core as mx
import numpy as np
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm_metal.v1.worker import MetalWorker
from vllm_metal.v1.sampling import Sampler
from vllm_metal.v1.sampling_batch import SamplingBatch, sample_from_logits
from shard import ROOT, Shard, send, receive, watch_peer


class StudioWorker(MetalWorker):
    def init_device(self):
        # A singleton torch control group is local. Use TCPStore on macOS
        # to avoid flock stalls in FileStore during Gloo initialization.
        self.distributed_init_method = "tcp://127.0.0.1:18714"
        os.environ["GLOO_SOCKET_IFNAME"] = "lo0"
        super().init_device()

    def load_model(self):
        peer = os.environ["DEEPSEEK_PEER"]
        deadline = time.monotonic() + 180
        while True:
            try:
                self.link = socket.create_connection((peer, 18712), timeout=3)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1)
        self.link.settimeout(300)
        self.link.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.control = self.link.makefile("rwb")
        send(self.control, (ROOT / "pair.auth").read_text().strip())
        if receive(self.control) != "ok":
            raise RuntimeError("Peer authentication failed")
        watch_peer(self.link)
        self.shard = Shard()
        if receive(self.control) != "ready":
            raise RuntimeError("Peer failed to load the checkpoint")
        self.states = {}
        self.sampler = Sampler()
        self.pending = None

    def get_supported_tasks(self):
        return ("generate",)

    def get_kv_cache_spec(self):
        # The scheduler accounts for sequence blocks. MLX allocates actual
        # compressed KV and Engram state per request on each rank.
        return {
            "deepseek.request_cache": FullAttentionSpec(
                block_size=self.cache_config.block_size,
                num_kv_heads=1,
                head_size=64,
                dtype=torch.float16,
            )
        }

    def determine_available_memory(self):
        spec = next(iter(self.get_kv_cache_spec().values()))
        blocks = (
            self.model_config.max_model_len + spec.block_size - 1
        ) // spec.block_size
        return spec.page_size_bytes * (blocks * self.scheduler_config.max_num_seqs + 2)

    def initialize_from_config(self, kv_cache_config):
        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks

    def _forward(self, jobs, drop, sample=None):
        command = {
            "jobs": jobs,
            "drop": list(drop),
            "sample": [key for key, _ in jobs] if sample is None else sample,
        }
        send(self.control, command)
        logits = self.shard.forward(command)
        if receive(self.control) != "ok":
            raise RuntimeError("Peer did not finish the model step")
        return logits

    def compile_or_warm_up_model(self):
        from vllm.v1.worker.worker_base import CompilationTimes

        start = time.monotonic()
        self._forward([("warmup", [0, 422])], [])
        self._forward([], ["warmup"])
        return CompilationTimes(language_model=time.monotonic() - start, encoder=0)

    def execute_model(self, scheduler_output):
        drop = set(scheduler_output.finished_req_ids)
        for key in drop:
            self.states.pop(key, None)
        for req in scheduler_output.scheduled_new_reqs:
            key, params = req.req_id, req.sampling_params
            seed = params.seed
            generator = (
                None
                if seed is None
                else torch.Generator(device="cpu").manual_seed(seed)
            )
            self.states[key] = dict(
                prompt=list(req.prompt_token_ids),
                output=[],
                pos=0,
                params=params,
                generator=generator,
            )
            drop.add(key)
        jobs = []
        sample = []
        for key, count in scheduler_output.num_scheduled_tokens.items():
            state = self.states[key]
            pos = state["pos"]
            known = state["prompt"] + state["output"]
            ids = known[pos : pos + count]
            if len(ids) != count:
                raise RuntimeError("Inconsistent vLLM schedule for the DeepSeek cache")
            jobs.append((key, ids))
            state["pos"] += count
            if state["pos"] == len(known):
                sample.append(key)
        logits = self._forward(jobs, drop, sample) if jobs or drop else {}
        self.pending = ([key for key, _ in jobs], sample, logits)
        return None

    def sample_tokens(self, grammar_output):
        if self.pending is None:
            return None
        keys, sample, logits = self.pending
        self.pending = None
        rows = {key: i for i, key in enumerate(keys)}
        tokens = [[] for _ in keys]
        logprobs = None
        if sample:
            states = [self.states[k] for k in sample]
            scores = mx.stack([logits[k] for k in sample])
            if grammar_output is not None:
                targets = dict(
                    zip(
                        grammar_output.structured_output_request_ids,
                        grammar_output.grammar_bitmask,
                    )
                )
                for i, key in enumerate(sample):
                    if key in targets:
                        bits = np.unpackbits(
                            targets[key].view(np.uint8), bitorder="little"
                        )[: scores.shape[-1]]
                        scores[i] = mx.where(
                            mx.array(bits.astype(bool)), scores[i], -float("inf")
                        )
            batch = SamplingBatch(
                [s["params"] for s in states],
                [s["prompt"] for s in states],
                [s["output"] for s in states],
                vocab_size=scores.shape[-1],
                generators={
                    i: s["generator"]
                    for i, s in enumerate(states)
                    if s["generator"] is not None
                },
            )
            result = sample_from_logits(scores, batch, self.sampler)
            logprobs = result.logprobs
            for key, token in zip(sample, result.token_ids):
                self.states[key]["output"].append(token)
                tokens[rows[key]] = [token]
        if logprobs is not None and len(sample) != len(keys):
            offsets = list(np.cumsum([0] + [len(row) for row in tokens]))
            logprobs = logprobs._replace(cu_num_generated_tokens=offsets)
        return ModelRunnerOutput(
            req_ids=keys,
            req_id_to_index=rows,
            sampled_token_ids=tokens,
            logprobs=logprobs,
        )

    def get_model(self):
        return self.shard.lm

    def shutdown(self):
        if hasattr(self, "control"):
            self.control.close()
            self.link.close()
        super().shutdown()
