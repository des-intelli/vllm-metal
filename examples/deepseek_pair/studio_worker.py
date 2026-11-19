"""vLLM V1 worker for the two-rank DeepSeek V4.1 MLX model."""

import os
import socket
import time

import mlx.core as mx
import numpy as np
import torch
from prefix_cache import prefix_keys, request_snapshots
from shard import ROOT, Shard, receive, send, watch_peer
from sharded_sampling import merge_candidates
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.outputs import DraftTokenIds, LogprobsLists, ModelRunnerOutput

from vllm_metal.v1.sampling import Sampler
from vllm_metal.v1.sampling_batch import SamplingBatch, sample_from_logits
from vllm_metal.v1.worker import MetalWorker


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
        expected = {
            "status": "ready",
            "protocol": 4,
            "signature": self.shard.runtime_signature,
            "prefix_enabled": self.shard.prefix_enabled,
            "features": self.shard.runtime_features,
        }
        if receive(self.control) != expected:
            raise RuntimeError("Peer runtime source or prefix configuration differs")
        self.states = {}
        self.sampler = Sampler()
        self.pending = None
        self.candidate_k = None
        self.draft_token_ids = None
        spec = self.vllm_config.speculative_config
        if self.shard.dspark_width:
            if (
                spec is None
                or spec.method != "dspark"
                or spec.num_speculative_tokens != self.shard.dspark_width
            ):
                raise ValueError(
                    "DSpark requires matching --speculative-config and DEEPSEEK_DSPARK_TOKENS"
                )
        elif spec is not None:
            raise ValueError(
                "Enable DEEPSEEK_DSPARK_TOKENS on both ranks for speculation"
            )

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

    def _forward(self, jobs, drop, sample=None, save=None, *, speculate=False):
        command = {
            "jobs": jobs,
            "drop": list(drop),
            "sample": [key for key, _ in jobs] if sample is None else sample,
            "save": {} if save is None else save,
            "speculate": speculate,
            "candidate_k": self.candidate_k,
            "draft_keys": [
                key
                for key, _ in jobs
                if key in self.states and self._can_draft(self.states[key]["params"])
            ],
        }
        send(self.control, command)
        logits = self.shard.forward(command)
        if receive(self.control) != "ok":
            raise RuntimeError("Peer did not finish the model step")
        return logits

    def _prepare_prefixes(self, new_keys, drop):
        store = self.shard.prefix_store
        candidates = {}
        for key in new_keys:
            state = self.states[key]
            prompt = state["prompt"]
            if len(prompt) <= 1024:
                continue
            checkpoint = len(prompt) - 1
            choices = prefix_keys(prompt, store.lengths | {checkpoint})
            candidates[key] = choices
            state["checkpoint"] = choices[-1]
        command = {"op": "probe", "drop": list(drop), "candidates": candidates}
        send(self.control, command)
        local = self.shard.probe(command)
        peer = receive(self.control)
        selected = {}
        for key, choices in local.items():
            common = set(map(tuple, choices)) & set(map(tuple, peer.get(key, [])))
            if common:
                selected[key] = max(common, key=lambda item: item[1])
        if not selected:
            return set()
        send(self.control, {"op": "restore", "selected": selected})
        local_ok = self.shard.restore(selected)
        peer_ok = receive(self.control)
        failed = set()
        for key, prefix in selected.items():
            if local_ok[key] and peer_ok.get(key):
                self.states[key]["reused"] = prefix[1]
                self.shard._prefix_event(
                    "prefix_hit", request=key, reused_tokens=prefix[1]
                )
            else:
                failed.add(key)
        # Drop BOTH rank copies on any restore failure before model execution.
        return failed

    def _snapshot_requests(self, keys, *, finished=False):
        if not self.shard.prefix_decode_end:
            return
        prefixes = request_snapshots(
            self.states, keys, stride=self.shard.prefix_stride, finished=finished
        )
        if not prefixes:
            return
        send(self.control, {"op": "snapshot", "prefixes": prefixes})
        self.shard.snapshot_requests(prefixes)
        if receive(self.control) != "ok":
            raise RuntimeError("Peer did not finish request snapshots")
        for key, prefix in prefixes.items():
            self.states[key]["saved_offset"] = prefix[1]

    def compile_or_warm_up_model(self):
        from vllm.v1.worker.worker_base import CompilationTimes

        start = time.monotonic()
        self._forward([("warmup", [0, 422])], [])
        self._forward([], ["warmup"])
        return CompilationTimes(language_model=time.monotonic() - start, encoder=0)

    def execute_model(self, scheduler_output):
        drop = set(scheduler_output.finished_req_ids)
        # Cache offset excludes the last sampled token, which has not been
        # evaluated. Save that exact processed prefix on completion or cancel.
        self._snapshot_requests(drop, finished=True)
        for key in drop:
            self.states.pop(key, None)
        new_keys = []
        for req in scheduler_output.scheduled_new_reqs:
            key, params = req.req_id, req.sampling_params
            seed = params.seed
            generator = (
                None
                if seed is None
                else torch.Generator(device="cpu").manual_seed(seed)
            )
            self.states[key] = {
                "prompt": list(req.prompt_token_ids),
                "output": [],
                "pos": 0,
                "params": params,
                "generator": generator,
                "reused": 0,
                "checkpoint": None,
                "saved_offset": 0,
            }
            new_keys.append(key)
            drop.add(key)
        if self.shard.prefix_enabled and new_keys:
            drop = self._prepare_prefixes(new_keys, drop)
        jobs, tails, sample, save, verify = [], [], [], {}, []
        scheduled_drafts = scheduler_output.scheduled_spec_decode_tokens
        keys = list(scheduler_output.num_scheduled_tokens)
        self.candidate_k = (
            self._candidate_width(keys) if self.shard.sharded_greedy else None
        )
        for key, count in scheduler_output.num_scheduled_tokens.items():
            state = self.states[key]
            pos = state["pos"]
            known = state["prompt"] + state["output"]
            drafts = scheduled_drafts.get(key, [])
            if drafts:
                if pos != len(known) - 1 or count != len(drafts) + 1:
                    raise RuntimeError("Inconsistent speculative scheduler positions")
                if any(token < 0 for token in drafts):
                    raise RuntimeError("Unresolved speculative placeholder tokens")
                verify.append((key, [known[-1]] + list(drafts)))
                continue
            end = pos + count
            if end > len(known):
                raise RuntimeError("Inconsistent vLLM schedule for the DeepSeek cache")
            start = max(pos, state["reused"])
            checkpoint = state["checkpoint"]
            if start < end:
                if checkpoint is not None and start < checkpoint[1] <= end:
                    split = checkpoint[1]
                    jobs.append((key, known[start:split]))
                    save[key] = checkpoint
                    if split < end:
                        tails.append((key, known[split:end]))
                else:
                    jobs.append((key, known[start:end]))
            state["pos"] = end
            if end == len(known):
                sample.append(key)
        tail_keys = {key for key, _ in tails}
        first_sample = [key for key in sample if key not in tail_keys]
        logits = self._forward(jobs, drop, first_sample, save) if jobs or drop else {}
        if tails:
            logits.update(
                self._forward(tails, [], [key for key in sample if key in tail_keys])
            )
        for key, prefix in save.items():
            self.states[key]["saved_offset"] = prefix[1]
        self._snapshot_requests([key for key in keys if key not in scheduled_drafts])
        verify_start = time.perf_counter()
        verify_logits = self._forward(verify, [], speculate=True) if verify else {}
        self.verification_seconds = time.perf_counter() - verify_start
        self.pending = (keys, sample, logits, verify_logits, scheduled_drafts)
        return None

    def sample_tokens(self, grammar_output):
        if self.pending is None:
            return None
        keys, sample, logits, verify_logits, drafts = self.pending
        self.pending = None
        if verify_logits:
            return self._sample_verification(
                keys, sample, logits, verify_logits, drafts, grammar_output
            )
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
                        strict=True,
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
            token_ids, logprobs = self._sample_rows(
                scores,
                [s["params"] for s in states],
                [s["prompt"] for s in states],
                [s["output"] for s in states],
                generators={
                    i: s["generator"]
                    for i, s in enumerate(states)
                    if s["generator"] is not None
                },
            )
            for key, token in zip(sample, token_ids, strict=True):
                self.states[key]["output"].append(token)
                tokens[rows[key]] = [token]
        if logprobs is not None and len(sample) != len(keys):
            offsets = list(np.cumsum([0] + [len(row) for row in tokens]))
            logprobs = logprobs._replace(cu_num_generated_tokens=offsets)
        self._propose_drafts(sample)
        return ModelRunnerOutput(
            req_ids=keys,
            req_id_to_index=rows,
            sampled_token_ids=tokens,
            logprobs=logprobs,
        )

    def _candidate_width(self, keys):
        """Use one representation for every normal and verification row in this step."""
        width = -1
        for key in keys:
            state = self.states[key]
            p = state["params"]
            if (
                p.temperature >= 1e-5
                or p.structured_outputs is not None
                or p.top_k > 0
                or p.top_p != 1.0
                or p.frequency_penalty
                or p.presence_penalty
                or p.repetition_penalty != 1.0
                or p.min_tokens > len(state["output"])
                or p.allowed_token_ids
                or p.bad_words_token_ids
                or p.logit_bias
                or p.logprob_token_ids
                or p.prompt_logprobs is not None
            ):
                return None
            if p.num_logprobs is not None:
                if not 0 <= p.num_logprobs <= 5:
                    return None
                width = max(width, p.num_logprobs)
        return width

    def _sample_rows(self, scores, params, prompts, outputs, generators):
        if self.candidate_k is not None:
            ids, fields = merge_candidates(scores, self.candidate_k)
            return ids, None if fields is None else LogprobsLists(*fields, None)
        batch = SamplingBatch(
            params, prompts, outputs, vocab_size=scores.shape[-1], generators=generators
        )
        result = sample_from_logits(scores, batch, self.sampler)
        return result.token_ids, result.logprobs

    @staticmethod
    def _can_draft(params):
        # Stochastic verification needs rejection sampling, and structured output
        # needs a grammar transition per accepted token. Keep their existing path.
        return (
            params.temperature < 1e-5
            and getattr(params, "structured_outputs", None) is None
            and getattr(params, "repetition_detection", None) is None
        )

    @staticmethod
    def _stopped(state):
        params = state["params"]
        token = state["output"][-1]
        return (
            len(state["output"]) >= params.max_tokens
            or token == params.eos_token_id
            or token in (params.stop_token_ids or [])
        )

    def _propose_drafts(self, keys):
        self.draft_token_ids = None
        if not self.shard.dspark_width:
            return
        requests, discard = [], []
        for key in keys:
            state = self.states[key]
            params = state["params"]
            if not self._can_draft(params) or self._stopped(state):
                discard.append(key)
                continue
            remaining = min(
                params.max_tokens - len(state["output"]),
                self.model_config.max_model_len
                - len(state["prompt"])
                - len(state["output"]),
            )
            width = min(self.shard.dspark_width, remaining - 1)
            if width > 0:
                requests.append((key, state["output"][-1], width))
            else:
                discard.append(key)
        # Empty requests are sent too, keeping both ranks on the same protocol.
        send(self.control, {"op": "draft", "requests": requests, "discard": discard})
        proposals = self.shard.draft_requests(requests, discard)
        if receive(self.control) != proposals:
            raise RuntimeError("DSpark ranks produced different drafts")
        if proposals:
            self.draft_token_ids = DraftTokenIds(
                list(proposals), list(proposals.values())
            )

    def take_draft_token_ids(self):
        result, self.draft_token_ids = self.draft_token_ids, None
        return result

    def _sample_verification(
        self, keys, normal, logits, verified, drafts, grammar_output
    ):
        sample_start = time.perf_counter()
        if grammar_output is not None:
            # A mixed grammar batch is never drafted by _can_draft. Applying its
            # single-position masks to normal rows still requires the same mask.
            targets = dict(
                zip(
                    grammar_output.structured_output_request_ids,
                    grammar_output.grammar_bitmask,
                    strict=True,
                )
            )
            for key in normal:
                if key in targets:
                    bits = np.unpackbits(
                        targets[key].view(np.uint8), bitorder="little"
                    )[: logits[key].shape[-1]]
                    logits[key] = mx.where(
                        mx.array(bits.astype(bool)), logits[key], -float("inf")
                    )
        flat, params, prompts, outputs, spans = [], [], [], [], {}
        generators = {}
        for key in keys:
            state = self.states[key]
            values = verified.get(key)
            if values is None:
                if key not in normal:
                    continue
                values = logits[key][None]
            start = len(flat)
            flat.extend(values)
            # Rank-local candidates already include everything required for greedy sampling.
            if self.candidate_k is None:
                for index in range(len(values)):
                    if key not in verified and state["generator"] is not None:
                        generators[start + index] = state["generator"]
                    params.append(state["params"])
                    prompts.append(state["prompt"])
                    outputs.append(state["output"] + list(drafts.get(key, []))[:index])
            spans[key] = (start, len(flat))
        scores = mx.stack(flat)
        token_ids, logprobs = self._sample_rows(
            scores, params, prompts, outputs, generators
        )
        generated, lengths, selected = [], {}, []
        for key in keys:
            if key not in spans:
                generated.append([])
                continue
            start, end = spans[key]
            choices = token_ids[start:end]
            if key in verified:
                accepted = 0
                for proposal, target in zip(drafts[key], choices, strict=False):
                    if proposal != target:
                        break
                    accepted += 1
                emitted = list(drafts[key][:accepted]) + [choices[accepted]]
            else:
                emitted = choices
            state = self.states[key]
            actual = []
            for token in emitted:
                state["output"].append(token)
                actual.append(token)
                if self._stopped(state):
                    break
            if key in verified:
                lengths[key] = len(actual)
                state["pos"] += len(actual)
                self.shard._prefix_event(
                    "dspark_verify",
                    request=key,
                    drafted=len(drafts[key]),
                    accepted=len(actual) - 1,
                    emitted=len(actual),
                )
            generated.append(actual)
            selected.extend(range(start, start + len(actual)))
        sample_seconds = time.perf_counter() - sample_start
        commit_start = time.perf_counter()
        send(self.control, {"op": "commit", "lengths": lengths})
        self.shard.commit_verification(lengths)
        if receive(self.control) != "ok":
            raise RuntimeError("Peer did not commit verification")
        self._snapshot_requests(list(lengths))
        commit_seconds = time.perf_counter() - commit_start
        if logprobs is not None:
            logprobs = LogprobsLists(
                logprobs.logprob_token_ids[selected],
                logprobs.logprobs[selected],
                logprobs.sampled_token_ranks[selected],
                list(np.cumsum([0] + [len(row) for row in generated])),
            )
        draft_start = time.perf_counter()
        self._propose_drafts(list(spans))
        self.shard._prefix_event(
            "dspark_step",
            requests=len(lengths),
            verify_s=getattr(self, "verification_seconds", 0),
            sample_s=sample_seconds,
            commit_s=commit_seconds,
            draft_s=time.perf_counter() - draft_start,
        )
        return ModelRunnerOutput(
            req_ids=keys,
            req_id_to_index={key: i for i, key in enumerate(keys)},
            sampled_token_ids=generated,
            logprobs=logprobs,
        )

    def get_model(self):
        return self.shard.lm

    def shutdown(self):
        if hasattr(self, "control"):
            self.control.close()
            self.link.close()
        super().shutdown()
