"""Verify multi-token outputs against fixed target logits without loading weights."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mlx.core as mx
import numpy as np
import torch
from draft_policy import DraftWidthPolicy
from sharded_sampling import merge_candidates, shard_candidates
from studio_worker import StudioWorker
from vllm.sampling_params import SamplingParams

from vllm_metal.v1.sampling import Sampler
from vllm_metal.v1.sampling_batch import SamplingBatch, sample_from_logits


def scores(*tokens):
    return mx.array(
        [[10.0 if i == token else -10.0 for i in range(16)] for token in tokens]
    )


class TestWorkerVerification(unittest.TestCase):
    @patch("studio_worker.send")
    @patch("studio_worker.receive")
    def test_adaptive_pause_and_resume_respects_remaining_budget(self, receive, send):
        worker = self.worker(
            {
                "a": SamplingParams(temperature=0, max_tokens=10),
                "b": SamplingParams(temperature=0, max_tokens=3),
            }
        )
        worker.model_config = SimpleNamespace(max_model_len=100)
        worker.shard.dspark_width = 5
        worker.draft_policy = Mock()
        worker.draft_policy.choose.side_effect = [0, 5]
        worker.shard.draft_requests = Mock(side_effect=[{}, {"a": [4] * 5, "b": [4]}])
        receive.side_effect = [{}, {"a": [4] * 5, "b": [4]}]
        StudioWorker._propose_drafts(worker, ["a", "b"])
        self.assertIsNone(worker.draft_token_ids)
        worker.shard.draft_requests.assert_called_with([], [], ["a", "b"])
        self.assertEqual(worker.last_draft[:2], ({"a", "b"}, 0))
        StudioWorker._propose_drafts(worker, ["a", "b"])
        worker.shard.draft_requests.assert_called_with(
            [("a", 7, 5), ("b", 7, 1)], [], []
        )
        self.assertEqual(worker.draft_token_ids.draft_token_ids, [[4] * 5, [4]])
        self.assertIsNone(worker.last_draft)

    def test_adaptive_draft_width_uses_observed_cost_per_emitted_token(self):
        policy = DraftWidthPolicy(5)
        emitted_per_row = {0: 1, 1: 2, 2: 2, 5: 3}
        for batch, best, costs in (
            (8, 0, {0: 0.1, 1: 0.24, 2: 0.32, 5: 0.6}),
            (4, 1, {0: 0.1, 1: 0.15, 2: 0.24, 5: 0.3}),
            (1, 5, {0: 0.05, 1: 0.09, 2: 0.08, 5: 0.1}),
        ):
            chosen = []
            for _ in range(60):
                width = policy.choose(batch)
                emitted = emitted_per_row[width] * batch
                policy.observe(batch, width, costs[width], emitted)
                chosen.append(width)
            self.assertEqual(set(chosen), set(costs))
            self.assertEqual(chosen[-10:], [best] * 10)
        # Reconsider all modes when a formerly expensive verification width
        # starts emitting more tokens at a lower cost.
        for _ in range(1400):
            width = policy.choose(8)
            policy.observe(
                8,
                width,
                {0: 0.2, 1: 0.18, 2: 0.06, 5: 0.6}[width],
                8 * emitted_per_row[width],
            )
        self.assertEqual(policy.choose(8), 2)

    def test_adaptive_width_maximizes_tokens_over_elapsed_time(self):
        policy = DraftWidthPolicy(5)
        for accepted_output in (1, 3, 1, 3):
            policy.observe(1, 0, 0.12, 1)
            policy.observe(1, 1, 0.16, 1)
            policy.observe(1, 2, 0.20, accepted_output)
            policy.observe(1, 5, 0.60, 3)
        # Width two emits eight tokens in 0.8 seconds, ahead of ordinary
        # decoding at one token per 0.12 seconds despite two rejected drafts.
        self.assertEqual(policy.choose(1), 2)

    def worker(self, params):
        worker = object.__new__(StudioWorker)
        worker.states = {
            key: {
                "prompt": [8, 9],
                "output": [7],
                "pos": 2,
                "params": value,
                "generator": torch.Generator().manual_seed(123),
            }
            for key, value in params.items()
        }
        worker.control = None
        worker.candidate_k = None
        worker.policy_sample = None
        worker.draft_policy = None
        worker.sampler = Sampler()
        worker.shard = SimpleNamespace(commit_verification=Mock(), _prefix_event=Mock())
        worker._snapshot_requests = Mock()
        worker._propose_drafts = Mock()
        return worker

    @patch("studio_worker.receive", return_value="ok")
    @patch("studio_worker.send")
    def test_independent_acceptance_stop_and_logprobs(self, send, receive):
        worker = self.worker(
            {
                "reject": SamplingParams(temperature=0, max_tokens=10, logprobs=3),
                "partial": SamplingParams(temperature=0, max_tokens=10, logprobs=3),
                "stop": SamplingParams(
                    temperature=0, max_tokens=10, stop_token_ids=[2], logprobs=3
                ),
                "budget": SamplingParams(temperature=0, max_tokens=3, logprobs=3),
            }
        )
        keys = list(worker.states)
        result = worker._sample_verification(
            keys,
            [],
            {},
            {
                "reject": scores(4, 5, 6),
                "partial": scores(1, 5, 6),
                "stop": scores(1, 2, 6),
                "budget": scores(1, 2, 6),
            },
            dict.fromkeys(keys, [1, 2]),
            None,
        )
        self.assertEqual(result.sampled_token_ids, [[4], [1, 5], [1, 2], [1, 2]])
        worker.shard.commit_verification.assert_called_once_with(
            {"reject": 1, "partial": 2, "stop": 2, "budget": 2}
        )
        self.assertEqual([worker.states[k]["pos"] for k in keys], [3, 4, 4, 4])
        self.assertEqual(result.logprobs.cu_num_generated_tokens, [0, 1, 3, 5, 7])
        self.assertEqual(
            result.logprobs.logprob_token_ids[:, 0].tolist(), [4, 1, 5, 1, 2, 1, 2]
        )

    @patch("studio_worker.receive", return_value="ok")
    @patch("studio_worker.send")
    def test_compact_verification_with_an_ordinary_row(self, send, receive):
        params = SamplingParams(temperature=0, max_tokens=16384, logprobs=0)
        worker = self.worker({"spec": params, "normal": params})
        worker.candidate_k = 0
        logits = scores(1, 5, 6, 9)
        packets = mx.stack(
            [
                shard_candidates(shard, rank, 0)
                for rank, shard in enumerate(mx.split(logits, 2, axis=-1))
            ],
            axis=-2,
        )
        result = worker._sample_verification(
            ["spec", "normal"],
            ["normal"],
            {"normal": packets[-1]},
            {"spec": packets[:3]},
            {"spec": [1, 2]},
            None,
        )
        self.assertEqual(result.sampled_token_ids, [[1, 5], [9]])
        self.assertEqual(result.logprobs.cu_num_generated_tokens, [0, 2, 3])
        self.assertEqual(result.logprobs.logprob_token_ids[:, 0].tolist(), [1, 5, 9])

    @patch("studio_worker.receive", return_value="ok")
    @patch("studio_worker.send")
    def test_mixed_random_request_preserves_rng_and_partial_prefill(
        self, send, receive
    ):
        params = SamplingParams(temperature=1, max_tokens=10, seed=123)
        worker = self.worker(
            {
                "spec": SamplingParams(temperature=0, max_tokens=10),
                "random": params,
                "prefill": params,
            }
        )
        random_scores = mx.zeros((1, 16))
        expected_rng = torch.Generator().manual_seed(123)
        expected = sample_from_logits(
            random_scores,
            SamplingBatch(
                [params], [[8, 9]], [[7]], vocab_size=16, generators={0: expected_rng}
            ),
            Sampler(),
        )
        result = worker._sample_verification(
            list(worker.states),
            ["random"],
            {"random": random_scores[0]},
            {"spec": scores(1, 2, 3)},
            {"spec": [1, 2]},
            None,
        )
        self.assertEqual(result.sampled_token_ids, [[1, 2, 3], expected.token_ids, []])
        self.assertTrue(
            torch.equal(
                worker.states["random"]["generator"].get_state(),
                expected_rng.get_state(),
            )
        )
        self.assertEqual(worker.states["prefill"]["output"], [7])

    def test_stochastic_and_grammar_requests_do_not_draft(self):
        self.assertTrue(StudioWorker._can_draft(SamplingParams(temperature=0)))
        self.assertFalse(StudioWorker._can_draft(SamplingParams(temperature=0.8)))
        self.assertFalse(
            StudioWorker._can_draft(
                SimpleNamespace(
                    temperature=0,
                    structured_outputs=object(),
                    repetition_detection=None,
                )
            )
        )

    def test_compact_shards_match_full_greedy_tokens_probabilities_and_tie_ranks(self):
        logits = mx.array(
            [
                [0.0, -3.0, 5.0, 2.0, 4.0, -1.0, 1.0, 7.0],
                [1000.0, 1002.0, 1008.0, 1003.0, 1005.0, 1008.0, 1008.0, 1001.0],
            ]
        )
        for k in (-1, 0, 1, 3):
            with self.subTest(k=k):
                p = SamplingParams(temperature=0, logprobs=None if k < 0 else k)
                worker = self.worker({"r": p})
                self.assertEqual(worker._candidate_width(["r"]), k)
                batch = SamplingBatch([p, p], [[], []], [[], []], vocab_size=8)
                full = sample_from_logits(logits, batch, worker.sampler)
                packets = mx.stack(
                    [
                        shard_candidates(row, rank, k)
                        for rank, row in enumerate(mx.split(logits, 2, axis=-1))
                    ],
                    axis=-2,
                )
                ids, fields = merge_candidates(packets, k)
                self.assertEqual(ids, full.token_ids)
                if k < 0:
                    self.assertIsNone(fields)
                    continue
                ids_table, probabilities, ranks = fields
                np.testing.assert_allclose(
                    probabilities, full.logprobs.logprobs, atol=1e-6
                )
                np.testing.assert_array_equal(ranks, full.logprobs.sampled_token_ranks)
                np.testing.assert_array_equal(
                    ids_table[0], full.logprobs.logprob_token_ids[0]
                )
                np.testing.assert_array_equal(ids_table[:, 0], ids)


if __name__ == "__main__":
    unittest.main()
