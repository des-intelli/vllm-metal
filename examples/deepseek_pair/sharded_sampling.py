"""Reduce a contiguous FP32 vocabulary shard to the statistics needed by greedy serving."""

import mlx.core as mx
import numpy as np


def shard_candidates(logits, rank, k):
    """k=-1 requests just the winner. k>=0 includes its probability and up to k alternatives."""
    width = logits.shape[-1]
    if (rank + 1) * width > 2**24:
        raise ValueError("Token IDs must fit exactly in the FP32 candidate packet")
    winner = mx.argmax(logits, axis=-1, keepdims=True)
    peak = mx.take_along_axis(logits, winner, axis=-1)
    base = rank * width
    fields = [(winner + base).astype(mx.float32), peak]
    if k >= 0:
        fields.extend(
            [
                mx.sum(logits == peak, axis=-1, keepdims=True).astype(mx.float32),
                mx.sum(mx.exp(logits - peak), axis=-1, keepdims=True),
            ]
        )
    if k > 1:
        count = min(width, k)
        indices = mx.argpartition(-logits, count - 1, axis=-1)[..., :count]
        fields.extend(
            [
                (indices + base).astype(mx.float32),
                mx.take_along_axis(logits, indices, axis=-1),
            ]
        )
    return mx.concatenate(fields, axis=-1)


def gather_candidates(logits, group, k):
    packet = shard_candidates(logits, group.rank(), k)
    # Preserve independent rank and vocabulary axes for both [B,V] and [B,T,V].
    ranks = mx.distributed.all_gather(packet[None], group=group)
    return mx.moveaxis(ranks, 0, -2)


def merge_candidates(packets, k):
    """Return sampled IDs, API logprob columns, and 1-based ranks including ties."""
    packets = np.asarray(packets)
    rows = np.arange(len(packets))
    peaks = packets[..., 1]
    winning_rank = np.argmax(peaks, axis=-1)
    winner = packets[rows, winning_rank, 0].astype(np.int32)
    if k < 0:
        return winner.tolist(), None
    peak = peaks[rows, winning_rank, None]
    offsets = peaks - peak
    denominator = np.log(
        (np.exp(offsets) * packets[..., 3]).sum(axis=-1, keepdims=True)
    )
    ranks = np.where(peaks == peak, packets[..., 2], 0).sum(axis=-1).astype(np.int32)
    ids = winner[:, None]
    values = -denominator
    if k == 1:
        ids = np.repeat(ids, 2, axis=-1)
        values = np.repeat(values, 2, axis=-1)
    elif k > 1:
        per_rank = (packets.shape[-1] - 4) // 2
        labels = packets[..., 4 : 4 + per_rank].reshape(len(rows), -1).astype(np.int32)
        scores = packets[..., 4 + per_rank :].reshape(len(rows), -1)
        order = np.lexsort((labels, -scores), axis=-1)[:, :k]
        ids = np.concatenate([ids, np.take_along_axis(labels, order, axis=-1)], axis=-1)
        alternatives = np.take_along_axis(scores, order, axis=-1) - peak - denominator
        values = np.concatenate([values, alternatives], axis=-1)
    return winner.tolist(), (ids, values, ranks)
