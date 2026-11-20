"""Select speculative width from measured seconds per emitted token at each batch size."""

from dataclasses import dataclass


@dataclass
class _Cost:
    mean_seconds: float
    mean_tokens: float
    samples: int
    last_step: int

    @property
    def seconds_per_token(self):
        return self.mean_seconds / self.mean_tokens


class DraftWidthPolicy:
    def __init__(self, limit):
        self.widths = tuple(sorted({0, min(1, limit), min(2, limit), limit}))
        self.costs = {}
        self.steps = {}

    def observe(self, batch, width, seconds, emitted):
        if width not in self.widths or emitted < batch or seconds <= 0:
            return
        step = self.steps[batch] = self.steps.get(batch, 0) + 1
        table = self.costs.setdefault(batch, {})
        if width not in table:
            table[width] = _Cost(seconds, emitted, 1, step)
        else:
            cost = table[width]
            cost.samples += 1
            # Estimate throughput over the same recency-weighted steps. Averaging
            # seconds per token directly overweights steps that reject drafts.
            window = min(cost.samples, 12)
            cost.mean_seconds += (seconds - cost.mean_seconds) / window
            cost.mean_tokens += (emitted - cost.mean_tokens) / window
            cost.last_step = step

    def choose(self, batch):
        table = self.costs.get(batch, {})
        for width in self.widths:
            if width not in table or table[width].samples < 3:
                return width
        oldest = min(self.widths, key=lambda w: table[w].last_step)
        if self.steps[batch] - table[oldest].last_step >= 96:
            return oldest
        return min(self.widths, key=lambda w: table[w].seconds_per_token)
