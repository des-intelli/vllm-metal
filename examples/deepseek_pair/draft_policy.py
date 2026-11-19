"""Select speculative width from measured seconds per emitted token at each batch size."""

from dataclasses import dataclass


@dataclass
class _Cost:
    seconds_per_token: float
    samples: int
    last_step: int


class DraftWidthPolicy:
    def __init__(self, limit):
        self.widths = tuple(sorted({0, min(2, limit), limit}))
        self.costs = {}
        self.steps = {}

    def observe(self, batch, width, seconds, emitted):
        if width not in self.widths or emitted < batch or seconds <= 0:
            return
        step = self.steps[batch] = self.steps.get(batch, 0) + 1
        table = self.costs.setdefault(batch, {})
        value = seconds / emitted
        if width not in table:
            table[width] = _Cost(value, 1, step)
        else:
            cost = table[width]
            cost.samples += 1
            # Early observations include compilation and cache warm-up. Keep
            # adapting as prompt mix and the number of accepted tokens change.
            cost.seconds_per_token += (value - cost.seconds_per_token) / min(
                cost.samples, 12
            )
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
