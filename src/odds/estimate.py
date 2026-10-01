"""Turn ranged stage probabilities into odds, and find the lever that matters.

Each stage's (low, high) is read as an 80% interval on a logit-normal
distribution. Logit space keeps samples inside (0, 1) and makes "5-15%" and
"85-95%" equally wide in the way people mean them, which a plain normal on the
probability scale does not.

Two simplifications are stated here so nobody files them as oversights:

* Stages are sampled independently. In reality a strong first impression
  raises every later stage too; ignoring that correlation narrows the spread.
* Attempts are independent: overall = 1 - (1 - p)^attempts, with p drawn once
  per sample (so it is the same person with the same skill every night). Real
  attempts learn and fatigue; this is the naive repeat.

The numbers are a screen, not a verdict -- the report says so.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from .models import Odds, Stage, Strategy

Z80 = 1.2815515655446004  # standard-normal quantile for the 90th percentile
EPS = 1e-4


def _logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def clean_range(low: float, high: float) -> tuple[float, float]:
    """Models write 30 for 30%, swap the ends, or give 0 and 1. Repair, don't trust."""
    if low > 1 or high > 1:
        low, high = low / 100.0, high / 100.0
    low, high = min(low, high), max(low, high)
    return min(max(low, EPS), 1 - EPS), min(max(high, EPS), 1 - EPS)


def sample_stage(stage: Stage, rng: random.Random) -> float:
    low, high = clean_range(stage.low, stage.high)
    lo, hi = _logit(low), _logit(high)
    mu, sigma = (lo + hi) / 2, max((hi - lo) / (2 * Z80), 1e-6)
    return _sigmoid(rng.gauss(mu, sigma))


def _summary(samples: Sequence[float]) -> Odds:
    ordered = sorted(samples)
    n = len(ordered)

    def pct(q: float) -> float:
        return ordered[min(n - 1, max(0, int(q * (n - 1) + 0.5)))]

    return Odds(p10=pct(0.10), p50=pct(0.50), p90=pct(0.90), mean=sum(ordered) / n)


def simulate(
    stages: Sequence[Stage],
    attempts: int = 1,
    *,
    n: int = 20000,
    seed: int = 7,
    pinned: dict[int, float] | None = None,
) -> tuple[Odds, Odds]:
    """(per-attempt odds, odds over `attempts` tries). Seeded, so reruns agree."""
    if not stages:
        return Odds(), Odds()
    rng = random.Random(seed)
    attempts = max(1, int(attempts))
    per, overall = [], []
    for _ in range(n):
        p = 1.0
        for i, stage in enumerate(stages):
            p *= pinned[i] if pinned and i in pinned else sample_stage(stage, rng)
        per.append(p)
        overall.append(1.0 - (1.0 - p) ** attempts)
    return _summary(per), _summary(overall)


def sensitivity(strategy: Strategy, *, n: int = 6000) -> dict[str, float]:
    """Tornado swing per stage: mean overall odds with the stage at its high end
    minus at its low end. The biggest swing is where effort pays most."""
    swings: dict[str, float] = {}
    for i, stage in enumerate(strategy.stages):
        low, high = clean_range(stage.low, stage.high)
        _, at_high = simulate(strategy.stages, strategy.attempts, n=n, pinned={i: high})
        _, at_low = simulate(strategy.stages, strategy.attempts, n=n, pinned={i: low})
        swings[stage.name] = round(at_high.mean - at_low.mean, 4)
    return dict(sorted(swings.items(), key=lambda kv: -kv[1]))


def estimate(strategy: Strategy) -> Strategy:
    for stage in strategy.stages:
        stage.low, stage.high = clean_range(stage.low, stage.high)
    strategy.per_attempt, strategy.overall = simulate(strategy.stages, strategy.attempts)
    strategy.sensitivity = sensitivity(strategy)
    return strategy
