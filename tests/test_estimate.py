from __future__ import annotations

from odds.estimate import clean_range, estimate, simulate
from odds.models import Stage, Strategy


def test_clean_range_repairs_percent_and_order() -> None:
    assert clean_range(30, 10) == (0.10, 0.30)
    low, high = clean_range(0.0, 1.0)
    assert 0 < low < 0.01 and 0.99 < high < 1


def test_single_stage_interval_roughly_matches_input() -> None:
    per, _ = simulate([Stage("s", 0.2, 0.4)], n=20000)
    assert 0.17 < per.p10 < 0.23
    assert 0.37 < per.p90 < 0.43


def test_chain_multiplies_and_attempts_raise_odds() -> None:
    stages = [Stage("a", 0.5, 0.5), Stage("b", 0.4, 0.4)]
    per, overall = simulate(stages, attempts=5, n=2000)
    assert abs(per.p50 - 0.2) < 0.01
    assert abs(overall.p50 - (1 - 0.8 ** 5)) < 0.01


def test_seeded_runs_agree() -> None:
    stages = [Stage("a", 0.1, 0.6), Stage("b", 0.3, 0.9)]
    assert simulate(stages, 3) == simulate(stages, 3)


def test_sensitivity_ranks_the_widest_stage_first() -> None:
    strategy = Strategy("P1", "x", stages=[Stage("narrow", 0.5, 0.55), Stage("wide", 0.1, 0.9)],
                        attempts=2)
    estimate(strategy)
    assert next(iter(strategy.sensitivity)) == "wide"
    assert strategy.sensitivity["wide"] > strategy.sensitivity["narrow"] >= 0
    assert strategy.overall.p50 >= strategy.per_attempt.p50


def test_empty_strategy_is_zero() -> None:
    per, overall = simulate([], 3)
    assert per.p50 == 0 and overall.p50 == 0
