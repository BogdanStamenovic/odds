"""The run record: everything one investigation produced, as plain dataclasses.

A run is saved as one `run.json`. Every stage of the pipeline reads and extends
it, which is what makes `odds followup` possible -- a follow-up is just another
pass over a record that already holds the evidence.

Probabilities are always *ranges* (`low`, `high`), never points. A point
estimate on a question like this would claim a precision the evidence does not
have; the range is the honest unit, and the Monte Carlo step turns ranges into
a distribution rather than pretending to multiply exact numbers.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = 1

# How a claim is known. The report colours by this, and the critic is told to
# hunt for speculation dressed up as sourced.
TAGS = ("sourced", "inferred", "speculation")

# What kind of thing a source is. Quality defaults hang off this (see
# `sources.default_quality`), and the critic discounts anecdote-heavy kinds.
SOURCE_KINDS = ("study", "statistic", "news", "guide", "forum", "reddit", "wiki", "other")


@dataclass
class Source:
    id: str
    url: str
    title: str = ""
    kind: str = "other"
    quality: float = 0.5  # 0..1, how much a claim from here should move belief
    lang: str = "en"
    note: str = ""


@dataclass
class Evidence:
    id: str
    claim: str
    tag: str = "sourced"
    source_ids: list[str] = field(default_factory=list)
    quote: str = ""
    subquestion_id: str = ""
    weight: float = 0.5  # researcher's own view of how diagnostic this is


@dataclass
class Subquestion:
    id: str
    text: str
    angle: str = ""  # base-rate | psychology | culture | logistics | risk | anecdote | ...
    queries: list[str] = field(default_factory=list)
    status: str = "open"  # open | done | failed
    engine: str = ""
    summary: str = ""
    round: int = 0


@dataclass
class Hypothesis:
    id: str
    statement: str
    prior: float = 0.5
    posterior: float = 0.5
    support: list[str] = field(default_factory=list)  # evidence ids
    against: list[str] = field(default_factory=list)  # evidence ids
    status: str = "open"  # supported | refuted | open
    reasoning: str = ""


@dataclass
class Stage:
    """One link in a strategy's chain. `low`/`high` are P(stage | previous stages)."""

    name: str
    low: float
    high: float
    evidence_ids: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass
class Odds:
    """Monte Carlo summary. `p*` are percentiles of the success probability."""

    p10: float = 0.0
    p50: float = 0.0
    p90: float = 0.0
    mean: float = 0.0


@dataclass
class Strategy:
    id: str
    name: str
    summary: str = ""
    steps: list[str] = field(default_factory=list)
    stages: list[Stage] = field(default_factory=list)
    attempts: int = 1  # independent tries inside the user's time window
    attempt_unit: str = "attempt"  # "night out", "week", ...
    effort: float = 0.5  # 0 trivial .. 1 enormous
    cost: str = ""
    risks: list[str] = field(default_factory=list)
    # Nothing is filtered for being frowned upon; it is priced instead. `lane`
    # says which kind of path this is, `exposure` what it costs if it goes wrong.
    lane: str = "clean"  # clean | grey | illegal (illegal for the asker)
    exposure: str = ""  # consequences and their likelihood, in plain words
    # The "play" half: what to watch for while running it. A strategy is not a
    # prediction to sit and wait on -- these let the user update mid-play.
    signals: list[str] = field(default_factory=list)  # tells that it is working
    bail: list[str] = field(default_factory=list)  # early signs to abandon or switch
    tests: list[str] = field(default_factory=list)  # cheap probes that split hypotheses
    per_attempt: Odds = field(default_factory=Odds)
    overall: Odds = field(default_factory=Odds)
    sensitivity: dict[str, float] = field(default_factory=dict)  # stage -> d(mean)


@dataclass
class Critique:
    round: int
    problems: list[str] = field(default_factory=list)
    new_subquestions: list[str] = field(default_factory=list)
    verdict: str = ""


@dataclass
class QA:
    question: str
    answer: str = ""
    why: str = ""
    phase: str = "intake"  # intake | gap


@dataclass
class Run:
    id: str
    question: str
    created: str
    depth: str = "normal"
    engines: dict[str, str] = field(default_factory=dict)
    framing: dict[str, Any] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    qa: list[QA] = field(default_factory=list)
    subquestions: list[Subquestion] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    strategies: list[Strategy] = field(default_factory=list)
    critiques: list[Critique] = field(default_factory=list)
    base_rate: dict[str, Any] = field(default_factory=dict)
    other_side: str = ""  # the "read the room" section: what the counterpart wants
    verdict: str = ""
    levers: list[str] = field(default_factory=list)
    followups: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    usage: dict[str, dict[str, float]] = field(default_factory=dict)  # engine -> calls/s/cost
    seconds: float = 0.0  # wall time across all passes, follow-ups included
    status: str = "running"  # running | done | failed
    schema: int = SCHEMA

    # ---- lookups ---------------------------------------------------------

    def source(self, source_id: str) -> Source | None:
        return next((s for s in self.sources if s.id == source_id), None)

    def evidence_by_id(self, evidence_id: str) -> Evidence | None:
        return next((e for e in self.evidence if e.id == evidence_id), None)

    def next_id(self, prefix: str, existing: list[str]) -> str:
        n = 1 + max((int(x[len(prefix):]) for x in existing
                     if x.startswith(prefix) and x[len(prefix):].isdigit()), default=0)
        return f"{prefix}{n}"

    # ---- persistence -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "run.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        tmp.replace(path)
        return path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Run:
        def build(kind: type, items: list[dict[str, Any]]) -> list[Any]:
            return [_build(kind, item) for item in items or []]

        strategies = []
        for item in data.get("strategies", []) or []:
            s = _build(Strategy, item)
            s.stages = build(Stage, item.get("stages", []))
            s.per_attempt = _build(Odds, item.get("per_attempt") or {})
            s.overall = _build(Odds, item.get("overall") or {})
            strategies.append(s)

        run = _build(cls, data)
        run.qa = build(QA, data.get("qa", []))
        run.subquestions = build(Subquestion, data.get("subquestions", []))
        run.sources = build(Source, data.get("sources", []))
        run.evidence = build(Evidence, data.get("evidence", []))
        run.hypotheses = build(Hypothesis, data.get("hypotheses", []))
        run.critiques = build(Critique, data.get("critiques", []))
        run.strategies = strategies
        return run

    @classmethod
    def load(cls, directory: Path) -> Run:
        return cls.from_dict(json.loads((directory / "run.json").read_text()))


def _build(kind: type, data: dict[str, Any]) -> Any:
    """Construct a dataclass from a dict, ignoring keys it does not know.

    Tolerant on purpose: run.json files outlive schema tweaks, and an old run
    that fails to load is worse than one that loads with a default.
    """
    names = {f.name for f in dataclasses.fields(kind)}
    return kind(**{k: v for k, v in data.items() if k in names})
