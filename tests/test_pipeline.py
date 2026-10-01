from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from odds import pipeline, sources
from odds.engines import Engine, EngineError
from odds.models import Run
from odds.pipeline import DEPTHS, Config, follow_up, investigate, new_run

FRAMING = {
    "success": "a one night stand within a 2 week trip", "horizon": "14 days",
    "reference_class": "male tourists in Seoul", "domain": "dating", "langs": ["ko"],
    "region": "Seoul", "counterparts": ["Korean women 20-30"],
    "unknowns": [
        {"question": "What is your gender?", "why": "changes the market", "blocking": True,
         "assume": "male"},
        {"question": "Do you speak Korean?", "why": "stage 3", "blocking": False,
         "assume": "no"},
    ],
}
PLAN = {
    "subquestions": [
        {"text": "base rate of casual sex among young Koreans", "angle": "base-rate",
         "queries_en": ["casual sex Korea survey"], "queries_local": ["한국 원나잇"]},
        {"text": "what do Korean women fear about foreign men", "angle": "other-side",
         "queries_en": ["Korea molka fear"], "queries_local": []},
        {"text": "where it happens", "angle": "market-map", "queries_en": ["Hongdae club"]},
    ],
    "hypotheses": [{"statement": "being foreign helps", "prior": 0.5},
                   {"statement": "being foreign hurts", "prior": 0.5}],
}


def research(n: int) -> dict[str, Any]:
    return {"summary": f"summary {n}", "dead_ends": ["naver cafe: login wall"], "findings": [
        {"claim": f"claim {n}a", "tag": "sourced", "quote": "q",
         "sources": [{"url": "https://example.org/study", "title": "Study", "kind": "study"}],
         "weight": 0.8, "bears_on": {"H1": "support", "H2": "against", "H9": "support"}},
        {"claim": f"claim {n}b", "tag": "sourced", "sources": [{"url": "javascript:alert(1)"}]},
        {"claim": f"claim {n}c", "tag": "nonsense", "sources": []},
    ]}


SYNTH = {
    "base_rate": {"reference_class": "tourists", "estimate": "1 in 6", "low": 0.1, "high": 0.25,
                  "evidence_ids": ["E1", "E999"]},
    "other_side": "they fear hidden cameras",
    "hypotheses": [{"id": "H1", "posterior": 0.7, "status": "supported", "reasoning": "E1"}],
    "strategies": [
        {"name": "Hongdae clubs", "stages": [
            {"name": "venue", "low": 0.8, "high": 0.95, "evidence_ids": ["E1"]},
            {"name": "mutual interest", "low": "10%", "high": "30%"}],
         "attempts": 6, "attempt_unit": "night out", "effort": 0.4,
         "signals": ["eye contact"], "bail": ["no-foreigner door policy"],
         "tests": ["night 1 in Itaewon"]},
        {"name": "apps", "stages": [{"name": "match", "low": 0.5, "high": 0.7},
                                    {"name": "meet", "low": 0.2, "high": 0.4}],
         "attempts": 10, "effort": 0.2},
    ],
    "verdict": "apps beat clubs", "levers": ["learn 50 phrases"],
    "gaps": [{"question": "How long are you staying?", "why": "attempt count"}],
    "followups": ["what about Busan"],
}


class Fake(Engine):
    def __init__(self, critic: list[str] | None = None, fail_research: int = 0) -> None:
        super().__init__("claude", "fake")
        self.calls: list[str] = []
        self.critic = critic or ["revise", "pass"]
        self.fail_research = fail_research
        self.n = 0
        self.final: dict[str, Any] = {"problems": [], "revisions": [],
                                      "verdict": "Best is {P1}; apps give {P2.per}.",
                                      "levers": ["learn 50 phrases"]}

    def ask(self, prompt: str, *, web: bool = False, timeout: float = 600.0) -> str:
        if "You wrote an analysis" in prompt:
            self.calls.append("finalize")
            return json.dumps(self.final)
        if "Reframe it" in prompt:
            self.calls.append("reframe")
            return json.dumps(FRAMING)
        if "Plan the research" in prompt:
            self.calls.append("plan")
            return json.dumps(PLAN)
        if "YOUR subquestion" in prompt:
            self.calls.append("research")
            self.n += 1
            if self.fail_research and self.n <= self.fail_research:
                raise EngineError("boom")
            return json.dumps(research(self.n))
        if "adversarial reviewer" in prompt:
            self.calls.append("critic")
            verdict = self.critic.pop(0)
            return json.dumps({"verdict": verdict, "problems": ["E2 is an anecdote"],
                               "evidence_fixes": [{"id": "E1", "tag": "speculation",
                                                   "weight": 0.1}],
                               "new_subquestions": [{"text": "check E2", "angle": "myth-test",
                                                     "queries_en": ["x"]}]
                               if verdict == "revise" else []})
        if "Synthesize." in prompt:
            self.calls.append("synth")
            return json.dumps(SYNTH)
        raise AssertionError("unexpected prompt")


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sources, "search", lambda *a, **k: [])


def config(engine: Engine, depth: str = "normal", answers: dict[str, str] | None = None,
           saved: list[str] | None = None) -> Config:
    answers = answers or {}
    return Config(engine, [engine], DEPTHS[depth], interactive=bool(answers),
                  ask_user=lambda q, why: answers.get(q, ""),
                  save=lambda r: saved.append(r.status) if saved is not None else None)


def test_full_investigation_with_critic_loop(tmp_path: Path) -> None:
    engine = Fake(critic=["revise"])
    run, _ = new_run("one night stand in Korea?", "normal", tmp_path)
    saved: list[str] = []
    investigate(run, config(engine, saved=saved))
    assert engine.calls[:2] == ["reframe", "plan"]
    assert engine.calls.count("critic") == 1
    assert engine.calls.count("synth") == 2  # first draft + after the critic's new research
    assert run.status == "done" and saved
    # 3 planned + 1 critic subquestion, all researched
    assert [s.status for s in run.subquestions] == ["done"] * 4
    # one shared study URL dedupes to a single source; javascript: URL dropped
    assert len(run.sources) == 1
    assert run.evidence[1].tag == "inferred"  # "sourced" with no usable URL is not sourced
    assert run.evidence[2].tag == "inferred"  # unknown tag repaired
    assert run.evidence[0].tag == "speculation"  # critic's fix applied
    h1, h2 = run.hypotheses
    assert "E1" in h1.support and "E1" in h2.against
    assert h1.posterior == 0.7 and h1.status == "supported"
    assert run.base_rate["evidence_ids"] == ["E1"]  # unknown ids filtered
    apps = next(s for s in run.strategies if s.name == "apps")
    clubs = next(s for s in run.strategies if s.name == "Hongdae clubs")
    assert clubs.stages[1].low == pytest.approx(0.10)
    assert clubs.signals == ["eye contact"] and clubs.tests
    assert run.strategies[0].overall.p50 >= run.strategies[1].overall.p50
    assert apps.sensitivity and clubs.overall.p90 >= clubs.overall.p10
    # non-interactive: blocking question became an assumption, gap recorded as open
    assert any("gender" in a for a in run.assumptions)
    assert any("How long" in a for a in run.assumptions)


def test_interactive_asks_blocking_and_gap_questions(tmp_path: Path) -> None:
    engine = Fake(critic=["pass"])
    run, _ = new_run("q", "normal", tmp_path)
    answers = {"What is your gender?": "male, straight",
               "How long are you staying?": "3 weeks"}
    investigate(run, config(engine, answers=answers))
    asked = {qa.question: qa.answer for qa in run.qa}
    assert asked == answers
    assert not any("gender" in a for a in run.assumptions)
    assert any("Korean" in a for a in run.assumptions)  # non-blocking -> assumed, not asked
    assert engine.calls.count("synth") == 2  # draft + re-synthesis with the gap answer


def test_quick_depth_skips_critic(tmp_path: Path) -> None:
    engine = Fake()
    run, _ = new_run("q", "quick", tmp_path)
    investigate(run, config(engine, "quick"))
    assert "critic" not in engine.calls


def test_one_failed_researcher_does_not_sink_the_run(tmp_path: Path) -> None:
    engine = Fake(critic=["pass"], fail_research=1)
    run, _ = new_run("q", "normal", tmp_path)
    investigate(run, config(engine))
    statuses = sorted(s.status for s in run.subquestions)
    assert statuses == ["done", "done", "failed"]
    assert run.status == "done"


def test_all_research_failing_fails_loudly_and_saves(tmp_path: Path) -> None:
    engine = Fake(fail_research=99)
    run, _ = new_run("q", "normal", tmp_path)
    saved: list[str] = []
    with pytest.raises(pipeline.PipelineError):
        investigate(run, config(engine, saved=saved))
    assert run.status == "failed" and saved[-1] == "failed"


def test_roundtrip_and_followup(tmp_path: Path) -> None:
    engine = Fake(critic=["pass", "pass"])
    run, directory = new_run("q", "normal", tmp_path)
    investigate(run, config(engine))
    run.save(directory)
    loaded = Run.load(directory)
    assert loaded.to_dict() == run.to_dict()
    before = len(loaded.subquestions)
    follow_up(loaded, "what about Busan", config(engine))
    assert len(loaded.subquestions) > before
    assert loaded.status == "done"
    assert any(qa.phase == "followup" for qa in loaded.qa)


def test_finalize_revises_model_and_fills_numbers(tmp_path: Path) -> None:
    engine = Fake(critic=["pass"])
    engine.final = {
        "problems": ["apps modelled with too few attempts"],
        "revisions": [{"id": "P2", "attempts": 30, "attempt_unit": "match",
                       "stages": [{"name": "match", "low": 0.5, "high": 0.7},
                                  {"name": "meet", "low": 0.2, "high": 0.4}]}],
        "verdict": "Apps win at {P2}, per match {P2.per}; clubs {P1}. Unknown {P9}.",
        "levers": [],
    }
    run, _ = new_run("q", "normal", tmp_path)
    investigate(run, config(engine))
    assert engine.calls[-1] == "finalize"
    apps = next(s for s in run.strategies if s.id == "P2")
    assert apps.attempts == 30 and apps.attempt_unit == "match"
    assert run.strategies[0].id == "P2"  # re-sorted after the revision
    assert "{P2}" not in run.verdict and "100%" not in run.verdict
    assert "over 30 matches" in run.verdict and "(likely range 9" in run.verdict
    assert "per match" in run.verdict and "{P9}" in run.verdict  # unknown ids left visible
    assert run.levers  # empty levers from finalize keep the earlier ones
    assert run.critiques[-1].verdict.startswith("finalize: revised P2")


def test_fill_formats_small_and_large_odds() -> None:
    from odds.models import Odds, Strategy
    from odds.pipeline import fill

    run = Run("r", "q", "now")
    run.strategies = [Strategy("P1", "a", attempts=4, attempt_unit="night out",
                               per_attempt=Odds(0.01, 0.042, 0.08, 0.04),
                               overall=Odds(0.1, 0.18, 0.3, 0.18))]
    out = fill("{P1} / {P1.per}", run)
    assert out == ("~18% over 4 night outs (likely range 10%-30%) / "
                   "~4.2% per night out (likely range 1.0%-8.0%)")
