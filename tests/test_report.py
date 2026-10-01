from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest

from odds.models import Evidence, Hypothesis, Run, Source, Stage, Strategy
from odds.report import render_html, render_terminal

FIXTURE = Path(__file__).parent / "fixtures" / "sample_run.json"


@pytest.fixture
def data() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


@pytest.fixture
def run(data: dict[str, Any]) -> Run:
    return Run.from_dict(data)


def test_fixture_is_labelled_as_fixture(data: dict[str, Any]) -> None:
    assert data["verdict"].startswith("FIXTURE")
    langs = {s["lang"] for s in data["sources"]}
    assert "ko" in langs
    for s in data["sources"]:
        assert "example" in s["url"] or not s["url"].startswith("http")


def test_html_has_every_section(run: Run) -> None:
    out = render_html(run)
    assert out.startswith("<!DOCTYPE html>")
    for sid in ("playbook", "strategies", "odds", "levers", "ground", "ach", "evidence",
                "sources", "framing", "qa", "critique", "followups", "method"):
        assert f'<section id="{sid}">' in out, sid
    assert 'class="tree"' in out and 'class="chart"' in out
    assert "FIXTURE" in out
    assert "not measurements" in out
    for s in run.strategies:
        assert s.name in out
    assert "외국인과의" in out  # Korean source title survives unmangled


def test_html_is_self_contained(run: Run) -> None:
    out = render_html(run)
    assert "<script" not in out
    assert "<link" not in out
    assert "@import" not in out
    assert "url(" not in out.split("</style>")[0]
    assert not re.search(r'\bsrc="', out)


def test_playbook_rendered_for_each_strategy(run: Run) -> None:
    out = render_html(run)
    playbook = out.split('<section id="playbook">')[1].split("</section>")[0]
    assert playbook.count('<article class="play') == len(run.strategies)
    for label in ("Do this", "It is working if", "Switch or stop if", "Cheap tests first"):
        assert label in playbook
    best = max(run.strategies, key=lambda s: s.overall.p50)
    assert "play-lead" in playbook.split(best.name)[0][-400:]
    assert best.signals[0].replace("'", "&#x27;") in playbook


def test_strategies_ranked_by_overall(run: Run) -> None:
    out = render_html(run)
    ranked = sorted(run.strategies, key=lambda s: -s.overall.p50)
    bars = out.split('class="ranked"')[1]
    positions = [bars.index(s.name) for s in ranked]
    assert positions == sorted(positions)


def test_everything_is_escaped(data: dict[str, Any]) -> None:
    evil = "<script>alert('x')</script>"
    d = copy.deepcopy(data)
    d["question"] = evil
    d["verdict"] = evil
    d["other_side"] = evil
    d["evidence"][0]["claim"] = evil
    d["evidence"][0]["quote"] = evil
    d["hypotheses"][0]["statement"] = evil
    d["strategies"][0]["name"] = evil
    d["strategies"][0]["stages"][0]["name"] = evil
    d["strategies"][0]["signals"] = [evil]
    d["sources"][0]["title"] = evil
    d["levers"] = [evil]
    d["engines"] = {evil: evil}
    d["framing"] = {evil: evil}
    d["base_rate"] = {"statement": evil, evil: evil}
    out = render_html(Run.from_dict(d))
    assert "<script" not in out
    assert "&lt;script&gt;" in out


def test_non_http_urls_are_not_linked() -> None:
    run = Run(id="r", question="q", created="2026-10-01")
    bad = ["javascript:alert(1)", "data:text/html,<b>x</b>", "file:///etc/passwd",
           "ftp://example.com/x", "  JavaScript:alert(2)", "http://", ""]
    run.sources = [Source(id=f"S{i}", url=u, title=f"title {i}") for i, u in enumerate(bad)]
    run.sources.append(Source(id="SOK", url="https://example.com/ok", title="ok"))
    run.evidence = [Evidence(id="E1", claim="c", source_ids=[s.id for s in run.sources])]
    out = render_html(run)
    hrefs = re.findall(r'href="([^"]*)"', out)
    external = [h for h in hrefs if not h.startswith("#")]
    assert external and all(h == "https://example.com/ok" for h in external)
    assert "javascript:" not in out.lower()
    assert "link withheld" in out


def test_empty_run_renders() -> None:
    out = render_html(Run(id="", question="", created=""))
    assert out.startswith("<!DOCTYPE html>")
    assert "No strategies were produced" in out
    assert "No verdict was reached" in out
    assert 'id="playbook"' not in out
    assert "not measurements" in out
    text = render_terminal(Run(id="", question="", created=""))
    assert "(none)" in text


def test_failed_run_shows_banner() -> None:
    run = Run(id="f", question="Will it rain?", created="2026-10-01", status="failed",
              log=["research crashed"])
    out = render_html(run)
    assert "This run failed" in out
    assert "research crashed" in out


def test_garbage_numbers_do_not_crash() -> None:
    st = Strategy(id="s", name="s", stages=[Stage(name="x", low="abc", high=None)],  # type: ignore[arg-type]
                  effort="lots", attempts="many")  # type: ignore[arg-type]
    run = Run(id="g", question="q", created="not a date", strategies=[st])
    run.strategies[0].sensitivity = {"x": "big"}  # type: ignore[dict-item]
    assert "<svg" in render_html(run)
    assert "q" in render_terminal(run)


def test_ach_groups_by_diagnosticity() -> None:
    run = Run(id="a", question="q", created="2026-10-01")
    run.evidence = [Evidence(id="E1", claim="splits"), Evidence(id="E2", claim="everywhere"),
                    Evidence(id="E3", claim="one-sided"), Evidence(id="E4", claim="unused")]
    run.hypotheses = [
        Hypothesis(id="H1", statement="a", support=["E1", "E2"], against=["E3"]),
        Hypothesis(id="H2", statement="b", support=["E2"], against=["E1"]),
    ]
    out = render_html(run)
    ach = out.split('<section id="ach">')[1].split("</section>")[0]
    order = [ach.index(label) for label in
             ("Splits hypotheses", "Bears on some", "Consistent with all")]
    assert order == sorted(order)
    assert ach.index('"#ev-E1"') < ach.index("Bears on some") < ach.index('"#ev-E3"')
    assert ach.index("Consistent with all") < ach.index('"#ev-E2"')
    assert "1 evidence item not linked" in ach


def test_speculation_is_visually_distinct(run: Run) -> None:
    out = render_html(run)
    assert 'class="ev ev-speculation"' in out
    assert "tag-speculation" in out and "tag-sourced" in out and "tag-inferred" in out


def test_terminal_summary(run: Run) -> None:
    text = render_terminal(run)
    assert "\x1b[" not in text
    assert run.question in text
    assert "FIXTURE" in text
    ranked = sorted(run.strategies, key=lambda s: -s.overall.p50)
    idx = [text.index(f"{i}. {s.name}") for i, s in enumerate(ranked, 1)]
    assert idx == sorted(idx)
    assert f"Playbook: {ranked[0].name}" in text
    assert ranked[0].bail[0] in text
    assert "Top levers" in text and run.levers[2] in text and run.levers[3] not in text
    assert "Most sensitive stages" in text
    assert f"{len(run.sources)} sources, {len(run.evidence)} evidence" in text
    assert all(len(line) <= 100 for line in text.splitlines() if not line.startswith("     "))


def test_terminal_color_and_control_chars(run: Run) -> None:
    assert "\x1b[" in render_terminal(run, color=True)
    run.verdict = "safe \x1b]0;pwned\x07 text"
    text = render_terminal(run)
    assert "\x1b" not in text and "\x07" not in text
    assert "safe" in text
