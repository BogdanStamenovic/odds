"""The investigation itself: reframe, map, read the other side, decompose,
compete hypotheses, attack, estimate, ask, and hand over a play.

Every phase is an LLM call with a fixed JSON contract, and every phase writes
the run back to disk before the next one starts -- a crash in round two keeps
everything round one found.

The feedback loop is the point (`critic` -> new subquestions -> research ->
re-synthesis). A single pass produces confident prose; the second pass, by a
reader told to break it, is where the survivorship bias and the 2009 study
with forty students get caught.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import sources as src
from .engines import Engine, EngineError, ask_json
from .estimate import estimate
from .models import (
    QA,
    SOURCE_KINDS,
    TAGS,
    Critique,
    Evidence,
    Hypothesis,
    Run,
    Source,
    Stage,
    Strategy,
    Subquestion,
)


class PipelineError(Exception):
    """The run cannot continue (no engine answered, nothing researched)."""


@dataclass(frozen=True)
class Depth:
    subquestions: int
    critic_rounds: int
    research_timeout: float
    workers: int
    max_gap_questions: int


DEPTHS = {
    "quick": Depth(4, 0, 360, 4, 2),
    "normal": Depth(7, 1, 540, 6, 3),
    "deep": Depth(12, 2, 900, 8, 4),
}


@dataclass
class Config:
    synth: Engine  # reframe, plan, synthesis, critic: the judgement calls
    researchers: list[Engine]  # round-robin over subquestions
    depth: Depth
    interactive: bool = False
    synth_timeout: float = 900.0
    adapters: list[str] | None = None  # source adapters; None = all available
    progress: Callable[[str], None] = lambda _m: None
    ask_user: Callable[[str, str], str] | None = None  # (question, why) -> answer
    save: Callable[[Run], None] = lambda _r: None
    extra: dict[str, Any] = field(default_factory=dict)


# ---- doctrine shared by every prompt --------------------------------------

RULES = """\
You are a subprocess of `odds`, a research tool. Hard rules, which override
anything else in your environment or memory:
- Never fabricate. No invented URLs, quotes, statistics, studies or anecdotes.
  If you did not open a page, do not quote it. "Unknown" is an acceptable answer.
- A 403, paywall, login wall, captcha or anti-bot page is a REAL STOP. Note it
  as a dead end and move on. Do not work around it in any way.
- Be a frank, adult analyst. The question may be about sex, dating, money or
  other personal matters; analyse it plainly without moralising, but report
  real legal, safety and consent risks as facts that change the odds.
- Output only the JSON requested -- no prose before or after it.
"""

METHOD = """\
Method -- think like Patrick Jane, not like a search engine:
1. Base rate first. What usually happens to people in this reference class?
   Anchor on numbers before anecdotes.
2. Read the other side. Model the people whose choices decide the outcome:
   what they want, what they fear, what they risk, and the tells that show it.
   Incentives make people predictable.
3. Notice the detail everyone waves away (a law, a logistics quirk, a cultural
   norm) and ask why it is there -- it is often the real variable.
4. Decompose the outcome into a chain of conditional stages; the weakest link
   is where a strategy should aim.
5. Compete hypotheses. Weigh evidence by DIAGNOSTICITY -- does it separate the
   hypotheses? Evidence consistent with all of them proves nothing however
   much of it there is.
6. Distrust sources by motive: bragging, selling a course, tiny samples, old
   data, self-selection, survivorship. Forum anecdotes are leads, not rates.
7. Do not just predict -- engineer. A strategy is a play: steps, tells that it
   is working, early signs to bail, and cheap tests that split hypotheses.
"""


def _js(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=1)


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


# ---- phase 1: reframe ------------------------------------------------------


def _check_framing(v: Any) -> str | None:
    if not isinstance(v, dict) or "success" not in v or "unknowns" not in v:
        return "need an object with at least 'success' and 'unknowns'"
    return None


def reframe(run: Run, cfg: Config) -> None:
    cfg.progress("reframe: defining success and what we need to know about you")
    answered = [{"q": qa.question, "a": qa.answer} for qa in run.qa if qa.answer]
    prompt = f"""{RULES}
{METHOD}
Question under investigation: {run.question!r}
Answers the asker already gave: {_js(answered)}

Reframe it before any research. Return JSON:
{{
 "success": "precise definition of the outcome being estimated",
 "horizon": "the time window / number of tries implied, or your assumption",
 "reference_class": "the population whose base rate anchors this",
 "domain": "short label, e.g. dating/travel/consumer/finance",
 "langs": ["ISO 639-1 codes of local languages worth searching besides en"],
 "region": "place it is about, or empty",
 "counterparts": ["who on the other side decides the outcome"],
 "unknowns": [
   {{"question": "what to ask the asker", "why": "how it changes the answer",
     "blocking": true/false, "assume": "default assumption if not asked"}}
 ]
}}
Mark an unknown "blocking" ONLY if the whole market changes with it (e.g. the
asker's gender and orientation for a dating question). Everything else gets a
sensible default assumption. At most 6 unknowns, at most 2 blocking."""
    framing = ask_json(cfg.synth, prompt, timeout=cfg.synth_timeout, check=_check_framing)
    run.framing = framing
    asked = {qa.question for qa in run.qa}
    for unknown in framing.get("unknowns", []):
        q = str(unknown.get("question", "")).strip()
        if not q or q in asked:
            continue
        if unknown.get("blocking") and cfg.interactive and cfg.ask_user:
            answer = cfg.ask_user(q, str(unknown.get("why", "")))
            if answer:
                run.qa.append(QA(q, answer, str(unknown.get("why", "")), "intake"))
                continue
        assume = str(unknown.get("assume", "")).strip()
        if assume:
            run.assumptions.append(f"{q} -> assumed: {assume}")
    _log(run, cfg, f"framed: success = {framing.get('success')}")


# ---- phase 2: plan (subquestions + competing hypotheses) -------------------


def _check_plan(v: Any) -> str | None:
    if not isinstance(v, dict) or not v.get("subquestions") or not v.get("hypotheses"):
        return "need 'subquestions' and 'hypotheses' arrays, both non-empty"
    return None


def plan(run: Run, cfg: Config, *, extra: str = "", count: int | None = None) -> None:
    count = count or cfg.depth.subquestions
    cfg.progress(f"plan: splitting into {count} subquestions and competing hypotheses")
    langs = run.framing.get("langs") or []
    existing = [s.text for s in run.subquestions]
    hyps = [{"id": h.id, "statement": h.statement} for h in run.hypotheses]
    prompt = f"""{RULES}
{METHOD}
Question: {run.question!r}
Framing: {_js(run.framing)}
Asker facts: {_js([{"q": q.question, "a": q.answer} for q in run.qa])}
Assumptions: {_js(run.assumptions)}
Existing subquestions (do not repeat): {_js(existing)}
Existing hypotheses: {_js(hyps)}
{extra}
Plan the research. Return JSON:
{{
 "subquestions": [
   {{"text": "...", "angle": "base-rate|other-side|culture|market-map|logistics|risk|myth-test|...",
     "queries_en": ["2-5 word search queries"],
     "queries_local": ["the same idea in the local language, if any: {langs}"]}}
 ],
 "hypotheses": [ {{"statement": "a claim that competes with the others", "prior": 0.0-1.0}} ]
}}
Exactly {count} subquestions. Cover, in this priority: the base rate (hard
numbers), the other side's psychology and incentives, the market map (where /
through which channels this actually happens), the local cultural rules,
logistics, legal and safety risk, and at least one subquestion that tests a
popular myth. Queries must be SHORT keyword strings -- scholarly indexes match
every word. {"Give 3-5 NEW competing hypotheses." if not hyps else
"Add hypotheses only if a genuinely new rival explanation is needed; else []."}"""
    result = ask_json(cfg.synth, prompt, timeout=cfg.synth_timeout, check=_check_plan)
    round_no = len(run.critiques)
    for item in result.get("subquestions", [])[:count]:
        sid = run.next_id("Q", [s.id for s in run.subquestions])
        queries = [str(x) for x in (item.get("queries_en") or [])]
        queries += [str(x) for x in (item.get("queries_local") or [])]
        run.subquestions.append(Subquestion(
            sid, str(item.get("text", "")), str(item.get("angle", "")),
            queries[:8], round=round_no,
        ))
    for item in result.get("hypotheses", []) or []:
        statement = str(item.get("statement", "")).strip()
        if not statement:
            continue
        hid = run.next_id("H", [h.id for h in run.hypotheses])
        prior = _prob(item.get("prior"), 0.5)
        run.hypotheses.append(Hypothesis(hid, statement, prior, prior))
    _log(run, cfg, f"planned {len(result.get('subquestions', []))} subquestions, "
                   f"{len(run.hypotheses)} hypotheses")


# ---- phase 3: research -----------------------------------------------------


def _check_research(v: Any) -> str | None:
    if not isinstance(v, dict) or not isinstance(v.get("findings"), list):
        return "need an object with a 'findings' array"
    return None


def _leads(sq: Subquestion, cfg: Config, langs: list[str]) -> list[src.Hit]:
    """Scholarly / encyclopedic leads from the keyless adapters.

    For a web-capable engine these are a head start; for Ollama they are the
    only sources it will ever see.
    """
    hits: list[src.Hit] = []
    warnings: list[str] = []
    english = [q for q in sq.queries if q.isascii()][:2]
    local = [q for q in sq.queries if not q.isascii()][:1]
    for query in english:
        hits += src.search(query, adapters=cfg.adapters, limit=3, warn=warnings.append)
    for query in local:
        lang = (langs or ["en"])[0]
        hits += src.search(query, adapters=["wikipedia"], lang=lang, limit=3,
                           warn=warnings.append)
    seen: set[str] = set()
    unique = []
    for hit in hits:
        key = src.normalize_url(hit.url)
        if key not in seen:
            seen.add(key)
            unique.append(hit)
    return unique[:10]


def research_one(run: Run, sq: Subquestion, engine: Engine, cfg: Config) -> dict[str, Any]:
    langs = [str(x) for x in run.framing.get("langs") or []]
    leads = _leads(sq, cfg, langs)
    lead_text = "\n".join(
        f"- [{h.kind}] {h.title} ({h.year or 'n.d.'}) {h.url}\n  {h.snippet[:300]}"
        for h in leads
    ) or "(none)"
    hyps = "\n".join(f"{h.id}: {h.statement}" for h in run.hypotheses)
    if engine.has_web:
        how = (
            "Use web search and fetch. Search in English AND in the local "
            f"language(s) {langs or '[]'} -- local forums, local news, local "
            "statistics offices. Open the pages you cite. 4-10 searches is right."
        )
    else:
        how = (
            "You have NO web access. Work only from the leads below; anything not "
            "in them must be tagged 'inferred' or 'speculation', with no source."
        )
    prompt = f"""{RULES}
{METHOD}
Overall question: {run.question!r}
Framing: {_js(run.framing)}
Asker facts: {_js([{"q": q.question, "a": q.answer} for q in run.qa])}

YOUR subquestion ({sq.angle}): {sq.text}
Suggested queries: {_js(sq.queries)}
Competing hypotheses -- hunt for evidence that SEPARATES them:
{hyps}

Leads from scholarly / encyclopedia indexes (may be off-topic; judge them):
{lead_text}

{how}

Return JSON:
{{
 "summary": "3-6 sentences: what you learned, numbers first",
 "findings": [
   {{"claim": "one specific, checkable claim (with numbers where possible)",
     "tag": "sourced|inferred|speculation",
     "quote": "short verbatim quote from the source (sourced claims only)",
     "sources": [{{"url": "...", "title": "...", "kind": "{'|'.join(SOURCE_KINDS)}", "lang": "en"}}],
     "weight": 0.0-1.0,
     "bears_on": {{"H1": "support|against"}}}}
 ],
 "dead_ends": ["what was blocked or empty, and why"]
}}
5-12 findings. Include at least one detail others would dismiss, if you found
one. 'bears_on' only for hypotheses the claim actually discriminates between.
Every 'sourced' claim needs a URL you opened (or a lead above), and the
source must STATE it. Your own generalisation from a source -- a rate you
derived, a pattern across anecdotes, an application to this asker -- is
'inferred'. A guess with nothing behind it is 'speculation'. Expect a mix."""
    return dict(ask_json(engine, prompt, web=True, timeout=cfg.depth.research_timeout,
                         check=_check_research))


def research(run: Run, cfg: Config) -> None:
    todo = [s for s in run.subquestions if s.status == "open"]
    if not todo:
        return
    cfg.progress(f"research: {len(todo)} subquestions over "
                 f"{', '.join(e.name for e in cfg.researchers)}")
    jobs: dict[Any, tuple[Subquestion, Engine]] = {}
    with ThreadPoolExecutor(max_workers=cfg.depth.workers) as pool:
        for i, sq in enumerate(todo):
            engine = cfg.researchers[i % len(cfg.researchers)]
            sq.engine = engine.name
            jobs[pool.submit(research_one, run, sq, engine, cfg)] = (sq, engine)
        for future in as_completed(jobs):
            sq, engine = jobs[future]
            try:
                result = future.result()
            except (EngineError, OSError) as exc:
                sq.status = "failed"
                sq.summary = f"failed: {exc}"
                _log(run, cfg, f"{sq.id} failed on {engine.name}: {exc}")
                cfg.save(run)
                continue
            added = merge_findings(run, sq, result)
            sq.status, sq.summary = "done", str(result.get("summary", ""))
            for dead in result.get("dead_ends", []) or []:
                _log(run, cfg, f"{sq.id} dead end: {dead}", quiet=True)
            _log(run, cfg, f"{sq.id} done on {engine.name}: {added} findings -- {sq.text}")
            cfg.save(run)
    if not any(s.status == "done" for s in run.subquestions):
        raise PipelineError("every research call failed; nothing to reason over")


def merge_findings(run: Run, sq: Subquestion, result: dict[str, Any]) -> int:
    by_url = {src.normalize_url(s.url): s for s in run.sources}
    hyp_ids = {h.id for h in run.hypotheses}
    added = 0
    for item in result.get("findings", []) or []:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim", "")).strip()
        if not claim:
            continue
        source_ids: list[str] = []
        for s in item.get("sources", []) or []:
            if not isinstance(s, dict):
                continue
            url = str(s.get("url", "")).strip()
            if not url.startswith(("http://", "https://")):
                continue
            key = src.normalize_url(url)
            known = by_url.get(key)
            if known is None:
                kind = str(s.get("kind", "other"))
                kind = kind if kind in SOURCE_KINDS else "other"
                known = Source(
                    run.next_id("S", [x.id for x in run.sources]), url,
                    str(s.get("title", ""))[:300], kind, src.default_quality(kind),
                    str(s.get("lang", "en"))[:8],
                )
                run.sources.append(known)
                by_url[key] = known
            source_ids.append(known.id)
        tag = str(item.get("tag", "inferred"))
        tag = tag if tag in TAGS else "inferred"
        if tag == "sourced" and not source_ids:
            tag = "inferred"  # a "sourced" claim with no usable URL is not sourced
        eid = run.next_id("E", [e.id for e in run.evidence])
        run.evidence.append(Evidence(
            eid, claim[:1000], tag, source_ids, str(item.get("quote", ""))[:600], sq.id,
            _prob(item.get("weight"), 0.5),
        ))
        for hid, direction in (item.get("bears_on") or {}).items():
            hyp = next((h for h in run.hypotheses if h.id == hid), None)
            if hyp is None or hid not in hyp_ids:
                continue
            target = hyp.support if str(direction).startswith("sup") else hyp.against
            if eid not in target:
                target.append(eid)
        added += 1
    return added


# ---- phase 4-6: synthesis --------------------------------------------------


def _ledger(run: Run) -> str:
    quality = {s.id: s for s in run.sources}
    lines = []
    for e in run.evidence:
        cites = ", ".join(
            f"{sid}({quality[sid].kind},q{quality[sid].quality:.2f})"
            for sid in e.source_ids if sid in quality
        )
        marks = " ".join(
            f"{h.id}:{'+' if e.id in h.support else '-'}"
            for h in run.hypotheses if e.id in h.support or e.id in h.against
        )
        lines.append(f"{e.id} [{e.tag}, w{e.weight:.1f}] {e.claim}"
                     + (f" | src {cites}" if cites else "") + (f" | {marks}" if marks else ""))
    return "\n".join(lines)


def _check_synthesis(v: Any) -> str | None:
    if not isinstance(v, dict):
        return "need a JSON object"
    strategies = v.get("strategies")
    if not isinstance(strategies, list) or not strategies:
        return "need a non-empty 'strategies' array"
    for s in strategies:
        stages = s.get("stages") if isinstance(s, dict) else None
        if not isinstance(stages, list) or not stages:
            return "every strategy needs a non-empty 'stages' array"
        for st in stages:
            if not isinstance(st, dict) or "low" not in st or "high" not in st:
                return "every stage needs numeric 'low' and 'high'"
    return None


def synthesize(run: Run, cfg: Config) -> None:
    cfg.progress("synthesis: base rate, the other side, hypotheses, strategies")
    critique = run.critiques[-1] if run.critiques else None
    prior_strats = [{"name": s.name, "stages": [(st.name, st.low, st.high) for st in s.stages]}
                    for s in run.strategies]
    prompt = f"""{RULES}
{METHOD}
Question: {run.question!r}
Framing: {_js(run.framing)}
Asker facts: {_js([{"q": q.question, "a": q.answer} for q in run.qa])}
Assumptions in force: {_js(run.assumptions)}

Research summaries:
{chr(10).join(f"{s.id} ({s.angle}): {s.summary}" for s in run.subquestions if s.status == "done")}

Evidence ledger (id [tag, weight] claim | sources(kind, quality) | hypothesis marks):
{_ledger(run)}

Hypotheses:
{chr(10).join(f"{h.id} (prior {h.prior:.2f}): {h.statement}" for h in run.hypotheses)}
{"Previous strategies: " + _js(prior_strats) if prior_strats else ""}
{"Critic's objections you MUST address: " + _js(critique.problems) if critique else ""}

Synthesize. Every number must be traceable to evidence ids or be honestly
labelled a judgement in its rationale. Return JSON:
{{
 "base_rate": {{"reference_class": "...", "estimate": "e.g. 'about 1 in 8 per month'",
               "low": 0.0-1.0, "high": 0.0-1.0, "evidence_ids": ["E.."], "note": "..."}},
 "other_side": "what the counterparts want, fear and risk; the tells that show interest or refusal (one rich paragraph)",
 "hypotheses": [{{"id": "H1", "posterior": 0.0-1.0, "status": "supported|refuted|open",
                  "reasoning": "which DIAGNOSTIC evidence moved it"}}],
 "strategies": [
   {{"name": "...", "summary": "...", "steps": ["..."],
     "stages": [{{"name": "...", "low": 0.0-1.0, "high": 0.0-1.0,
                  "evidence_ids": ["E.."], "rationale": "..."}}],
     "attempts": <integer tries inside the asker's horizon>, "attempt_unit": "night out|week|...",
     "effort": 0.0-1.0, "cost": "money/time in plain words", "risks": ["..."],
     "signals": ["tells it is working"], "bail": ["early signs to switch"],
     "tests": ["cheap probe that splits hypotheses"]}}
 ],
 "verdict": "4-8 sentences: the honest bottom line, best play first. Do NOT type odds for a strategy -- write {{P1}}, {{P2}}... (overall odds over its attempts) or {{P1.per}} (per attempt); the tool fills in the simulated number WITH its unit and range (e.g. '~19% over 4 night outs (likely range 7%-41%)'), so do not repeat the unit or horizon around it. P-numbers follow the order of your strategies array.",
 "levers": ["the changes that move the odds most, most powerful first"],
 "gaps": [{{"question": "for the asker", "why": "what it changes"}}],
 "followups": ["questions worth a follow-up run"]
}}
2-5 strategies, each 3-6 stages, stage probabilities CONDITIONAL on the stages
before. A strategy's stages are ONE attempt (one night out, one app match,
one week); 'attempts' is how many such tries fit in the horizon -- never model
a whole trip as a single attempt, and a combined plan must be its own chain. Ranges must be honest: wide where evidence is thin. Include the
do-nothing / default approach as a baseline strategy when it is meaningful."""
    v = ask_json(cfg.synth, prompt, timeout=cfg.synth_timeout, check=_check_synthesis)
    apply_synthesis(run, v)
    for strategy in run.strategies:
        estimate(strategy)
    run.verdict = fill(run.verdict, run)  # before sorting: P-ids follow synthesis order
    run.strategies.sort(key=lambda s: -s.overall.p50)
    _log(run, cfg, "synthesized: " + "; ".join(
        f"{s.name} {s.overall.p50:.0%}" for s in run.strategies))


def apply_synthesis(run: Run, v: dict[str, Any]) -> None:
    valid_e = {e.id for e in run.evidence}

    def eids(items: Any) -> list[str]:
        return [str(x) for x in items or [] if str(x) in valid_e]

    base = v.get("base_rate") or {}
    if isinstance(base, dict):
        base["evidence_ids"] = eids(base.get("evidence_ids"))
        run.base_rate = base
    run.other_side = str(v.get("other_side", ""))
    for item in v.get("hypotheses", []) or []:
        hyp = next((h for h in run.hypotheses if h.id == item.get("id")), None)
        if hyp is None:
            continue
        hyp.posterior = _prob(item.get("posterior"), hyp.posterior)
        status = str(item.get("status", "open"))
        hyp.status = status if status in ("supported", "refuted", "open") else "open"
        hyp.reasoning = str(item.get("reasoning", ""))
    strategies = []
    for i, item in enumerate(v.get("strategies", []) or [], 1):
        stages = [
            Stage(str(st.get("name", f"stage {j}")), _num(st.get("low")), _num(st.get("high")),
                  eids(st.get("evidence_ids")), str(st.get("rationale", "")))
            for j, st in enumerate(item.get("stages", []) or [], 1) if isinstance(st, dict)
        ]
        strategies.append(Strategy(
            f"P{i}", str(item.get("name", f"strategy {i}")), str(item.get("summary", "")),
            _strs(item.get("steps")), stages,
            max(1, int(_num(item.get("attempts"), 1))), str(item.get("attempt_unit", "attempt")),
            _prob(item.get("effort"), 0.5), str(item.get("cost", "")), _strs(item.get("risks")),
            _strs(item.get("signals")), _strs(item.get("bail")), _strs(item.get("tests")),
        ))
    run.strategies = strategies
    run.verdict = str(v.get("verdict", ""))
    run.levers = _strs(v.get("levers"))
    run.followups = _strs(v.get("followups"))
    run.framing["gaps"] = [g for g in v.get("gaps", []) or [] if isinstance(g, dict)]


# ---- phase 7b: finalize -- reconcile the story with the simulation ----------

_PLACEHOLDER = re.compile(r"\{(P\d+)(\.per)?\}")


def fill(text: str, run: Run) -> str:
    """Replace {P1} / {P1.per} with simulated odds, so prose cannot drift from them."""
    by_id = {s.id: s for s in run.strategies}

    def sub(match: re.Match[str]) -> str:
        s = by_id.get(match.group(1))
        if s is None:
            return match.group(0)
        if match.group(2):
            o, unit = s.per_attempt, f"per {s.attempt_unit}"
        else:
            o, unit = s.overall, f"over {s.attempts} {_plural(s.attempt_unit, s.attempts)}"
        return f"~{_pct(o.p50)} {unit} (likely range {_pct(o.p10)}-{_pct(o.p90)})"

    return _PLACEHOLDER.sub(sub, text)


def _pct(p: float) -> str:
    # A simulated 99.97% printed as "100%" claims a certainty no ranged input supports.
    if p >= 0.995:
        return ">99%"
    if p < 0.001:
        return "<0.1%"
    return f"{p * 100:.0f}%" if p >= 0.095 else f"{p * 100:.1f}%"


def _plural(unit: str, n: int) -> str:
    # "day of active app use" -> "days of active app use": pluralize the head noun.
    head, sep, tail = unit.partition(" of ")
    if n == 1 or not head or head.endswith("s"):
        return unit
    return head + ("es" if head.endswith(("x", "ch", "sh")) else "s") + sep + tail


def _check_finalize(v: Any) -> str | None:
    if not isinstance(v, dict) or not str(v.get("verdict", "")).strip():
        return "need an object with a non-empty 'verdict'"
    return None


def finalize(run: Run, cfg: Config) -> None:
    """Show the judge its own numbers after simulation and make it reconcile.

    The synthesis writes a story and a set of stage ranges in one breath; the
    simulation then turns the ranges into odds the story never saw. In the first
    live run the prose recommended the strategy the numbers ranked last. This
    pass catches that: it may correct a strategy's stages or attempt count
    (re-simulated here), and it rewrites the verdict with placeholders that the
    code fills, so the final text cannot state a number the model did not make.
    """
    cfg.progress("finalize: checking the story against the simulated odds")
    table = [{
        "id": s.id, "name": s.name, "attempts": s.attempts, "attempt_unit": s.attempt_unit,
        "per_attempt_p50": round(s.per_attempt.p50, 4), "overall_p50": round(s.overall.p50, 4),
        "overall_p10_p90": [round(s.overall.p10, 4), round(s.overall.p90, 4)],
        "stages": [{"name": st.name, "low": st.low, "high": st.high} for st in s.stages],
        "biggest_swing": next(iter(s.sensitivity), ""),
    } for s in run.strategies]
    prompt = f"""{RULES}
You wrote an analysis; the tool then simulated your stage ranges. Here are the
simulated odds, which are now the truth about your own model:
{_js(table)}

Your draft verdict: {run.verdict}
Your draft levers: {_js(run.levers)}

Check: does the verdict recommend a strategy the numbers do not support? Is any
strategy mis-specified (a whole trip modelled as one attempt, an attempt count
that does not match the horizon {run.framing.get("horizon", "")!r}, a stage
range you would not defend)? Fix the MODEL if it is wrong, not the story to
match a broken model.

Return JSON:
{{
 "problems": ["what was inconsistent, if anything"],
 "revisions": [{{"id": "P1", "attempts": <int>, "attempt_unit": "...",
                "stages": [{{"name": "...", "low": 0.0-1.0, "high": 0.0-1.0}}]}}],
 "verdict": "4-8 sentences, best play first. Never type a strategy's odds -- write {{P1}} (overall) or {{P1.per}} (per attempt); the tool fills in the number WITH its unit and range, e.g. '~19% over 4 night outs (likely range 7%-41%)' -- so never repeat the unit or horizon around a placeholder.",
 "levers": ["most powerful first"]
}}
'revisions' may be empty. Only include 'stages' in a revision if you change them
(then give the full list for that strategy, names and evidence order kept)."""
    v = ask_json(cfg.synth, prompt, timeout=cfg.synth_timeout, check=_check_finalize)
    revised = []
    for rev in v.get("revisions", []) or []:
        s = next((x for x in run.strategies if x.id == rev.get("id")), None)
        if s is None:
            continue
        if rev.get("attempts") is not None:
            s.attempts = max(1, int(_num(rev.get("attempts"), s.attempts)))
        if rev.get("attempt_unit"):
            s.attempt_unit = str(rev["attempt_unit"])
        new_stages = rev.get("stages")
        if isinstance(new_stages, list) and new_stages:
            old = {st.name: st for st in s.stages}
            s.stages = [
                Stage(str(st.get("name", "")), _num(st.get("low")), _num(st.get("high")),
                      old[str(st.get("name"))].evidence_ids if str(st.get("name")) in old else [],
                      old[str(st.get("name"))].rationale if str(st.get("name")) in old
                      else "revised in finalize")
                for st in new_stages if isinstance(st, dict)
            ]
        estimate(s)
        revised.append(s.id)
    problems = _strs(v.get("problems"))
    run.verdict = fill(str(v.get("verdict", "")), run)
    run.levers = _strs(v.get("levers")) or run.levers
    run.strategies.sort(key=lambda s: -s.overall.p50)
    if problems or revised:
        run.critiques.append(Critique(len(run.critiques) + 1, problems, [],
                                      "finalize: revised " + (", ".join(revised) or "verdict")))
    _log(run, cfg, f"finalized: {len(problems)} inconsistencies, revised {revised or 'none'}")


# ---- phase 5: the critic ---------------------------------------------------


def _check_critique(v: Any) -> str | None:
    if not isinstance(v, dict) or v.get("verdict") not in ("pass", "revise"):
        return "need an object with verdict 'pass' or 'revise'"
    return None


def criticize(run: Run, cfg: Config, round_no: int) -> Critique:
    cfg.progress(f"critic round {round_no}: trying to break the analysis")
    strategies = [{
        "name": s.name, "overall_p50": round(s.overall.p50, 3),
        "stages": [{"name": st.name, "low": st.low, "high": st.high,
                    "evidence": st.evidence_ids, "why": st.rationale} for st in s.stages],
    } for s in run.strategies]
    prompt = f"""{RULES}
You are the adversarial reviewer. Your job is to BREAK this analysis, not to
polish it. Look for: numbers with no evidence behind them; anecdotes treated as
rates; survivorship and self-selection (who posts success stories?); sources
with a motive (selling courses, bragging); stale or tiny-sample studies;
evidence counted as support that is consistent with every hypothesis; missing
counterpart psychology; missing local-language or local-law angles; stage
probabilities that are not actually conditional; ranges narrower than the
evidence justifies; a strategy that ignores a real risk.

Question: {run.question!r}
Evidence ledger:
{_ledger(run)}
Hypotheses: {_js([{"id": h.id, "statement": h.statement, "posterior": h.posterior,
                   "status": h.status, "why": h.reasoning} for h in run.hypotheses])}
Strategies: {_js(strategies)}
Verdict as drafted: {run.verdict}

Return JSON:
{{
 "verdict": "pass|revise",
 "problems": ["specific, actionable objection, citing ids"],
 "evidence_fixes": [{{"id": "E..", "tag": "sourced|inferred|speculation", "weight": 0.0-1.0, "why": "..."}}],
 "new_subquestions": [{{"text": "...", "angle": "...", "queries_en": ["..."], "queries_local": ["..."]}}]
}}
'revise' only if fixing it would materially change the odds or the best play.
At most 4 new subquestions, only for gaps research can actually close."""
    v = ask_json(cfg.synth, prompt, timeout=cfg.synth_timeout, check=_check_critique)
    for fix in v.get("evidence_fixes", []) or []:
        e = run.evidence_by_id(str(fix.get("id")))
        if e is None:
            continue
        tag = str(fix.get("tag", e.tag))
        e.tag = tag if tag in TAGS else e.tag
        e.weight = _prob(fix.get("weight"), e.weight)
    new = [str(x.get("text", "")) for x in v.get("new_subquestions", []) or []
           if isinstance(x, dict)]
    critique = Critique(round_no, _strs(v.get("problems")), [t for t in new if t],
                        str(v.get("verdict")))
    run.critiques.append(critique)
    for item in (v.get("new_subquestions", []) or [])[:4]:
        if not isinstance(item, dict) or not item.get("text"):
            continue
        queries = _strs(item.get("queries_en")) + _strs(item.get("queries_local"))
        run.subquestions.append(Subquestion(
            run.next_id("Q", [s.id for s in run.subquestions]), str(item["text"]),
            str(item.get("angle", "critic")), queries[:8], round=round_no,
        ))
    _log(run, cfg, f"critic {round_no}: {critique.verdict}, {len(critique.problems)} problems, "
                   f"{len(critique.new_subquestions)} new subquestions")
    return critique


# ---- phase 8: ask about gaps ------------------------------------------------


def ask_gaps(run: Run, cfg: Config) -> bool:
    """Ask the user what would change the answer. True if anything was answered."""
    gaps = run.framing.get("gaps") or []
    asked = {qa.question for qa in run.qa}
    gaps = [g for g in gaps if g.get("question") and g["question"] not in asked]
    gaps = gaps[: cfg.depth.max_gap_questions]
    if not gaps:
        return False
    if not (cfg.interactive and cfg.ask_user):
        for g in gaps:
            run.assumptions.append(f"open question (not asked): {g['question']}")
        return False
    answered = False
    for g in gaps:
        answer = cfg.ask_user(str(g["question"]), str(g.get("why", "")))
        if answer:
            run.qa.append(QA(str(g["question"]), answer, str(g.get("why", "")), "gap"))
            answered = True
    return answered


# ---- the whole investigation ----------------------------------------------


def investigate(run: Run, cfg: Config) -> Run:
    try:
        if not run.framing:
            reframe(run, cfg)
            cfg.save(run)
        if not run.subquestions:
            plan(run, cfg)
            cfg.save(run)
        research(run, cfg)
        synthesize(run, cfg)
        cfg.save(run)
        for round_no in range(1, cfg.depth.critic_rounds + 1):
            critique = criticize(run, cfg, round_no)
            cfg.save(run)
            if critique.verdict == "pass":
                break
            research(run, cfg)
            synthesize(run, cfg)
            cfg.save(run)
        if ask_gaps(run, cfg):
            _log(run, cfg, "re-synthesizing with your answers")
            synthesize(run, cfg)
        finalize(run, cfg)
        run.status = "done"
    except (EngineError, PipelineError) as exc:
        run.status = "failed"
        _log(run, cfg, f"FAILED: {exc}")
        cfg.save(run)
        raise
    cfg.save(run)
    return run


def follow_up(run: Run, question: str, cfg: Config) -> Run:
    """Continue a finished run: new subquestions aimed at `question`, then the
    same research -> synthesis -> critic loop over the enlarged ledger."""
    run.status = "running"
    _log(run, cfg, f"follow-up: {question}")
    run.qa.append(QA(f"follow-up: {question}", "(asked)", "user follow-up", "followup"))
    count = max(2, cfg.depth.subquestions // 2)
    plan(run, cfg, count=count, extra=(
        f"FOLLOW-UP from the asker, focus every new subquestion on it: {question!r}\n"
        f"Current verdict: {run.verdict}"))
    cfg.save(run)
    research(run, cfg)
    synthesize(run, cfg)
    if cfg.depth.critic_rounds:
        critique = criticize(run, cfg, len(run.critiques) + 1)
        if critique.verdict == "revise":
            research(run, cfg)
            synthesize(run, cfg)
    finalize(run, cfg)
    run.status = "done"
    cfg.save(run)
    return run


# ---- helpers ---------------------------------------------------------------


def _log(run: Run, cfg: Config, message: str, *, quiet: bool = False) -> None:
    run.log.append(f"{_now()} {message}")
    if not quiet:
        cfg.progress(message)


def _num(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value or ""))
    if not match:
        return default
    number = float(match.group())
    return number / 100 if "%" in str(value) else number


def _prob(value: Any, default: float) -> float:
    if value is None:
        return default
    p = _num(value, default)
    if p > 1:
        p /= 100
    return min(max(p, 0.0), 1.0)


def _strs(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(x) for x in value or [] if str(x).strip()]


def new_run(question: str, depth: str, root: Path) -> tuple[Run, Path]:
    stamp = _dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")[:40].strip("-") or "question"
    run_id = f"{stamp}-{slug}"
    return Run(run_id, question, _now(), depth), root / run_id
