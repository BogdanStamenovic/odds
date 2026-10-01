"""Render a run as a self-contained HTML brief or a short terminal summary.

The HTML is one file with inline CSS and inline SVG and nothing fetched from
anywhere: it has to open from disk, offline, years later. Every string that
came out of a run is untrusted (LLM output, scraped web text), so all of it
goes through `_e` and only http(s) URLs ever become links.
"""

from __future__ import annotations

import html
import math
import re
import textwrap
from collections.abc import Iterable
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from odds.models import (
    SOURCE_KINDS,
    TAGS,
    Evidence,
    Hypothesis,
    Odds,
    Run,
    Source,
    Strategy,
)

PALETTE_SIZE = 6

# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)


def _num(value: Any, default: float = 0.0) -> float:
    # Run records are LLM-produced; a "0.4" string or a None must not crash the render.
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(x) else x


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _safe_url(url: Any) -> str | None:
    if not isinstance(url, str):
        return None
    url = url.strip()
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    if parsed.scheme.lower() in ("http", "https") and parsed.netloc:
        return url
    return None


def _anchor(prefix: str, ident: Any) -> str:
    return f"{prefix}-{re.sub(r'[^A-Za-z0-9_-]', '_', str(ident)) or '_'}"


def _pct(p: Any) -> str:
    x = _clamp(_num(p))
    if x == 0:
        return "0%"
    if x < 0.001:
        return "<0.1%"
    if x < 0.1:
        return f"{x * 100:.1f}%"
    if 0.995 < x < 1:
        return ">99%"
    return f"{x * 100:.0f}%"


def _range(lo: Any, hi: Any, dash: str = "–") -> str:
    a, b = _pct(lo), _pct(hi)
    if a.endswith("%") and b.endswith("%") and a[0].isdigit():
        a = a[:-1]
    return f"{a}{dash}{b}"


def _plural(unit: str, n: int) -> str:
    unit = unit or "attempt"
    if n == 1 or unit.endswith("s"):
        return unit
    return unit + "s"


def _wrap(text: str, width: int, max_lines: int) -> list[str]:
    lines = textwrap.wrap(" ".join(str(text).split()), width=width) or [""]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        lines[-1] = (last[: width - 1].rstrip() if len(last) >= width else last) + "…"
    return lines


def _truncate(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _date(created: str) -> str:
    try:
        dt = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
    except ValueError:
        return str(created)
    out = dt.strftime("%Y-%m-%d %H:%M")
    return out + (f" {dt.tzname()}" if dt.tzinfo else "")


def _ranked(run: Run) -> list[tuple[int, Strategy]]:
    """Strategies with their original index (which fixes their colour), best first."""
    pairs = list(enumerate(run.strategies))
    return sorted(pairs, key=lambda p: (-_num(p[1].overall.p50), -_num(p[1].overall.mean)))


def _color(i: int) -> str:
    return f"var(--c{i % PALETTE_SIZE + 1})"


def _link(url: Any, label: str, cls: str = "") -> str:
    safe = _safe_url(url)
    c = f' class="{cls}"' if cls else ""
    if safe is None:
        return f"<span{c}>{_e(label)}</span>"
    return (f'<a{c} href="{_e(safe)}" target="_blank" rel="noopener noreferrer nofollow">'
            f"{_e(label)}</a>")


def _domain(url: Any) -> str:
    if _safe_url(url) is None:
        return ""
    host = urlparse(str(url).strip()).netloc
    return host.removeprefix("www.")


def _tag_chip(tag: str) -> str:
    t = tag if tag in TAGS else "other"
    return f'<span class="tag tag-{t}">{_e(tag or "untagged")}</span>'


def _ev_ref(run: Run, eid: str) -> str:
    ev = run.evidence_by_id(eid)
    title = f' title="{_e(_truncate(ev.claim, 200))}"' if ev else ' title="unknown evidence id"'
    cls = "ref" if ev else "ref ref-missing"
    return f'<a class="{cls}" href="#{_anchor("ev", eid)}"{title}>{_e(eid)}</a>'


def _src_ref(run: Run, sid: str) -> str:
    src = run.source(sid)
    title = f' title="{_e(_truncate(src.title or src.url, 200))}"' if src else ""
    cls = "ref" if src else "ref ref-missing"
    return f'<a class="{cls}" href="#{_anchor("src", sid)}"{title}>{_e(sid)}</a>'


def _paras(text: str) -> str:
    blocks = [b.strip() for b in re.split(r"\n\s*\n", str(text)) if b.strip()]
    return "".join(f"<p>{_e(b)}</p>" for b in blocks)


def _list(items: Iterable[Any], ordered: bool = False) -> str:
    tag = "ol" if ordered else "ul"
    body = "".join(f"<li>{_e(x)}</li>" for x in items)
    return f"<{tag}>{body}</{tag}>"


def _none(what: str) -> str:
    return f'<p class="none">No {_e(what)}.</p>'


# --------------------------------------------------------------------------
# SVG: strategy decision tree
# --------------------------------------------------------------------------

_NW, _NH, _GX, _GY = 124, 72, 20, 14
_ROOT_W, _ST_W, _LEAF_W = 140, 164, 138


def _svg_lines(lines: list[str], x: float, y: float, step: float, cls: str,
               anchor: str = "start") -> str:
    out = []
    for i, line in enumerate(lines):
        out.append(f'<text x="{x:.1f}" y="{y + i * step:.1f}" class="{cls}" '
                   f'text-anchor="{anchor}">{_e(line)}</text>')
    return "".join(out)


def _tree_svg(run: Run) -> str:
    strats = run.strategies
    if not strats:
        return ""
    pad = 8
    n = len(strats)
    max_stages = max(len(s.stages) for s in strats)
    x_root = pad
    x_st = x_root + _ROOT_W + 38
    x_stage0 = x_st + _ST_W + _GX
    x_leaf = x_stage0 + max_stages * (_NW + _GX)
    width = x_leaf + _LEAF_W + pad

    q_lines = _wrap(run.question or "(no question)", 18, 7)
    root_h = 30 + len(q_lines) * 16
    rows_h = n * (_NH + _GY) - _GY
    inner_h = max(rows_h, root_h)
    height = inner_h + 2 * pad
    rows_top = pad + (inner_h - rows_h) / 2
    root_y = pad + (inner_h - root_h) / 2
    root_cy = root_y + root_h / 2

    parts: list[str] = [
        # Shrinks to fit down to 85% of natural size, then the container scrolls instead:
        # any smaller and the 11px node text stops being readable.
        (f'<svg class="tree" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} '
         f'{height:.0f}" style="width:100%;max-width:{width}px;min-width:{width * 0.85:.0f}px" '
         f'role="img" aria-label="Strategy decision tree">'),
        ('<defs><marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" '
         'markerHeight="7" orient="auto"><path d="M0,0 L8,4 L0,8 z" class="arrow"/></marker>'
         "</defs>"),
    ]

    parts.append(f'<g class="node root"><title>{_e(run.question)}</title>'
                 f'<rect x="{x_root}" y="{root_y:.1f}" width="{_ROOT_W}" height="{root_h}" '
                 f'rx="10"/>'
                 f'<text x="{x_root + 12}" y="{root_y + 20:.1f}" class="t-eyebrow">QUESTION'
                 f"</text>"
                 + _svg_lines(q_lines, x_root + 12, root_y + 38, 16, "t-root") + "</g>")

    for r, s in enumerate(strats):
        color = _color(r)
        y = rows_top + r * (_NH + _GY)
        cy = y + _NH / 2
        mid = (x_root + _ROOT_W + x_st) / 2
        parts.append(f'<path class="edge" d="M{x_root + _ROOT_W},{root_cy:.1f} C{mid:.1f},'
                     f'{root_cy:.1f} {mid:.1f},{cy:.1f} {x_st},{cy:.1f}"/>')

        att = max(1, int(_num(s.attempts, 1)))
        meta = f"×{att} {_plural(s.attempt_unit, att)} · effort {_num(s.effort):.2f}"
        parts.append(f'<g class="node strat"><title>{_e(s.name)} — {_e(s.summary)}</title>'
                     f'<rect x="{x_st}" y="{y:.1f}" width="{_ST_W}" height="{_NH}" rx="8"/>'
                     f'<rect x="{x_st}" y="{y:.1f}" width="5" height="{_NH}" rx="2" '
                     f'style="fill:{color}"/>'
                     + _svg_lines(_wrap(s.name, 20, 3), x_st + 14, y + 19, 15, "t-strat")
                     + f'<text x="{x_st + 14}" y="{y + _NH - 10:.1f}" class="t-meta">'
                     f"{_e(_truncate(meta, 30))}</text></g>")

        prev_right = x_st + _ST_W
        for i, st in enumerate(s.stages):
            x = x_stage0 + i * (_NW + _GX)
            parts.append(f'<line class="edge" x1="{prev_right}" y1="{cy:.1f}" '
                         f'x2="{x - 2}" y2="{cy:.1f}" marker-end="url(#arr)"/>')
            lo, hi = _clamp(_num(st.low)), _clamp(_num(st.high))
            lo, hi = min(lo, hi), max(lo, hi)
            bx, bw = x + 10, _NW - 20
            tip = f"{st.name} — {_range(lo, hi)}"
            if st.rationale:
                tip += f"\n{st.rationale}"
            parts.append(
                f'<g class="node stage"><title>{_e(tip)}</title>'
                f'<rect x="{x}" y="{y:.1f}" width="{_NW}" height="{_NH}" rx="8"/>'
                + _svg_lines(_wrap(st.name, 19, 3), x + 10, y + 16, 13, "t-stage")
                + f'<text x="{x + 10}" y="{y + 59:.1f}" class="t-range">{_e(_range(lo, hi))}'
                f"</text>"
                f'<rect x="{bx}" y="{y + 64:.1f}" width="{bw}" height="4" rx="2" class="track"/>'
                f'<rect x="{bx + lo * bw:.1f}" y="{y + 64:.1f}" '
                f'width="{max(2.0, (hi - lo) * bw):.1f}" height="4" rx="2" '
                f'style="fill:{color}"/></g>')
            prev_right = x + _NW

        dashed = " dashed" if len(s.stages) < max_stages else ""
        parts.append(f'<line class="edge{dashed}" x1="{prev_right}" y1="{cy:.1f}" '
                     f'x2="{x_leaf - 2}" y2="{cy:.1f}" marker-end="url(#arr)"/>')
        o, pa = s.overall, s.per_attempt
        parts.append(
            f'<g class="node leaf"><title>{_e(s.name)}: overall p50 {_pct(o.p50)}, '
            f'p10–p90 {_range(o.p10, o.p90)}</title>'
            f'<rect x="{x_leaf}" y="{y:.1f}" width="{_LEAF_W}" height="{_NH}" rx="8" '
            f'style="stroke:{color}"/>'
            f'<text x="{x_leaf + 12}" y="{y + 28:.1f}" class="t-leaf">{_e(_pct(o.p50))}'
            f'<tspan class="t-leafsub" dx="5">overall</tspan></text>'
            f'<text x="{x_leaf + 12}" y="{y + 47:.1f}" class="t-meta">p10–p90 '
            f"{_e(_range(o.p10, o.p90))}</text>"
            f'<text x="{x_leaf + 12}" y="{y + 62:.1f}" class="t-meta">per '
            f"{_e(_truncate(s.attempt_unit or 'attempt', 12))} {_e(_pct(pa.p50))}</text></g>")

    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------
# SVG: odds vs effort
# --------------------------------------------------------------------------


def _nice_max(x: float) -> float:
    for step in (0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.4, 0.5, 0.6, 0.8, 1.0):
        if x <= step:
            return step
    return 1.0


def _odds_scale(run: Run) -> float:
    top = max((_num(s.overall.p90) for s in run.strategies), default=0.0)
    return _nice_max(top * 1.08 if top else 0.01)


def _effort_chart(run: Run) -> str:
    if not run.strategies:
        return ""
    w, h = 760, 360
    ml, mr, mt, mb = 56, 24, 16, 50
    pw, ph = w - ml - mr, h - mt - mb
    ymax = _odds_scale(run)

    def px(e: float) -> float:
        return ml + _clamp(e) * pw

    def py(p: float) -> float:
        return mt + ph - min(_clamp(p), ymax) / ymax * ph

    out = [(f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" '
            f'aria-label="Overall odds versus effort, per strategy">')]
    for i in range(5):
        v = ymax * i / 4
        y = py(v)
        out.append(f'<line class="grid" x1="{ml}" x2="{w - mr}" y1="{y:.1f}" y2="{y:.1f}"/>'
                   f'<text class="axis" x="{ml - 8}" y="{y + 4:.1f}" text-anchor="end">'
                   f"{_e(_pct(v))}</text>")
    for i in range(5):
        v = i / 4
        x = px(v)
        out.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{mt}" y2="{mt + ph}"/>'
                   f'<text class="axis" x="{x:.1f}" y="{mt + ph + 18}" text-anchor="middle">'
                   f"{v:g}</text>")
    out.append(f'<text class="axis-title" x="{ml + pw / 2:.0f}" y="{h - 8}" '
               f'text-anchor="middle">effort  (0 trivial → 1 enormous)</text>'
               f'<text class="axis-title" transform="translate(14 {mt + ph / 2:.0f}) rotate(-90)" '
               f'text-anchor="middle">overall odds over the window</text>')

    labels: list[tuple[float, float, str, str, bool]] = []
    for i, s in enumerate(run.strategies):
        o, color = s.overall, _color(i)
        x = px(_num(s.effort))
        y50, y10, y90 = py(_num(o.p50)), py(_num(o.p10)), py(_num(o.p90))
        out.append(f'<g><title>{_e(s.name)}: p50 {_pct(o.p50)}, p10–p90 '
                   f'{_range(o.p10, o.p90)}, effort {_num(s.effort):.2f}</title>'
                   f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y10:.1f}" y2="{y90:.1f}" '
                   f'class="whisker" style="stroke:{color}"/>'
                   f'<line x1="{x - 5:.1f}" x2="{x + 5:.1f}" y1="{y10:.1f}" y2="{y10:.1f}" '
                   f'class="whisker" style="stroke:{color}"/>'
                   f'<line x1="{x - 5:.1f}" x2="{x + 5:.1f}" y1="{y90:.1f}" y2="{y90:.1f}" '
                   f'class="whisker" style="stroke:{color}"/>'
                   f'<circle cx="{x:.1f}" cy="{y50:.1f}" r="5.5" class="dot" '
                   f'style="fill:{color}"/></g>')
        left = x > ml + pw * 0.62
        labels.append((x, y50, f"{_truncate(s.name, 30)}  {_pct(o.p50)}", color, left))

    # Push labels apart vertically so close points stay readable.
    labels.sort(key=lambda t: t[1])
    placed: list[float] = []
    for x, y, text, _color_, left in labels:
        ly = y + 4
        for prev in placed:
            if abs(ly - prev) < 15:
                ly = prev + 15
        placed.append(ly)
        dx = -10 if left else 10
        anchor = "end" if left else "start"
        out.append(f'<text class="pt-label" x="{x + dx:.1f}" y="{ly:.1f}" '
                   f'text-anchor="{anchor}">{_e(text)}</text>')
    out.append("</svg>")
    return "".join(out)


def _ranked_bars(run: Run) -> str:
    if not run.strategies:
        return ""
    ymax = _odds_scale(run)
    rows = []
    for rank, (i, s) in enumerate(_ranked(run), 1):
        o, pa = s.overall, s.per_attempt
        lo = _clamp(_num(o.p10)) / ymax * 100
        hi = _clamp(_num(o.p90)) / ymax * 100
        mid = _clamp(_num(o.p50)) / ymax * 100
        att = max(1, int(_num(s.attempts, 1)))
        rows.append(
            f'<div class="rank-row"><div class="rank-n">{rank}</div>'
            f'<div class="rank-name"><a href="#{_anchor("st", s.id)}">{_e(s.name)}</a>'
            f'<span class="sub">×{att} {_e(_plural(s.attempt_unit, att))} · per '
            f'{_e(s.attempt_unit or "attempt")} {_e(_pct(pa.p50))} '
            f'({_e(_range(pa.p10, pa.p90))})</span></div>'
            f'<div class="rank-bar"><div class="band" style="left:{min(lo, 100):.1f}%;'
            f'width:{max(0.6, min(hi, 100) - min(lo, 100)):.1f}%;background:{_color(i)}">'
            f'</div><div class="tick" style="left:{min(mid, 100):.1f}%"></div></div>'
            f'<div class="rank-val"><b>{_e(_pct(o.p50))}</b> '
            f'<span class="sub">{_e(_range(o.p10, o.p90))}</span></div></div>')
    return (f'<div class="ranked"><div class="rank-head"><span></span><span>strategy</span>'
            f'<span>overall odds, p10–p90 band, p50 tick (scale 0–{_e(_pct(ymax))})</span>'
            f'<span>p50</span></div>{"".join(rows)}</div>')


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------


def _section(sid: str, title: str, body: str, lede: str = "") -> str:
    lede_html = f'<p class="lede">{lede}</p>' if lede else ""
    return (f'<section id="{sid}"><h2>{_e(title)}</h2>{lede_html}{body}</section>')


def _header(run: Run) -> str:
    status = run.status or "unknown"
    engines = " · ".join(f"{_e(k)}: {_e(v)}" for k, v in run.engines.items()) or "none"
    meta = (f'<span>{_e(_date(run.created))}</span><span>depth {_e(run.depth)}</span>'
            f"<span>engines {engines}</span>"
            f'<span class="status status-{_e(re.sub(r"[^a-z]", "", status.lower()))}">'
            f"{_e(status)}</span>")
    banner = ""
    if status == "failed":
        banner = ('<div class="banner bad">This run failed. The report shows whatever was '
                  "gathered before it stopped; treat every number as incomplete.</div>")
    elif status == "running":
        banner = ('<div class="banner">This run is still in progress; sections may be '
                  "missing or provisional.</div>")
    verdict = (f'<div class="verdict"><div class="eyebrow">Verdict</div>{_paras(run.verdict)}'
               f"</div>" if run.verdict.strip() else
               '<div class="verdict empty"><div class="eyebrow">Verdict</div>'
               "<p>No verdict was reached.</p></div>")

    tiles = []
    ranked = _ranked(run)
    if ranked:
        _, best = ranked[0]
        tiles.append(("Best strategy", _e(best.name), ""))
        tiles.append(("Its overall odds", _e(_pct(best.overall.p50)),
                      f"p10–p90 {_e(_range(best.overall.p10, best.overall.p90))}"))
    n_spec = sum(1 for ev in run.evidence if ev.tag == "speculation")
    tiles.append(("Evidence", str(len(run.evidence)),
                  f"{len(run.sources)} sources · {n_spec} speculation"))
    tiles.append(("Hypotheses", str(len(run.hypotheses)),
                  f"{len(run.critiques)} critique round{'s' if len(run.critiques) != 1 else ''}"))
    tile_html = "".join(f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div>'
                        f'<div class="s">{s}</div></div>' for k, v, s in tiles)
    return (f'<header><div class="eyebrow">odds · research brief · run {_e(run.id)}</div>'
            f"<h1>{_e(run.question or '(no question)')}</h1>"
            f'<div class="meta">{meta}</div>{banner}{verdict}'
            f'<div class="tiles">{tile_html}</div></header>')


def _framing(run: Run) -> str:
    if not run.framing and not run.assumptions:
        return ""
    body = ""
    if run.framing:
        body += _kv(run.framing)
    if run.assumptions:
        body += f"<h3>Assumptions</h3>{_list(run.assumptions)}"
    return _section("framing", "Framing & assumptions", body)


def _kv(d: dict[str, Any]) -> str:
    rows = []
    for k, v in d.items():
        if isinstance(v, (list, tuple)):
            v = ", ".join(str(x) for x in v)
        elif isinstance(v, dict):
            v = "; ".join(f"{a}: {b}" for a, b in v.items())
        rows.append(f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>")
    return f'<dl class="kv">{"".join(rows)}</dl>'


def _base_and_other(run: Run) -> str:
    cards = []
    br = dict(run.base_rate or {})
    if br:
        text = ""
        for key in ("statement", "summary", "text", "rate", "description"):
            if isinstance(br.get(key), str) and br[key].strip():
                text = br.pop(key)
                break
        big = ""
        if "low" in br and "high" in br:
            big = f'<div class="big">{_e(_range(br.pop("low"), br.pop("high")))}</div>'
        elif "value" in br:
            big = f'<div class="big">{_e(_pct(br.pop("value")))}</div>'
        refs = ""
        sids = br.pop("source_ids", None)
        if isinstance(sids, list) and sids:
            refs = '<p class="refs">sources ' + " ".join(_src_ref(run, str(s)) for s in sids) \
                + "</p>"
        note = br.pop("note", "")
        rest = _kv(br) if br else ""
        cards.append(f'<div class="card"><h3>Base rate</h3>{big}'
                     f"{f'<p>{_e(text)}</p>' if text else ''}"
                     f"{f'<p class=muted>{_e(note)}</p>' if note else ''}{refs}{rest}</div>")
    if run.other_side.strip():
        cards.append(f'<div class="card other"><h3>The other side</h3>'
                     f'<p class="muted small">What the counterpart is likely weighing.</p>'
                     f"{_paras(run.other_side)}</div>")
    if not cards:
        return ""
    return _section("ground", "Base rate & the other side",
                    f'<div class="two">{"".join(cards)}</div>')


def _strategies(run: Run) -> str:
    if not run.strategies:
        return _section("strategies", "Strategies", _none("strategies were produced"))
    tree = (f'<figure><div class="wide"><div class="scroll">{_tree_svg(run)}</div></div>'
            '<figcaption>Each row is a strategy as a chain of stages; a stage shows '
            "P(stage | all previous stages) as a range. The right column is the Monte Carlo "
            "result over the whole time window. Hover a node for its rationale.</figcaption>"
            "</figure>")
    details = []
    for i, s in _ranked(run):
        att = max(1, int(_num(s.attempts, 1)))
        stage_rows = "".join(
            f"<tr><td>{_e(st.name)}</td><td class='num'>{_e(_range(st.low, st.high))}</td>"
            f"<td>{_e(st.rationale)}</td>"
            f"<td>{' '.join(_ev_ref(run, x) for x in st.evidence_ids) or '—'}</td></tr>"
            for st in s.stages)
        table = (f'<div class="scroll"><table class="stages"><thead><tr><th>stage</th>'
                 f"<th>range</th><th>rationale</th><th>evidence</th></tr></thead>"
                 f"<tbody>{stage_rows}</tbody></table></div>" if s.stages else
                 _none("stages"))
        facts = [f"<dt>attempts</dt><dd>{att} × {_e(s.attempt_unit)}</dd>",
                 f"<dt>effort</dt><dd>{_num(s.effort):.2f}</dd>",
                 (f"<dt>per {_e(s.attempt_unit or 'attempt')}</dt>"
                  f"<dd>{_e(_odds_text(s.per_attempt))}</dd>"),
                 f"<dt>overall</dt><dd>{_e(_odds_text(s.overall))}</dd>"]
        if s.cost:
            facts.append(f"<dt>cost</dt><dd>{_e(s.cost)}</dd>")
        risks = f"<h4>Risks</h4>{_list(s.risks)}" if s.risks else ""
        details.append(
            f'<details id="{_anchor("st", s.id)}" style="--sc:{_color(i)}">'
            f"<summary><span class='swatch'></span><b>{_e(s.name)}</b>"
            f"<span class='sub'>overall {_e(_pct(s.overall.p50))} "
            f"({_e(_range(s.overall.p10, s.overall.p90))})</span></summary>"
            f"<div class='det'>{f'<p>{_e(s.summary)}</p>' if s.summary else ''}"
            f'<dl class="kv">{"".join(facts)}</dl>{table}{risks}</div></details>')
    return _section("strategies", "Strategy decision tree",
                    tree + f'<div class="details-list">{"".join(details)}</div>')


_PLAY_PARTS = (
    ("steps", "Do this", "step", ""),
    ("signals", "It is working if", "sig", "✓"),
    ("bail", "Switch or stop if", "bail", "✗"),
    ("tests", "Cheap tests first", "test", "?"),
)


def _playbook(run: Run) -> str:
    """The play itself: what to do, how to tell it works, when to quit, what to check first."""
    cards = []
    for rank, (i, s) in enumerate(_ranked(run), 1):
        cols = []
        for attr, label, cls, mark in _PLAY_PARTS:
            items = [x for x in getattr(s, attr, []) or [] if str(x).strip()]
            if not items:
                continue
            tag = "ol" if attr == "steps" else "ul"
            lis = "".join(f'<li><span class="mk">{mark}</span>{_e(x)}</li>' if mark else
                          f"<li>{_e(x)}</li>" for x in items)
            cols.append(f'<div class="play-col play-{cls}"><h4>{label}</h4>'
                        f'<{tag}>{lis}</{tag}></div>')
        att = max(1, int(_num(s.attempts, 1)))
        body = (f'<div class="play-cols">{"".join(cols)}</div>' if cols else
                '<p class="none">No playbook was written for this strategy.</p>')
        lead = " play-lead" if rank == 1 else ""
        cards.append(
            f'<article class="play{lead}" style="--sc:{_color(i)}">'
            f'<div class="play-head"><div><div class="eyebrow">#{rank}'
            f'{" · best odds" if rank == 1 else ""}</div><h3>{_e(s.name)}</h3>'
            f"{f'<p class=muted>{_e(s.summary)}</p>' if s.summary else ''}</div>"
            f'<div class="play-odds"><b>{_e(_pct(s.overall.p50))}</b>'
            f'<span>p10–p90 {_e(_range(s.overall.p10, s.overall.p90))}<br>over {att} '
            f"{_e(_plural(s.attempt_unit, att))}</span></div></div>{body}</article>")
    if not cards:
        return ""
    return _section("playbook", "Playbook", "".join(cards),
                    lede="Each strategy as a play: the moves, the tells that it is working, "
                    "the early signs to switch, and cheap real-world tests that settle the "
                    "open hypotheses before you commit.")


def _odds_text(o: Odds) -> str:
    return f"p50 {_pct(o.p50)} (p10–p90 {_range(o.p10, o.p90)}), mean {_pct(o.mean)}"


def _odds_section(run: Run) -> str:
    if not run.strategies:
        return ""
    body = (f'<figure><div class="scroll">{_effort_chart(run)}</div><figcaption>Dot = p50 of overall odds over the '
            "window; whisker = p10–p90. Up and to the left is better.</figcaption></figure>"
            f"{_ranked_bars(run)}")
    return _section("odds", "Odds vs effort", body)


def _levers(run: Run) -> str:
    sens = [(i, s) for i, s in enumerate(run.strategies) if s.sensitivity]
    if not run.levers and not sens:
        return ""
    body = ""
    if run.levers:
        body += f'<h3>Levers</h3><ol class="levers">' \
                f'{"".join(f"<li>{_e(x)}</li>" for x in run.levers)}</ol>'
    if sens:
        top = max((abs(_num(v)) for _, s in sens for v in s.sensitivity.values()), default=0.0)
        top = top or 1.0
        cards = []
        for i, s in sorted(sens, key=lambda p: -max(_num(v) for v in p[1].sensitivity.values())):
            items = sorted(s.sensitivity.items(), key=lambda kv: -_num(kv[1]))
            rows = "".join(
                f'<div class="sens-row"><span class="sens-name">{_e(k)}</span>'
                f'<span class="sens-track"><span class="sens-bar" style="width:'
                f'{max(1.0, abs(_num(v)) / top * 100):.1f}%;background:{_color(i)}"></span>'
                f'</span><span class="sens-val">{_num(v) * 100:+.1f} pts</span></div>'
                for k, v in items)
            cards.append(f'<div class="card"><h4>{_e(s.name)}</h4>{rows}</div>')
        body += ("<h3>Sensitivity</h3><p class='muted small'>How many percentage points the "
                 "mean overall odds rise when that one stage is improved, everything else held. "
                 "Bars share one scale across strategies.</p>"
                 f'<div class="grid">{"".join(cards)}</div>')
    return _section("levers", "Levers & sensitivity", body)


def _ach(run: Run) -> str:
    hyps = run.hypotheses
    if not hyps:
        return _section("ach", "Competing hypotheses", _none("hypotheses were formed"))

    cards = []
    for h in hyps:
        status = h.status if h.status in ("supported", "refuted", "open") else "open"
        prior, post = _clamp(_num(h.prior)), _clamp(_num(h.posterior))
        cards.append(
            f'<div class="hyp hyp-{status}" id="{_anchor("hyp", h.id)}">'
            f'<div class="hyp-top"><b>{_e(h.id)}</b><span class="badge badge-{status}">'
            f"{_e(h.status or 'open')}</span></div><p>{_e(h.statement)}</p>"
            f'<div class="pp"><span class="pp-track"><span class="pp-prior" style="left:'
            f'{prior * 100:.1f}%"></span><span class="pp-post" style="left:{post * 100:.1f}%">'
            f'</span></span><span class="pp-text">prior {_e(_pct(prior))} → '
            f"<b>posterior {_e(_pct(post))}</b></span></div>"
            f"{f'<p class=muted small>{_e(h.reasoning)}</p>' if h.reasoning else ''}</div>")

    marks: dict[str, dict[str, str]] = {}
    for h in hyps:
        for eid in h.support:
            marks.setdefault(eid, {})[h.id] = "+"
        for eid in h.against:
            marks.setdefault(eid, {})[h.id] = "-"
    known = [ev.id for ev in run.evidence]
    order = [eid for eid in known if eid in marks] + [eid for eid in marks if eid not in known]
    unlinked = len([eid for eid in known if eid not in marks])

    # Heuer: what matters is evidence that discriminates. A row that is + for one hypothesis
    # and - for another splits them; a row with the same sign on every hypothesis cannot
    # move the ranking however strong it looks.
    def diag(eid: str) -> int:
        signs = set(marks[eid].values())
        if len(signs) == 2:
            return 0
        if len(marks[eid]) == len(hyps) and len(hyps) > 1:
            return 2
        return 1

    def weight(eid: str) -> float:
        ev = run.evidence_by_id(eid)
        return _num(ev.weight) if ev else 0.0

    groups = (
        (0, "Splits hypotheses", "supports one, counts against another: the diagnostic rows"),
        (1, "Bears on some", "one-sided: moves some hypotheses, silent on the rest"),
        (2, "Consistent with all", "same sign everywhere: cannot change the ranking"),
    )
    head = "".join(f'<th class="hcol" title="{_e(h.statement)}"><a href="#'
                   f'{_anchor("hyp", h.id)}">{_e(h.id)}</a></th>' for h in hyps)
    rows = []
    for level, label, hint in groups:
        members = sorted((eid for eid in order if diag(eid) == level), key=lambda x: -weight(x))
        if not members:
            continue
        rows.append(f'<tr class="grp grp-{level}"><th colspan="{len(hyps) + 1}">{_e(label)} '
                    f'<span class="count">{len(members)} · {_e(hint)}</span></th></tr>')
        rows.extend(_ach_row(run, eid, hyps, marks, level) for eid in members)
    foot = "".join(f'<td class="num">{_e(_pct(h.posterior))}</td>' for h in hyps)
    table = (f'<div class="scroll"><table class="ach"><thead><tr><th>evidence</th>{head}</tr>'
             f'</thead><tbody>{"".join(rows)}</tbody><tfoot><tr><th>posterior</th>{foot}</tr>'
             f"</tfoot></table></div>")
    note = (f'<p class="muted small">{unlinked} evidence item{"s" if unlinked != 1 else ""} '
            "not linked to any hypothesis are left out of the matrix (see the ledger).</p>"
            if unlinked else "")
    legend = ('<p class="legend"><span class="c-sup lg">+</span> supports '
              '<span class="c-ag lg">−</span> counts against <span class="c-neu lg">·</span> '
              "neutral / not assessed</p>")
    return _section("ach", "Competing hypotheses (ACH)",
                    f'<div class="grid hyps">{"".join(cards)}</div>{legend}{table}{note}',
                    lede="Heuer's Analysis of Competing Hypotheses: evidence that separates "
                    "hypotheses matters; evidence consistent with all of them does not.")


def _ach_row(run: Run, eid: str, hyps: list[Hypothesis], marks: dict[str, dict[str, str]],
             level: int) -> str:
    ev = run.evidence_by_id(eid)
    claim = _e(_truncate(ev.claim, 140)) if ev else '<i class="muted">unknown evidence id</i>'
    tag = _tag_chip(ev.tag) if ev else ""
    cells = []
    for h in hyps:
        m = marks[eid].get(h.id, "")
        if m == "+":
            cells.append(f'<td class="c-sup" title="{_e(eid)} supports {_e(h.id)}">+</td>')
        elif m == "-":
            cells.append(f'<td class="c-ag" title="{_e(eid)} counts against {_e(h.id)}">'
                         "−</td>")
        else:
            cells.append('<td class="c-neu">·</td>')
    classes = [f"d{level}"]
    if ev and ev.tag == "speculation":
        classes.append("spec-row")
    return (f'<tr class="{" ".join(classes)}"><th scope="row" class="evcell">'
            f"{_ev_ref(run, eid)} {tag} <span class=\"cl\">{claim}</span></th>"
            f'{"".join(cells)}</tr>')


def _sources(run: Run) -> str:
    if not run.sources:
        return _section("sources", "Source map", _none("sources were collected"))
    cited: dict[str, list[str]] = {}
    for ev in run.evidence:
        for sid in ev.source_ids:
            cited.setdefault(sid, []).append(ev.id)
    kinds = list(SOURCE_KINDS) + sorted({s.kind for s in run.sources} - set(SOURCE_KINDS))
    groups = []
    for kind in kinds:
        items = [s for s in run.sources if s.kind == kind]
        if not items:
            continue
        rows = "".join(_source_row(s, cited.get(s.id, [])) for s in
                       sorted(items, key=lambda s: -_num(s.quality)))
        groups.append(f'<div class="src-group"><h3>{_e(kind)} <span class="count">'
                      f"{len(items)}</span></h3>{rows}</div>")
    return _section("sources", "Source map", f'<div class="src-groups">{"".join(groups)}</div>',
                    lede="Grouped by kind, best first. The bar is how much a claim from that "
                    "source should move belief (0–1).")


def _source_row(s: Source, cited_by: list[str]) -> str:
    q = _clamp(_num(s.quality))
    lang = (s.lang or "?").lower()
    lang_cls = "lang" if lang == "en" else "lang lang-other"
    title = s.title or s.url or s.id
    safe = _safe_url(s.url)
    where = (f'<span class="domain">{_e(_domain(s.url))}</span>' if safe else
             '<span class="domain warn">link withheld: not http(s)</span>')
    refs = (" ".join(f'<a class="ref" href="#{_anchor("ev", e)}">{_e(e)}</a>' for e in cited_by)
            if cited_by else '<span class="muted">not cited</span>')
    note = f'<div class="muted small">{_e(s.note)}</div>' if s.note else ""
    return (f'<div class="src" id="{_anchor("src", s.id)}">'
            f'<div class="q" title="quality {q:.2f}"><span style="width:{q * 100:.0f}%"></span>'
            f"</div>"
            f'<div class="src-main"><span class="sid">{_e(s.id)}</span> '
            f"{_link(s.url, title)} <span class=\"{lang_cls}\">{_e(lang)}</span>"
            f'<div class="small">{where} · cited by {refs}</div>{note}</div></div>')


def _evidence(run: Run) -> str:
    if not run.evidence:
        return _section("evidence", "Evidence ledger", _none("evidence was recorded"))
    counts = {t: sum(1 for ev in run.evidence if ev.tag == t) for t in TAGS}
    summary = " · ".join(f"{_tag_chip(t)} {n}" for t, n in counts.items())
    items = "".join(_evidence_item(run, ev) for ev in run.evidence)
    return _section("evidence", "Evidence ledger",
                    f'<p class="tagsum">{summary}</p><div class="ledger">{items}</div>',
                    lede="<b>sourced</b> = a source says it; <b>inferred</b> = reasoned from "
                    "sources; <b>speculation</b> = nobody measured it. Treat speculation as a "
                    "hypothesis, not a fact.")


def _evidence_item(run: Run, ev: Evidence) -> str:
    tag = ev.tag if ev.tag in TAGS else "other"
    srcs = []
    for sid in ev.source_ids:
        src = run.source(sid)
        ext = ""
        if src and _safe_url(src.url):
            ext = (f' <a class="ext" href="{_e(_safe_url(src.url))}" target="_blank" '
                   f'rel="noopener noreferrer nofollow" title="{_e(src.title)}">'
                   f"{_e(_domain(src.url))} ↗</a>")
        srcs.append(_src_ref(run, sid) + ext)
    warn = ""
    if ev.tag == "sourced" and not ev.source_ids:
        warn = '<span class="warn small">tagged sourced but no source attached</span>'
    quote = f"<blockquote>{_e(ev.quote)}</blockquote>" if ev.quote else ""
    meta = [f"weight {_num(ev.weight):.2f}"]
    if ev.subquestion_id:
        meta.append(f"subquestion {ev.subquestion_id}")
    return (f'<div class="ev ev-{tag}" id="{_anchor("ev", ev.id)}">'
            f'<div class="ev-top"><b class="eid">{_e(ev.id)}</b>{_tag_chip(ev.tag)}'
            f'<span class="muted small">{_e(" · ".join(meta))}</span>{warn}</div>'
            f'<p class="claim">{_e(ev.claim)}</p>{quote}'
            f'<div class="small ev-src">{" · ".join(srcs) or "<span class=muted>no sources</span>"}'
            f"</div></div>")


def _qa(run: Run) -> str:
    if not run.qa:
        return ""
    items = []
    for qa in run.qa:
        ans = (f"<p>{_e(qa.answer)}</p>" if qa.answer.strip()
               else '<p class="muted"><i>unanswered</i></p>')
        why = f'<p class="muted small">why asked: {_e(qa.why)}</p>' if qa.why else ""
        items.append(f'<div class="qa"><div class="qa-q"><span class="chip">{_e(qa.phase)}'
                     f"</span> {_e(qa.question)}</div>{ans}{why}</div>")
    return _section("qa", "Questions asked of you", "".join(items))


def _critiques(run: Run) -> str:
    if not run.critiques:
        return ""
    cards = []
    for c in sorted(run.critiques, key=lambda c: _num(c.round)):
        n = len(c.problems)
        probs = _list(c.problems, ordered=True) if c.problems else _none("problems found")
        newq = (f"<h4>New subquestions</h4>{_list(c.new_subquestions)}"
                if c.new_subquestions else "")
        verdict = f'<p class="cverdict">{_e(c.verdict)}</p>' if c.verdict else ""
        cards.append(f'<div class="card"><h3>Round {_e(c.round)} <span class="count">{n} '
                     f'problem{"s" if n != 1 else ""}</span></h3>{probs}{newq}{verdict}</div>')
    return _section("critique", "Critic rounds", f'<div class="grid">{"".join(cards)}</div>',
                    lede="An adversarial pass that hunts for weak ranges, speculation passed "
                    "off as fact, and missing perspectives.")


def _followups(run: Run) -> str:
    if not run.followups:
        return ""
    return _section("followups", "Suggested follow-ups", _list(run.followups))


def _method(run: Run) -> str:
    trail = ""
    if run.subquestions:
        rows = "".join(
            f"<tr><td>{_e(q.id)}</td><td>{_e(q.text)}</td><td>{_e(q.angle)}</td>"
            f"<td><span class='status status-{_e(re.sub(r'[^a-z]', '', q.status.lower()))}'>"
            f"{_e(q.status)}</span></td><td>{_e(q.engine)}</td><td>{_e(q.round)}</td></tr>"
            for q in run.subquestions)
        trail += (f"<details><summary>Research questions ({len(run.subquestions)})</summary>"
                  f'<div class="scroll"><table class="stages"><thead><tr><th>id</th>'
                  f"<th>question</th><th>angle</th><th>status</th><th>engine</th>"
                  f"<th>round</th></tr></thead><tbody>{rows}</tbody></table></div></details>")
    if run.log:
        trail += (f"<details><summary>Run log ({len(run.log)} lines)</summary>"
                  f'<pre class="log">{_e(chr(10).join(str(x) for x in run.log))}</pre>'
                  "</details>")
    body = (
        "<p>These odds are <b>estimates built from ranged inputs, not measurements</b>. Each "
        "stage of a strategy carries a probability range judged from the evidence above; a "
        "Monte Carlo simulation samples those ranges, chains the stages, and repeats over the "
        "number of attempts in your window. The p10/p50/p90 figures describe the spread that "
        "the input ranges imply. They do not capture errors in how the chain was drawn, "
        "correlations between stages, or evidence nobody found.</p>"
        "<p>Sources are mostly whatever is findable on the open web; anecdote-heavy kinds "
        "(forums, reddit) are discounted but still over-represent people with a story to tell. "
        "Speculation-tagged evidence is shown, not hidden, so you can see exactly where the "
        "argument is thinnest.</p>" + trail)
    return _section("method", "Method & limits", body)


def _nav(present: list[tuple[str, str]]) -> str:
    links = "".join(f'<a href="#{sid}">{_e(label)}</a>' for sid, label in present)
    return f'<nav class="toc">{links}</nav>'


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------


def render_html(run: Run) -> str:
    """Render a run as one self-contained HTML document."""
    sections = [
        ("playbook", "Playbook", _playbook(run)),
        ("strategies", "Strategies", _strategies(run)),
        ("odds", "Odds vs effort", _odds_section(run)),
        ("levers", "Levers", _levers(run)),
        ("ground", "Base rate", _base_and_other(run)),
        ("ach", "Hypotheses", _ach(run)),
        ("evidence", "Evidence", _evidence(run)),
        ("sources", "Sources", _sources(run)),
        ("framing", "Framing", _framing(run)),
        ("qa", "Q&A", _qa(run)),
        ("critique", "Critique", _critiques(run)),
        ("followups", "Follow-ups", _followups(run)),
        ("method", "Method", _method(run)),
    ]
    present = [(sid, label) for sid, label, body in sections if body]
    body = "".join(b for _, _, b in sections if b)
    title = _truncate(run.question or "odds report", 80)
    return (
        "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="color-scheme" content="light dark">'
        f"<title>{_e(title)} · odds</title><style>{_CSS}</style></head><body>"
        f'<main>{_header(run)}{_nav(present)}{body}'
        f'<footer>Generated by odds from run {_e(run.id)} · schema {_e(run.schema)}</footer>'
        "</main></body></html>\n"
    )


_ANSI = {"bold": "1", "dim": "2", "green": "32", "yellow": "33", "red": "31", "cyan": "36"}
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def render_terminal(run: Run, *, color: bool = False, width: int = 100) -> str:
    """A short plain-text summary for stdout. File paths are the caller's to print."""
    width = max(40, width)

    def c(text: str, *styles: str) -> str:
        if not color or not styles:
            return text
        codes = ";".join(_ANSI[s] for s in styles)
        return f"\x1b[{codes}m{text}\x1b[0m"

    def clean(text: Any) -> str:
        # Strip control characters so LLM/web text cannot inject terminal escapes.
        return _CTRL.sub("", " ".join(str(text).split()))

    def wrap(text: Any, indent: str = "  ") -> list[str]:
        return textwrap.wrap(clean(text), width=width, initial_indent=indent,
                             subsequent_indent=indent) or [indent.rstrip()]

    def pct(p: float) -> str:
        return _pct(p)

    def rng(o: Odds) -> str:
        return _range(o.p10, o.p90, dash="-")

    status = run.status or "unknown"
    status_style = {"done": "green", "failed": "red", "running": "yellow"}.get(status, "dim")
    out: list[str] = []
    out += [c(line, "bold") for line in wrap(run.question or "(no question)", "")]
    meta = [c(status, status_style, "bold"), f"depth {clean(run.depth)}", _date(run.created)]
    if run.engines:
        meta.append("engines " + ", ".join(f"{clean(k)}={clean(v)}"
                                           for k, v in run.engines.items()))
    out.append(c(" | ", "dim").join(meta))
    out.append("")

    out.append(c("Verdict", "bold"))
    out += wrap(run.verdict) if run.verdict.strip() else ["  (no verdict)"]
    out.append("")

    out.append(c("Strategies, ranked by overall odds (p50, p10-p90)", "bold"))
    if not run.strategies:
        out.append("  (none)")
    def labels(s: Strategy) -> tuple[str, str]:
        att = max(1, int(_num(s.attempts, 1)))
        unit = clean(s.attempt_unit or "attempt")
        return f"per {unit}", f"over {att} {_plural(unit, att)}"

    lw = max((len(x) for s in run.strategies for x in labels(s)), default=0)
    for rank, (_, s) in enumerate(_ranked(run), 1):
        out += wrap(f"{rank}. {s.name}", "  ")
        label_a, label_o = labels(s)
        out.append(f"       {label_a:<{lw}}  {pct(s.per_attempt.p50):>6}  ({rng(s.per_attempt)})")
        out.append(f"       {label_o:<{lw}}  {c(f'{pct(s.overall.p50):>6}', 'cyan', 'bold')}  "
                   f"({rng(s.overall)})   effort {_num(s.effort):.2f}")
    out.append("")

    ranked = _ranked(run)
    if ranked:
        _, best = ranked[0]
        parts = [(attr, mark) for attr, _, _, mark in _PLAY_PARTS
                 if [x for x in getattr(best, attr, []) or [] if str(x).strip()]]
        if parts:
            out.append(c(f"Playbook: {clean(best.name)}", "bold"))
            heads = {"steps": "Do this", "signals": "It is working if",
                     "bail": "Switch or stop if", "tests": "Cheap tests first"}
            marks = {"steps": "", "signals": "+", "bail": "x", "tests": "?"}
            for attr, _ in parts:
                out.append(c(f"  {heads[attr]}", "dim"))
                for n, item in enumerate(getattr(best, attr), 1):
                    if not str(item).strip():
                        continue
                    bullet = f"{n}." if attr == "steps" else marks[attr]
                    lines = textwrap.wrap(clean(item), width=width,
                                          initial_indent=f"    {bullet} ",
                                          subsequent_indent=" " * (5 + len(bullet)))
                    out += lines
            out.append("")

    if run.levers:
        out.append(c("Top levers", "bold"))
        for i, lever in enumerate(run.levers[:3], 1):
            out += wrap(f"{i}. {lever}", "  ")
        out.append("")
    sens = sorted(((_num(v), str(k), s.name) for s in run.strategies
                   for k, v in s.sensitivity.items()), reverse=True)[:3]
    if sens:
        out.append(c("Most sensitive stages", "bold"))
        for v, stage, sname in sens:
            out += wrap(f"- {stage} ({sname}): {v * 100:+.1f} pts", "  ")
        out.append("")

    n_spec = sum(1 for ev in run.evidence if ev.tag == "speculation")
    supported = sum(1 for h in run.hypotheses if h.status == "supported")
    out.append(c(f"{len(run.sources)} sources, {len(run.evidence)} evidence "
                 f"({n_spec} speculation), {len(run.hypotheses)} hypotheses "
                 f"({supported} supported), {len(run.critiques)} critique rounds", "dim"))
    return "\n".join(out) + "\n"


_CSS = """
:root{
  --bg:#f6f5f1;--surface:#ffffff;--surface2:#f0eee8;--ink:#1d1e20;--muted:#5d6166;
  --faint:#8b8f94;--line:#e1ded6;--accent:#2d5b88;--good:#2e7a4d;--good-bg:#e3f1e8;
  --bad:#b03a2e;--bad-bg:#f8e4e1;--warn:#b0620c;--spec:#c2410c;--spec-bg:#fdeee3;
  --inf:#3d63a8;--inf-bg:#e7eef9;--edge:#a7a49c;
  --c1:#3a6ea5;--c2:#c4712f;--c3:#3f8a5f;--c4:#8659a8;--c5:#b4475a;--c6:#3f8b9b;
  --serif:"Iowan Old Style","Charter","Source Serif Pro","Georgia",serif;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans","Noto Sans KR",
    "Apple SD Gothic Neo",sans-serif;
  --mono:ui-monospace,"SFMono-Regular","JetBrains Mono",Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#131416;--surface:#1b1c1f;--surface2:#232529;--ink:#e7e5e0;--muted:#a4a7ab;
  --faint:#7c7f84;--line:#2f3236;--accent:#86b4e3;--good:#69c08f;--good-bg:#1d3326;
  --bad:#ee7f73;--bad-bg:#3a201d;--warn:#e5a352;--spec:#f39256;--spec-bg:#3a2518;
  --inf:#8eb0ec;--inf-bg:#1f2a3d;--edge:#5a5d62;
  --c1:#6d9fd6;--c2:#e39a5c;--c3:#6cbf8c;--c4:#b38ad6;--c5:#e07a8c;--c6:#6cbccb;
}}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 var(--sans)}
main{max-width:1080px;margin:0 auto;padding:32px 16px 48px}
a{color:var(--accent);text-decoration-thickness:1px;text-underline-offset:2px}
h1,h2,h3{font-family:var(--serif);font-weight:600;line-height:1.2;letter-spacing:-.01em}
h1{font-size:clamp(26px,4vw,38px);margin:6px 0 12px}
h2{font-size:24px;margin:0 0 6px}
h3{font-size:17px;margin:18px 0 8px}
h4{font-size:13px;margin:14px 0 6px;text-transform:uppercase;letter-spacing:.06em;
  color:var(--muted);font-family:var(--sans)}
p{margin:0 0 10px}
ul,ol{margin:0 0 10px;padding-left:22px}
li{margin:3px 0}
.eyebrow{font-size:12px;text-transform:uppercase;letter-spacing:.09em;color:var(--muted)}
.muted{color:var(--muted)} .small{font-size:13px} .num{font-variant-numeric:tabular-nums}
.none{color:var(--faint);font-style:italic}
.lede{color:var(--muted);max-width:75ch;margin-bottom:16px}
header{padding-bottom:8px}
.meta{display:flex;flex-wrap:wrap;gap:6px 16px;color:var(--muted);font-size:13px}
.status{display:inline-block;padding:0 8px;border-radius:999px;font-size:12px;
  font-weight:600;background:var(--surface2);color:var(--muted)}
.status-done{background:var(--good-bg);color:var(--good)}
.status-failed{background:var(--bad-bg);color:var(--bad)}
.status-running,.status-open{background:var(--inf-bg);color:var(--inf)}
.banner{margin:16px 0 0;padding:10px 14px;border-radius:8px;background:var(--inf-bg);
  color:var(--inf);font-size:14px}
.banner.bad{background:var(--bad-bg);color:var(--bad)}
.verdict{margin:22px 0 18px;padding:18px 22px;background:var(--surface);
  border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:10px}
.verdict p{font-family:var(--serif);font-size:clamp(17px,2.2vw,20px);line-height:1.5;
  margin:6px 0 0;max-width:72ch}
.verdict.empty p{color:var(--faint);font-style:italic}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:10px;
  padding:10px 14px}
.tile .k{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.tile .v{font-size:20px;font-weight:600;font-variant-numeric:tabular-nums;line-height:1.3;
  margin-top:2px}
.tile .s{font-size:12.5px;color:var(--muted)}
.toc{display:flex;flex-wrap:wrap;gap:4px 14px;margin:20px 0 4px;padding:10px 0;
  border-top:1px solid var(--line);border-bottom:1px solid var(--line);font-size:13.5px}
.toc a{text-decoration:none;color:var(--muted)} .toc a:hover{color:var(--accent)}
section{margin-top:44px}
figure{margin:12px 0 0}
figcaption{font-size:12.5px;color:var(--muted);margin-top:6px;max-width:90ch}
.wide{width:min(calc(100vw - 32px),1500px);margin-left:50%;transform:translateX(-50%)}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
.wide .scroll{background:var(--surface);border:1px solid var(--line);border-radius:12px;
  padding:12px}
.wide .scroll svg{display:block;margin:0 auto;height:auto}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;
  padding:14px 18px;min-width:0}
.card h3,.card h4{margin-top:0}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:12px}
.big{font-size:30px;font-weight:600;font-variant-numeric:tabular-nums;margin:0 0 6px}
.other p:not(.small){font-family:var(--serif);font-size:16.5px}
.kv{display:grid;grid-template-columns:max-content 1fr;gap:4px 16px;margin:8px 0 12px;
  font-size:14px}
.kv dt{color:var(--muted)} .kv dd{margin:0}
.ref{font-family:var(--mono);font-size:12px;padding:0 4px;border-radius:4px;
  background:var(--surface2);text-decoration:none;color:var(--ink)}
.ref:hover{background:var(--inf-bg)} .ref-missing{color:var(--bad)}
.refs{font-size:13px;color:var(--muted)}
.chip{font-size:11px;padding:1px 7px;border-radius:999px;background:var(--surface2);
  color:var(--muted);text-transform:uppercase;letter-spacing:.05em;vertical-align:1px}
.count{font-family:var(--sans);font-size:12px;color:var(--muted);font-weight:500;
  margin-left:6px}
.warn{color:var(--warn)}
/* tree */
svg text{font-family:var(--sans)}
.tree .node rect{fill:var(--surface);stroke:var(--line);stroke-width:1}
.tree .root rect{fill:var(--surface2)}
.tree .leaf rect{stroke-width:2}
.tree .edge{fill:none;stroke:var(--edge);stroke-width:1.4}
.tree .edge.dashed{stroke-dasharray:4 4}
.tree .arrow{fill:var(--edge)}
.tree .track{fill:var(--surface2);stroke:none}
.tree .node .track{stroke:none}
.t-eyebrow{font-size:10px;letter-spacing:.1em;fill:var(--muted)}
.t-root{font-size:13px;font-weight:600;fill:var(--ink)}
.t-strat{font-size:12.5px;font-weight:600;fill:var(--ink)}
.t-stage{font-size:11px;fill:var(--ink)}
.t-range{font-size:11.5px;font-weight:600;fill:var(--ink);font-variant-numeric:tabular-nums}
.t-meta{font-size:11px;fill:var(--muted);font-variant-numeric:tabular-nums}
.t-leaf{font-size:19px;font-weight:700;fill:var(--ink);font-variant-numeric:tabular-nums}
.t-leafsub{font-size:11px;font-weight:400;fill:var(--muted)}
/* chart */
.chart{width:100%;height:auto;max-width:860px;min-width:540px;display:block;background:var(--surface);
  border:1px solid var(--line);border-radius:12px}
.chart .grid{stroke:var(--line);stroke-width:1}
.chart .axis{font-size:11.5px;fill:var(--muted);font-variant-numeric:tabular-nums}
.chart .axis-title{font-size:12px;fill:var(--muted)}
.chart .whisker{stroke-width:2.2;stroke-linecap:round}
.chart .dot{stroke:var(--surface);stroke-width:2}
.chart .pt-label{font-size:12.5px;font-weight:600;fill:var(--ink);paint-order:stroke;
  stroke:var(--surface);stroke-width:4px;stroke-linejoin:round}
.ranked{margin-top:18px;background:var(--surface);border:1px solid var(--line);
  border-radius:12px;padding:6px 16px}
.rank-head,.rank-row{display:grid;grid-template-columns:22px minmax(0,1.3fr) minmax(0,1fr) 78px;
  gap:12px;align-items:center}
.rank-head{font-size:11.5px;color:var(--faint);padding:6px 0;border-bottom:1px solid var(--line)}
.rank-row{padding:9px 0;border-bottom:1px solid var(--line)}
.rank-row:last-child{border-bottom:0}
.rank-n{font-weight:700;color:var(--muted);font-variant-numeric:tabular-nums}
.rank-name a{color:var(--ink);font-weight:600;text-decoration:none}
.rank-name .sub,.rank-val .sub{display:block;font-size:12px;color:var(--muted);
  font-variant-numeric:tabular-nums}
.rank-bar{position:relative;height:14px;background:var(--surface2);border-radius:7px}
.rank-bar .band{position:absolute;top:2px;bottom:2px;border-radius:5px;opacity:.85}
.rank-bar .tick{position:absolute;top:-3px;bottom:-3px;width:3px;margin-left:-1.5px;
  background:var(--ink);border-radius:2px}
.rank-val{text-align:right;font-variant-numeric:tabular-nums}
.rank-val b{font-size:17px}
/* strategy details */
.details-list{margin-top:16px;display:grid;gap:8px}
details{background:var(--surface);border:1px solid var(--line);border-radius:10px}
details>summary{cursor:pointer;padding:10px 16px;list-style:none}
details>summary>*{margin-right:8px}
details>summary::-webkit-details-marker{display:none}
details>summary::before{content:"▸";color:var(--muted);font-size:12px;margin-right:10px}
details[open]>summary::before{content:"▾"}
details .sub{color:var(--muted);font-size:13px;font-variant-numeric:tabular-nums}
.swatch{width:10px;height:10px;border-radius:3px;background:var(--sc,var(--accent));
  display:inline-block;vertical-align:0}
.det{padding:0 18px 14px}
section#method details{margin-top:10px}
section#method p{max-width:80ch}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{text-align:left;vertical-align:top;padding:6px 8px;border-bottom:1px solid var(--line)}
thead th{font-size:11.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);
  font-weight:600}
table.stages td.num{white-space:nowrap;font-weight:600}
/* levers */
ol.levers li{margin:6px 0;font-size:15.5px}
.sens-row{display:grid;grid-template-columns:minmax(0,1fr) 90px 64px;gap:10px;
  align-items:center;font-size:13px;padding:3px 0}
.sens-track{height:8px;background:var(--surface2);border-radius:4px;overflow:hidden}
.sens-bar{display:block;height:100%;border-radius:4px}
.sens-val{text-align:right;font-variant-numeric:tabular-nums;color:var(--muted)}
/* ACH */
.hyps{margin-bottom:14px}
.hyp{background:var(--surface);border:1px solid var(--line);border-radius:10px;
  padding:12px 16px}
.hyp-refuted p:not(.small){color:var(--muted)}
.hyp-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px}
.badge{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;
  padding:1px 8px;border-radius:999px}
.badge-supported{background:var(--good-bg);color:var(--good)}
.badge-refuted{background:var(--bad-bg);color:var(--bad)}
.badge-open{background:var(--surface2);color:var(--muted)}
.pp{display:flex;align-items:center;gap:10px;font-size:12.5px;margin:6px 0;
  font-variant-numeric:tabular-nums}
.pp-track{position:relative;flex:0 0 110px;height:6px;background:var(--surface2);
  border-radius:3px}
.pp-prior,.pp-post{position:absolute;top:-3px;width:12px;height:12px;margin-left:-6px;
  border-radius:50%}
.pp-prior{border:2px solid var(--faint);background:var(--surface)}
.pp-post{background:var(--accent)}
.legend{font-size:12.5px;color:var(--muted)}
.lg{display:inline-block;width:20px;text-align:center;border-radius:4px;margin-left:8px}
table.ach{background:var(--surface);border:1px solid var(--line);border-radius:10px;
  border-collapse:separate;border-spacing:0;overflow:hidden}
table.ach th.hcol{text-align:center;width:52px}
table.ach .evcell{font-weight:400;min-width:300px}
table.ach td{text-align:center;font-weight:700;font-size:15px;width:52px}
.c-sup{background:var(--good-bg);color:var(--good)}
.c-ag{background:var(--bad-bg);color:var(--bad)}
.c-neu{color:var(--faint)}
table.ach tfoot th,table.ach tfoot td{border-bottom:0;font-size:13px;color:var(--ink);
  background:var(--surface2)}
tr.spec-row .cl{font-style:italic}
/* evidence */
.tagsum{font-size:13px;color:var(--muted)}
.tag{display:inline-block;font-size:11px;font-weight:600;padding:0 7px;border-radius:4px;
  letter-spacing:.02em;line-height:18px;vertical-align:1px;white-space:nowrap}
.tag-sourced{background:var(--good-bg);color:var(--good)}
.tag-inferred{background:transparent;color:var(--inf);box-shadow:inset 0 0 0 1px var(--inf)}
.tag-speculation{color:var(--spec);text-transform:uppercase;letter-spacing:.06em;
  box-shadow:inset 0 0 0 1px var(--spec);
  background:repeating-linear-gradient(-45deg,var(--spec-bg) 0 4px,transparent 4px 8px)}
.tag-other{background:var(--surface2);color:var(--muted)}
.ledger{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:8px;
  align-items:start}
.ev{background:var(--surface);border:1px solid var(--line);border-left:4px solid var(--good);
  border-radius:8px;padding:8px 12px}
.ev-inferred{border-left-color:var(--inf)}
.ev-speculation{border-left:4px dashed var(--spec);
  background:linear-gradient(90deg,var(--spec-bg),var(--surface) 40%)}
.ev-other{border-left-color:var(--faint)}
.ev-top{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.eid{font-family:var(--mono);font-size:12.5px}
.claim{margin:3px 0;font-size:14.5px;line-height:1.45}
.ev-speculation .claim{font-style:italic}
blockquote{margin:4px 0 6px;padding:2px 0 2px 12px;border-left:2px solid var(--line);
  color:var(--muted);font-family:var(--serif);font-size:14.5px}
.ev-src{color:var(--muted)} .ext{font-size:12px}
/* sources */
.src-groups{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}
.src-group{background:var(--surface);border:1px solid var(--line);border-radius:10px;
  padding:4px 16px 8px}
.src-group h3{text-transform:capitalize;font-size:16px;margin:10px 0 4px}
.src{display:grid;grid-template-columns:44px 1fr;gap:12px;padding:8px 0;
  border-top:1px solid var(--line);font-size:14px}
.src:first-of-type{border-top:0}
.q{height:6px;background:var(--surface2);border-radius:3px;margin-top:8px;overflow:hidden}
.q span{display:block;height:100%;background:var(--accent)}
.sid{font-family:var(--mono);font-size:12px;color:var(--muted)}
.domain{color:var(--faint)}
.lang{font-size:10.5px;text-transform:uppercase;padding:0 5px;border-radius:3px;
  background:var(--surface2);color:var(--muted);margin-left:4px;vertical-align:1px}
.lang-other{background:var(--inf-bg);color:var(--inf);font-weight:700}
/* playbook */
.play{background:var(--surface);border:1px solid var(--line);border-top:4px solid var(--sc);
  border-radius:12px;padding:16px 20px;margin-bottom:12px}
.play-lead{box-shadow:0 1px 0 var(--line),0 8px 24px -16px rgba(0,0,0,.25)}
.play-head{display:flex;justify-content:space-between;gap:16px;align-items:flex-start}
.play-head h3{font-size:20px;margin:2px 0 4px}
.play-head p{margin:0;font-size:14px}
.play-odds{text-align:right;flex:0 0 auto;font-variant-numeric:tabular-nums}
.play-odds b{display:block;font-size:28px;line-height:1.1}
.play-odds span{font-size:12px;color:var(--muted)}
.play-cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px 22px;
  margin-top:10px}
.play-col h4{margin:6px 0 6px}
.play-col ul{list-style:none;padding-left:0}
.play-col li{position:relative;font-size:14px;margin:5px 0}
.play-col ul li{padding-left:22px}
.mk{position:absolute;left:0;top:1px;width:16px;height:16px;border-radius:4px;font-size:11px;
  font-weight:700;line-height:16px;text-align:center}
.play-sig .mk{background:var(--good-bg);color:var(--good)}
.play-bail .mk{background:var(--bad-bg);color:var(--bad)}
.play-test .mk{background:var(--inf-bg);color:var(--inf)}
.play-test h4{color:var(--inf)}
.play-bail h4{color:var(--bad)}
.play-sig h4{color:var(--good)}
/* ACH groups */
table.ach tr.grp th{background:var(--surface2);font-family:var(--sans);font-size:12px;
  font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:var(--ink);padding:7px 10px}
table.ach tr.grp .count{text-transform:none;letter-spacing:0;font-weight:400}
table.ach tr.grp-0 th{box-shadow:inset 3px 0 0 var(--accent)}
table.ach tr.d0 .evcell{box-shadow:inset 3px 0 0 var(--accent)}
table.ach tr.d2 td,table.ach tr.d2 .evcell{opacity:.6}
/* qa, critique */
.qa{padding:10px 0;border-bottom:1px solid var(--line)}
.qa-q{font-weight:600;margin-bottom:2px}
.qa p{margin:0 0 2px}
.cverdict{font-family:var(--serif);font-style:italic;border-top:1px solid var(--line);
  padding-top:8px;margin-top:8px}
pre.log{font:12px/1.5 var(--mono);background:var(--surface2);padding:10px 14px;
  border-radius:8px;overflow-x:auto;white-space:pre-wrap;margin:0 16px 14px}
footer{margin-top:56px;padding-top:12px;border-top:1px solid var(--line);font-size:12px;
  color:var(--faint)}
@media (max-width:640px){
  main{padding-top:20px}
  .ledger{grid-template-columns:1fr}
  .play-head{flex-direction:column}
  .play-odds{text-align:left}
  .src-groups{grid-template-columns:1fr}
  .rank-head{display:none}
  .rank-row{grid-template-columns:22px 1fr 70px;row-gap:6px}
  .rank-n,.rank-name,.rank-val{grid-row:1}
  .rank-val{grid-column:3}
  .rank-bar{grid-column:2 / 4;grid-row:2}
  .sens-row{grid-template-columns:minmax(0,1fr) 60px 58px}
  .verdict{padding:14px 16px}
  table.ach .evcell{min-width:220px}
}
@media print{.wide{width:auto;transform:none;margin-left:0}details{break-inside:avoid}}
"""
