"""Command-line interface for odds.

stdout carries only the real output (the summary, a path, or JSON); all
progress, questions to the user, warnings and errors go to stderr.
Exit codes: 0 success, 1 the investigation failed, 2 usage error / aborted,
130 interrupted.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import webbrowser
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

from . import __version__
from . import engines as eng
from . import sources as src
from .models import Run
from .pipeline import DEPTHS, Config, PipelineError, follow_up, investigate, new_run


class _UsageError(Exception):
    pass


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _UsageError(message)


def runs_root() -> Path:
    base = os.environ.get("ODDS_HOME") or os.path.join(
        os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share"), "odds")
    return Path(base) / "runs"


def _engine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--depth", choices=sorted(DEPTHS), default="normal",
                   help="quick ~4 subquestions no critic; normal 7 + 1 critic round; "
                        "deep 12 + 2 rounds (default: normal)")
    p.add_argument("--engine", default="claude:opus",
                   help="engine for the judgement calls: claude[:model], codex[:model], "
                        "ollama[:model] (default: claude:opus)")
    p.add_argument("--research", default="claude:sonnet",
                   help="comma-separated researcher engines, used round-robin "
                        "(default: claude:sonnet; try claude:sonnet,codex)")
    p.add_argument("--sources", default=None,
                   help="comma-separated source adapters for leads (default: all available)")
    p.add_argument("--no-ask", action="store_true",
                   help="never ask questions; record assumptions instead")
    p.add_argument("--open", action="store_true", help="open the HTML report when done")


def _build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="odds",
        description="Research a silly question like a market: sources, hypotheses, "
                    "strategies, and the odds.")
    parser.add_argument("-v", "--verbose", action="store_true", help="print detailed progress")
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress non-error output")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", parser_class=_ArgumentParser)

    ask = sub.add_parser("ask", help="investigate a new question")
    ask.add_argument("question", nargs="+")
    _engine_args(ask)

    fu = sub.add_parser("followup", help="continue a saved run with a new angle")
    fu.add_argument("run", help="run id, unique prefix, or 'last'")
    fu.add_argument("question", nargs="+")
    _engine_args(fu)

    sub.add_parser("list", help="list saved runs")

    show = sub.add_parser("show", help="print a saved run's summary")
    show.add_argument("run", nargs="?", default="last")
    show.add_argument("--json", action="store_true", help="print run.json instead")

    rep = sub.add_parser("report", help="re-render a run's HTML report and print its path")
    rep.add_argument("run", nargs="?", default="last")
    rep.add_argument("--open", action="store_true", help="open it in a browser")

    sub.add_parser("engines", help="show which engines and source adapters are usable")
    return parser


def resolve(ref: str) -> Path:
    root = runs_root()
    runs = sorted(p for p in root.glob("*") if (p / "run.json").exists()) if root.exists() else []
    if not runs:
        raise _UsageError(f"no saved runs under {root}")
    if ref == "last":
        return runs[-1]
    matches = [p for p in runs if p.name == ref] or [p for p in runs if p.name.startswith(ref)]
    if len(matches) != 1:
        raise _UsageError(f"{'no' if not matches else 'ambiguous'} run matching {ref!r}")
    return matches[0]


def _render(run: Run, directory: Path) -> Path:
    from .report import render_html

    path = directory / "report.html"
    path.write_text(render_html(run), encoding="utf-8")
    return path


def _summary(run: Run) -> str:
    from .report import render_terminal

    return render_terminal(run, color=sys.stdout.isatty())


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
        if not args.command:
            raise _UsageError("a command is required (ask, followup, list, show, report, engines)")
        return _dispatch(args)
    except _UsageError as exc:
        print(f"odds: error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("odds: interrupted (the run so far is saved)", file=sys.stderr)
        return 130


def _dispatch(args: argparse.Namespace) -> int:
    def log(message: str) -> None:
        if not args.quiet:
            print(f"odds: {message}", file=sys.stderr, flush=True)

    if args.command == "engines":
        for name, state in eng.available().items():
            print(f"engine  {name:16} {state}")
        for name, state in src.available().items():
            print(f"source  {name:16} {state}")
        return 0

    if args.command == "list":
        root = runs_root()
        for path in sorted(root.glob("*/run.json")) if root.exists() else []:
            try:
                run = Run.load(path.parent)
            except (OSError, ValueError) as exc:
                print(f"odds: skipping {path.parent.name}: {exc}", file=sys.stderr)
                continue
            best = run.strategies[0].overall.p50 if run.strategies else 0.0
            print(f"{run.id}\t{run.status}\t{best:.0%}\t{run.question}")
        return 0

    if args.command in ("show", "report"):
        directory = resolve(args.run)
        run = Run.load(directory)
        if args.command == "show":
            print((directory / "run.json").read_text() if args.json else _summary(run))
            return 0
        path = _render(run, directory)
        print(path)
        if args.open:
            webbrowser.open(path.as_uri())
        return 0

    try:
        cfg = _config(args, log)
    except eng.EngineError as exc:
        raise _UsageError(str(exc)) from exc

    if args.command == "ask":
        run, directory = new_run(" ".join(args.question).strip(), args.depth, runs_root())
    else:
        directory = resolve(args.run)
        run = Run.load(directory)
    run.engines = {"judgement": cfg.synth.name,
                   "research": ",".join(e.name for e in cfg.researchers)}

    # Usage is per process; a follow-up adds to what the run already spent.
    base_usage = {k: dict(v) for k, v in run.usage.items()}
    base_seconds, started = run.seconds, time.monotonic()

    def save(r: Run) -> None:
        merged = {k: dict(v) for k, v in base_usage.items()}
        for name, row in eng.USAGE.snapshot().items():
            into = merged.setdefault(name, {})
            for key, value in row.items():
                into[key] = round(into.get(key, 0) + value, 4)
        r.usage = merged
        r.seconds = round(base_seconds + time.monotonic() - started, 1)
        r.save(directory)

    cfg.save = save
    log(f"run {run.id} -> {directory}")
    try:
        if args.command == "ask":
            investigate(run, cfg)
        else:
            follow_up(run, " ".join(args.question).strip(), cfg)
    except (eng.EngineError, PipelineError) as exc:
        print(f"odds: error: {exc}", file=sys.stderr)
        print(f"odds: partial run saved in {directory}", file=sys.stderr)
        return 1
    path = _render(run, directory)
    print(_summary(run))
    log(f"report: {path}")
    if args.open:
        webbrowser.open(path.as_uri())
    return 0


def _config(args: argparse.Namespace, log: object) -> Config:
    synth = eng.parse(args.engine)
    researchers = [eng.parse(s) for s in args.research.split(",") if s.strip()]
    if not researchers:
        raise eng.EngineError("--research needs at least one engine")
    interactive = not args.no_ask and sys.stdin.isatty()

    def ask_user(question: str, why: str) -> str:
        print(f"\nodds asks: {question}", file=sys.stderr)
        if why:
            print(f"  (why: {why})", file=sys.stderr)
        print("  answer, or Enter to let it assume > ", end="", file=sys.stderr, flush=True)
        try:
            return input().strip()
        except EOFError:
            return ""

    adapters = [a.strip() for a in args.sources.split(",")] if args.sources else None
    progress = log if callable(log) else (lambda _m: None)
    return Config(synth, researchers, DEPTHS[args.depth], interactive=interactive,
                  adapters=adapters, progress=progress, ask_user=ask_user)
