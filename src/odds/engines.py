"""LLM engines: `claude -p`, `codex exec`, and a local Ollama model.

Every engine is one blocking call, prompt in, text out. Keyless on purpose --
the two CLIs ride the user's existing subscriptions, and Ollama is local -- so
there is no API key anywhere in this tool.

Web access differs per engine and that difference is real, not cosmetic:

| engine | web | how |
|---|---|---|
| claude | yes | WebSearch + WebFetch, nothing else |
| codex  | yes | `codex --search exec`, read-only sandbox |
| ollama | no  | gets the source adapters' hits pasted into the prompt instead |

Containment for `claude -p` (measured, see the README's limitations): the
child inherits the user's global CLAUDE.md and cannot be stopped from doing so
while staying keyless, so the tool list is the actual fence. `--tools` sets
what exists, `--allowedTools` what is auto-approved -- both are needed.
`--tools ""` does not reliably mean "none", so a reasoning-only call gets the
inert `TodoWrite` instead.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class EngineError(Exception):
    """An engine could not produce an answer (missing, timed out, crashed)."""


class Usage:
    """Calls, wall seconds and the CLIs' reported cost, per engine, for this process.

    `cost_usd` is what `claude -p` reports for the call. On a subscription it is
    the notional API-price equivalent, not money charged -- still the honest
    yardstick for comparing depths. codex and ollama report none.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.by_engine: dict[str, dict[str, float]] = {}

    def add(self, name: str, seconds: float, cost: float = 0.0, failed: bool = False) -> None:
        with self._lock:
            row = self.by_engine.setdefault(
                name, {"calls": 0, "failed": 0, "seconds": 0.0, "cost_usd": 0.0})
            row["calls"] += 1
            row["failed"] += int(failed)
            row["seconds"] = round(row["seconds"] + seconds, 1)
            row["cost_usd"] = round(row["cost_usd"] + cost, 4)

    def snapshot(self) -> dict[str, dict[str, float]]:
        with self._lock:
            return {k: dict(v) for k, v in self.by_engine.items()}


USAGE = Usage()
_COST = threading.local()


@dataclass
class Engine:
    kind: str  # claude | codex | ollama
    model: str = ""

    @property
    def name(self) -> str:
        return f"{self.kind}:{self.model}" if self.model else self.kind

    @property
    def has_web(self) -> bool:
        return self.kind in ("claude", "codex")

    def ask(self, prompt: str, *, web: bool = False, timeout: float = 600.0) -> str:
        web = web and self.has_web
        started = time.monotonic()
        _COST.value = 0.0
        try:
            if self.kind == "claude":
                text = _claude(prompt, self.model, web, timeout)
            elif self.kind == "codex":
                text = _codex(prompt, self.model, web, timeout)
            elif self.kind == "ollama":
                text = _ollama(prompt, self.model or "qwen3:8b", timeout)
            else:
                raise EngineError(f"unknown engine {self.kind!r}")
        except EngineError:
            USAGE.add(self.name, time.monotonic() - started, failed=True)
            raise
        USAGE.add(self.name, time.monotonic() - started, float(getattr(_COST, "value", 0.0)))
        return text


def parse(spec: str) -> Engine:
    """`claude`, `claude:opus`, `codex`, `codex:gpt-5`, `ollama:qwen3:8b`."""
    kind, _, model = spec.strip().partition(":")
    if kind not in ("claude", "codex", "ollama"):
        raise EngineError(f"unknown engine {spec!r} (want claude, codex or ollama)")
    return Engine(kind, model)


def available() -> dict[str, str]:
    """engine -> "ok" or why not. Cheap: no model calls."""
    out = {
        "claude": "ok" if shutil.which("claude") else "claude CLI not on PATH",
        "codex": "ok" if shutil.which("codex") else "codex CLI not on PATH",
    }
    try:
        tags = _ollama_get("/api/tags", 3.0)
        names = [m.get("name", "") for m in tags.get("models", [])]
        out["ollama"] = "ok: " + ", ".join(names) if names else "ollama up but no models pulled"
    except EngineError as exc:
        out["ollama"] = str(exc)
    return out


# ---- claude ----------------------------------------------------------------


def _claude(prompt: str, model: str, web: bool, timeout: float) -> str:
    tools = "WebSearch,WebFetch" if web else "TodoWrite"
    cmd = [
        "claude", "-p",
        "--tools", tools, "--allowedTools", tools,
        "--strict-mcp-config", "--setting-sources", "",
        "--output-format", "json",
    ]
    if model:
        cmd += ["--model", model]
    raw = _run(cmd, prompt, timeout, cwd=tempfile.gettempdir())
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EngineError(f"claude returned non-JSON envelope: {raw[:200]!r}") from exc
    if envelope.get("is_error"):
        raise EngineError(f"claude error: {str(envelope.get('result'))[:300]}")
    _COST.value = float(envelope.get("total_cost_usd") or 0.0)
    return str(envelope.get("result", ""))


# ---- codex -----------------------------------------------------------------


def _codex(prompt: str, model: str, web: bool, timeout: float) -> str:
    # `--search` is a top-level flag: `codex exec --search` is rejected outright.
    with tempfile.TemporaryDirectory(prefix="odds-codex-") as work:
        last = os.path.join(work, "last.txt")
        cmd = ["codex"] + (["--search"] if web else []) + [
            "exec", "-s", "read-only", "--skip-git-repo-check", "--ephemeral",
            "-o", last,
        ]
        if model:
            cmd += ["-m", model]
        cmd.append("-")
        _run(cmd, prompt, timeout, cwd=work)
        try:
            with open(last, encoding="utf-8") as fh:
                return fh.read()
        except OSError as exc:
            raise EngineError("codex finished without a final message") from exc


# ---- ollama ----------------------------------------------------------------

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if not OLLAMA.startswith("http"):
    OLLAMA = "http://" + OLLAMA


def _ollama_get(path: str, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(OLLAMA + path, timeout=timeout) as response:
            return dict(json.loads(response.read()))
    except (OSError, ValueError) as exc:
        raise EngineError(f"ollama unreachable at {OLLAMA}: {exc}") from exc


def _ollama(prompt: str, model: str, timeout: float) -> str:
    body = {
        "model": model,
        "stream": False,
        "format": "json",
        # qwen3 thinks by default; with format=json the thinking eats the
        # budget and the answer comes back truncated.
        "think": False,
        "options": {"num_ctx": 32768, "temperature": 0.4},
        "messages": [{"role": "user", "content": prompt}],
    }
    request = urllib.request.Request(
        OLLAMA + "/api/chat", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise EngineError(f"ollama {exc.code}: {exc.read()[:200]!r}") from exc
    except (OSError, ValueError) as exc:
        raise EngineError(f"ollama call failed: {exc}") from exc
    return str(data.get("message", {}).get("content", ""))


# ---- shared ----------------------------------------------------------------


def _run(cmd: list[str], stdin: str, timeout: float, cwd: str) -> str:
    if not shutil.which(cmd[0]):
        raise EngineError(f"{cmd[0]} not on PATH")
    try:
        done = subprocess.run(
            cmd, input=stdin, capture_output=True, text=True, timeout=timeout, cwd=cwd, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise EngineError(f"{cmd[0]} timed out after {timeout:.0f}s") from exc
    if done.returncode != 0:
        tail = (done.stderr or done.stdout).strip()[-400:]
        raise EngineError(f"{cmd[0]} exited {done.returncode}: {tail}")
    return done.stdout


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Pull the JSON value out of a model reply that may wrap it in prose or fences."""
    text = text.strip()
    candidates = [text] + _FENCE.findall(text)
    for start_char in "{[":
        start = text.find(start_char)
        if start != -1:
            candidates.append(_balanced(text, start))
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
    raise ValueError("no JSON value found in reply")


def _balanced(text: str, start: int) -> str:
    """The substring from `start` to its matching close bracket, string-aware."""
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]


def ask_json(
    engine: Engine,
    prompt: str,
    *,
    web: bool = False,
    timeout: float = 600.0,
    repairs: int = 1,
    check: Callable[[Any], str | None] | None = None,
) -> Any:
    """Ask for JSON; on a bad reply, show the model its own output and ask again.

    `check` returns a complaint string when the shape is wrong, None when fine.
    The repair turn is a fresh, tool-less call: it only reformats, it must not
    go back to the web and come home with different facts.
    """
    reply = engine.ask(prompt, web=web, timeout=timeout)
    for attempt in range(repairs + 1):
        try:
            value = extract_json(reply)
            complaint = check(value) if check else None
            if complaint is None:
                return value
        except ValueError as exc:
            complaint = str(exc)
        if attempt == repairs:
            raise EngineError(f"{engine.name}: unusable reply after repair: {complaint}")
        reply = engine.ask(
            "Your previous reply could not be used: " + complaint + ".\n"
            "Re-emit the SAME content as a single valid JSON value matching the "
            "requested shape. Output only the JSON, nothing else.\n\n"
            "Original instructions (for the shape):\n" + prompt[-6000:] +
            "\n\nYour previous reply:\n" + reply[:30000],
            web=False, timeout=timeout,
        )
    raise EngineError("unreachable")
