from __future__ import annotations

from typing import Any

import pytest

from odds.engines import Engine, EngineError, ask_json, extract_json, parse


def test_extract_json_from_prose_and_fences() -> None:
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure!\n```json\n{"a": [1, 2]}\n```\nDone.') == {"a": [1, 2]}
    assert extract_json('noise {"a": "x}y", "b": {"c": 2}} trailing') == {"a": "x}y", "b": {"c": 2}}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_parse_specs() -> None:
    assert parse("claude:opus").model == "opus"
    assert parse("ollama:qwen3:8b").model == "qwen3:8b"
    assert parse("codex").has_web and not parse("ollama").has_web
    with pytest.raises(EngineError):
        parse("gpt")


class Scripted(Engine):
    def __init__(self, replies: list[str]) -> None:
        super().__init__("claude", "fake")
        self.replies = replies
        self.prompts: list[tuple[str, bool]] = []

    def ask(self, prompt: str, *, web: bool = False, timeout: float = 600.0) -> str:
        self.prompts.append((prompt, web))
        return self.replies.pop(0)


def test_ask_json_repairs_once_without_web() -> None:
    engine = Scripted(["I think the answer is great", '{"ok": true}'])
    assert ask_json(engine, "give json", web=True) == {"ok": True}
    assert engine.prompts[0][1] is True
    assert engine.prompts[1][1] is False  # the repair must not go back to the web
    assert "could not be used" in engine.prompts[1][0]


def test_ask_json_shape_check_and_give_up() -> None:
    def check(v: Any) -> str | None:
        return None if "x" in v else "missing x"

    engine = Scripted(['{"y": 1}', '{"y": 2}'])
    with pytest.raises(EngineError):
        ask_json(engine, "p", check=check)
