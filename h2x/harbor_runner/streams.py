"""Claude Code stream-json helpers (pure Python): final message and per-request cost."""

from __future__ import annotations

import json
from pathlib import Path

from .. import pricing


def events(path: Path) -> list[dict]:
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace").replace("\x00", "")
    except OSError:
        return []
    out = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            out.append(event)
    return out


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def final_message(path: Path) -> str:
    last, result = "", None
    for event in events(path):
        if event.get("type") == "result" and isinstance(event.get("result"), str):
            result = event["result"]
        elif event.get("type") == "assistant" and isinstance(event.get("message"), dict):
            text = _text(event["message"].get("content"))
            if text.strip():
                last = text
    return result if result is not None else last


def reported_cost(path: Path) -> float | None:
    for event in reversed(events(path)):
        if event.get("type") == "result":
            value = event.get("total_cost_usd")
            return float(value) if isinstance(value, (int, float)) else None
    return None


def usage_cost(path: Path, model: str) -> float:
    """Per-request cost from assistant-message usage, de-duplicated by message id."""
    seen: dict[str, dict] = {}
    for event in events(path):
        if event.get("type") != "assistant":
            continue
        message = event.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("usage"), dict):
            continue
        key = str(message.get("id") or len(seen))
        seen[key] = {"usage": message["usage"], "model": message.get("model") or model}
    total = 0.0
    for item in seen.values():
        try:
            total += pricing.anthropic_usage_cost(item["model"], item["usage"])
        except KeyError:
            total += pricing.anthropic_usage_cost(model, item["usage"])
    return total


def cost(path: Path, model: str) -> float:
    reported = reported_cost(path)
    if reported is not None and reported > 0:
        return reported
    return usage_cost(path, model)


def final_error(path: Path) -> str:
    """The final result event's text when the session ended in an error, else ''.

    Transient API errors that Claude Code retried past do not count: only how the session ended.
    """
    for event in reversed(events(path)):
        if event.get("type") == "result":
            if event.get("is_error") or event.get("subtype") not in (None, "success"):
                return json.dumps(event)[:4000]
            return ""
    return ""
