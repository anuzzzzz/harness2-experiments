"""Render an h2x tau transcript (transcript.json) for the improver. Pure Python, no tau2 import.

transcript.json schema (written by h2x.tau_runner.cli):

    {"domain": str, "task_id": str, "termination_reason": str,
     "messages": [
        {"role": "assistant"|"user", "content": str|None,
         "tool_calls": [{"id": str, "name": str, "arguments": {...}}] | None},
        {"role": "tool", "id": str, "requestor": "assistant"|"user", "content": str, "error": bool},
        {"role": "harness", "event": "open_guide", "name": str, "found": bool, "chars": int}
     ]}

The grade is never part of this file.
"""

from __future__ import annotations

import json
from pathlib import Path

TRUNC = "\n[... {n} more characters truncated by the renderer, not by the agent ...]"

ENDINGS = {
    "user_stop": "the customer ended the conversation",
    "agent_stop": "the agent ended the conversation",
    "max_steps": "the conversation hit the step limit before anyone ended it",
    "timeout": "the conversation hit the wall-clock limit",
    "too_many_errors": "the conversation stopped after too many consecutive tool errors",
    "agent_error": "the conversation stopped because the agent broke the chat protocol",
    "user_error": "the conversation stopped because the customer simulator broke the protocol",
}


def clip(text: object, limit: int) -> str:
    s = "" if text is None else str(text)
    if limit and limit > 0 and len(s) > limit:
        return s[:limit] + TRUNC.format(n=len(s) - limit)
    return s


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _args(arguments: object, limit: int = 2000) -> str:
    return clip(json.dumps(arguments, ensure_ascii=False, sort_keys=True), limit)


def render_trajectory(transcript: dict, *, obs_trunc: int = 50_000, max_chars: int = 400_000) -> str:
    lines = [
        f"[HARNESS] customer-service chat | domain={transcript.get('domain')} "
        f"| conversation={transcript.get('task_id')}"
    ]
    for m in transcript.get("messages", []):
        role = m.get("role")
        if role == "harness":
            if m.get("event") == "open_guide":
                status = "opened" if m.get("found") else "NOT FOUND"
                lines.append(f"[ACTION] open_guide: {json.dumps({'name': m.get('name')})}")
                lines.append(f"[OBSERVATION] (guide {m.get('name')!r} {status}, {m.get('chars', 0)} chars)")
            elif m.get("event") == "dropped_local_call":
                lines.append(
                    f"[HARNESS] open_guide was requested together with system tools in one turn; "
                    f"only the system tools ran ({m.get('name')!r} was not opened)"
                )
            continue
        if role in ("assistant", "user"):
            who = "ASSISTANT" if role == "assistant" else "USER"
            if m.get("content"):
                lines.append(f"[{who}] {clip(m['content'], obs_trunc)}")
            for call in m.get("tool_calls") or []:
                tag = "ACTION" if role == "assistant" else "USER ACTION"
                lines.append(f"[{tag}] {call.get('name')}: {_args(call.get('arguments'))}")
        elif role == "tool":
            prefix = "OBSERVATION" if m.get("requestor", "assistant") == "assistant" else "USER OBSERVATION"
            if m.get("error"):
                prefix += " ERROR"
            lines.append(f"[{prefix}] {clip(m.get('content'), obs_trunc)}")
    ending = transcript.get("termination_reason") or "unknown"
    lines.append(f"[HARNESS] conversation ended: {ending} ({ENDINGS.get(ending, ending)})")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + TRUNC.format(n=len(text) - max_chars)
    return text


def render_conversation(transcript: dict) -> str:
    """The customer-facing resolution: what was said, and which actions the agent took."""
    out = ["# Conversation", ""]
    actions = []
    for m in transcript.get("messages", []):
        role = m.get("role")
        if role == "assistant":
            if m.get("content"):
                out += [f"**Agent:** {m['content']}", ""]
            for call in m.get("tool_calls") or []:
                actions.append(f"- `{call.get('name')}` {_args(call.get('arguments'), 600)}")
        elif role == "user" and m.get("content"):
            out += [f"**Customer:** {m['content']}", ""]
    out += ["# Tool calls made by the agent", ""]
    out += actions or ["(none)"]
    return "\n".join(out).rstrip() + "\n"


def n_agent_messages(transcript: dict) -> int:
    return sum(1 for m in transcript.get("messages", []) if m.get("role") == "assistant")
