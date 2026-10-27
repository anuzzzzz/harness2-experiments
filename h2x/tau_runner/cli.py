"""python -m h2x.tau_runner run ...  -- one tau2-bench conversation with an h2x harness.

Writes into --out (agent-visible evidence only):
    transcript.json   the conversation, tool calls/results and guide openings
    policy.md         the policy text exactly as the agent received it
    meta.json         ok, termination reason, token cost (agent + simulated customer)
Writes to --grade (hidden from the improver): the tau2 reward and its breakdown.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
import traceback
from pathlib import Path

from .. import hidden, pricing

EXIT_INFRA = 75
EXIT_STOP = 76
CREDIT = re.compile(r"credit balance is too low|insufficient[_ ]quota|billing", re.I)
RATE = re.compile(r"rate[ _-]?limit|\b429\b|overloaded|\b529\b", re.I)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m h2x.tau_runner")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one conversation")
    r.add_argument("--domain", required=True)
    r.add_argument("--task-id", required=True)
    r.add_argument("--harness", required=True, type=Path)
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--grade", required=True, type=Path)
    r.add_argument("--agent-model", required=True)
    r.add_argument("--user-model", required=True)
    r.add_argument("--seed", type=int, default=300)
    r.add_argument("--max-steps", type=int, default=200)
    r.add_argument("--max-errors", type=int, default=10)
    r.add_argument("--timeout", type=float, default=900.0)
    r.add_argument("--retrieval-config", default="")
    r.add_argument("--communicate-scorer", default="substring", choices=("substring", "semantic"))
    r.add_argument("--communicate-judge", default="anthropic/claude-haiku-5-5")
    r.add_argument("--max-tokens", type=int, default=16000)
    r.add_argument("--retries", type=int, default=2)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.cmd == "run":
        return run(args)
    return 2


def _fake(args) -> int:
    """Offline stand-in used by CI plumbing tests (H2X_FAKE=1). No model is called."""
    agent_md = (args.harness / "AGENT.md")
    text = agent_md.read_text(encoding="utf-8").lower() if agent_md.is_file() else ""
    guides = sorted(p.stem for p in (args.harness / "guides").glob("*.md")) if (args.harness / "guides").is_dir() else []
    h = int(hashlib.sha1(f"{args.task_id}:{args.seed}".encode()).hexdigest(), 16)
    reward = 1.0 if ("verify" in text or h % 3 == 0) else 0.0
    messages = [
        {"role": "assistant", "content": "Hi! How can I help you today?", "tool_calls": None},
        {"role": "user", "content": f"(fake customer for task {args.task_id}) I need help.", "tool_calls": None},
    ]
    if guides:
        messages.append({"role": "harness", "event": "open_guide", "name": guides[0], "found": True, "chars": 10})
    messages += [
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "t1", "name": "get_user_details", "arguments": {"user_id": "u1"}}]},
        {"role": "tool", "id": "t1", "requestor": "assistant", "content": "{\"name\": \"Fake\"}", "error": False},
        {"role": "assistant", "content": "Done. Anything else?", "tool_calls": None},
        {"role": "user", "content": "No thanks ###STOP###", "tool_calls": None},
    ]
    transcript = {"domain": args.domain, "task_id": args.task_id,
                  "termination_reason": "user_stop", "messages": messages}
    (args.out / "transcript.json").write_text(json.dumps(transcript, indent=2), encoding="utf-8")
    (args.out / "policy.md").write_text(f"# Fake {args.domain} policy\n", encoding="utf-8")
    hidden.write(args.grade, {"reward": reward, "fake": True})
    _meta(args.out, ok=True, cost=0.001, termination="user_stop", fake=True)
    return 0


def _meta(out: Path, **fields) -> None:
    data = {"ok": fields.pop("ok"), "cost_usd": round(float(fields.pop("cost")), 6),
            "termination_reason": fields.pop("termination", None), **fields}
    (out / "meta.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def _quiet() -> None:
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    try:
        from loguru import logger

        logger.remove()
        logger.add(sys.stderr, level="WARNING")
    except Exception:
        pass


def _calls(message) -> list[dict] | None:
    calls = getattr(message, "tool_calls", None) or []
    return [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in calls] or None


def build_transcript(sim, recorder, domain: str, task_id: str) -> dict:
    events = sorted(recorder.events, key=lambda e: e["before_agent_message"])
    out, produced, cursor = [], 0, 0

    def flush(limit: int) -> None:
        nonlocal cursor
        while cursor < len(events) and events[cursor]["before_agent_message"] <= limit:
            event = {k: v for k, v in events[cursor].items() if k != "before_agent_message"}
            out.append({"role": "harness", **event})
            cursor += 1

    for m in sim.messages or []:
        kind = type(m).__name__
        if kind == "AssistantMessage":
            if getattr(m, "usage", None) is not None or getattr(m, "raw_data", None) is not None:
                flush(produced)
                produced += 1
            out.append({"role": "assistant", "content": m.content, "tool_calls": _calls(m)})
        elif kind == "UserMessage":
            out.append({"role": "user", "content": m.content, "tool_calls": _calls(m)})
        elif kind == "ToolMessage":
            out.append({"role": "tool", "id": m.id, "requestor": m.requestor,
                        "content": m.content, "error": bool(m.error)})
        elif kind == "MultiToolMessage":
            for t in m.tool_messages:
                out.append({"role": "tool", "id": t.id, "requestor": t.requestor,
                            "content": t.content, "error": bool(t.error)})
    flush(10**9)
    reason = sim.termination_reason
    return {
        "domain": domain,
        "task_id": task_id,
        "termination_reason": getattr(reason, "value", reason),
        "messages": out,
    }


def _user_cost(sim, model: str) -> float:
    total = 0.0
    for m in sim.messages or []:
        if type(m).__name__ == "UserMessage":
            total += pricing.litellm_usage_cost(model, getattr(m, "usage", None))
    return total


def run(args) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    if os.environ.get("H2X_FAKE", "").strip() not in ("", "0", "false"):
        return _fake(args)
    _quiet()
    import litellm

    litellm.register_model(pricing.litellm_registration([args.agent_model, args.user_model]))

    from tau2.data_model.semantic_communication import (
        CommunicateScoring,
        SemanticCommunicateSettings,
    )
    from tau2.data_model.simulation import TextRunConfig
    from tau2.run import get_tasks, run_single_task

    from .agent import HarnessBundle, Recorder, register

    bundle = HarnessBundle.load(args.harness)
    task = get_tasks(args.domain, task_split_name=None, task_ids=[args.task_id])[0]
    if args.communicate_scorer == "substring":
        scoring = CommunicateScoring(scorer="substring")
    else:
        scoring = CommunicateScoring(
            scorer="semantic",
            judge=SemanticCommunicateSettings(model=args.communicate_judge, model_args={}),
        )
    # No temperature/top_p: Claude 5.5 models reject non-default sampling parameters.
    llm_args = {"max_tokens": args.max_tokens}
    attempts: list[dict] = []
    spent = 0.0
    for attempt in range(args.retries + 1):
        recorder = Recorder()
        agent_name = register(bundle, recorder)
        config = TextRunConfig(
            domain=args.domain,
            agent=agent_name,
            llm_agent=args.agent_model,
            llm_args_agent=dict(llm_args),
            user="user_simulator",
            llm_user=args.user_model,
            llm_args_user=dict(llm_args),
            max_steps=args.max_steps,
            max_errors=args.max_errors,
            seed=args.seed,
            timeout=args.timeout,
            communicate_scoring=scoring,
            retrieval_config=args.retrieval_config or None,
        )
        started = time.time()
        try:
            sim = run_single_task(config, task, seed=args.seed)
        except Exception as exc:  # API errors, provider outages, protocol bugs
            cost = sum(pricing.litellm_usage_cost(args.agent_model, u) for u in recorder.usages)
            spent += cost
            err = f"{type(exc).__name__}: {exc}"
            attempts.append({"error": err[:4000], "cost_usd": round(cost, 6),
                             "traceback": traceback.format_exc()[-4000:]})
            print(f"attempt {attempt + 1} failed: {err[:500]}", file=sys.stderr, flush=True)
            if CREDIT.search(err):
                _meta(args.out, ok=False, cost=spent, error=err[:2000], attempts=attempts)
                return EXIT_STOP
            if attempt < args.retries:
                time.sleep(20 * (attempt + 1))
                continue
            _meta(args.out, ok=False, cost=spent, error=err[:2000], attempts=attempts)
            return EXIT_STOP if RATE.search(err) else EXIT_INFRA
        agent_cost = sum(pricing.litellm_usage_cost(args.agent_model, u) for u in recorder.usages)
        user_cost = _user_cost(sim, args.user_model)
        spent += agent_cost + user_cost
        reason = getattr(sim.termination_reason, "value", sim.termination_reason)
        if reason == "infrastructure_error" and attempt < args.retries:
            attempts.append({"error": "infrastructure_error", "cost_usd": agent_cost + user_cost})
            time.sleep(20 * (attempt + 1))
            continue
        transcript = build_transcript(sim, recorder, args.domain, args.task_id)
        (args.out / "transcript.json").write_text(
            json.dumps(transcript, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (args.out / "policy.md").write_text(recorder.policy, encoding="utf-8")
        reward_info = sim.reward_info.model_dump(mode="json") if sim.reward_info else None
        reward = sim.reward_info.reward if sim.reward_info else None
        if reason == "infrastructure_error":
            reward = None  # not the agent's doing: excluded, never scored as a zero
        hidden.write(
            args.grade,
            {"reward": reward, "reward_info": reward_info, "termination_reason": reason,
             "seed": args.seed},
        )
        _meta(
            args.out,
            ok=reason != "infrastructure_error",
            cost=spent,
            termination=reason,
            agent_cost_usd=round(agent_cost, 6),
            user_cost_usd=round(user_cost, 6),
            agent_calls=len(recorder.usages),
            guide_opens=sum(1 for e in recorder.events if e.get("event") == "open_guide"),
            seconds=round(time.time() - started, 1),
            attempts=attempts,
        )
        return 0 if reason != "infrastructure_error" else EXIT_INFRA
    return EXIT_INFRA
