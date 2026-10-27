"""python -m h2x.harbor_runner run ...  -- one Harbor trial with Claude Code and an h2x harness.

Writes into --out (agent-visible evidence only):
    stream.jsonl        Claude Code's stream-json transcript
    final.md            the agent's final message
    deliverable.patch   the repository diff the agent left (when the workdir is a git repo)
    meta.json           ok, cost, timing, Harbor exception type (never the reward)
Writes to --grade (hidden): the verifier reward. Harbor's job directory (verifier output,
test logs) stays under --scratch, which the adapter never reads as evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .. import hidden
from . import streams

EXIT_INFRA = 75
EXIT_STOP = 76
CREDIT = re.compile(r"credit balance is too low|insufficient[_ ]quota", re.I)
RATE = re.compile(r"rate[ _-]?limit|\b429\b|overloaded_error|\b529\b", re.I)
# Claude Code stopped because the API refused, not because of anything the agent did.
API_FAIL = re.compile(
    r"API Error|authentication_error|permission_error|invalid_request_error|not_found_error|"
    r"overloaded_error|rate_limit_error|api_error|Credit balance",
    re.I,
)
AGENT_CLASS = "h2x.harbor_runner.agent:HarnessClaudeCode"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m h2x.harbor_runner")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--task-dir", required=True, type=Path, help="local Harbor task directory")
    r.add_argument("--harness", required=True, type=Path)
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--grade", required=True, type=Path)
    r.add_argument("--scratch", required=True, type=Path)
    r.add_argument("--model", required=True)
    r.add_argument("--max-budget-usd", type=float, default=0.0)
    r.add_argument("--max-turns", type=int, default=0)
    r.add_argument("--effort", default="")
    r.add_argument("--agent-timeout-multiplier", type=float, default=0.0)
    r.add_argument("--claude-version", default="")
    r.add_argument("--retries", type=int, default=1)
    r.add_argument("--harbor", default="harbor", help="harbor executable")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.cmd == "run":
        return run(args)
    return 2


def _meta(out: Path, **fields) -> None:
    data = {"ok": fields.pop("ok"), "cost_usd": round(float(fields.pop("cost")), 6), **fields}
    (out / "meta.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _fake(args) -> int:
    claude_md = args.harness / "CLAUDE.md"
    text = claude_md.read_text(encoding="utf-8").lower() if claude_md.is_file() else ""
    name = args.task_dir.name
    h = int(hashlib.sha1(f"{name}:{args.out.name}".encode()).hexdigest(), 16)
    reward = 1.0 if ("verify" in text or h % 3 == 0) else 0.0
    lines = [
        {"type": "system", "subtype": "init", "model": args.model, "cwd": "/app"},
        {"type": "assistant", "message": {"id": "m1", "content": [
            {"type": "tool_use", "id": "u1", "name": "Bash", "input": {"command": "ls"}}],
            "usage": {"input_tokens": 1000, "output_tokens": 50}}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "u1", "content": "README.md"}]}},
        {"type": "result", "subtype": "success", "result": f"(fake) finished {name}",
         "total_cost_usd": 0.002},
    ]
    (args.out / "stream.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    (args.out / "final.md").write_text(f"(fake) finished {name}\n", encoding="utf-8")
    (args.out / "deliverable.patch").write_text("", encoding="utf-8")
    hidden.write(args.grade, {"reward": reward, "fake": True})
    _meta(args.out, ok=True, cost=0.002, fake=True)
    return 0


def _trial_result(job_dir: Path) -> tuple[Path | None, dict]:
    for path in sorted(job_dir.rglob("result.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and "trial_name" in data:
            return path.parent, data
    return None, {}


def _reward(result: dict) -> float | None:
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    if not isinstance(rewards, dict) or not rewards:
        return None
    if "reward" in rewards:
        return float(rewards["reward"])
    values = [float(v) for v in rewards.values() if isinstance(v, (int, float))]
    return sum(values) / len(values) if values else None


def _command(args, job_dir: Path) -> list[str]:
    cmd = [
        args.harbor, "run",
        "--path", str(args.task_dir),
        "--agent", AGENT_CLASS,
        "-m", args.model,
        "--jobs-dir", str(job_dir.parent),
        "--job-name", job_dir.name,
        "-n", "1", "-k", "1",
        "-y", "-q",
    ]
    if args.max_budget_usd > 0:
        cmd += ["--ak", f'max_budget_usd="{args.max_budget_usd:.2f}"']
    if args.max_turns > 0:
        cmd += ["--ak", f"max_turns={args.max_turns}"]
    if args.claude_version:
        cmd += ["--ak", f'version="{args.claude_version}"']
    if args.effort:
        cmd += ["--effort", args.effort]
    if args.agent_timeout_multiplier > 0:
        cmd += ["--agent-timeout-multiplier", str(args.agent_timeout_multiplier)]
    return cmd


def run(args) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    if os.environ.get("H2X_FAKE", "").strip() not in ("", "0", "false"):
        return _fake(args)
    env = dict(os.environ)
    env["H2X_HARNESS_DIR"] = str(args.harness.resolve())
    env.setdefault("HARBOR_TELEMETRY", "0")
    attempts: list[dict] = []
    spent = 0.0
    for attempt in range(args.retries + 1):
        job_dir = args.scratch / f"job-{attempt}"
        shutil.rmtree(job_dir, ignore_errors=True)
        job_dir.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        proc = subprocess.run(
            _command(args, job_dir), env=env, stdin=subprocess.DEVNULL,
            stdout=sys.stdout, stderr=sys.stderr,
        )
        trial_dir, result = _trial_result(job_dir)
        stream = trial_dir / "agent" / "claude-code.txt" if trial_dir else None
        cost = 0.0
        if stream and stream.is_file():
            cost = streams.cost(stream, args.model)
        elif result:
            cost = float((result.get("agent_result") or {}).get("cost_usd") or 0.0)
        spent += cost
        exc = result.get("exception_info") or {}
        exc_type = str(exc.get("exception_type") or "")
        exc_msg = str(exc.get("exception_message") or "")
        reward = _reward(result)
        api_errors = (streams.final_error(stream) if stream and stream.is_file() else "") + exc_msg
        record = {"attempt": attempt, "harbor_exit": proc.returncode, "exception": exc_type,
                  "message": exc_msg[:1500], "reward_present": reward is not None,
                  "cost_usd": round(cost, 6), "seconds": round(time.time() - started, 1)}
        attempts.append(record)
        if CREDIT.search(api_errors):
            _meta(args.out, ok=False, cost=spent, error="provider credit exhausted", attempts=attempts)
            return EXIT_STOP
        agent_ran = stream is not None and stream.is_file() and stream.stat().st_size > 0
        api_fail = bool(API_FAIL.search(api_errors))
        if agent_ran and not api_fail:
            if reward is None:
                # The agent ran but no reward was produced (timeout, crash, tests hung on its
                # changes): Harbor's convention is to count the trial as a failure.
                reward = 0.0
                record["reward_defaulted_to_zero"] = True
            shutil.copyfile(stream, args.out / "stream.jsonl")
            (args.out / "final.md").write_text(streams.final_message(stream), encoding="utf-8")
            diff = trial_dir / "agent" / "h2x-diff.patch"
            if diff.is_file():
                shutil.copyfile(diff, args.out / "deliverable.patch")
            hidden.write(args.grade, {"reward": reward, "rewards": (result.get("verifier_result") or {}).get("rewards"),
                                      "exception": exc_type, "trial_dir": str(trial_dir)})
            _meta(args.out, ok=True, cost=spent, exception=exc_type or None,
                  agent_timed_out=exc_type.startswith("AgentTimeout"), attempts=attempts,
                  timings={k: result.get(k) for k in ("environment_setup", "agent_setup",
                                                       "agent_execution", "verifier")})
            return 0
        print(f"attempt {attempt + 1}: no usable result ({exc_type or 'no exception'}: "
              f"{exc_msg[:300]})", file=sys.stderr, flush=True)
        if RATE.search(api_errors) and attempt == args.retries:
            _meta(args.out, ok=False, cost=spent, error="rate limited", attempts=attempts)
            return EXIT_STOP
        if attempt < args.retries:
            time.sleep(30)
    _meta(args.out, ok=False, cost=spent, error="no verifier result after retries",
          attempts=attempts)
    return EXIT_INFRA
