"""Phase 2: the transfer test. Base harness vs final harness on held-out test tasks.

Every (test task, repeat) is run once with the starting harness ("base") and once with the
harness Harness2 consolidated from the train tasks ("final"). Both arms of a pair go to the
same shard so they run on the same machine at about the same time. Results are appended to
results/<exp>/eval/shard-<i>.json after every rollout, so an interrupted shard resumes.

A rollout that failed for infrastructure reasons (API outage, container failure) or whose grade
could not be computed is recorded with value None and excluded from the comparison -- never
scored as a zero.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from . import h2, ledger
from .config import Experiment

ARMS = ("base", "final")


def plan_jobs(exp: Experiment) -> list[dict]:
    jobs = []
    tasks = exp.split("test")
    for ti, task in enumerate(tasks):
        for rep in range(exp.repeats):
            shard = (ti * exp.repeats + rep) % max(1, exp.shards)
            for arm in ARMS:
                jobs.append({"task": task, "rep": rep, "arm": arm, "shard": shard,
                             "key": f"{arm}/{task}/r{rep}"})
    return jobs


def shard_path(exp: Experiment, shard: int) -> Path:
    return exp.results_dir / "eval" / f"shard-{shard}.json"


def load_shard(exp: Experiment, shard: int) -> dict:
    try:
        return json.loads(shard_path(exp, shard).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"experiment": exp.id, "shard": shard, "jobs": {}}


def _lab_infra(run_dir: Path) -> bool:
    """A Claude Code rollout (LAB, Harness2's own executor) that died on an API error."""
    try:
        meta = json.loads((run_dir / "meta.json").read_text())
    except (OSError, ValueError):
        return True
    if meta.get("ok"):
        return False
    from .harbor_runner import streams

    tail = streams.final_error(run_dir / "stream.jsonl")
    return bool(tail) and any(
        s in tail for s in ("API Error", "api_error", "overloaded", "rate_limit", "authentication")
    )


def value_of(exp: Experiment, task: str, score: dict | None) -> tuple[float | None, str]:
    if not score or not score.get("n_criteria"):
        return None, "grade_failed"
    if "reward" in score:
        return float(score["reward"]), "ok"
    if exp.suite == "h2x-lab":
        from .lab.importer import expected_criteria

        want = expected_criteria(exp, task)
        if want and score["n_criteria"] != want:
            return None, "grade_failed"
    return score["n_passed"] / score["n_criteria"], "ok"


def run_shard(exp: Experiment, shard: int, *, budget_usd: float | None = None,
              concurrency: int | None = None) -> dict:
    from harness2.core import events

    adapter = h2.adapter(exp)
    final = h2.final_harness_for_eval(exp, adapter)
    if final is None:
        raise SystemExit(f"{exp.id}: no final harness yet -- run the train phase first")
    state = load_shard(exp, shard)
    state.update(experiment=exp.id, shard=shard, shards=exp.shards)
    done = state.setdefault("jobs", {})
    mine = [j for j in plan_jobs(exp) if j["shard"] == shard]
    todo = [j for j in mine if (done.get(j["key"]) or {}).get("status") not in ("ok",)]
    total_eval = budget_usd if budget_usd is not None else exp.eval_budget_usd
    # never more than what the whole experiment has left (eval spend so far included)
    total_eval = min(total_eval, ledger.remaining(exp) + ledger.spent_in(exp, "eval"))
    cap = max(0.0, total_eval) / max(1, exp.shards)
    lock = threading.Lock()
    stop = threading.Event()
    path = shard_path(exp, shard)
    path.parent.mkdir(parents=True, exist_ok=True)

    def spent() -> float:
        return sum(float(j.get("cost_usd", 0.0)) + float(j.get("judge_cost_usd", 0.0))
                   for j in done.values())

    def save() -> None:
        state["spent_usd"] = round(spent(), 6)
        state["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)

    def work(job: dict) -> None:
        key = job["key"]
        with lock:
            if stop.is_set() or spent() >= cap:
                done[key] = {**job, "status": "skipped_budget" if not stop.is_set() else "stopped"}
                save()
                return
        run_id = f"h2x-eval/{job['arm']}/{job['task']}/r{job['rep']}"
        started = time.monotonic()
        record = {**job, "run_id": run_id}
        try:
            run_dir = adapter.execute(job["task"], None if job["arm"] == "base" else final, run_id)
        except events.Stopped as exc:
            stop.set()
            record.update(status="stopped", error=str(exc))
            run_dir = None
        except Exception as exc:
            record.update(status="infra", error=f"{type(exc).__name__}: {exc}"[:1500])
            run_dir = None
        if run_dir is not None:
            record["run_dir"] = run_dir.name
            record["cost_usd"] = float(adapter.rollout_cost(run_dir))
            if exp.suite == "h2x-lab" and _lab_infra(run_dir):
                record.update(status="infra", error="API failure inside Claude Code")
            else:
                had_grade = (run_dir / "judge.json").is_file()
                try:
                    score = adapter.score(run_dir, job["task"])
                except Exception as exc:
                    score = None
                    record["error"] = f"grading: {type(exc).__name__}: {exc}"[:1500]
                judge = getattr(adapter, "judge_cost", None)
                if callable(judge) and not had_grade:
                    record["judge_cost_usd"] = float(judge(run_dir))
                value, status = value_of(exp, job["task"], score)
                if status == "grade_failed":
                    (run_dir / "judge.json").unlink(missing_ok=True)  # re-grade on resume
                record.update(status=status, value=value,
                              score={k: v for k, v in (score or {}).items() if k != "output_tail"})
        record["seconds"] = round(time.monotonic() - started, 1)
        with lock:
            done[key] = record
            save()
        print(f"[h2x eval {exp.id} s{shard}] {key}: {record['status']} "
              f"value={record.get('value')} cost=${record.get('cost_usd', 0):.3f}",
              file=sys.stderr, flush=True)

    state["started"] = state.get("started") or datetime.now(timezone.utc).isoformat(timespec="seconds")
    save()
    width = concurrency or exp.eval_concurrency
    # Interleave arms so a budget stop leaves complete pairs rather than one arm.
    with ThreadPoolExecutor(max_workers=max(1, width)) as pool:
        list(pool.map(work, todo))
    save()
    ledger.record(exp, f"eval.shard-{shard}", spent())
    return state


def collect(exp: Experiment) -> dict[str, dict[str, list]]:
    """values[task][arm] -> list of per-repeat values (None = excluded)."""
    values: dict[str, dict[str, list]] = {}
    status_counts: dict[str, int] = {}
    for path in sorted((exp.results_dir / "eval").glob("shard-*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for job in (data.get("jobs") or {}).values():
            status_counts[job.get("status", "?")] = status_counts.get(job.get("status", "?"), 0) + 1
            if job.get("status") != "ok" or job.get("value") is None:
                continue
            values.setdefault(job["task"], {}).setdefault(job["arm"], []).append(
                (job["rep"], float(job["value"]))
            )
    # keep only repeats present in both arms, so the comparison stays paired
    paired: dict[str, dict[str, list]] = {}
    for task, arms in values.items():
        reps = {r for r, _ in arms.get("base", [])} & {r for r, _ in arms.get("final", [])}
        paired[task] = {
            arm: [v for r, v in sorted(arms.get(arm, [])) if r in reps] for arm in ARMS
        }
    return {"values": paired, "status_counts": status_counts}
