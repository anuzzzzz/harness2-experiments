"""Phase 1: run Harness2 on the train split and keep what it produced.

1. ``harness2 run --split train --budget-usd <cap>`` (k=1, parallel mode, resumable). A watchdog
   interrupts it before the CI job's 6-hour limit; rerunning resumes from the saved state.
2. If the run stopped early but some tasks finished, the final harness is still consolidated
   from those tasks (one improver session), budget permitting.
3. The final harness, its diff from the base harness, the improver's notes, each task's adopted
   diff, and the in-task comparison on the train tasks (base rollout vs the adopted rollout --
   the paper's protocol) are written to results/<exp>/.
"""

from __future__ import annotations

import json
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from . import h2, ledger
from .config import RUN_NAME, Experiment


def _watchdog(proc: subprocess.Popen, soft_s: float, hard_s: float) -> None:
    start = time.monotonic()
    sent = False
    while proc.poll() is None:
        elapsed = time.monotonic() - start
        if not sent and elapsed > soft_s:
            print(f"[h2x] deadline {soft_s / 60:.0f} min reached: asking Harness2 to stop "
                  "(in-flight calls finish, state is kept)", file=sys.stderr, flush=True)
            proc.send_signal(signal.SIGINT)
            sent = True
        if elapsed > hard_s:
            print("[h2x] hard deadline: killing Harness2", file=sys.stderr, flush=True)
            proc.kill()
            return
        time.sleep(5)


def run_harness2(exp: Experiment, cap_usd: float, *, soft_min: float, hard_min: float) -> int:
    cmd = [
        sys.executable, "-m", "harness2", "-C", str(exp.project), "--json",
        "run", "--name", RUN_NAME, "--split", "train", "--budget-usd", f"{cap_usd:.2f}",
    ]
    print("[h2x] " + " ".join(cmd), file=sys.stderr, flush=True)
    exp.run_root.mkdir(parents=True, exist_ok=True)
    with (exp.run_root.parent / f"{RUN_NAME}.stdout.json").open("w") as out:
        proc = subprocess.Popen(cmd, stdout=out, stderr=None, stdin=subprocess.DEVNULL)
        guard = threading.Thread(
            target=_watchdog, args=(proc, soft_min * 60, hard_min * 60), daemon=True
        )
        guard.start()
        return proc.wait()


def salvage_consolidation(exp: Experiment, adapter) -> float:
    """Build the final harness from the tasks that finished; returns its cost."""
    from harness2.cli._final import consolidate

    ctx = h2.context(exp)
    result = consolidate(ctx, adapter, exp.run_root, "parallel")
    info = json.loads((Path(result["path"]).parent / "consolidation.json").read_text())
    return float(info.get("cost_usd") or 0.0)


def _value(score: dict | None) -> float | None:
    if not score or not score.get("n_criteria"):
        return None
    if "reward" in score:
        return float(score["reward"])
    return score["n_passed"] / score["n_criteria"]


def in_task_scores(exp: Experiment, adapter, rows: dict[str, dict]) -> tuple[dict, float]:
    """Base rollout vs adopted rollout on each train task (the paper's per-task protocol)."""
    out, judge = {}, 0.0
    judge_cost = getattr(adapter, "judge_cost", None)
    for task, row in sorted(rows.items()):
        sel = (row.get("selections") or {}).get("agg") or {}
        base = (row.get("base") or [{}])[0].get("run_dir")
        chosen = sel.get("chosen_run_dir")
        entry = {"status": row.get("status"), "adopted_is_base": sel.get("chosen_tree_is_base")}
        for arm, rel in (("base", base), ("adopted", chosen)):
            if not rel:
                entry[arm] = None
                continue
            run_dir = h2.resolve(adapter, rel)
            had_grade = (run_dir / "judge.json").is_file()
            try:
                score = adapter.score(run_dir, task)
            except Exception as exc:  # grading problems are reported, never scored as zero
                score = {"error": f"{type(exc).__name__}: {exc}"}
            if callable(judge_cost) and not had_grade:
                judge += judge_cost(run_dir)
            entry[arm] = _value(score if isinstance(score, dict) else None)
            entry[f"{arm}_score"] = score
        out[task] = entry
    return out, judge


def write_outputs(exp: Experiment, adapter) -> dict:
    res = exp.results_dir
    res.mkdir(parents=True, exist_ok=True)
    final = h2.final_harness(exp)
    base = adapter.base_harness()
    info: dict = {"final_harness": final.is_dir()}
    if final.is_dir():
        native = res / "final_harness"
        shutil.rmtree(native, ignore_errors=True)
        h2.export_native(adapter, final, native)
        changes, diff = h2.tree_diff(adapter, base, final)
        (res / "final_harness.diff").write_text(diff or "(no changes)\n", encoding="utf-8")
        info["final_changes"] = [{"path": p, "change": c} for p, c in changes]
        notes = final.parent / "notes.md"
        if notes.is_file():
            shutil.copyfile(notes, res / "final_harness_notes.md")
    adopted_dir = res / "adopted"
    shutil.rmtree(adopted_dir, ignore_errors=True)
    rows = h2.summaries(exp, adapter)
    for task, row in rows.items():
        sel = (row.get("selections") or {}).get("agg") or {}
        label = sel.get("chosen_label")
        if not label or sel.get("chosen_tree_is_base"):
            continue
        harness = next(exp.run_root.glob(f"adapt/*-agg*/{task.replace('/', '__')}/harnesses/{label}"), None)
        if harness and harness.is_dir():
            _, diff = h2.tree_diff(adapter, base, harness)
            adopted_dir.mkdir(exist_ok=True)
            (adopted_dir / f"{task.replace('/', '__')}.diff").write_text(diff, encoding="utf-8")
    return info


def train(exp: Experiment, *, soft_min: float = 300, hard_min: float = 345) -> dict:
    other = ledger.total(exp) - ledger.spent_in(exp, "train")
    cap = round(min(exp.train_budget_usd, exp.budget_usd - other), 2)
    if cap <= 0.5:
        raise SystemExit(f"{exp.id}: no budget left for training (cap ${cap:.2f})")
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    code = run_harness2(exp, cap, soft_min=soft_min, hard_min=hard_min)
    adapter = h2.adapter(exp)
    rows = h2.summaries(exp, adapter)
    finished = [t for t, r in rows.items() if ((r.get("selections") or {}).get("agg") or {}).get("ok")]
    salvage = float(ledger.phases(exp).get("train_salvage", 0.0))
    totals = h2.event_totals(exp)
    if not h2.final_harness(exp).is_dir() and finished and code in (3, 4, 130):
        left = exp.budget_usd - (other + totals["spent_usd"] + salvage)
        if left >= 1.5:
            print(f"[h2x] consolidating from {len(finished)} finished task(s)", file=sys.stderr)
            try:
                salvage += salvage_consolidation(exp, adapter)
            except Exception as exc:
                print(f"[h2x] consolidation failed: {exc}", file=sys.stderr)
            ledger.record(exp, "train_salvage", salvage)
    totals = h2.event_totals(exp)
    ledger.record(exp, "train", totals["spent_usd"], calls=totals["calls"], harness2_exit=code)
    info = write_outputs(exp, adapter)
    scores, judge = in_task_scores(exp, adapter, rows)
    if judge:
        ledger.record(exp, "train_scoring", ledger.phases(exp).get("train_scoring", 0.0) + judge)
    summary = {
        "experiment": exp.id,
        "harness2_exit": code,
        "started": started,
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_tasks": exp.split("train"),
        "tasks_finished": finished,
        "spend_usd": {"train": totals["spent_usd"], "salvage": salvage, "scoring": judge},
        "model_calls": totals["calls"],
        "in_task": scores,
        **info,
    }
    (exp.results_dir / "train.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary
