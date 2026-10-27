"""Phase 0: one real base-harness rollout of the first train task, graded, priced. No improver.

Confirms, for a few cents to a dollar, that model access, the benchmark runner, grading and
cost accounting all work before anything bigger is spent. Writes results/<exp>/smoke.json.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from . import h2, ledger
from .config import Experiment


def smoke(exp: Experiment) -> dict:
    from harness2.core import runmark
    from harness2.core.library import slug

    adapter = h2.adapter(exp)
    task = exp.split("train")[0]
    run_id = f"h2x-smoke/{task}"
    cached = runmark.is_complete(adapter.spec.runs_root / "rollouts" / slug(run_id), harness_dir=None)
    started = time.monotonic()
    out = {"experiment": exp.id, "task": task, "cached": cached,
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    try:
        run_dir = adapter.execute(task, None, run_id)
    except Exception as exc:
        out.update(ok=False, error=f"{type(exc).__name__}: {exc}"[:3000])
        failed = adapter.spec.runs_root / "rollouts" / slug(run_id) / "meta.json"
        try:
            cost = float(json.loads(failed.read_text()).get("cost_usd") or 0.0)
        except (OSError, ValueError):
            cost = 0.0
    else:
        cost = 0.0 if cached else float(adapter.rollout_cost(run_dir))
        had_grade = (run_dir / "judge.json").is_file()
        try:
            score = adapter.score(run_dir, task)
        except Exception as exc:
            score = {"error": f"{type(exc).__name__}: {exc}"}
        judge = getattr(adapter, "judge_cost", None)
        if callable(judge) and not had_grade:
            cost += float(judge(run_dir))
        out.update(
            ok=bool(score) and "error" not in (score or {}),
            delivered=adapter.delivered(run_dir),
            score={k: v for k, v in (score or {}).items() if k != "output_tail"},
            meta=adapter.read_meta(run_dir) if hasattr(adapter, "read_meta") else None,
        )
    out["cost_usd"] = round(cost, 6)
    out["seconds"] = round(time.monotonic() - started, 1)
    exp.results_dir.mkdir(parents=True, exist_ok=True)
    (exp.results_dir / "smoke.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    ledger.record(exp, "smoke", ledger.phases(exp).get("smoke", 0.0) + cost)
    return out
