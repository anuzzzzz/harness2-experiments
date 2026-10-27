"""Per-experiment spend ledger, committed with the results: results/<exp>/spend.json.

Each phase records its cumulative spend (idempotent re-writes, no double counting):
    smoke, train (Harness2 events: rollouts + improver/reflector sessions + lost infra costs),
    train_salvage (consolidation outside Harness2's budget gate), train_scoring (judge calls),
    eval.shard-<i> (rollouts + judge calls of one eval shard).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .config import Experiment


def path(exp: Experiment) -> Path:
    return exp.results_dir / "spend.json"


def load(exp: Experiment) -> dict:
    try:
        data = json.loads(path(exp).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def phases(exp: Experiment) -> dict[str, float]:
    return {k: float(v.get("usd", 0.0)) for k, v in (load(exp).get("phases") or {}).items()}


def total(exp: Experiment) -> float:
    return round(sum(phases(exp).values()), 6)


def spent_in(exp: Experiment, prefix: str) -> float:
    return round(sum(v for k, v in phases(exp).items() if k == prefix or k.startswith(prefix + ".")), 6)


def record(exp: Experiment, phase: str, usd: float, **extra) -> dict:
    data = load(exp)
    data.setdefault("experiment", exp.id)
    data.setdefault("phases", {})
    data["phases"][phase] = {
        "usd": round(float(usd), 6),
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "github_run": os.environ.get("GITHUB_RUN_ID"),
        **extra,
    }
    data["total_usd"] = round(sum(v["usd"] for v in data["phases"].values()), 6)
    data["budget_usd"] = exp.budget_usd
    p = path(exp)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data


def remaining(exp: Experiment) -> float:
    return round(exp.budget_usd - total(exp), 6)
