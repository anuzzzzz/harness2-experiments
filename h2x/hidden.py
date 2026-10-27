"""Grades live outside every rollout directory, so no evidence path can ever serve one.

Harness2 builds the improver's evidence only from what an adapter's ``trajectory`` and
``deliverable_*`` methods return. h2x adapters read fixed files inside the rollout directory,
and runners write grades to ``<runs_root>/.h2x_hidden/<rollout>.json`` instead, a sibling tree
that no adapter method reads. The improver, selector and reflector therefore never see a reward.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

HIDDEN = ".h2x_hidden"


def hidden_root(runs_root: Path) -> Path:
    return Path(runs_root) / HIDDEN


def grade_path(runs_root: Path, run_dir: Path) -> Path:
    return hidden_root(runs_root) / f"{Path(run_dir).name}.json"


def scratch_dir(runs_root: Path, run_dir: Path) -> Path:
    """Private working space for a rollout (e.g. Harbor job dirs holding verifier output)."""
    return hidden_root(runs_root) / "scratch" / Path(run_dir).name


def write(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def read(path: Path) -> dict | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def clear(path: Path) -> None:
    Path(path).unlink(missing_ok=True)
