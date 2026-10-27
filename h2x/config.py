"""Experiment definitions: experiments/<id>/experiment.toml plus the pre-registered task lock."""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(os.environ.get("H2X_REPO") or Path(__file__).resolve().parents[1])
EXPERIMENTS = REPO / "experiments"
RESULTS = REPO / "results"
RUN_NAME = "main"


class ExperimentError(RuntimeError):
    pass


@dataclass
class Experiment:
    id: str
    root: Path
    raw: dict
    tasks: dict = field(default_factory=dict)

    # ---- identity -------------------------------------------------------------------------
    @property
    def title(self) -> str:
        return str(self.raw.get("title", self.id))

    @property
    def suite(self) -> str:
        return str(self.raw["suite"])

    @property
    def question(self) -> str:
        return str(self.raw.get("question", ""))

    # ---- money ----------------------------------------------------------------------------
    @property
    def budget_usd(self) -> float:
        return float(self.raw["budget"]["total_usd"])

    @property
    def train_budget_usd(self) -> float:
        return float(self.raw["budget"]["train_usd"])

    @property
    def eval_budget_usd(self) -> float:
        return float(self.raw["budget"]["eval_usd"])

    @property
    def smoke_budget_usd(self) -> float:
        return float(self.raw["budget"].get("smoke_usd", 2.0))

    # ---- models ---------------------------------------------------------------------------
    def model(self, role: str) -> str:
        return str(self.raw["models"][role])

    # ---- tasks ----------------------------------------------------------------------------
    def split(self, name: str) -> list[str]:
        if name == "all":
            return list(dict.fromkeys([*self.tasks.get("train", []), *self.tasks.get("test", [])]))
        return list(self.tasks.get(name, []))

    @property
    def repeats(self) -> int:
        return int(self.raw.get("eval", {}).get("repeats", 1))

    @property
    def shards(self) -> int:
        return int(self.raw.get("eval", {}).get("shards", 1))

    @property
    def eval_concurrency(self) -> int:
        return int(self.raw.get("eval", {}).get("concurrency", 2))

    def section(self, name: str) -> dict:
        return dict(self.raw.get(name, {}))

    # ---- paths ----------------------------------------------------------------------------
    @property
    def project(self) -> Path:
        """The Harness2 project directory (holds .harness2/)."""
        return self.root

    @property
    def run_root(self) -> Path:
        return self.root / ".harness2" / "runs" / RUN_NAME

    @property
    def harness_dir(self) -> Path:
        return self.root / "harness"

    @property
    def results_dir(self) -> Path:
        return RESULTS / self.id


def load(exp_id: str | Path) -> Experiment:
    root = Path(exp_id)
    if not root.is_dir():
        root = EXPERIMENTS / str(exp_id)
    path = root / "experiment.toml"
    if not path.is_file():
        known = sorted(p.name for p in EXPERIMENTS.iterdir() if (p / "experiment.toml").is_file())
        raise ExperimentError(f"unknown experiment {exp_id!r}; known: {', '.join(known)}")
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    lock = root / "tasks.lock.json"
    tasks = json.loads(lock.read_text(encoding="utf-8")) if lock.is_file() else {}
    exp = Experiment(id=raw.get("id", root.name), root=root.resolve(), raw=raw, tasks=tasks)
    _validate(exp)
    return exp


def for_project(project: Path) -> Experiment:
    """The experiment whose Harness2 project directory is `project`."""
    return load(Path(project))


def all_ids() -> list[str]:
    return sorted(p.name for p in EXPERIMENTS.iterdir() if (p / "experiment.toml").is_file())


def _validate(exp: Experiment) -> None:
    problems = []
    for key in ("suite", "budget", "models"):
        if key not in exp.raw:
            problems.append(f"{exp.id}: experiment.toml needs [{key}]")
    budget = exp.raw.get("budget", {})
    for key in ("total_usd", "train_usd", "eval_usd"):
        if key not in budget:
            problems.append(f"{exp.id}: budget.{key} is required")
    if not problems and exp.train_budget_usd + exp.eval_budget_usd > exp.budget_usd + 1e-9:
        problems.append(f"{exp.id}: train_usd + eval_usd exceeds total_usd")
    if exp.tasks:
        overlap = set(exp.tasks.get("train", [])) & set(exp.tasks.get("test", []))
        if overlap:
            problems.append(f"{exp.id}: train and test overlap: {sorted(overlap)}")
        groups = exp.tasks.get("groups") or {}
        if groups:
            train_groups = {groups.get(t) for t in exp.tasks.get("train", [])}
            leaked = [t for t in exp.tasks.get("test", []) if groups.get(t) in train_groups]
            if leaked:
                problems.append(f"{exp.id}: test tasks share a group with train tasks: {leaked}")
    if problems:
        raise ExperimentError("\n".join(problems))
