"""Thin access to Harness2 internals (pinned fork 0c81134): settings, adapter, run records."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from .config import RUN_NAME, Experiment


def settings(exp: Experiment):
    from harness2 import settings as S

    return S.load(exp.project)


def adapter(exp: Experiment):
    from harness2 import registry
    from harness2.registry import SuiteContext

    cfg = settings(exp)
    suite = registry.load(cfg.get("suite.name"))
    return suite.make_adapter(SuiteContext(cfg, exp.project, exp.run_root, None))


def context(exp: Experiment):
    """A harness2 CLI Context, for reusing its run helpers (consolidate, diff)."""
    from harness2.cli._common import Context

    args = Namespace(
        project=str(exp.project), profile=None, config=[], quiet=True, verbose=False,
        json=False, argv=[],
    )
    return Context(args, {})


def final_harness(exp: Experiment, mode: str = "parallel") -> Path:
    return exp.run_root / "consolidated" / mode / "harness"


def final_harness_for_eval(exp: Experiment, adapter_obj) -> Path | None:
    """The consolidated harness from the run state, or -- when the state cache is gone -- the
    committed copy in results/<exp>/final_harness/ re-imported into Harness2's tree format."""
    path = final_harness(exp)
    if path.is_dir():
        return path
    native = exp.results_dir / "final_harness"
    if not native.is_dir():
        return None
    from harness2.core import tree as HT

    dest = exp.run_root / "imported_final" / "harness"
    if not dest.is_dir():
        tree = adapter_obj.substrate.import_tree(native)
        HT.write_tree(adapter_obj.substrate.surface(), tree, dest, stubs=False)
    return dest


def event_totals(exp: Experiment) -> dict:
    from harness2.cli._common import event_totals as totals

    return totals(exp.run_root) if exp.run_root.is_dir() else {"spent_usd": 0.0, "calls": 0}


def summaries(exp: Experiment, adapter_obj) -> dict[str, dict]:
    """task_id -> summary.json row, for this run's (k, parallel) namespace."""
    from harness2.cli import _common as C

    if not (exp.run_root / "run.json").is_file():
        return {}
    record = C.load_run(exp.run_root)
    plan = record["schedule"]
    ns = C.summary_namespace(adapter_obj, plan["k"], seed=plan.get("seed"))
    return {
        row["task_id"]: row
        for path, row in C.summaries(exp.run_root)
        if path.parent.parent.name == ns
    }


def resolve(adapter_obj, rel: str) -> Path:
    from harness2.core import runmark

    return runmark.resolve_path(rel, adapter_obj.spec.runs_root)


def tree_diff(adapter_obj, before_dir: Path, after_dir: Path) -> tuple[list, str]:
    from harness2.cli import _adopt as A
    from harness2.core.tree import read_tree

    surface = adapter_obj.substrate.surface()
    before, after = read_tree(surface, before_dir), read_tree(surface, after_dir)
    changes = A.changes(before, after)
    return changes, "".join(A.unified(rel, before, after) for rel, _ in changes)


def export_native(adapter_obj, tree_dir: Path, dest: Path) -> None:
    from harness2.core.tree import read_tree

    dest.mkdir(parents=True, exist_ok=True)
    tree = read_tree(adapter_obj.substrate.surface(), tree_dir)
    adapter_obj.substrate.export_tree(tree, dest, base={})


RUN = RUN_NAME
