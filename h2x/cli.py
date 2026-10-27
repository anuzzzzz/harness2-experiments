"""h2x command line (used by .github/workflows/experiment.yml; works locally too).

    h2x list                         experiments, budgets, spend so far
    h2x plan  <exp> [--json]         tasks, model calls and an estimate of what each phase costs
    h2x setup <exp>                  fetch / import the pre-registered tasks (no model calls)
    h2x budget-check <exp> <phase>   exit 3 if the phase's cap no longer fits the experiment budget
    h2x smoke <exp>                  one real base rollout, graded and priced
    h2x train <exp>                  Harness2 on the train split -> final harness
    h2x eval  <exp> --shard I        base vs final harness on this shard of the test split
    h2x report [<exp> ...]           summaries, README table, docs/data/summary.json
    h2x matrix <exp>                 the eval shard list as JSON (for the workflow matrix)
    h2x lab-import <exp>             LAB tasks -> .harness2/tasks
    h2x lab-grade ...                grade one LAB deliverable set (Harness2 check command)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import config, ledger


def estimate(exp: config.Experiment) -> dict:
    """Expected spend from per-unit guesses in [estimate]; the caps in [budget] are what bind.

    Harness2 at k=1 makes, per train task: 4 rollouts (base, INV, DOM, composed), 3 improver
    sessions (propose INV, propose DOM, compose) and 1 reflector session; plus one
    consolidation session per run.
    """
    est = exp.section("estimate")
    rollout = float(est.get("rollout_usd", 0.1))
    session = float(est.get("improver_session_usd", 0.75))
    reflector = float(est.get("reflector_session_usd", 0.25))
    consolidation = float(est.get("consolidation_usd", 1.0))
    judge = float(est.get("judge_usd", 0.0))
    n_train, n_test = len(exp.split("train")), len(exp.split("test"))
    train_rollouts = n_train * 4
    train_sessions = n_train * 4 + 1
    train_usd = (train_rollouts * rollout + n_train * (3 * session + reflector) + consolidation
                 + 2 * n_train * judge)
    eval_rollouts = n_test * exp.repeats * 2
    return {
        "train": {"tasks": n_train, "rollouts": train_rollouts, "sessions": train_sessions,
                  "usd": round(train_usd, 2)},
        "eval": {"tasks": n_test, "repeats": exp.repeats, "rollouts": eval_rollouts,
                 "usd": round(eval_rollouts * (rollout + judge), 2)},
        "smoke": {"usd": round(rollout + judge, 2)},
        "assumptions": {"rollout_usd": rollout, "improver_session_usd": session,
                        "reflector_session_usd": reflector, "consolidation_usd": consolidation,
                        "judge_usd": judge},
    }


def cmd_list(args) -> int:
    for exp_id in config.all_ids():
        exp = config.load(exp_id)
        print(f"{exp.id:24s} ${ledger.total(exp):7.2f} / ${exp.budget_usd:5.0f}  {exp.title}")
    return 0


def cmd_plan(args) -> int:
    exp = config.load(args.exp)
    data = {
        "experiment": exp.id,
        "title": exp.title,
        "suite": exp.suite,
        "models": exp.raw.get("models"),
        "train_tasks": exp.split("train"),
        "test_tasks": exp.split("test"),
        "repeats": exp.repeats,
        "shards": exp.shards,
        "budget": {"total": exp.budget_usd, "train_cap": exp.train_budget_usd,
                   "eval_cap": exp.eval_budget_usd, "spent": ledger.total(exp),
                   "remaining": ledger.remaining(exp)},
        "estimate": estimate(exp),
    }
    if args.json:
        print(json.dumps(data, indent=2))
        return 0
    e = data["estimate"]
    print(f"{exp.id}: {exp.title}")
    print(f"  train: {len(data['train_tasks'])} tasks -> {e['train']['rollouts']} rollouts + "
          f"{e['train']['sessions']} improver sessions  (~${e['train']['usd']}, cap ${exp.train_budget_usd})")
    print(f"  eval:  {len(data['test_tasks'])} tasks x {exp.repeats} repeats x 2 harnesses = "
          f"{e['eval']['rollouts']} rollouts  (~${e['eval']['usd']}, cap ${exp.eval_budget_usd})")
    print(f"  spent ${data['budget']['spent']:.2f} of ${exp.budget_usd:.0f}")
    return 0


def cmd_setup(args) -> int:
    exp = config.load(args.exp)
    if exp.suite == "h2x-harbor":
        from .fetch import fetch_harbor_tasks

        got = fetch_harbor_tasks(exp, force=args.force)
        print(f"fetched {len(got)} task directories")
    elif exp.suite == "h2x-lab":
        from .lab.importer import import_tasks

        root = Path(os.environ.get("H2X_LAB_ROOT", config.REPO / "vendor" / "harvey-labs"))
        got = import_tasks(exp, root, force=args.force)
        print(f"imported {len(got)} LAB tasks")
    else:
        print("nothing to set up")
    return 0


def cmd_budget_check(args) -> int:
    exp = config.load(args.exp)
    remaining = ledger.remaining(exp)
    need = {
        "smoke": min(exp.smoke_budget_usd, 0.5),
        "train": 2.0,
        "eval": 1.0,
    }.get(args.phase, 0.0)
    ok = remaining >= need
    print(json.dumps({"experiment": exp.id, "phase": args.phase, "remaining_usd": remaining,
                      "needs_at_least_usd": need, "ok": ok}))
    return 0 if ok else 3


def cmd_smoke(args) -> int:
    from .smoke import smoke

    out = smoke(config.load(args.exp))
    print(json.dumps(out, indent=2, default=str))
    return 0 if out.get("ok") else 1


def cmd_train(args) -> int:
    from .train import train

    out = train(config.load(args.exp), soft_min=args.soft_min, hard_min=args.hard_min)
    print(json.dumps({k: v for k, v in out.items() if k != "in_task"}, indent=2, default=str))
    return 0 if out.get("final_harness") else 2


def cmd_eval(args) -> int:
    from .evalx import run_shard

    exp = config.load(args.exp)
    state = run_shard(exp, args.shard, budget_usd=args.budget_usd, concurrency=args.concurrency)
    counts: dict[str, int] = {}
    for job in state["jobs"].values():
        counts[job.get("status", "?")] = counts.get(job.get("status", "?"), 0) + 1
    print(json.dumps({"experiment": exp.id, "shard": args.shard, "status": counts,
                      "spent_usd": state.get("spent_usd")}, indent=2))
    return 0


def cmd_report(args) -> int:
    from .report import write_all

    rows = write_all([] if args.global_only else (args.exp or None))
    for s in rows:
        t = s["transfer"]
        print(f"{s['id']:24s} {s['verdict']:22s} n={t['n_tasks']} delta={t['delta']}")
    return 0


def cmd_matrix(args) -> int:
    exp = config.load(args.exp)
    print(json.dumps(list(range(exp.shards))))
    return 0


def cmd_lab_import(args) -> int:
    from .lab.importer import import_tasks

    exp = config.load(args.exp)
    got = import_tasks(exp, Path(args.lab_root), force=args.force)
    print(f"imported {len(got)} tasks into {exp.project / '.harness2' / 'tasks'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["lab-grade"]:
        from .lab.grade import main as grade

        return grade(argv[1:])
    p = argparse.ArgumentParser(prog="h2x", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    s = sub.add_parser("plan"); s.add_argument("exp"); s.add_argument("--json", action="store_true"); s.set_defaults(fn=cmd_plan)
    s = sub.add_parser("setup"); s.add_argument("exp"); s.add_argument("--force", action="store_true"); s.set_defaults(fn=cmd_setup)
    s = sub.add_parser("budget-check"); s.add_argument("exp"); s.add_argument("phase"); s.set_defaults(fn=cmd_budget_check)
    s = sub.add_parser("smoke"); s.add_argument("exp"); s.set_defaults(fn=cmd_smoke)
    s = sub.add_parser("train"); s.add_argument("exp")
    s.add_argument("--soft-min", type=float, default=300); s.add_argument("--hard-min", type=float, default=345)
    s.set_defaults(fn=cmd_train)
    s = sub.add_parser("eval"); s.add_argument("exp"); s.add_argument("--shard", type=int, required=True)
    s.add_argument("--budget-usd", type=float, default=None); s.add_argument("--concurrency", type=int, default=None)
    s.set_defaults(fn=cmd_eval)
    s = sub.add_parser("report"); s.add_argument("exp", nargs="*")
    s.add_argument("--global-only", action="store_true",
                   help="only rebuild the README table and docs/data from existing summaries")
    s.set_defaults(fn=cmd_report)
    s = sub.add_parser("matrix"); s.add_argument("exp"); s.set_defaults(fn=cmd_matrix)
    s = sub.add_parser("lab-import"); s.add_argument("exp")
    s.add_argument("--lab-root", default=os.environ.get("H2X_LAB_ROOT", "vendor/harvey-labs"))
    s.add_argument("--force", action="store_true"); s.set_defaults(fn=cmd_lab_import)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
