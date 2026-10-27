"""Phase 3: numbers. Per-experiment summary + the cross-experiment table in README and docs/."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from . import config, evalx, ledger, stats
from .config import Experiment

START, END = "<!-- results:start -->", "<!-- results:end -->"


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def verdict(cmp: stats.Comparison) -> str:
    if cmp.n_tasks == 0 or cmp.delta is None:
        return "not run yet"
    if cmp.ci_low is not None and cmp.ci_low > 0:
        return "improved"
    if cmp.ci_high is not None and cmp.ci_high < 0:
        return "worse"
    return "no clear difference"


def summarize(exp: Experiment) -> dict:
    train = _load(exp.results_dir / "train.json")
    collected = evalx.collect(exp)
    values = collected["values"]
    cmp = stats.compare(values, seed=7)
    out = {
        "id": exp.id,
        "title": exp.title,
        "question": exp.question,
        "metric": exp.raw.get("metric", "reward"),
        "models": dict(exp.raw.get("models", {})),
        "n_train_tasks": len(exp.split("train")),
        "n_test_tasks": len(exp.split("test")),
        "repeats": exp.repeats,
        "transfer": cmp.as_dict(),
        "verdict": verdict(cmp),
        "eval_status_counts": collected["status_counts"],
        "spend_usd": ledger.total(exp),
        "budget_usd": exp.budget_usd,
        "train": {
            "harness2_exit": train.get("harness2_exit"),
            "tasks_finished": train.get("tasks_finished"),
            "final_harness": train.get("final_harness"),
            "final_changes": train.get("final_changes"),
        },
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    in_task = train.get("in_task") or {}
    pairs = [(v.get("base"), v.get("adopted")) for v in in_task.values()
             if v.get("base") is not None and v.get("adopted") is not None]
    out["in_task"] = {
        "n_tasks": len(pairs),
        "base_mean": stats.mean(b for b, _ in pairs),
        "adopted_mean": stats.mean(a for _, a in pairs),
    }
    if exp.suite in ("h2x-tau", "h2x-harbor") and exp.repeats > 1:
        out["pass_hat_k"] = {
            arm: {k: stats.mean_pass_hat_k({t: a.get(arm, []) for t, a in values.items()}, k)
                  for k in range(1, exp.repeats + 1)}
            for arm in evalx.ARMS
        }
    return out


def pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def pts(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:+.1f}"


CHANGE_WORDS = {"A": "added", "M": "modified", "D": "deleted"}


def experiment_markdown(exp: Experiment, s: dict) -> str:
    t = s["transfer"]
    p_value = "–" if t["sign_test_p"] is None else f"{t['sign_test_p']:.3f}"
    lines = [
        f"# {exp.title}",
        "",
        f"**Question.** {s['question']}",
        "",
        f"**Verdict:** {s['verdict']}  ",
        f"Held-out tasks: {t['n_tasks']} (×{s['repeats']} repeats) · "
        f"base {pct(t['base_mean'])} → final {pct(t['final_mean'])} "
        f"(Δ {pts(t['delta'])} pts, 95% CI {pts(t['ci95'][0])} to {pts(t['ci95'][1])}; "
        f"tasks better/worse/same {t['wins']}/{t['losses']}/{t['ties']}, "
        f"sign test p={p_value})  ",
        f"In-task (paper protocol, train tasks): base {pct(s['in_task']['base_mean'])} → "
        f"adopted {pct(s['in_task']['adopted_mean'])} over {s['in_task']['n_tasks']} task(s)  ",
        f"Spend: ${s['spend_usd']:.2f} of ${s['budget_usd']:g}",
        "",
        f"Metric: {s['metric']}. Models: "
        + ", ".join(f"{k} `{v}`" for k, v in s["models"].items()),
        "",
        f"Pre-registered tasks ([`tasks.lock.json`](../../experiments/{exp.id}/tasks.lock.json)): "
        f"{s['n_train_tasks']} train ({', '.join(f'`{t}`' for t in exp.split('train'))}), "
        f"{s['n_test_tasks']} held-out test. Budget caps: train ${exp.train_budget_usd:g}, "
        f"eval ${exp.eval_budget_usd:g}.",
        "",
    ]
    if s.get("pass_hat_k"):
        lines += ["| arm | " + " | ".join(f"pass^{k}" for k in s["pass_hat_k"]["base"]) + " |",
                  "|---|" + "---|" * len(s["pass_hat_k"]["base"])]
        for arm, row in s["pass_hat_k"].items():
            lines.append(f"| {arm} | " + " | ".join(pct(v) for v in row.values()) + " |")
        lines.append("")
    changes = s["train"].get("final_changes") or []
    if changes:
        lines += ["## What Harness² changed", "",
                  "The final harness is in [`final_harness/`](final_harness/) "
                  "([diff](final_harness.diff), [improver notes](final_harness_notes.md)).", ""]
        lines += [f"- `{c['path']}` ({CHANGE_WORDS.get(c['change'], c['change'])})"
                  for c in changes]
        lines.append("")
    lines += ["## Per task", "", "| task | base | final |", "|---|---|---|"]
    for task, row in t["per_task"].items():
        lines.append(f"| `{task}` | {pct(row['base'])} | {pct(row['final'])} |")
    counts = s.get("eval_status_counts") or {}
    if counts:
        lines += ["", "Rollout status counts: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())),
                  "Rollouts with status other than `ok` (infrastructure failures, budget stops) are "
                  "excluded from the comparison, never counted as zeros."]
    return "\n".join(lines) + "\n"


def sync_eval_ledger(exp: Experiment) -> None:
    """Eval shards run on separate machines; their spend lives in their shard files."""
    for path in sorted((exp.results_dir / "eval").glob("shard-*.json")):
        data = _load(path)
        if not data:
            continue
        phase = f"eval.shard-{data.get('shard', path.stem.split('-')[-1])}"
        spent = float(data.get("spent_usd") or 0.0)
        if abs(ledger.phases(exp).get(phase, -1.0) - spent) > 1e-9:
            ledger.record(exp, phase, spent)


def write(exp: Experiment) -> dict:
    sync_eval_ledger(exp)
    s = summarize(exp)
    exp.results_dir.mkdir(parents=True, exist_ok=True)
    (exp.results_dir / "summary.json").write_text(json.dumps(s, indent=2, default=str) + "\n")
    (exp.results_dir / "README.md").write_text(experiment_markdown(exp, s), encoding="utf-8")
    return s


def table(rows: list[dict]) -> str:
    out = [
        "| # | Experiment | Held-out tasks | Base | Harness² | Δ (95% CI) | Verdict | Spend |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, s in enumerate(rows, 1):
        t = s["transfer"]
        ci = f"{pts(t['delta'])} ({pts(t['ci95'][0])}, {pts(t['ci95'][1])})" if t["delta"] is not None else "–"
        out.append(
            f"| {i} | [{s['title']}](results/{s['id']}/README.md) | "
            f"{t['n_tasks'] or '–'}×{s['repeats']} | {pct(t['base_mean'])} | {pct(t['final_mean'])} | "
            f"{ci} | {s['verdict']} | ${s['spend_usd']:.2f} / ${s['budget_usd']:g} |"
        )
    return "\n".join(out)


def update_global(rows: list[dict]) -> None:
    readme = config.REPO / "README.md"
    text = readme.read_text(encoding="utf-8")
    if START in text and END in text:
        body = f"{START}\n{table(rows)}\n\n_Last updated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC._\n{END}"
        text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: body, text, flags=re.S)
        readme.write_text(text, encoding="utf-8")
    data = config.REPO / "docs" / "data" / "summary.json"
    data.parent.mkdir(parents=True, exist_ok=True)
    data.write_text(json.dumps({"experiments": rows,
                                "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                               indent=2, default=str) + "\n")


def write_all(ids: list[str] | None = None) -> list[dict]:
    """Rewrite the pages of `ids` (None = every experiment); the global table always reflects
    every experiment's committed files (spend, eval shards), never a stale summary.json."""
    rows = []
    for exp_id in config.all_ids():
        exp = config.load(exp_id)
        rows.append(write(exp) if (ids is None or exp_id in ids) else summarize(exp))
    update_global(rows)
    return rows
