"""Paired comparisons for base-vs-final harness results (pure Python, no numpy needed).

Unit of analysis: a task. Each arm's per-task value is the mean over its repeats; the effect is
the mean over tasks of (final - base), with a percentile bootstrap CI that resamples tasks, and
an exact two-sided sign test on per-task differences. For binary rewards with one repeat the
sign test is McNemar's exact test.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class Comparison:
    n_tasks: int
    base_mean: float | None
    final_mean: float | None
    delta: float | None
    ci_low: float | None
    ci_high: float | None
    wins: int
    losses: int
    ties: int
    sign_p: float | None
    per_task: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n_tasks": self.n_tasks,
            "base_mean": self.base_mean,
            "final_mean": self.final_mean,
            "delta": self.delta,
            "ci95": [self.ci_low, self.ci_high],
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "sign_test_p": self.sign_p,
            "per_task": self.per_task,
        }


def mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def sign_test(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
    return min(1.0, 2 * tail)


def bootstrap_ci(diffs: list[float], *, iters: int = 10_000, seed: int = 0, alpha: float = 0.05):
    if not diffs:
        return None, None
    rng = random.Random(seed)
    n = len(diffs)
    stats = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    lo = stats[int((alpha / 2) * iters)]
    hi = stats[min(iters - 1, int((1 - alpha / 2) * iters))]
    return lo, hi


def compare(values: dict[str, dict[str, list[float]]], *, seed: int = 0) -> Comparison:
    """values[task][arm] = list of per-repeat scores in [0,1]; arms 'base' and 'final'.

    Only tasks with at least one valid value in BOTH arms enter the comparison.
    """
    per_task = {}
    diffs = []
    for task, arms in sorted(values.items()):
        b = mean(arms.get("base", []))
        f = mean(arms.get("final", []))
        per_task[task] = {"base": b, "final": f, "n_base": len(arms.get("base", [])),
                          "n_final": len(arms.get("final", []))}
        if b is not None and f is not None:
            diffs.append(f - b)
    paired = [t for t, v in per_task.items() if v["base"] is not None and v["final"] is not None]
    wins = sum(1 for t in paired if per_task[t]["final"] > per_task[t]["base"] + 1e-12)
    losses = sum(1 for t in paired if per_task[t]["final"] < per_task[t]["base"] - 1e-12)
    lo, hi = bootstrap_ci(diffs, seed=seed)
    return Comparison(
        n_tasks=len(paired),
        base_mean=mean(per_task[t]["base"] for t in paired),
        final_mean=mean(per_task[t]["final"] for t in paired),
        delta=mean(diffs),
        ci_low=lo,
        ci_high=hi,
        wins=wins,
        losses=losses,
        ties=len(paired) - wins - losses,
        sign_p=sign_test(wins, losses),
        per_task=per_task,
    )


def pass_hat_k(successes: int, trials: int, k: int) -> float | None:
    """tau-bench pass^k for one task: C(c,k)/C(n,k)."""
    if trials < k or k <= 0:
        return None
    return math.comb(successes, k) / math.comb(trials, k)


def mean_pass_hat_k(values: dict[str, list[float]], k: int) -> float | None:
    """Mean pass^k over tasks; values[task] = per-trial binary rewards."""
    per = []
    for rewards in values.values():
        rewards = [r for r in rewards if r is not None]
        p = pass_hat_k(sum(1 for r in rewards if r >= 0.999), len(rewards), k)
        if p is not None:
            per.append(p)
    return sum(per) / len(per) if per else None
