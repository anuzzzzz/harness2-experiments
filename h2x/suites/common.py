"""The adapter skeleton shared by suites whose rollouts run in an external benchmark runner.

A rollout is one subprocess (``python -m h2x.<runner> run ...``) that executes the agent with a
staged harness and writes, into the rollout directory, only what the improver may see
(trajectory, deliverable, meta.json), and into ``.h2x_hidden/`` the grade it must never see.

Runner exit codes:
    0   the rollout finished (the agent may have done well or badly; a grade was written)
    75  infrastructure failure after retries (API outage, container failure) -- not evidence
    76  the provider refused for quota/credit/rate reasons -- the whole run stops, resumable
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from harness2 import providers
from harness2.core import events, runmark
from harness2.core import tree as HT
from harness2.core.library import slug
from harness2.core.spec import BenchPrompts, BenchSpec, ChannelSpec
from harness2.registry import SuiteContext
from harness2.settings import ConfigError, Settings

from .. import config as h2x_config
from .. import hidden

EXIT_INFRA = 75
EXIT_STOP = 76


def fake_mode() -> bool:
    """Offline mode for CI plumbing tests: runners fabricate rollouts, no model is called."""
    return os.environ.get("H2X_FAKE", "").strip() not in ("", "0", "false")


def channel(settings: Settings, role: str) -> ChannelSpec:
    values = settings.role(role)
    provider = None
    if values["channel"] != "human":
        provider_name, _ = providers.split_model(values["model"])
        provider = settings.provider(provider_name)
    try:
        return ChannelSpec(
            kind=values["channel"],
            model=values["model"],
            bin=values["bin"],
            effort=values["effort"],
            timeout=values["timeout"],
            steps=values["steps"],
            provider=provider,
        )
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def seed_from_run_id(run_id: str, base: int = 300) -> int:
    """Eval rollouts end in /r<k>; repeat k gets seed base+k so arms share seeds per repeat."""
    match = re.search(r"/r(\d+)$", run_id)
    return base + (int(match.group(1)) if match else 0)


class RunnerAdapter:
    """BenchmarkAdapter for rollouts executed by an h2x runner subprocess."""

    suite_name = "h2x"
    install_mode = "command_files"
    domain_word = "area"
    leak_globs: tuple[str, ...] = ()
    allow_subagents = False
    runner_module = ""

    def __init__(self, ctx: SuiteContext, *, substrate, prompts: BenchPrompts):
        self.ctx = ctx
        self.settings = ctx.settings
        self.exp = h2x_config.for_project(ctx.project)
        self.substrate = substrate
        self.spec = BenchSpec(
            name=self.suite_name,
            runs_root=ctx.run_root,
            install_mode=self.install_mode,
            prompts=prompts,
            channels={
                "proposer": channel(self.settings, "improver"),
                "selector": channel(self.settings, "selector"),
                "reflector": channel(self.settings, "reflector"),
            },
            domain_word=self.domain_word,
            leak_globs=self.leak_globs,
            allow_subagents=self.allow_subagents,
            obs_trunc=self.settings.get("run.obs_trunc"),
        )
        self._base_lock = threading.Lock()

    # ---- tasks ---------------------------------------------------------------------------
    def tasks(self, split: str = "all") -> list[str]:
        if split not in ("all", "train", "test"):
            raise ConfigError(f"unknown task split {split!r}; use all, train or test")
        return self.exp.split(split)

    def domain_of(self, task_id: str) -> str:
        return str(self.exp.raw.get("domain", "default"))

    def task_prompt(self, task_id: str) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def task_files(self, task_id: str) -> Path | None:
        return None

    def reference_domain_order(self) -> list[str]:
        return [str(self.exp.raw.get("domain", "default"))]

    # ---- harness -------------------------------------------------------------------------
    def base_harness(self) -> Path:
        dest = self.spec.runs_root / "base_harness"
        surface = self.substrate.surface()
        with self._base_lock:
            if not dest.exists():
                current = self.substrate.import_tree(self.exp.harness_dir)
                HT.write_tree(surface, current, dest, stubs=True)
        return dest

    def stage(self, harness_dir: Path | None) -> Path:
        """Export the canonical tree as the files the agent will actually read."""
        source = harness_dir if harness_dir is not None else self.base_harness()
        tree = HT.read_tree(self.substrate.surface(), source)
        staged = Path(tempfile.mkdtemp(prefix="h2x-harness-"))
        try:
            self.substrate.export_tree(tree, staged, base={})
        except BaseException:
            shutil.rmtree(staged, ignore_errors=True)
            raise
        return staged

    def validate_tree(self, tree: dict[str, str]) -> None:
        pass

    # ---- execution -----------------------------------------------------------------------
    def runner_command(
        self, task_id: str, staged: Path, run_dir: Path, grade: Path, *, run_id: str
    ) -> list[str]:  # pragma: no cover - abstract
        raise NotImplementedError

    def runner_timeout(self, task_id: str) -> int:
        return int(self.settings.get("executor.timeout"))

    def after_rollout(self, task_id: str, run_dir: Path) -> None:
        """Hook for suites that publish agent-visible source material from a rollout."""

    def execute(
        self, task_id: str, harness_dir: Path | None, run_id: str, *, timeout: int | None = None
    ) -> Path:
        run_name = slug(run_id)
        if not run_name or run_name in (".", "..") or "\\" in run_name:
            raise ConfigError(f"invalid rollout id {run_id!r}")
        run_dir = self.spec.runs_root / "rollouts" / run_name
        if runmark.is_complete(run_dir, harness_dir=harness_dir):
            return run_dir
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        grade = hidden.grade_path(self.spec.runs_root, run_dir)
        hidden.clear(grade)
        # The runner's console output (Harbor prints result tables) stays out of the run dir.
        log_path = hidden.scratch_dir(self.spec.runs_root, run_dir) / "runner.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        staged = self.stage(harness_dir)
        wall = timeout or self.runner_timeout(task_id)
        started = time.monotonic()
        timed_out = False
        try:
            argv = self.runner_command(task_id, staged, run_dir, grade, run_id=run_id)
            with log_path.open("wb") as log:
                code, timed_out = _run_group(argv, log, wall, self.runner_env())
        finally:
            shutil.rmtree(staged, ignore_errors=True)
        meta = self.read_meta(run_dir)
        meta.update(
            task_id=task_id,
            run_id=run_id,
            exit_code=code,
            runner_timed_out=timed_out,
            wall_s=round(time.monotonic() - started, 3),
        )
        meta.setdefault("cost_usd", 0.0)
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if code == EXIT_STOP:
            self._record_lost_cost(task_id, run_id, meta)
            shutil.rmtree(run_dir, ignore_errors=True)
            events.request_stop("usage_limit")
            raise events.Stopped("usage_limit")
        if code != 0:
            self._record_lost_cost(task_id, run_id, meta)
            tail = _tail(log_path)
            # Not marked complete: a resumed run retries it, and it is never shown as evidence.
            raise RuntimeError(
                f"rollout {run_id} failed outside the agent (exit {code}"
                f"{', runner timeout' if timed_out else ''}): {tail}"
            )
        self.after_rollout(task_id, run_dir)
        runmark.mark_complete(run_dir, run_id=run_id, harness_dir=harness_dir)
        return run_dir

    def runner_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        return env

    def _record_lost_cost(self, task_id: str, run_id: str, meta: dict) -> None:
        cost = float(meta.get("cost_usd") or 0.0)
        if cost:
            # Counted against --budget-usd without counting as a model call.
            events.emit("h2x_infra", task=task_id, run_id=run_id, cost_usd=cost)
        path = hidden.hidden_root(self.spec.runs_root) / "infra_failures.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps({"task": task_id, "run_id": run_id, "cost_usd": cost,
                            "error": str(meta.get("error", ""))[:500], "t": time.time()})
                + "\n"
            )

    @staticmethod
    def read_meta(run_dir: Path) -> dict:
        try:
            data = json.loads((Path(run_dir) / "meta.json").read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def rollout_cost(self, run_dir: Path) -> float:
        return float(self.read_meta(run_dir).get("cost_usd") or 0.0)

    # ---- evidence ------------------------------------------------------------------------
    def delivered(self, run_dir: Path) -> bool:  # pragma: no cover - abstract
        raise NotImplementedError

    def trajectory(self, run_dir: Path, *, obs_trunc: int) -> str:  # pragma: no cover
        raise NotImplementedError

    def deliverable_files(self, run_dir: Path) -> dict[str, str]:  # pragma: no cover
        raise NotImplementedError

    def deliverable_text(self, run_dir: Path) -> str:
        return "\n\n".join(self.deliverable_files(run_dir).values())

    # ---- measurement (never called by the method) ---------------------------------------
    def has_judge(self, task_id: str) -> bool:
        return True

    def score(self, run_dir: Path, task_id: str) -> dict | None:
        grade = hidden.read(hidden.grade_path(self.spec.runs_root, run_dir))
        if not grade or grade.get("reward") is None:
            return None
        reward = float(grade["reward"])
        passed = reward >= float(grade.get("pass_threshold", 0.999))
        return {
            "n_passed": int(passed),
            "n_criteria": 1,
            "all_pass": passed,
            "reward": reward,
        }


def runner_argv(module: str, *args: str) -> list[str]:
    return [sys.executable, "-m", module, *args]


def _run_group(argv: list[str], log, wall: float, env: dict) -> tuple[int, bool]:
    """Run in a new process group so a timeout stops the runner and everything it started."""
    proc = subprocess.Popen(
        argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env,
        start_new_session=True,
    )
    try:
        return proc.wait(timeout=wall), False
    except subprocess.TimeoutExpired:
        for sig, grace in ((signal.SIGTERM, 60), (signal.SIGKILL, 10)):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                break
            try:
                proc.wait(timeout=grace)
                break
            except subprocess.TimeoutExpired:
                continue
        return EXIT_INFRA, True
    except BaseException:
        # Interrupted (e.g. the run is being stopped): take the runner down with us.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        raise


def _tail(path: Path, chars: int = 1200) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "(no runner log)"
    return text[-chars:].strip().replace("\n", " | ")


def check_models(ctx: SuiteContext, roles: tuple[str, ...]) -> list[str]:
    """Provider-key problems for the improver-side roles (runners check their own models)."""
    if fake_mode():
        return []
    problems: list[str] = []
    for role in roles:
        try:
            values = ctx.settings.role(role)
            provider_name, _ = providers.split_model(values["model"])
            problems.extend(providers.check(ctx.settings.provider(provider_name)))
        except ConfigError as exc:
            problems.append(f"{role}: {exc}")
    return list(dict.fromkeys(problems))
