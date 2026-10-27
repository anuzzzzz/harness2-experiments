"""h2x-harbor: Harbor task datasets (Terminal-Bench 2.0, SWE-bench Pro) as a Harness2 suite.

The agent being tuned is Harbor's Claude Code agent inside the task's container; the harness
is Claude Code's own native surface (CLAUDE.md, skills, subagents, scripts, hooks), installed in
the container's working directory for the agent run and removed before the tests run. Task
containers, instructions and tests are Harbor's, unchanged; the verifier's reward is hidden.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from harness2.core.spec import BenchPrompts
from harness2.registry import SuiteContext
from harness2.substrate.claude import ClaudeSubstrate, render_events

from .. import hidden
from ..harbor_runner import streams
from .common import RunnerAdapter, check_models, fake_mode, runner_argv

PROMPTS = {
    "terminal": BenchPrompts(
        work=(
            "Hands-on technical work in a Linux container, driven from a terminal: building and "
            "fixing software, configuring services, recovering or transforming data, and "
            "scientific or security computations. The request controls. The result is the "
            "state the agent leaves in the container -- files, programs, services -- which must "
            "do exactly what the request specifies, at the paths and in the formats it names, "
            "when checked independently after the agent has finished."
        ),
        domain_word="kind of system task",
        deliverable_word="result",
        inv_levers=(
            "- read the request for every concrete requirement (paths, names, formats, "
            "commands that must work) and restate them as a checklist before acting\n"
            "- inspect the environment first: what is installed, what versions, what already "
            "exists at the named paths\n"
            "- after building, run the result yourself the way the request describes and "
            "compare the output with each requirement before stopping\n"
            "- when a command fails, read the error and change approach instead of repeating it; "
            "keep long-running work bounded"
        ),
        dom_levers=(
            "- the conventions of the tools involved (build systems, package managers, service "
            "managers, file formats) that a practitioner relies on\n"
            "- the failure modes typical for this kind of task and how a practitioner checks "
            "for them\n"
            "- what 'done' means for this kind of system: what must persist, be reachable, or "
            "produce exact output"
        ),
        yardstick=(
            "Did the run identify every concrete requirement, inspect the environment before "
            "changing it, produce the requested state at the requested paths, and verify it by "
            "running it the way the request describes before stopping?"
        ),
        selection_checks=(
            "Prefer the result that meets every concrete requirement of the request exactly, "
            "leaves the system in a working state, and shows it was verified."
        ),
    ),
    "repo": BenchPrompts(
        work=(
            "Software engineering in one large open-source repository: implementing the change "
            "a pull-request description asks for (bug fixes and features), including any stated "
            "interface requirements. The request controls. The result is the code change: it "
            "must implement the described behaviour completely, keep existing behaviour that "
            "the request did not ask to change, follow the repository's conventions, and pass "
            "the project's own tests for that area."
        ),
        domain_word="repository",
        deliverable_word="change",
        inv_levers=(
            "- locate the code, tests and conventions that govern the request before editing; "
            "read before writing\n"
            "- find the project's own way to run its tests for the affected area and run them "
            "before finishing, reading what they report\n"
            "- implement every requirement and named interface in the request (exact names, "
            "signatures, return values); keep the change scoped to it\n"
            "- when a command fails, read the error and change approach instead of repeating it"
        ),
        dom_levers=(
            "- the repository's architecture, module boundaries and the patterns existing code "
            "follows in the area the request touches\n"
            "- the invariants a change here must keep: interfaces, data formats, error "
            "handling, compatibility\n"
            "- where the tests for this area live, how they are run, and what a convincing "
            "test of this kind of change looks like"
        ),
        yardstick=(
            "Did the run find the code that governs the request, implement every stated "
            "requirement and interface and nothing else, follow the repository's conventions, "
            "and run the relevant tests before stopping?"
        ),
        selection_checks=(
            "Prefer the change that implements everything the request specifies, fits the "
            "surrounding code's conventions, preserves behaviour the request did not ask to "
            "change, and shows it was tested."
        ),
    ),
}


class HarborAdapter(RunnerAdapter):
    suite_name = "h2x-harbor"
    install_mode = "claude_root"
    domain_word = "area"
    allow_subagents = True
    leak_globs = ("tests/**", "solution/**", "reward.txt", "reward.json")
    runner_module = "h2x.harbor_runner"

    def __init__(self, ctx: SuiteContext):
        from .. import config as h2x_config

        cfg = h2x_config.for_project(ctx.project).section("harbor")
        self.cfg = cfg
        kind = cfg.get("kind", "terminal")
        self.domain_word = "repository" if kind == "repo" else "kind of system task"
        super().__init__(ctx, substrate=ClaudeSubstrate(), prompts=PROMPTS[kind])
        self.task_root = self.exp.root / ".cache" / "tasks"

    # ---- tasks ---------------------------------------------------------------------------
    def task_dir(self, task_id: str) -> Path:
        return self.task_root / task_id

    def _task_toml(self, task_id: str) -> dict:
        path = self.task_dir(task_id) / "task.toml"
        try:
            return tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def task_prompt(self, task_id: str) -> str:
        path = self.task_dir(task_id) / "instruction.md"
        if path.is_file():
            return path.read_text(encoding="utf-8")
        if fake_mode():
            return f"(fake instruction for {task_id})\n"
        raise FileNotFoundError(
            f"{path} is missing; fetch the task directories first (python scripts/fetch_tasks.py "
            f"{self.exp.id})"
        )

    # ---- execution -----------------------------------------------------------------------
    def runner_timeout(self, task_id: str) -> int:
        t = self._task_toml(task_id)
        mult = float(self.cfg.get("agent_timeout_multiplier", 1.0) or 1.0)
        agent = float((t.get("agent") or {}).get("timeout_sec", 1800)) * mult
        verifier = float((t.get("verifier") or {}).get("timeout_sec", 900))
        build = float((t.get("environment") or {}).get("build_timeout_sec", 900))
        per_attempt = agent + verifier + build + 900
        return int(per_attempt * (int(self.cfg.get("retries", 1)) + 1))

    def runner_command(self, task_id, staged, run_dir, grade, *, run_id):
        argv = runner_argv(
            "h2x.harbor_runner",
            "run",
            "--task-dir", str(self.task_dir(task_id)),
            "--harness", str(staged),
            "--out", str(run_dir),
            "--grade", str(grade),
            "--scratch", str(hidden.scratch_dir(self.spec.runs_root, run_dir)),
            "--model", self.exp.model("executor"),
            "--retries", str(self.cfg.get("retries", 1)),
        )
        for flag, key in (
            ("--max-budget-usd", "max_budget_usd"),
            ("--max-turns", "max_turns"),
            ("--agent-timeout-multiplier", "agent_timeout_multiplier"),
            ("--effort", "effort"),
            ("--claude-version", "claude_version"),
        ):
            if self.cfg.get(key):
                argv += [flag, str(self.cfg[key])]
        return argv

    # ---- evidence ------------------------------------------------------------------------
    def delivered(self, run_dir: Path) -> bool:
        stream = Path(run_dir) / "stream.jsonl"
        return stream.is_file() and stream.stat().st_size > 0

    def trajectory(self, run_dir: Path, *, obs_trunc: int) -> str:
        return render_events(
            Path(run_dir) / "stream.jsonl", obs_trunc=obs_trunc, max_chars=300_000
        )

    def deliverable_files(self, run_dir: Path) -> dict[str, str]:
        run_dir = Path(run_dir)
        files = {}
        patch = run_dir / "deliverable.patch"
        if patch.is_file() and patch.read_text(encoding="utf-8", errors="replace").strip():
            files["deliverable.patch"] = patch.read_text(encoding="utf-8", errors="replace")[:200_000]
        final = run_dir / "final.md"
        if final.is_file():
            files["final_message.md"] = final.read_text(encoding="utf-8", errors="replace")
        if not files and (run_dir / "stream.jsonl").is_file():
            files["final_message.md"] = streams.final_message(run_dir / "stream.jsonl")
        return files


def make_adapter(ctx: SuiteContext) -> HarborAdapter:
    return HarborAdapter(ctx)


def check(ctx: SuiteContext) -> list[str]:
    import shutil

    problems = check_models(ctx, ("improver", "reflector"))
    adapter = HarborAdapter(ctx)
    if not fake_mode():
        if shutil.which("harbor") is None:
            problems.append("harbor is not installed (pip install -r requirements/harbor.txt)")
        if shutil.which("docker") is None:
            problems.append("docker is required for Harbor tasks")
        missing = [t for t in adapter.tasks("all") if not (adapter.task_dir(t) / "task.toml").is_file()]
        if missing:
            problems.append(
                f"{len(missing)} task directories not fetched (e.g. {missing[0]}); run "
                f"python scripts/fetch_tasks.py {adapter.exp.id}"
            )
    return problems
