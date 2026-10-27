"""h2x-tau: tau2-bench customer-service domains (airline, banking_knowledge) as a Harness2 suite.

The agent being tuned is tau2's own ``llm_agent`` (same system prompt, tools and loop) plus an
editable harness: ``AGENT.md`` (always in the system prompt) and ``guides/<name>.md`` (read on
demand through a local ``open_guide`` tool). The simulated customer, the policy, the tools and
the database checks are tau2's, unchanged. Grades come from tau2's evaluator and are hidden.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from harness2.core.spec import BenchPrompts
from harness2.registry import SuiteContext
from harness2.settings import ConfigError
from harness2.substrate.command import CommandSubstrate

from ..tau_runner import render
from .common import RunnerAdapter, check_models, fake_mode, runner_argv, seed_from_run_id

EDITABLE = ("AGENT.md", "guides/*.md")
GUIDE = re.compile(r"guides/([a-z0-9][a-z0-9_-]*)\.md")
MAX_AGENT_MD = 12_000
MAX_GUIDE = 15_000
MAX_GUIDES = 12

COMPANIES = {
    "airline": "an airline",
    "banking_knowledge": "a retail bank (Rho-Bank) whose product rules live in a searchable "
    "knowledge base",
}


def prompts(domain: str) -> BenchPrompts:
    company = COMPANIES.get(domain, f"a {domain} business")
    return BenchPrompts(
        work=(
            f"Customer-service chat for {company}. The agent talks with one customer at a time "
            "and acts on the company's systems through tools: look-ups, and changes such as "
            "bookings, cancellations, refunds, applications or account updates. The written "
            "policy (context/source/policy.md) controls what the agent may do and what it must "
            "check first. The result of a conversation is what the agent changed in the "
            "systems and what it told the customer: exactly the changes the customer asked for "
            "and is entitled to under the policy, after the checks and confirmations the policy "
            "requires, with every stated fact confirmed by a tool. How the harness reaches this "
            "agent: AGENT.md is appended to its system prompt on every turn; each "
            "guides/<name>.md is listed in that prompt by name and its first line, and the agent "
            "reads it only when it calls its open_guide tool with that name."
        ),
        domain_word="service area",
        deliverable_word="resolution",
        inv_levers=(
            "- identify and verify the customer, and look up the records a request depends on, "
            "before proposing or making any change\n"
            "- before every action that changes a record, state the exact change and get the "
            "customer's explicit confirmation, as the policy requires\n"
            "- make exactly the changes requested and permitted -- no extra, missing or "
            "duplicated actions; recompute amounts with the tools instead of guessing\n"
            "- when a tool returns an error, read it, correct the arguments once, and tell the "
            "customer plainly if the request cannot be done"
        ),
        dom_levers=(
            "- the policy rules that decide eligibility (membership tiers, fare classes, time "
            "windows, product terms) and the order in which they apply\n"
            "- which requests must be refused, transferred to a human, or partly fulfilled, and "
            "how to say so\n"
            "- the facts a customer must be told (amounts, fees, terms) and where to find them "
            "in the systems or the knowledge base"
        ),
        yardstick=(
            "Did the agent verify the customer and the records first, apply the policy's "
            "eligibility rules correctly, confirm before every change, make exactly the "
            "changes the customer asked for and is entitled to, and tell the customer only "
            "facts the tools confirmed?"
        ),
        selection_checks=(
            "Prefer the conversation whose actions are exactly what the policy allows for "
            "this customer's request, confirmed before execution, with no extra or missing "
            "changes, and whose statements to the customer are accurate."
        ),
    )


class TauAdapter(RunnerAdapter):
    suite_name = "h2x-tau"
    install_mode = "command_files"
    domain_word = "service area"
    leak_globs = ("grade*.json", "*.hidden.json")
    runner_module = "h2x.tau_runner"

    def __init__(self, ctx: SuiteContext):
        self.tau = None
        super().__init__(
            ctx,
            substrate=CommandSubstrate("(h2x tau runner)", EDITABLE),
            prompts=prompts(_domain(ctx)),
        )
        self.tau = self.exp.section("tau")
        self.domain = self.tau["domain"]

    # ---- tasks ---------------------------------------------------------------------------
    def domain_of(self, task_id: str) -> str:
        return self.domain

    def reference_domain_order(self) -> list[str]:
        return [self.domain]

    def task_prompt(self, task_id: str) -> str:
        return (
            f"Customer conversation {task_id} ({self.domain}).\n\n"
            "You are the customer-service agent whose policy is in context/source/policy.md. "
            "A customer contacts you by chat; help them with what they ask for, following the "
            "policy and using your tools. What this customer wanted, and how the conversation "
            "went, is in the trajectories.\n"
        )

    def task_files(self, task_id: str) -> Path | None:
        path = self._source_dir(task_id)
        return path if (path / "policy.md").is_file() else None

    def _source_dir(self, task_id: str) -> Path:
        return self.spec.runs_root / "source" / task_id.replace("/", "__")

    def after_rollout(self, task_id: str, run_dir: Path) -> None:
        policy = run_dir / "policy.md"
        dest = self._source_dir(task_id) / "policy.md"
        if policy.is_file() and not dest.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(policy, dest)

    # ---- execution -----------------------------------------------------------------------
    def runner_timeout(self, task_id: str) -> int:
        sim = int(self.tau.get("timeout_s", 900))
        return sim * (int(self.tau.get("retries", 2)) + 1) + 180

    def runner_command(self, task_id, staged, run_dir, grade, *, run_id):
        argv = runner_argv(
            "h2x.tau_runner",
            "run",
            "--domain", self.domain,
            "--task-id", task_id,
            "--harness", str(staged),
            "--out", str(run_dir),
            "--grade", str(grade),
            "--agent-model", self.exp.model("executor"),
            "--user-model", self.exp.model("user"),
            "--seed", str(seed_from_run_id(run_id, int(self.tau.get("seed", 300)))),
            "--max-steps", str(self.tau.get("max_steps", 200)),
            "--max-errors", str(self.tau.get("max_errors", 10)),
            "--timeout", str(self.tau.get("timeout_s", 900)),
            "--communicate-scorer", str(self.tau.get("communicate_scorer", "substring")),
            "--max-tokens", str(self.tau.get("max_tokens", 16000)),
            "--retries", str(self.tau.get("retries", 2)),
        )
        if self.tau.get("retrieval_config"):
            argv += ["--retrieval-config", str(self.tau["retrieval_config"])]
        return argv

    # ---- evidence ------------------------------------------------------------------------
    def _transcript(self, run_dir: Path) -> dict:
        return render.load(Path(run_dir) / "transcript.json")

    def delivered(self, run_dir: Path) -> bool:
        try:
            return render.n_agent_messages(self._transcript(run_dir)) > 0
        except (OSError, ValueError):
            return False

    def trajectory(self, run_dir: Path, *, obs_trunc: int) -> str:
        return render.render_trajectory(self._transcript(run_dir), obs_trunc=obs_trunc)

    def deliverable_files(self, run_dir: Path) -> dict[str, str]:
        try:
            return {"conversation.md": render.render_conversation(self._transcript(run_dir))}
        except (OSError, ValueError):
            return {}

    def validate_tree(self, tree: dict[str, str]) -> None:
        problems = []
        guides = []
        for rel, body in tree.items():
            if rel == "AGENT.md":
                if len(body) > MAX_AGENT_MD:
                    problems.append(
                        f"AGENT.md is {len(body)} characters; keep it under {MAX_AGENT_MD} "
                        "(it is sent on every turn) and move situational procedures to guides/"
                    )
                continue
            match = GUIDE.fullmatch(rel)
            if not match:
                problems.append(
                    f"{rel}: guides must be guides/<name>.md with a lowercase name of letters, "
                    "digits, '-' or '_' (no subdirectories)"
                )
                continue
            guides.append(match.group(1))
            if not body.strip():
                problems.append(f"{rel} is empty; delete it instead")
            if len(body) > MAX_GUIDE:
                problems.append(f"{rel} is {len(body)} characters; keep a guide under {MAX_GUIDE}")
        if len(guides) > MAX_GUIDES:
            problems.append(f"{len(guides)} guides; keep at most {MAX_GUIDES}")
        agent_md = tree.get("AGENT.md", "")
        for name in guides:
            if name not in agent_md:
                problems.append(
                    f"guide '{name}' is never named in AGENT.md: the agent sees every guide's "
                    "name and first line, but say in AGENT.md when to open it so the trigger "
                    "is unambiguous"
                )
        if problems:
            raise ValueError(" | ".join(problems))


def _domain(ctx: SuiteContext) -> str:
    from .. import config as h2x_config

    return h2x_config.for_project(ctx.project).section("tau").get("domain", "airline")


def make_adapter(ctx: SuiteContext) -> TauAdapter:
    return TauAdapter(ctx)


def check(ctx: SuiteContext) -> list[str]:
    problems = check_models(ctx, ("improver", "reflector"))
    if not fake_mode():
        import importlib.util

        if importlib.util.find_spec("tau2") is None:
            problems.append("tau2 is not installed; run scripts/fetch_tau2.sh and pip install it")
    try:
        exp = TauAdapter(ctx).exp
        if not exp.split("train"):
            problems.append(f"{exp.id}: tasks.lock.json has no train tasks")
    except (ConfigError, KeyError) as exc:
        problems.append(str(exc))
    return problems
