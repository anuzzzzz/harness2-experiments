"""Harbor API contract (offline): the agent class and CLI flags h2x relies on.

Skipped unless harbor is installed. Starts no container and calls no model.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

try:
    import harbor  # noqa: F401

    HAVE_HARBOR = True
except ImportError:  # pragma: no cover
    HAVE_HARBOR = False


@unittest.skipUnless(HAVE_HARBOR, "harbor is not installed")
class HarborContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("ANTHROPIC_API_KEY", "offline-test-not-a-key")
        os.environ.setdefault("HARBOR_TELEMETRY", "0")

    def test_agent_subclasses_claude_code(self):
        from harbor.agents.installed.claude_code import ClaudeCode

        from h2x.harbor_runner.agent import HarnessClaudeCode

        self.assertTrue(issubclass(HarnessClaudeCode, ClaudeCode))
        for hook in ("configure", "run", "exec_as_root"):
            self.assertTrue(callable(getattr(HarnessClaudeCode, hook, None)), hook)

    def test_agent_reads_harness_env(self):
        from h2x.harbor_runner.agent import HarnessClaudeCode

        harness = Path(tempfile.mkdtemp())
        (harness / "CLAUDE.md").write_text("Always run the tests.\n")
        os.environ["H2X_HARNESS_DIR"] = str(harness)
        try:
            agent = HarnessClaudeCode(Path(tempfile.mkdtemp()), model_name="anthropic/claude-haiku-5-5")
        finally:
            os.environ.pop("H2X_HARNESS_DIR", None)
        self.assertTrue(agent._has_harness())
        empty = Path(tempfile.mkdtemp())
        (empty / "CLAUDE.md").write_text("")
        os.environ["H2X_HARNESS_DIR"] = str(empty)
        try:
            agent = HarnessClaudeCode(Path(tempfile.mkdtemp()), model_name="anthropic/claude-haiku-5-5")
        finally:
            os.environ.pop("H2X_HARNESS_DIR", None)
        self.assertFalse(agent._has_harness())  # empty harness = Harbor's claude-code exactly

    def test_shell_scripts_parse(self):
        from h2x.harbor_runner import agent

        for script in (agent.INSTALL, agent.CAPTURE, agent.REMOVE):
            subprocess.run(["bash", "-n", "-c", script], check=True)

    def test_runner_command_line_parses(self):
        """The exact `harbor run` command line h2x builds is accepted by Harbor's own CLI parser,
        and its --ak values are valid Claude Code agent options."""
        from argparse import Namespace

        from harbor.agents.installed.claude_code import ClaudeCodeOptions
        from harbor.cli.main import app
        from harbor.cli.utils import parse_kwargs
        from typer.main import get_command

        from h2x.harbor_runner.cli import _command

        args = Namespace(harbor="harbor", task_dir=Path(tempfile.mkdtemp()),
                         model="anthropic/claude-haiku-5-5", max_budget_usd=1.0, max_turns=40,
                         claude_version="2.1.0", effort="medium", agent_timeout_multiplier=0.5)
        cmd = _command(args, Path(tempfile.mkdtemp()) / "job")
        self.assertEqual(cmd[:2], ["harbor", "run"])
        run = get_command(app).commands["run"]
        run.make_context("run", list(cmd[2:]))  # raises click.UsageError on any unknown flag/value
        aks = [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "--ak"]
        opts = ClaudeCodeOptions(**parse_kwargs(aks))
        self.assertEqual((opts.max_budget_usd, opts.max_turns, opts.version), ("1.00", 40, "2.1.0"))

    def test_claude_code_options(self):
        from harbor.agents.installed.claude_code import ClaudeCodeOptions

        fields = set(ClaudeCodeOptions.model_fields)
        self.assertTrue({"max_budget_usd", "max_turns"} <= fields, fields)


if __name__ == "__main__":
    unittest.main()
