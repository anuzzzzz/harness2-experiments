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

    def test_run_cli_flags_exist(self):
        out = subprocess.run(["harbor", "run", "--help"], capture_output=True, text=True,
                             env={**os.environ, "COLUMNS": "250", "HARBOR_TELEMETRY": "0"})
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        text = out.stdout
        for flag in ("--path", "--agent", "--model", "--jobs-dir", "--job-name", "--n-concurrent",
                     "--n-attempts", "--yes", "--quiet", "--agent-kwarg", "--effort",
                     "--agent-timeout-multiplier"):
            self.assertIn(flag, text, flag)

    def test_claude_code_options(self):
        from harbor.agents.installed.claude_code import ClaudeCodeOptions

        fields = set(ClaudeCodeOptions.model_fields)
        self.assertTrue({"max_budget_usd", "max_turns"} <= fields, fields)


if __name__ == "__main__":
    unittest.main()
