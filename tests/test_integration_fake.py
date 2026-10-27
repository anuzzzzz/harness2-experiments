"""End-to-end plumbing with fakes: Harness2 runs for real, every model call is faked.

H2X_FAKE=1 makes the tau/Harbor runners and the LAB grader fabricate rollouts and grades;
tests/fakes/claude stands in for the Claude Code CLI (improver, reflector, consolidator, and
the LAB executor). Nothing here needs an API key or spends money. Skipped when harness2-cli is
not installed (CI installs it).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import harness2  # noqa: F401

    HAVE_HARNESS2 = True
except ImportError:  # pragma: no cover
    HAVE_HARNESS2 = False

REPO = Path(__file__).resolve().parents[1]
FAKES = REPO / "tests" / "fakes"


@unittest.skipUnless(HAVE_HARNESS2, "harness2-cli is not installed")
class FakeEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="h2x-it-"))
        shutil.copytree(
            REPO / "experiments", cls.tmp / "experiments",
            ignore=shutil.ignore_patterns(".cache", "runs", "tasks", "*.stdout.json"),
        )
        (cls.tmp / "results").mkdir()
        (cls.tmp / "docs" / "data").mkdir(parents=True)
        (cls.tmp / "README.md").write_text("table\n<!-- results:start -->\n<!-- results:end -->\n")
        os.chmod(FAKES / "claude", 0o755)
        cls.env = {
            **os.environ,
            "H2X_REPO": str(cls.tmp),
            "H2X_FAKE": "1",
            "HARNESS2_IMPROVER__BIN": str(FAKES / "claude"),
            "ANTHROPIC_API_KEY": "offline-test-not-a-key",
            "PATH": f"{FAKES}{os.pathsep}{os.environ.get('PATH', '')}",
            "XDG_CONFIG_HOME": str(cls.tmp / "xdg"),
        }

    def h2x(self, *args: str) -> subprocess.CompletedProcess:
        proc = subprocess.run(
            [sys.executable, "-m", "h2x", *args], env=self.env, capture_output=True, text=True,
            timeout=1800,
        )
        if proc.returncode:
            print("STDOUT:", proc.stdout[-6000:], "\nSTDERR:", proc.stderr[-12000:])
        return proc

    def results(self, exp: str) -> Path:
        return self.tmp / "results" / exp

    def assert_no_grade_reached_improver(self, exp: str) -> None:
        adapt = self.tmp / "experiments" / exp / ".harness2" / "runs" / "main" / "adapt"
        self.assertTrue(adapt.is_dir())
        for path in adapt.rglob("*"):
            if path.is_file() and path.suffix in (".md", ".json", ".txt"):
                body = path.read_text(encoding="utf-8", errors="replace")
                self.assertNotIn('"reward":', body, f"grade leaked into {path}")
                self.assertNotIn("n_passed", body, f"grade leaked into {path}")

    def run_flow(self, exp: str, *, setup: bool = False) -> dict:
        if setup:
            self.assertEqual(self.h2x("setup", exp).returncode, 0)
        self.assertEqual(self.h2x("smoke", exp).returncode, 0)
        self.assertEqual(self.h2x("train", exp).returncode, 0)
        train = json.loads((self.results(exp) / "train.json").read_text())
        self.assertTrue(train["final_harness"], train)
        shards = json.loads(self.h2x("matrix", exp).stdout)
        for shard in shards:
            self.assertEqual(self.h2x("eval", exp, "--shard", str(shard)).returncode, 0)
        self.assertEqual(self.h2x("report", exp).returncode, 0)
        summary = json.loads((self.results(exp) / "summary.json").read_text())
        self.assert_no_grade_reached_improver(exp)
        spend = json.loads((self.results(exp) / "spend.json").read_text())
        self.assertGreater(spend["total_usd"], 0)
        return summary

    def test_tau_airline(self):
        summary = self.run_flow("e3-tau-airline")
        final = (self.results("e3-tau-airline") / "final_harness" / "AGENT.md").read_text()
        self.assertIn("verify", final.lower())
        t = summary["transfer"]
        self.assertEqual(t["n_tasks"], 20)
        self.assertEqual(t["final_mean"], 1.0)  # the fake rewards a harness that says 'verify'
        self.assertLess(t["base_mean"], 1.0)
        self.assertIn("pass_hat_k", summary)

    def test_terminal_bench(self):
        summary = self.run_flow("e4-terminal-bench")
        final = (self.results("e4-terminal-bench") / "final_harness" / "CLAUDE.md").read_text()
        self.assertIn("verify", final.lower())
        self.assertEqual(summary["transfer"]["n_tasks"], 12)

    def test_lab(self):
        lab = self.tmp / "lab"
        lock = json.loads((self.tmp / "experiments" / "e1-lab-lpa" / "tasks.lock.json").read_text())
        for task_id, rel in lock["lab_paths"].items():
            d = lab / "tasks" / rel
            (d / "documents").mkdir(parents=True)
            (d / "documents" / "precedent.txt").write_text("Greenfield Fund LP precedent\n")
            (d / "task.json").write_text(json.dumps({
                "title": f"Draft LPA {task_id}",
                "instructions": "Adapt the precedent. Output: `lpa.docx`.",
                "deliverables": {"lpa.docx": "lpa.docx"},
                "criteria": [{"id": f"C-{i}", "title": "t", "match_criteria": "PASS if x",
                              "deliverables": ["lpa.docx"]} for i in range(4)],
            }))
        self.env["H2X_LAB_ROOT"] = str(lab)
        summary = self.run_flow("e1-lab-lpa", setup=True)
        self.assertEqual(summary["transfer"]["n_tasks"], 6)
        final = (self.results("e1-lab-lpa") / "final_harness" / "CLAUDE.md").read_text()
        self.assertIn("verify", final.lower())


@unittest.skipUnless(HAVE_HARNESS2, "harness2-cli is not installed")
class AdapterContracts(unittest.TestCase):
    """Wording and evidence rules the method relies on."""

    def test_prompt_wording_has_no_measurement_terms(self):
        # BenchPrompts raises at construction if a visible field mentions measurement.
        from h2x.suites import harbor, lab, tau

        self.assertTrue(tau.prompts("airline").work)
        self.assertTrue(tau.prompts("banking_knowledge").work)
        self.assertEqual(set(harbor.PROMPTS), {"terminal", "repo"})
        self.assertTrue(lab.LAB_PROMPTS.work)

    def test_tau_validate_tree(self):
        from h2x.suites.tau import TauAdapter

        validate = TauAdapter.validate_tree
        validate(None, {"AGENT.md": "Open the refunds guide for refunds.", "guides/refunds.md": "x"})
        with self.assertRaises(ValueError):
            validate(None, {"AGENT.md": "", "guides/refunds.md": "x"})  # never named in AGENT.md
        with self.assertRaises(ValueError):
            validate(None, {"AGENT.md": "a", "guides/Bad Name.md": "x"})
        with self.assertRaises(ValueError):
            validate(None, {"AGENT.md": "x" * 20_000})


if __name__ == "__main__":
    unittest.main()
