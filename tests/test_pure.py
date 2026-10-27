"""Tests that need nothing but this repository (no Harness2, no benchmark packages)."""

from __future__ import annotations

import json
import tempfile
import tomllib
import unittest
from pathlib import Path

from h2x import config, ledger, pricing, stats
from h2x.harbor_runner import streams
from h2x.lab.grade import parse_json
from h2x.tau_runner import render

FORBIDDEN = ("benchmark", "evaluation", "grader", "rubric", "score")


class Pricing(unittest.TestCase):
    def test_haiku_tiers_per_request(self):
        small = pricing.request_cost("anthropic/claude-haiku-5-5", input_tokens=100_000, output_tokens=1_000)
        self.assertAlmostEqual(small, 100_000 * 0.10 / 1e6 + 1_000 * 0.50 / 1e6)
        big = pricing.request_cost("claude-haiku-5-5", input_tokens=100_001, output_tokens=1_000)
        self.assertAlmostEqual(big, 100_001 * 0.50 / 1e6 + 1_000 * 2.50 / 1e6)

    def test_cache_reads_count_toward_prompt_length(self):
        cost = pricing.request_cost("claude-haiku-5-5", input_tokens=1_000, cache_read=150_000)
        self.assertAlmostEqual(cost, 1_000 * 0.50 / 1e6 + 150_000 * 0.05 / 1e6)

    def test_sonnet_flat(self):
        self.assertAlmostEqual(
            pricing.request_cost("claude-sonnet-5-5", input_tokens=1e6, output_tokens=1e6), 12.0)

    def test_anthropic_usage(self):
        usage = {"input_tokens": 10, "output_tokens": 20, "cache_creation_input_tokens": 3000,
                 "cache_read_input_tokens": 5000}
        expect = (10 * 0.10 + 20 * 0.50 + 3000 * 0.125 + 5000 * 0.01) / 1e6
        self.assertAlmostEqual(pricing.anthropic_usage_cost("claude-haiku-5-5", usage), expect)

    def test_litellm_usage_and_unknown_model(self):
        self.assertAlmostEqual(
            pricing.litellm_usage_cost("anthropic/claude-haiku-5-5",
                                       {"prompt_tokens": 1000, "completion_tokens": 100}),
            (1000 * 0.10 + 100 * 0.50) / 1e6)
        with self.assertRaises(KeyError):
            pricing.rate("gpt-unknown", 10)

    def test_normalize(self):
        self.assertEqual(pricing.normalize("anthropic/claude-sonnet-5-5-20261001"), "claude-sonnet-5-5")


class Stats(unittest.TestCase):
    def test_compare_paired(self):
        values = {"a": {"base": [0, 0], "final": [1, 1]},
                  "b": {"base": [1], "final": [1]},
                  "c": {"base": [1], "final": [0]},
                  "d": {"base": [0], "final": []}}
        c = stats.compare(values)
        self.assertEqual(c.n_tasks, 3)
        self.assertEqual((c.wins, c.losses, c.ties), (1, 1, 1))
        self.assertAlmostEqual(c.delta, 0.0)
        self.assertAlmostEqual(c.base_mean, 2 / 3)

    def test_sign_test(self):
        self.assertEqual(stats.sign_test(0, 0), None)
        self.assertAlmostEqual(stats.sign_test(5, 0), 2 / 32)
        self.assertAlmostEqual(stats.sign_test(3, 3), 1.0)

    def test_pass_hat_k(self):
        self.assertAlmostEqual(stats.pass_hat_k(2, 3, 2), 1 / 3)
        self.assertAlmostEqual(stats.mean_pass_hat_k({"t": [1, 1, 0], "u": [1, 1, 1]}, 2), (1 / 3 + 1) / 2)

    def test_bootstrap_brackets_mean(self):
        lo, hi = stats.bootstrap_ci([0.0, 1.0, 1.0, 0.0, 1.0], iters=2000)
        self.assertLessEqual(lo, 0.6)
        self.assertGreaterEqual(hi, 0.6)


class Render(unittest.TestCase):
    TRANSCRIPT = {
        "domain": "airline", "task_id": "7", "termination_reason": "user_stop",
        "messages": [
            {"role": "assistant", "content": "Hi! How can I help you today?", "tool_calls": None},
            {"role": "user", "content": "Cancel my trip", "tool_calls": None},
            {"role": "harness", "event": "open_guide", "name": "cancellations", "found": True, "chars": 50},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "1", "name": "get_reservation_details", "arguments": {"reservation_id": "X1"}}]},
            {"role": "tool", "id": "1", "requestor": "assistant", "content": "x" * 100, "error": False},
            {"role": "assistant", "content": "Done.", "tool_calls": None},
        ],
    }

    def test_trajectory_grammar_and_truncation(self):
        text = render.render_trajectory(self.TRANSCRIPT, obs_trunc=20)
        self.assertIn("[USER] Cancel my trip", text)
        self.assertIn("[ACTION] open_guide", text)
        self.assertIn('[ACTION] get_reservation_details: {"reservation_id": "X1"}', text)
        self.assertIn("truncated by the renderer", text)
        self.assertIn("conversation ended: user_stop", text)
        self.assertNotIn("reward", text.lower())

    def test_conversation(self):
        text = render.render_conversation(self.TRANSCRIPT)
        self.assertIn("**Customer:** Cancel my trip", text)
        self.assertIn("`get_reservation_details`", text)
        self.assertEqual(render.n_agent_messages(self.TRANSCRIPT), 3)


class Streams(unittest.TestCase):
    def write(self, events):
        path = Path(tempfile.mkdtemp()) / "s.jsonl"
        path.write_text("noise line\n" + "\n".join(json.dumps(e) for e in events) + "\n")
        return path

    def test_reported_cost_and_final(self):
        p = self.write([
            {"type": "assistant", "message": {"id": "a", "content": [{"type": "text", "text": "hello"}],
                                              "usage": {"input_tokens": 10, "output_tokens": 1}}},
            {"type": "result", "subtype": "success", "result": "final words", "total_cost_usd": 0.5},
        ])
        self.assertEqual(streams.final_message(p), "final words")
        self.assertAlmostEqual(streams.cost(p, "claude-haiku-5-5"), 0.5)
        self.assertEqual(streams.final_error(p), "")

    def test_usage_cost_dedupes_message_ids(self):
        usage = {"input_tokens": 1000, "output_tokens": 100}
        p = self.write([
            {"type": "assistant", "message": {"id": "a", "usage": usage, "content": []}},
            {"type": "assistant", "message": {"id": "a", "usage": usage, "content": []}},
            {"type": "result", "subtype": "error_during_execution", "is_error": True, "result": "API Error: 500"},
        ])
        self.assertAlmostEqual(streams.cost(p, "claude-haiku-5-5"), (1000 * 0.10 + 100 * 0.50) / 1e6)
        self.assertIn("API Error", streams.final_error(p))


class LabGrade(unittest.TestCase):
    def test_parse_json_variants(self):
        self.assertEqual(parse_json('```json\n{"verdict": "pass", "reasoning": "x"}\n```')["verdict"], "pass")
        self.assertEqual(parse_json('blah {"verdict": "fail", "reasoning": "y"} tail')["verdict"], "fail")
        with self.assertRaises(ValueError):
            parse_json("no json here")


class Experiments(unittest.TestCase):
    def test_all_load_and_budget(self):
        ids = config.all_ids()
        self.assertEqual(len(ids), 5)
        total = 0.0
        for exp_id in ids:
            exp = config.load(exp_id)
            total += exp.budget_usd
            self.assertTrue(exp.split("train"), exp_id)
            self.assertTrue(exp.split("test"), exp_id)
            self.assertFalse(set(exp.split("train")) & set(exp.split("test")), exp_id)
            self.assertLessEqual(exp.train_budget_usd + exp.eval_budget_usd + exp.smoke_budget_usd,
                                 exp.budget_usd + 1e-9, exp_id)
        self.assertLessEqual(total, 100.0 + 1e-9)

    def test_models_agree_with_harness2_config(self):
        for exp_id in config.all_ids():
            exp = config.load(exp_id)
            h2 = tomllib.loads((exp.root / ".harness2" / "config.toml").read_text())
            self.assertEqual(h2["executor"]["model"], exp.model("executor"), exp_id)
            self.assertEqual(h2["improver"]["model"], exp.model("improver"), exp_id)
            self.assertEqual(h2["suite"]["name"], exp.suite, exp_id)
            self.assertEqual(h2["run"]["k"], 1, exp_id)

    def test_base_harness_is_empty(self):
        for exp_id in config.all_ids():
            exp = config.load(exp_id)
            files = [p for p in exp.harness_dir.rglob("*") if p.is_file()]
            self.assertTrue(files, exp_id)
            self.assertTrue(all(p.stat().st_size == 0 for p in files), exp_id)

    def test_harbor_sources_cover_tasks(self):
        for exp_id in ("e4-terminal-bench", "e5-swe-pro"):
            exp = config.load(exp_id)
            self.assertEqual(set(exp.tasks["sources"]), set(exp.split("all")))

    def test_lab_groups_do_not_leak(self):
        exp = config.load("e1-lab-lpa")
        groups = exp.tasks["groups"]
        train = {groups[t] for t in exp.split("train")}
        self.assertFalse(any(groups[t] in train for t in exp.split("test")))

    def test_tau_airline_uses_official_test_split(self):
        exp = config.load("e3-tau-airline")
        self.assertEqual(len(exp.split("test")), 20)

    def test_estimates_fit_caps(self):
        from h2x.cli import estimate

        for exp_id in config.all_ids():
            exp = config.load(exp_id)
            est = estimate(exp)
            self.assertLessEqual(est["eval"]["usd"], exp.eval_budget_usd * 1.1, exp_id)


class Ledger(unittest.TestCase):
    def test_record_is_idempotent_per_phase(self):
        exp = config.load("e3-tau-airline")
        with tempfile.TemporaryDirectory() as tmp:
            original = config.RESULTS
            try:
                config.RESULTS = Path(tmp)
                exp_copy = config.Experiment(id=exp.id, root=exp.root, raw=exp.raw, tasks=exp.tasks)
                ledger.record(exp_copy, "train", 3.0)
                ledger.record(exp_copy, "train", 4.0)
                ledger.record(exp_copy, "eval.shard-0", 1.5)
                self.assertAlmostEqual(ledger.total(exp_copy), 5.5)
                self.assertAlmostEqual(ledger.spent_in(exp_copy, "eval"), 1.5)
                self.assertAlmostEqual(ledger.remaining(exp_copy), exp.budget_usd - 5.5)
            finally:
                config.RESULTS = original


class Report(unittest.TestCase):
    def test_global_table_uses_current_spend_not_a_stale_summary(self):
        from h2x import report

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "README.md").write_text(f"x\n{report.START}\nold\n{report.END}\n")
            original = (config.RESULTS, config.REPO)
            try:
                config.RESULTS, config.REPO = tmp / "results", tmp
                exp = config.load("e3-tau-airline")
                exp.results_dir.mkdir(parents=True)
                (exp.results_dir / "summary.json").write_text(json.dumps({"spend_usd": 0.0}))
                ledger.record(exp, "smoke", 0.25)
                rows = report.write_all([])  # what publish_results.sh runs
                table = (tmp / "README.md").read_text()
            finally:
                config.RESULTS, config.REPO = original
        row = next(r for r in rows if r["id"] == "e3-tau-airline")
        self.assertAlmostEqual(row["spend_usd"], 0.25)
        self.assertIn("$0.25 / $15.5", table)


if __name__ == "__main__":
    unittest.main()
