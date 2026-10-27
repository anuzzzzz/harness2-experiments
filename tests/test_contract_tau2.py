"""tau2-bench API contract (offline): the agent, factory and run config h2x relies on.

Skipped unless tau2 is installed. No model is called: `generate` is replaced by a script.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import tau2  # noqa: F401

    HAVE_TAU2 = True
except ImportError:  # pragma: no cover
    HAVE_TAU2 = False


@unittest.skipUnless(HAVE_TAU2, "tau2 is not installed")
class Tau2Contract(unittest.TestCase):
    def env(self):
        from tau2.runner.build import build_environment

        return build_environment("airline")

    def bundle(self, agent_md="", guides=None):
        from h2x.tau_runner.agent import HarnessBundle

        root = Path(tempfile.mkdtemp())
        (root / "AGENT.md").write_text(agent_md)
        for name, body in (guides or {}).items():
            (root / "guides").mkdir(exist_ok=True)
            (root / "guides" / f"{name}.md").write_text(body)
        return HarnessBundle.load(root)

    def test_tasks_load_offline(self):
        from tau2.run import get_tasks

        tasks = get_tasks("airline", task_split_name=None, task_ids=["11"])
        self.assertEqual(tasks[0].id, "11")
        banking = get_tasks("banking_knowledge", task_split_name=None, task_ids=["task_001"])
        self.assertEqual(banking[0].id, "task_001")

    def test_empty_harness_is_llm_agent(self):
        from tau2.agent.llm_agent import LLMAgent

        from h2x.tau_runner.agent import HarnessAgent, Recorder

        env = self.env()
        tools, policy = env.get_tools(), env.get_policy()
        plain = LLMAgent(tools=tools, domain_policy=policy, llm="anthropic/claude-haiku-5-5")
        ours = HarnessAgent(tools, policy, llm="anthropic/claude-haiku-5-5",
                            harness=self.bundle(), recorder=Recorder())
        self.assertEqual(ours.system_prompt, plain.system_prompt)
        self.assertEqual([t.name for t in ours.tools], [t.name for t in plain.tools])

    def test_guides_are_local_and_recorded(self):
        import h2x.tau_runner.agent as agent_mod
        from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage

        env = self.env()
        recorder = agent_mod.Recorder()
        bundle = self.bundle("Open the refunds guide before any refund.",
                             {"refunds": "description: refund rules\nCheck the fare class."})
        agent = agent_mod.HarnessAgent(env.get_tools(), env.get_policy(),
                                       llm="anthropic/claude-haiku-5-5",
                                       harness=bundle, recorder=recorder)
        self.assertIn("<operating_guide>", agent.system_prompt)
        self.assertIn("- refunds: refund rules", agent.system_prompt)
        self.assertIn("open_guide", [t.name for t in agent.tools])
        script = [
            AssistantMessage(role="assistant", content=None,
                             tool_calls=[ToolCall(id="g1", name="open_guide", arguments={"name": "refunds"})],
                             usage={"prompt_tokens": 10, "completion_tokens": 2}),
            AssistantMessage(role="assistant", content="Which reservation?",
                             usage={"prompt_tokens": 20, "completion_tokens": 3}),
        ]
        calls = []

        def fake_generate(**kwargs):
            calls.append(kwargs)
            return script.pop(0)

        original = agent_mod.generate
        agent_mod.generate = fake_generate
        try:
            state = agent.get_init_state()
            msg, state = agent.generate_next_message(
                UserMessage(role="user", content="I want a refund"), state)
        finally:
            agent_mod.generate = original
        self.assertEqual(msg.content, "Which reservation?")
        self.assertEqual(len(calls), 2)
        self.assertEqual([e["event"] for e in recorder.events], ["open_guide"])
        self.assertEqual(recorder.returned, 1)
        self.assertEqual(len(recorder.usages), 2)
        # the guide text reached the model as a tool result, in a well-formed pair
        roles = [type(m).__name__ for m in state.messages]
        self.assertEqual(roles, ["UserMessage", "AssistantMessage", "ToolMessage", "AssistantMessage"])

    def test_run_config_accepts_h2x_fields(self):
        from tau2.data_model.semantic_communication import CommunicateScoring
        from tau2.data_model.simulation import TextRunConfig

        cfg = TextRunConfig(
            domain="banking_knowledge", agent="llm_agent", llm_agent="anthropic/claude-haiku-5-5",
            llm_args_agent={"max_tokens": 16000}, user="user_simulator",
            llm_user="anthropic/claude-haiku-5-5", llm_args_user={"max_tokens": 16000},
            max_steps=200, max_errors=10, seed=300, timeout=900,
            communicate_scoring=CommunicateScoring(scorer="substring"), retrieval_config="bm25",
        )
        self.assertEqual(cfg.retrieval_config, "bm25")
        self.assertNotIn("temperature", cfg.llm_args_agent)

    def test_transcript_alignment(self):
        from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage

        from h2x.tau_runner.agent import Recorder
        from h2x.tau_runner.cli import build_transcript

        rec = Recorder()
        rec.events = [{"event": "open_guide", "name": "refunds", "found": True, "chars": 5,
                       "before_agent_message": 1}]
        msgs = [
            AssistantMessage(role="assistant", content="Hi! How can I help you today?", cost=0.0),
            UserMessage(role="user", content="refund", usage={"prompt_tokens": 1, "completion_tokens": 1}),
            AssistantMessage(role="assistant", content="Sure", usage={"prompt_tokens": 1, "completion_tokens": 1}),
            UserMessage(role="user", content="thanks", usage={"prompt_tokens": 1, "completion_tokens": 1}),
            AssistantMessage(role="assistant", content="Done", usage={"prompt_tokens": 1, "completion_tokens": 1}),
            ToolMessage(id="x", role="tool", content="{}"),
        ]
        sim = SimpleNamespace(messages=msgs, termination_reason=SimpleNamespace(value="user_stop"))
        t = build_transcript(sim, rec, "airline", "11")
        roles = [m["role"] for m in t["messages"]]
        # the guide was opened before the agent's 2nd produced message ("Done")
        self.assertEqual(roles, ["assistant", "user", "assistant", "user", "harness", "assistant", "tool"])
        self.assertEqual(t["termination_reason"], "user_stop")


if __name__ == "__main__":
    unittest.main()
