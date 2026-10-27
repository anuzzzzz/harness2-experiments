"""tau2's standard LLMAgent plus an editable harness (imports tau2; runs in the runner process).

With an empty harness the agent is byte-for-byte tau2's ``llm_agent``: same system prompt,
same tools, same message handling. The harness adds:

* ``AGENT.md``  -> appended to the system prompt as <operating_guide>...</operating_guide>
* ``guides/<name>.md`` -> listed by name and first line in a <guides> block, opened on demand
  through a local ``open_guide`` tool that the agent answers itself (it never reaches the
  environment and has no side effects on the database).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT, LLMAgent
from tau2.data_model.message import MultiToolMessage, ToolMessage, UserMessage
from tau2.environment.tool import as_tool
from tau2.utils.llm_utils import generate

AGENT_NAME = "h2x_harness_agent"
GUIDE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
MAX_LOCAL_ROUNDS = 6


@dataclass
class HarnessBundle:
    agent_md: str = ""
    guides: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, root: Path) -> "HarnessBundle":
        root = Path(root)
        agent_path = root / "AGENT.md"
        agent_md = agent_path.read_text(encoding="utf-8").strip() if agent_path.is_file() else ""
        guides = {}
        guide_dir = root / "guides"
        if guide_dir.is_dir():
            for path in sorted(guide_dir.glob("*.md")):
                if GUIDE_NAME.match(path.stem):
                    guides[path.stem] = path.read_text(encoding="utf-8")
        return cls(agent_md=agent_md, guides=guides)

    def description(self, name: str) -> str:
        body = self.guides[name]
        match = re.search(r"^description:\s*(.+)$", body[:1000], re.M)
        if match:
            return match.group(1).strip()[:300]
        for line in body.splitlines():
            line = line.strip().lstrip("#").strip()
            if line and line != "---":
                return line[:300]
        return name

    def system_suffix(self) -> str:
        parts = []
        if self.agent_md:
            parts.append(f"<operating_guide>\n{self.agent_md}\n</operating_guide>")
        if self.guides:
            listing = "\n".join(f"- {n}: {self.description(n)}" for n in self.guides)
            parts.append(
                "<guides>\nOpen a guide with the open_guide tool when it applies to the "
                f"conversation. Available guides:\n{listing}\n</guides>"
            )
        return ("\n" + "\n".join(parts)) if parts else ""


@dataclass
class Recorder:
    """Everything the agent did that the orchestrator's trajectory does not show."""

    usages: list[dict | None] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    returned: int = 0
    policy: str = ""
    system_prompt: str = ""

    def event(self, data: dict) -> None:
        self.events.append({**data, "before_agent_message": self.returned})


def open_guide(name: str) -> str:
    """Open one of your operating guides and read it in full.

    Args:
        name: The guide's name, exactly as listed under <guides> in your instructions.
    """
    raise RuntimeError("open_guide is answered by the agent itself, never by the environment")


class HarnessAgent(LLMAgent):
    def __init__(
        self,
        tools,
        domain_policy: str,
        llm: str,
        llm_args: dict | None = None,
        *,
        harness: HarnessBundle,
        recorder: Recorder,
        additional_stop_tool_names: frozenset[str] = frozenset(),
    ):
        self.harness = harness
        self.recorder = recorder
        local = [as_tool(open_guide)] if harness.guides else []
        self.local_names = frozenset(tool.name for tool in local)
        super().__init__(
            tools=list(tools) + local,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
            additional_stop_tool_names=additional_stop_tool_names,
        )
        recorder.policy = domain_policy
        recorder.system_prompt = self.system_prompt

    @property
    def system_prompt(self) -> str:
        base = SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION
        )
        return base + self.harness.system_suffix()

    def _call(self, state):
        msg = generate(
            model=self.llm,
            tools=self.tools,
            messages=state.system_messages + state.messages,
            call_name="agent_response",
            **self.llm_args,
        )
        self.recorder.usages.append(msg.usage)
        return msg

    def generate_next_message(self, message, state):
        if isinstance(message, UserMessage) and message.is_audio:
            raise ValueError("User message cannot be audio.")
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        msg = self._call(state)
        for _ in range(MAX_LOCAL_ROUNDS):
            calls = msg.tool_calls or []
            local = [c for c in calls if c.name in self.local_names]
            if not local:
                break
            others = [c for c in calls if c.name not in self.local_names]
            if others:
                # A guide cannot be read in the same turn as a system action: run the actions.
                for c in local:
                    self.recorder.event(
                        {"event": "dropped_local_call", "name": (c.arguments or {}).get("name")}
                    )
                msg.tool_calls = others
                break
            state.messages.append(msg)
            for c in local:
                name = str((c.arguments or {}).get("name", "")).strip()
                body = self.harness.guides.get(name)
                self.recorder.event(
                    {"event": "open_guide", "name": name, "found": body is not None,
                     "chars": len(body or "")}
                )
                content = (
                    body
                    if body is not None
                    else f"No guide named {name!r}. Available: {', '.join(self.harness.guides)}"
                )
                state.messages.append(
                    ToolMessage(
                        id=c.id, role="tool", content=content, requestor="assistant",
                        error=body is None,
                    )
                )
            msg = self._call(state)
        else:
            # The model kept opening guides; drop the calls so the turn still reaches the user.
            leftover = [c for c in (msg.tool_calls or []) if c.name not in self.local_names]
            msg.tool_calls = leftover or None
            if not leftover and not (msg.content or "").strip():
                msg.content = "Could you tell me more about what you need?"
            self.recorder.event({"event": "guide_loop_cut"})
        self.recorder.returned += 1
        state.messages.append(msg)
        return msg, state


_STATE: dict = {"bundle": HarnessBundle(), "recorder": Recorder()}


def _factory(tools, domain_policy, **kwargs):
    return HarnessAgent(
        tools,
        domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
        harness=_STATE["bundle"],
        recorder=_STATE["recorder"],
        additional_stop_tool_names=kwargs.get("additional_stop_tool_names", frozenset()),
    )


def register(bundle: HarnessBundle, recorder: Recorder) -> str:
    """Point the (once-registered) factory at this attempt's harness and recorder."""
    from tau2.registry import registry

    _STATE["bundle"], _STATE["recorder"] = bundle, recorder
    if registry.get_agent_factory(AGENT_NAME) is None:
        registry.register_agent_factory(_factory, AGENT_NAME)
    return AGENT_NAME
