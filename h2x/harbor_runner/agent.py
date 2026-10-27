"""Harbor's Claude Code agent with an h2x harness installed in the task's working directory.

The harness is Claude Code's native project files (CLAUDE.md, .claude/skills, .claude/agents,
.claude/scripts, hooks in .claude/settings.json), exported by the h2x adapter into a local
directory named by $H2X_HARNESS_DIR. Before Claude Code starts, the files are copied into the
container's working directory (a pre-existing CLAUDE.md/.claude is set aside). When Claude Code
exits -- normally, by timeout or by error -- the repository diff is saved to
/logs/agent/h2x-diff.patch, then the harness is removed and the originals restored, all before
Harbor runs the task's tests. With an empty harness this agent is Harbor's ``claude-code``.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path

from harbor.agents.installed.claude_code import ClaudeCode

HARNESS_ENV = "H2X_HARNESS_DIR"
STAGE = "/tmp/h2x-harness"
BACKUP = "/tmp/h2x-backup"
WORKDIR_FILE = "/tmp/h2x-workdir"

INSTALL = f"""
set -e
WD="$(pwd)"
printf '%s' "$WD" > {WORKDIR_FILE}
mkdir -p {BACKUP}
for p in CLAUDE.md .claude; do
  if [ -e "$WD/$p" ]; then rm -rf "{BACKUP}/$p"; mv "$WD/$p" "{BACKUP}/$p"; fi
done
cp -a {STAGE}/. "$WD/"
chmod -R a+rX "$WD/CLAUDE.md" "$WD/.claude" 2>/dev/null || true
find "$WD/.claude" -name '*.py' -exec chmod a+x {{}} + 2>/dev/null || true
"""

CAPTURE = f"""
WD="$(cat {WORKDIR_FILE} 2>/dev/null || pwd)"
cd "$WD" || exit 0
mkdir -p /logs/agent
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  {{
    git -c core.quotepath=off diff HEAD -- . ':(exclude)CLAUDE.md' ':(exclude).claude' 2>/dev/null
    git -c core.quotepath=off ls-files --others --exclude-standard -z -- . \
        ':(exclude)CLAUDE.md' ':(exclude).claude' 2>/dev/null \
      | while IFS= read -r -d '' f; do git diff --no-index -- /dev/null "$f" 2>/dev/null; done
  }} > /logs/agent/h2x-diff.patch
fi
true
"""

REMOVE = f"""
WD="$(cat {WORKDIR_FILE} 2>/dev/null)"
[ -n "$WD" ] || exit 0
rm -rf "$WD/CLAUDE.md" "$WD/.claude"
for p in CLAUDE.md .claude; do
  if [ -e "{BACKUP}/$p" ]; then mv "{BACKUP}/$p" "$WD/$p"; fi
done
rm -rf {STAGE} {WORKDIR_FILE}
true
"""


class HarnessClaudeCode(ClaudeCode):
    """``harbor run --agent h2x.harbor_runner.agent:HarnessClaudeCode``"""

    def __init__(self, logs_dir: Path, *args, **kwargs):
        raw = os.environ.get(HARNESS_ENV, "").strip()
        self._h2x_harness = Path(raw) if raw else None
        self._h2x_installed = False
        super().__init__(logs_dir, *args, **kwargs)

    def _has_harness(self) -> bool:
        return bool(
            self._h2x_harness
            and self._h2x_harness.is_dir()
            and any(p.is_file() and p.stat().st_size for p in self._h2x_harness.rglob("*"))
        )

    async def _bash(self, environment, script: str) -> None:
        await self.exec_as_root(environment, command=f"bash -c {shlex.quote(script)}")

    async def configure(self, environment) -> None:
        await super().configure(environment)
        if not self._has_harness():
            return
        await self.exec_as_root(environment, command=f"rm -rf {STAGE} && mkdir -p {STAGE}")
        await environment.upload_dir(self._h2x_harness, STAGE)
        await self._bash(environment, INSTALL)
        self._h2x_installed = True

    async def run(self, instruction, environment, context) -> None:
        try:
            await super().run(instruction, environment, context)
        finally:
            for script, needed in ((CAPTURE, True), (REMOVE, self._h2x_installed)):
                if not needed:
                    continue
                try:
                    await self._bash(environment, script)
                except Exception as exc:  # never mask the agent's own outcome
                    self.logger.warning(f"h2x: post-run step failed: {exc}")
