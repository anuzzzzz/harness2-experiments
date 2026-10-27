"""Grade one LAB deliverable set against its rubric -- the Harness2 ``check`` command.

Follows harvey-labs evaluation/scoring.py exactly where it matters: the same judge prompt
(evaluation/prompts/rubric_criterion.txt), the same document extraction (pandoc for .docx,
etc.), the same per-criterion deliverable scoping and filename matching, one pass/fail verdict
per criterion. Deviations, all for cost and model compatibility:

* the judge model is configurable (default Claude Haiku 5.5, not Sonnet 4.6) and is called
  without ``temperature``, which Claude 5.5 models reject;
* the prompt prefix shared by a deliverable's criteria (task + agent output) is prompt-cached;
* LAB's LLM fallback for unmatched deliverable filenames is disabled (an unmatched file is
  graded as missing);
* the cost of every judge call is recorded.

Prints {"n_passed": .., "n_criteria": ..} as its last line (Harness2's partial-credit format).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import pricing

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "fail"]},
        "reasoning": {"type": "string"},
    },
    "required": ["verdict", "reasoning"],
    "additionalProperties": False,
}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="h2x lab-grade")
    p.add_argument("--task-json", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--judge-model", default=os.environ.get("H2X_LAB_JUDGE", "claude-haiku-5-5"))
    p.add_argument("--lab-root", type=Path, default=Path(os.environ.get("H2X_LAB_ROOT", "vendor/harvey-labs")))
    p.add_argument("--cost-file", type=Path, default=None)
    p.add_argument("--details", type=Path, default=None)
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=8192)
    return p


def parse_json(text: str) -> dict:
    """harvey-labs Judge._parse_json, verbatim in behaviour."""
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    for i, ch in enumerate(text):
        if ch == "{":
            depth = 0
            for j in range(i, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[i : j + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"No JSON found in judge response: {text[:200]}")


class Judge:
    def __init__(self, model: str, max_tokens: int):
        import anthropic

        self.anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=4)
        self.model = model
        self.max_tokens = max_tokens
        self.cost = 0.0
        self.calls = 0
        self._lock = threading.Lock()

    def verdict(self, prefix: str, suffix: str) -> dict:
        content = [
            {"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": suffix},
        ]
        last: Exception | None = None
        for attempt in range(3):
            kwargs = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": content}],
            }
            if attempt == 0:
                kwargs["output_config"] = {"format": {"type": "json_schema", "schema": VERDICT_SCHEMA}}
            try:
                response = self.client.messages.create(**kwargs)
            except self.anthropic.BadRequestError as exc:
                last = exc  # e.g. structured output unsupported: retry without it
                continue
            except self.anthropic.InternalServerError as exc:
                last = exc
                continue
            with self._lock:
                self.calls += 1
                self.cost += pricing.anthropic_usage_cost(self.model, response.usage)
            text = "".join(
                getattr(block, "text", "") for block in response.content if getattr(block, "type", "") == "text"
            )
            try:
                return parse_json(text)
            except ValueError as exc:
                last = exc
        raise RuntimeError(f"judge failed after 3 attempts: {last}")


def _fake(args) -> int:
    task = json.loads(args.task_json.read_text(encoding="utf-8"))
    n = len(task["criteria"])
    files = sorted(p.name for p in args.output_dir.rglob("*") if p.is_file()) if args.output_dir.is_dir() else []
    seed = int(hashlib.sha1("|".join(files).encode()).hexdigest(), 16)
    passed = min(n, (seed % (n + 1)) if files else 0)
    if args.cost_file:
        args.cost_file.write_text(json.dumps({"judge_cost_usd": 0.0, "calls": 0, "fake": True}))
    print(json.dumps({"n_passed": passed, "n_criteria": n, "notes": "fake grade"}))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if os.environ.get("H2X_FAKE", "").strip() not in ("", "0", "false"):
        return _fake(args)
    lab_root = args.lab_root.resolve()
    sys.path.insert(0, str(lab_root))
    import evaluation.scoring as scoring  # harvey-labs, pinned checkout

    scoring._llm_match_deliverables = lambda *a, **k: {}  # no ungraded-cost LLM fallback
    template = (lab_root / "evaluation" / "prompts" / "rubric_criterion.txt").read_text(encoding="utf-8")
    task = json.loads(args.task_json.read_text(encoding="utf-8"))
    criteria, task_desc = task["criteria"], task["title"]
    output_dir = args.output_dir

    filenames = {d for c in criteria for d in c.get("deliverables", [])}
    deliverables_map = {f: f for f in filenames} if filenames else None
    if deliverables_map and output_dir.exists():
        actual = [f.name for f in output_dir.rglob("*") if f.is_file()]
        resolved = scoring._match_deliverables(deliverables_map, actual, output_dir=output_dir)
    else:
        resolved = None

    text_cache: dict[tuple[str, str], str] = {}
    cache_lock = threading.Lock()

    def read(path: Path, mode) -> str:
        key = (str(path), str(mode))
        with cache_lock:
            if key in text_cache:
                return text_cache[key]
        value = scoring._read_file_as_text(path, track_changes=mode)
        with cache_lock:
            text_cache[key] = value
        return value

    full_output = None
    if any(not (c.get("deliverables") and resolved) for c in criteria):
        full_output = scoring._load_all_output(output_dir)

    def agent_output(criterion: dict) -> tuple[str, bool]:
        """(text the judge sees, whether any relevant deliverable exists at all)."""
        names = criterion.get("deliverables", [])
        if not (names and resolved):
            return full_output, full_output != "(No agent output found)"
        sections, found = [], False
        for name in names:
            filename = resolved[name]
            path = output_dir / filename
            if not path.exists():
                sections.append(f"## Agent Output: {name}\n(File not found: {filename})")
                continue
            found = True
            redlines = criterion.get("evaluation_options", {}).get("include_docx_redlines", False)
            mode = scoring.DocxTrackChanges.ALL if redlines else scoring.DocxTrackChanges.ACCEPT
            sections.append(f"## Agent Output: {name}\n{read(path, mode)}")
        return ("\n\n".join(sections) if sections else "(No agent output found)"), found

    def split(criterion: dict) -> tuple[str, str, bool]:
        output, found = agent_output(criterion)
        prompt = template.format(
            task_description=task_desc,
            agent_output=output,
            criterion_title=criterion["title"],
            match_criteria=criterion["match_criteria"],
        )
        cut = prompt.rfind("## Criterion")
        prefix, suffix = (prompt[:cut], prompt[cut:]) if cut > 0 else (prompt, "")
        return prefix, suffix, found

    judge = Judge(args.judge_model, args.max_tokens)
    prepared = [(c, *split(c)) for c in criteria]
    results: dict[str, dict] = {}

    def grade(item) -> None:
        criterion, prefix, suffix, found = item
        if not found:
            # Nothing to judge: LAB's judge sees only "(File not found ...)" and fails it.
            results[criterion["id"]] = {"id": criterion["id"], "title": criterion["title"],
                                        "verdict": "fail", "reasoning": "deliverable missing"}
            return
        try:
            out = judge.verdict(prefix, suffix)
            verdict = str(out.get("verdict", "fail")).lower()
            reasoning = str(out.get("reasoning", ""))
        except Exception as exc:  # one failed criterion must not sink the whole grade
            verdict, reasoning = "error", f"{type(exc).__name__}: {exc}"
        results[criterion["id"]] = {"id": criterion["id"], "title": criterion["title"],
                                    "verdict": verdict, "reasoning": reasoning}

    # Warm the cache once per distinct prefix, then grade the rest concurrently.
    seen, first, rest = set(), [], []
    for item in prepared:
        key = hashlib.sha1(item[1].encode()).hexdigest()
        (rest if key in seen else first).append(item)
        seen.add(key)
    for item in first[:1]:
        grade(item)
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        list(pool.map(grade, first[1:] + rest))

    errors = [r for r in results.values() if r["verdict"] == "error"]
    n_criteria = len(criteria)
    n_passed = sum(1 for r in results.values() if r["verdict"] == "pass")
    if args.cost_file:
        args.cost_file.write_text(
            json.dumps({"judge_cost_usd": round(judge.cost, 6), "calls": judge.calls,
                        "judge_model": args.judge_model, "errors": len(errors)}),
            encoding="utf-8",
        )
    if args.details:
        args.details.write_text(json.dumps({"criteria": list(results.values())}, indent=2), encoding="utf-8")
    if len(errors) > max(2, n_criteria // 10):
        print(f"too many judge errors ({len(errors)}/{n_criteria}); not grading", file=sys.stderr)
        return 1
    print(json.dumps({"n_passed": n_passed, "n_criteria": n_criteria,
                      "notes": f"{n_passed}/{n_criteria} criteria; {len(errors)} judge errors counted as fail"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
