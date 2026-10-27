#!/usr/bin/env python3
"""GitHub Actions helpers: report test results and step output as annotations.

Annotations appear on the run's summary page and are served by the ordinary checks API
(``gh api repos/<owner>/<repo>/check-runs/<job id>/annotations``), so a failure can be read
without downloading the full job log.

  python scripts/gha.py tests [unittest args]                  e.g. tests discover -s tests -t .
  python scripts/gha.py annotate --title T --outcome O LOGFILE  tail of a step log
"""

from __future__ import annotations

import argparse
import re
import sys
import unittest
from collections import deque
from pathlib import Path

MAX_CHARS = 8000  # per annotation (GitHub allows far more; keeps the run page readable)
MAX_ERRORS = 9  # GitHub keeps at most 10 error annotations per step
ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
KEYS = re.compile(r"sk-ant-[A-Za-z0-9_-]{6,}")


def _data(s: str) -> str:
    return s.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _prop(s: str) -> str:
    return _data(s).replace(":", "%3A").replace(",", "%2C")


def clean(text: str) -> str:
    text = KEYS.sub("sk-ant-***", ANSI.sub("", text))
    # progress bars redraw a line with \r: keep what was last drawn
    return "\n".join(line.rsplit("\r", 1)[-1] for line in text.split("\n"))


def annotate(level: str, title: str, message: str) -> None:
    message = clean(message).strip() or "(no output)"
    if len(message) > MAX_CHARS:
        message = "...\n" + message[-MAX_CHARS:]
    print(f"::{level} title={_prop(clean(title))}::{_data(message)}", flush=True)


class _Result(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed: list[str] = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.passed.append(test.id())


def run_tests(argv: list[str]) -> int:
    runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=2, resultclass=_Result)
    prog = unittest.main(module=None, argv=["unittest", *argv], testRunner=runner, exit=False)
    r = prog.result
    problems = [("FAIL", t, tb) for t, tb in r.failures] + [("ERROR", t, tb) for t, tb in r.errors]
    summary = (f"{r.testsRun} run, {len(r.passed)} passed, {len(r.skipped)} skipped, "
               f"{len(r.failures)} failed, {len(r.errors)} errors")
    lines = [f"ok       {t}" for t in r.passed]
    lines += [f"skipped  {t.id()} ({why})" for t, why in r.skipped]
    lines += [f"{kind:<8} {t.id()}" for kind, t, _ in problems]
    ok = r.wasSuccessful() and r.testsRun > 0
    annotate("notice" if ok else "error", f"tests: {summary}", "\n".join(lines))
    for kind, test, tb in problems[: MAX_ERRORS - 1]:
        annotate("error", f"{kind} {test.id()}", tb)
    return 0 if ok else 1


def annotate_log(title: str, outcome: str, log: Path, n_lines: int) -> int:
    try:
        raw = log.read_bytes().decode("utf-8", errors="replace")  # keep \r (progress redraws)
        tail = deque(clean(raw).split("\n"), maxlen=n_lines)
        text = "\n".join(tail)
    except OSError:
        text = f"(no log at {log})"
    level = "error" if outcome in ("failure", "cancelled") else "notice"
    annotate(level, f"{title}: {outcome}", text)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "tests":
        return run_tests(argv[1:])
    ap = argparse.ArgumentParser(prog="gha.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("annotate")
    a.add_argument("log", type=Path)
    a.add_argument("--title", required=True)
    a.add_argument("--outcome", default="failure")
    a.add_argument("--lines", type=int, default=80)
    args = ap.parse_args(argv)
    return annotate_log(args.title, args.outcome, args.log, args.lines)


if __name__ == "__main__":
    sys.exit(main())
