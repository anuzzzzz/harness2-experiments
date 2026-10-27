"""Fetch pre-registered Harbor task directories (pinned git commits) into experiments/<exp>/.cache."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from .config import Experiment


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, stdout=subprocess.DEVNULL)


def fetch_harbor_tasks(exp: Experiment, *, force: bool = False) -> list[str]:
    sources = exp.tasks.get("sources") or {}
    dest_root = exp.root / ".cache" / "tasks"
    wanted = [t for t in exp.split("all") if force or not (dest_root / t / "task.toml").is_file()]
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for task in wanted:
        src = sources[task]
        groups[(src["git_url"], src["commit"])].append(task)
    for (url, commit), tasks in groups.items():
        repo = exp.root / ".cache" / "repos" / hashlib.sha1(f"{url}@{commit}".encode()).hexdigest()[:12]
        shutil.rmtree(repo, ignore_errors=True)
        repo.mkdir(parents=True)
        _git("init", "-q", cwd=repo)
        _git("remote", "add", "origin", url, cwd=repo)
        _git("config", "core.sparseCheckout", "true", cwd=repo)
        patterns = "".join(f"/{sources[t]['path'].strip('/')}/\n" for t in tasks)
        (repo / ".git" / "info" / "sparse-checkout").write_text(patterns)
        _git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit, cwd=repo)
        _git("checkout", "-q", "FETCH_HEAD", cwd=repo)
        for task in tasks:
            target = dest_root / task
            shutil.rmtree(target, ignore_errors=True)
            shutil.copytree(repo / sources[task]["path"].strip("/"), target)
        shutil.rmtree(repo, ignore_errors=True)
    return wanted
