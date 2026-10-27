#!/usr/bin/env python3
"""Pre-register the train/test task lists (experiments/<id>/tasks.lock.json).

Run once, before any model call, from local checkouts of the pinned benchmark data; the lock
files are committed so the selection is fixed and auditable. Every selection is a seeded shuffle
of an explicitly filtered pool; nothing here looks at any model's results.

    python scripts/select_tasks.py \
        --tau2 /path/tau2-bench@989d328 --tb2 /path/terminal-bench-2@69671fb \
        --swepro /path/harbor-datasets@c8e8f3f --lab /path/harvey-labs@b58a28a \
        --registry /path/harbor@e77a18b/registry.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261027

PINS = {
    "tau2-bench": "sierra-research/tau2-bench@989d3285ae669e49503324bdb9981bf14fccedeb (v2.0.0)",
    "terminal-bench-2": "laude-institute/terminal-bench-2@69671fbaac6d67a7ef0dfec016cc38a64ef7a77c",
    "swebenchpro": "laude-institute/harbor-datasets@c8e8f3fac7097accaacf261d74c3d6f441de45b1",
    "harvey-labs": "harveyai/harvey-labs@b58a28ae9fde5ee4b810e53b4fd0a1915325a35d",
}


def shuffled(items, tag: str) -> list:
    items = sorted(items)
    random.Random(f"{SEED}:{tag}").shuffle(items)
    return items


def write(exp: str, data: dict) -> None:
    path = ROOT / "experiments" / exp / "tasks.lock.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"{exp}: {len(data['train'])} train, {len(data['test'])} test")


def tau_airline(tau2: Path) -> None:
    split = json.loads((tau2 / "data/tau2/domains/airline/split_tasks.json").read_text())
    train = shuffled(split["train"], "airline")[:3]
    write("e3-tau-airline", {
        "train": sorted(train, key=int),
        "test": sorted(split["test"], key=int),
        "selection": {
            "rule": "train: 3 tasks drawn (seeded shuffle) from tau2's official airline 'train' "
                    "split (30 tasks); test: tau2's complete official airline 'test' split (20)",
            "seed": SEED, "pool": {"train": len(split["train"]), "test": len(split["test"])},
            "source": PINS["tau2-bench"],
        },
    })


def tau_banking(tau2: Path) -> None:
    ids = [json.loads(p.read_text())["id"]
           for p in sorted((tau2 / "data/tau2/domains/banking_knowledge/tasks").glob("task_*.json"))]
    order = shuffled(ids, "banking_knowledge")
    write("e2-tau-banking", {
        "train": sorted(order[:3]),
        "test": sorted(order[3:27]),
        "selection": {
            "rule": "seeded shuffle of all 97 banking_knowledge tasks (no official split exists): "
                    "first 3 -> train, next 24 -> test",
            "seed": SEED, "pool": len(ids), "source": PINS["tau2-bench"],
        },
    })


TB2_EXCLUDE_CATEGORIES = {"model-training", "machine-learning"}
TB2_EXCLUDE_NAMES = re.compile(r"qemu|windows|mteb|hf-model|caffe|reshard|dataset-tokens|video")


def terminal_bench(tb2: Path, registry: dict) -> None:
    pool = []
    for task_toml in sorted(tb2.glob("*/task.toml")):
        t = tomllib.loads(task_toml.read_text())
        name, meta = task_toml.parent.name, t.get("metadata", {})
        env, agent = t.get("environment", {}), t.get("agent", {})
        mem = str(env.get("memory", "2G")).upper().rstrip("G")
        if (meta.get("difficulty") in ("easy", "medium")
                and float(agent.get("timeout_sec", 0)) <= 1200
                and meta.get("category") not in TB2_EXCLUDE_CATEGORIES
                and not TB2_EXCLUDE_NAMES.search(name)
                and float(env.get("cpus", 1)) <= 2 and float(mem or 2) <= 4):
            pool.append(name)
    order = shuffled(pool, "terminal-bench")
    sources = {t["name"]: {"git_url": t["git_url"], "commit": t["git_commit_id"], "path": t["path"]}
               for t in registry["terminal-bench@2.0"]}
    train, test = order[:3], order[3:15]
    write("e4-terminal-bench", {
        "train": sorted(train), "test": sorted(test),
        "sources": {n: sources[n] for n in sorted(train + test)},
        "selection": {
            "rule": "Terminal-Bench 2.0 tasks with difficulty easy|medium, agent timeout <= 1200 s, "
                    "<= 2 CPUs and <= 4 GB, excluding model-training/machine-learning categories and "
                    "tasks dominated by large downloads or VMs; seeded shuffle: 3 train, 12 test",
            "seed": SEED, "pool": len(pool), "pool_tasks": pool, "source": PINS["terminal-bench-2"],
        },
    })


def swe_pro(swepro: Path, registry: dict) -> None:
    pool = []
    for d in sorted((swepro / "datasets/swebenchpro").glob("instance_internetarchive__openlibrary-*")):
        cfg = json.loads((d / "tests/config.json").read_text())
        patch = cfg.get("patch", "")
        changed = sum(1 for line in patch.splitlines()
                      if line[:1] in "+-" and not line.startswith(("+++", "---")))
        files = len(re.findall(r"^diff --git", patch, re.M))
        if changed <= 80 and files <= 3:
            pool.append(d.name)
    order = shuffled(pool, "swebenchpro-openlibrary")
    sources = {t["name"]: {"git_url": t["git_url"], "commit": t["git_commit_id"], "path": t["path"]}
               for t in registry["swebenchpro@1.0"]}
    train, test = order[:3], order[3:13]
    write("e5-swe-pro", {
        "train": sorted(train), "test": sorted(test),
        "sources": {n: sources[n] for n in sorted(train + test)},
        "selection": {
            "rule": "SWE-bench Pro (public, via Harbor) instances from internetarchive/openlibrary "
                    "whose reference patch changes <= 80 lines in <= 3 files (mid-sized issues); "
                    "seeded shuffle: 3 train, 10 test. One repository on purpose: a harness can "
                    "carry repository knowledge from train issues to new ones.",
            "seed": SEED, "pool": len(pool), "source": PINS["swebenchpro"],
        },
    })


def lab(lab_root: Path) -> None:
    fam = "funds-asset-management/draft-lpa"
    scen = sorted(p.name for p in (lab_root / "tasks" / fam).glob("scenario-*"))
    # scenario-k and scenario-(k+11) share a fund and documents: one group, never split.
    groups = {s: f"g{((int(s.split('-')[1]) - 1) % 11) + 1:02d}" for s in scen}
    order = shuffled(sorted(set(groups.values())), "lab-draft-lpa")
    train_groups, test_groups = order[:3], order[3:9]
    pick = {g: f"scenario-{int(g[1:]):02d}" for g in order}  # the first scenario of each group
    def tid(s: str) -> str:
        return "lpa-" + s.split("-")[1]
    train = [tid(pick[g]) for g in train_groups]
    test = [tid(pick[g]) for g in test_groups]
    write("e1-lab-lpa", {
        "train": sorted(train), "test": sorted(test),
        "lab_paths": {tid(s): f"{fam}/{s}" for s in scen if tid(s) in train + test},
        "groups": {tid(s): g for s, g in groups.items()},
        "selection": {
            "rule": "Harvey LAB 'draft-lpa' family (22 scenarios = 11 funds x 2 variants). The two "
                    "variants of a fund form a group and never straddle train/test. Seeded shuffle "
                    "of the 11 groups: 3 -> train, 6 -> test, using each group's first scenario.",
            "seed": SEED, "source": PINS["harvey-labs"],
        },
    })


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tau2", type=Path, required=True)
    p.add_argument("--tb2", type=Path, required=True)
    p.add_argument("--swepro", type=Path, required=True)
    p.add_argument("--lab", type=Path, required=True)
    p.add_argument("--registry", type=Path, required=True, help="Harbor registry.json")
    a = p.parse_args()
    reg = {f"{d['name']}@{d['version']}": d["tasks"] for d in json.loads(a.registry.read_text())}
    tau_airline(a.tau2)
    tau_banking(a.tau2)
    terminal_bench(a.tb2, reg)
    swe_pro(a.swepro, reg)
    lab(a.lab)


if __name__ == "__main__":
    main()
