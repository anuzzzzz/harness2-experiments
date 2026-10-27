"""Turn pre-registered Harvey LAB tasks into Harness2 files-workspace tasks.

    .harness2/tasks/<id>/task.md            LAB instructions + where the documents are
    .harness2/tasks/<id>/task.toml          split, workspace=files, check=h2x lab-grade ...
    .harness2/tasks/<id>/files/documents/   the matter documents (what the agent works from)
    .harness2/tasks/<id>/check/task.json    the full LAB task incl. rubric (never shown)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..config import Experiment

CHECK = (
    "h2x lab-grade --task-json check/task.json --output-dir output "
    "--judge-model {judge} "
    '--cost-file "$HARNESS2_RUN_DIR/.h2x_judge.json" '
    '--details "$HARNESS2_RUN_DIR/.h2x_lab_scores.json"'
)


def import_tasks(exp: Experiment, lab_root: Path, *, force: bool = False) -> list[str]:
    lab_root = Path(lab_root)
    tasks_dir = exp.project / ".harness2" / "tasks"
    paths = exp.tasks.get("lab_paths") or {}
    cfg = exp.section("lab")
    judge = exp.model("judge")
    done = []
    for split in ("train", "test"):
        for task_id in exp.split(split):
            src = lab_root / "tasks" / paths[task_id]
            config = json.loads((src / "task.json").read_text(encoding="utf-8"))
            dest = tasks_dir / task_id
            if dest.exists():
                if not force:
                    done.append(task_id)
                    continue
                shutil.rmtree(dest)
            (dest / "files").mkdir(parents=True)
            shutil.copytree(src / "documents", dest / "files" / "documents")
            (dest / "check").mkdir()
            shutil.copyfile(src / "task.json", dest / "check" / "task.json")
            prompt = config["instructions"].rstrip() + (
                "\n\nThe attached documents are in the documents/ directory."
            )
            (dest / "task.md").write_text(prompt + "\n", encoding="utf-8")
            values = {
                "domain": cfg.get("domain", "practice"),
                "split": split,
                "workspace": "files",
                "deliverable": "files",
                "check": CHECK.format(judge=judge),
                "timeout": int(cfg.get("timeout_s", 2400)),
            }
            (dest / "task.toml").write_text(
                "".join(f"{k} = {json.dumps(v)}\n" for k, v in values.items()), encoding="utf-8"
            )
            done.append(task_id)
    return done


def expected_criteria(exp: Experiment, task_id: str) -> int | None:
    path = exp.project / ".harness2" / "tasks" / task_id / "check" / "task.json"
    try:
        return len(json.loads(path.read_text(encoding="utf-8"))["criteria"])
    except (OSError, ValueError, KeyError):
        return None
