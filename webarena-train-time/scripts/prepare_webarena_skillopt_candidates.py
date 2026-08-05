#!/usr/bin/env python
"""Create an initial WebArena SkillOpt candidate pool and run manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SPLIT_ROOT = "data/webarena/legacy_skillopt_splits"
DEFAULT_OUTPUT = "runs/webarena_skillopt/skillopt_seed"


DEFAULT_SKILLS = [
    {
        "name": "empty_skill",
        "description": "No additional skill text. This is the control arm.",
        "skill": "",
    },
    {
        "name": "observe_plan_act",
        "description": "Generic web-task discipline for observation, planning, and action.",
        "skill": "\n".join(
            [
                "Before each action, identify the current page, the user's target, and the smallest next action that changes state.",
                "Prefer visible navigation, search, filters, table rows, and form fields over guessing hidden URLs.",
                "If the page content is insufficient, use one exploratory click or search before answering.",
                "Stop only after the target is satisfied or the requested information is explicitly visible.",
            ]
        ),
    },
    {
        "name": "evidence_first_answer",
        "description": "Useful for string-match and information extraction tasks.",
        "skill": "\n".join(
            [
                "For answer tasks, collect the exact visible evidence before producing the final answer.",
                "Preserve product names, user names, issue titles, dates, counts, and addresses exactly as shown.",
                "For top-k, comparison, and count tasks, verify the ordering or count on the relevant page before stopping.",
                "Do not answer from memory when the website can be searched or filtered.",
            ]
        ),
    },
    {
        "name": "navigation_recovery",
        "description": "Recovery skill for long-horizon browsing and wrong-page states.",
        "skill": "\n".join(
            [
                "When stuck, use breadcrumbs, site search, browser back, or the homepage to return to a known state.",
                "If a click opens an irrelevant page, recover immediately instead of continuing deeper.",
                "For multi-site tasks, finish the information gathering on one site before switching to the next.",
                "Keep track of completed subgoals so later actions do not undo earlier progress.",
            ]
        ),
    },
    {
        "name": "form_state_safety",
        "description": "Conservative interaction skill for tasks that edit carts, issues, accounts, or admin state.",
        "skill": "\n".join(
            [
                "For forms and state-changing tasks, inspect required fields before submitting.",
                "Use exact values from the instruction for names, quantities, assignees, labels, dates, and options.",
                "After submitting, verify that the target state changed on the confirmation page or updated object.",
                "Avoid extra destructive actions beyond the requested task.",
            ]
        ),
    },
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=True)
        f.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def validate_split_root(split_root: Path, train_split: str, dev_split: str) -> None:
    required = [
        split_root / "public_tasks" / f"{train_split}.json",
        split_root / "private_eval" / f"{train_split}.json",
        split_root / "public_tasks" / f"{dev_split}.json",
        split_root / "private_eval" / f"{dev_split}.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing split files: " + ", ".join(missing))


def render_skill_card(candidate: dict[str, str]) -> str:
    if not candidate["skill"]:
        return "# empty_skill\n\nNo additional skill text.\n"
    return "\n".join(
        [
            f"# {candidate['name']}",
            "",
            candidate["description"],
            "",
            "## Skill",
            "",
            candidate["skill"],
            "",
        ]
    )


def prepare_candidates(
    *,
    split_root: Path,
    output_dir: Path,
    train_split: str,
    dev_split: str,
) -> dict[str, Any]:
    validate_split_root(split_root, train_split, dev_split)
    train_public = load_json(split_root / "public_tasks" / f"{train_split}.json")
    dev_public = load_json(split_root / "public_tasks" / f"{dev_split}.json")

    candidates = []
    for index, candidate in enumerate(DEFAULT_SKILLS):
        candidate_id = f"{index:02d}_{candidate['name']}"
        skill_path = output_dir / "candidates" / f"{candidate_id}.md"
        write_text(skill_path, render_skill_card(candidate))
        candidates.append(
            {
                "id": candidate_id,
                "name": candidate["name"],
                "description": candidate["description"],
                "skill_path": str(skill_path),
                "agent_visible": candidate["skill"],
            }
        )

    manifest = {
        "split_root": str(split_root),
        "train_split": train_split,
        "dev_split": dev_split,
        "train_public_tasks": str(split_root / "public_tasks" / f"{train_split}.json"),
        "train_private_eval": str(split_root / "private_eval" / f"{train_split}.json"),
        "dev_public_tasks": str(split_root / "public_tasks" / f"{dev_split}.json"),
        "dev_private_eval": str(split_root / "private_eval" / f"{dev_split}.json"),
        "train_task_count": len(train_public),
        "dev_task_count": len(dev_public),
        "selection_metric": "site_balanced_success_rate_then_average_success_rate",
        "leakage_rule": "Only candidate.agent_visible and public task fields may enter prompts.",
        "candidates": candidates,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--train-split", default="es_train_tiny")
    parser.add_argument("--dev-split", default="es_dev")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = prepare_candidates(
        split_root=Path(args.split_root),
        output_dir=Path(args.output_dir),
        train_split=args.train_split,
        dev_split=args.dev_split,
    )
    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "train_task_count": manifest["train_task_count"],
                "dev_task_count": manifest["dev_task_count"],
                "candidates": [candidate["id"] for candidate in manifest["candidates"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
