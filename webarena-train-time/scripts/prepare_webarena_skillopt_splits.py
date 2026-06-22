#!/usr/bin/env python
"""Prepare WebArena-style task splits for skill optimization and model ES.

The script separates agent-visible task fields from evaluator-only fields and
drops oracle trajectory fields, so WebArena-Lite training tasks can be used as
black-box interaction tasks without leaking demonstrations.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = "data/webarena/source/config_files/test.raw.json"
DEFAULT_OUTPUT = "data/webarena/legacy_skillopt_splits"
DEFAULT_SPLITS = "es_train_tiny=24,es_train=96,es_dev=96"

PUBLIC_KEYS = {
    "sites",
    "task_id",
    "require_login",
    "storage_state",
    "start_url",
    "geolocation",
    "intent_template",
    "instantiation_dict",
    "intent",
    "require_reset",
    "intent_template_id",
}

PRIVATE_KEYS = {
    "eval",
    "reward",
    "reward_fn",
    "reward_function",
    "reference_answers",
    "reference_url",
    "program_html",
    "string_note",
    "reference_answer_raw_annotation",
}

ORACLE_KEY_FRAGMENTS = (
    "trajectory",
    "trajectories",
    "demonstration",
    "demonstrations",
    "expert",
    "oracle",
    "action_trace",
    "action_history",
    "gold_action",
    "gold_actions",
)


def parse_split_sizes(raw: str) -> list[tuple[str, int]]:
    splits: list[tuple[str, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid split spec {item!r}; expected name=size.")
        name, size = item.split("=", 1)
        name = name.strip()
        size_int = int(size.strip())
        if not name:
            raise ValueError("Split name cannot be empty.")
        if size_int < 0:
            raise ValueError(f"Split size for {name!r} must be non-negative.")
        splits.append((name, size_int))
    if not splits:
        raise ValueError("At least one split must be requested.")
    return splits


def load_tasks(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "tasks" in data and isinstance(data["tasks"], list):
            data = data["tasks"]
        else:
            data = list(data.values())
    if not isinstance(data, list):
        raise ValueError(f"Expected a list or dict of tasks in {path}.")
    tasks = []
    for index, task in enumerate(data):
        if not isinstance(task, dict):
            raise ValueError(f"Task at index {index} is not an object.")
        copied = dict(task)
        copied.setdefault("task_id", index)
        tasks.append(copied)
    return tasks


def has_oracle_name(key: str) -> bool:
    lowered = key.lower()
    return any(fragment in lowered for fragment in ORACLE_KEY_FRAGMENTS)


def strip_oracle_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_oracle_fields(item)
            for key, item in value.items()
            if not has_oracle_name(str(key))
        }
    if isinstance(value, list):
        return [strip_oracle_fields(item) for item in value]
    return value


def public_task(task: dict[str, Any]) -> dict[str, Any]:
    return {
        key: strip_oracle_fields(task[key])
        for key in PUBLIC_KEYS
        if key in task and not has_oracle_name(key)
    }


def private_task(task: dict[str, Any]) -> dict[str, Any]:
    private = public_task(task)
    for key in PRIVATE_KEYS:
        if key in task and not has_oracle_name(key):
            private[key] = strip_oracle_fields(task[key])
    return private


def full_evaluator_config(task: dict[str, Any]) -> dict[str, Any]:
    cleaned = strip_oracle_fields(task)
    for key in list(cleaned):
        if has_oracle_name(key):
            cleaned.pop(key, None)
    return cleaned


def stratum_key(task: dict[str, Any]) -> tuple[str, str, str]:
    sites = "|".join(str(site) for site in task.get("sites", ["unknown"]))
    eval_block = task.get("eval", {})
    eval_types = eval_block.get("eval_types", task.get("eval_types", ["unknown"]))
    eval_key = "|".join(str(item) for item in eval_types)
    template_id = task.get("intent_template_id", task.get("intent_template", "unknown"))
    return sites, eval_key, str(template_id)


def stratified_order(tasks: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        strata[stratum_key(task)].append(task)
    for bucket in strata.values():
        rng.shuffle(bucket)

    keys = list(strata.keys())
    rng.shuffle(keys)
    ordered: list[dict[str, Any]] = []
    while keys:
        next_keys = []
        for key in keys:
            bucket = strata[key]
            if bucket:
                ordered.append(bucket.pop())
            if bucket:
                next_keys.append(key)
        keys = next_keys
        rng.shuffle(keys)
    return ordered


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=True)
        f.write("\n")


def prepare_splits(
    *,
    source: Path,
    output_dir: Path,
    splits: list[tuple[str, int]],
    seed: int,
    include_heldout: bool,
) -> dict[str, Any]:
    tasks = load_tasks(source)
    ordered = stratified_order(tasks, seed)
    offset = 0
    manifest_splits = []

    for name, size in splits:
        selected = ordered[offset : offset + size]
        offset += size
        write_split(output_dir, name, selected)
        manifest_splits.append(split_manifest(name, selected))

    if include_heldout and offset < len(ordered):
        selected = ordered[offset:]
        write_split(output_dir, "heldout", selected)
        manifest_splits.append(split_manifest("heldout", selected))

    manifest = {
        "source": str(source),
        "seed": seed,
        "total_tasks": len(tasks),
        "oracle_fields_dropped_if_name_contains": list(ORACLE_KEY_FRAGMENTS),
        "agent_visible_dir": "public_tasks",
        "evaluator_private_dir": "private_eval",
        "private_webarena_config_dir": "private_webarena_configs",
        "splits": manifest_splits,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def split_manifest(name: str, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "name": name,
        "size": len(tasks),
        "task_ids": [task.get("task_id") for task in tasks],
    }


def write_split(output_dir: Path, name: str, tasks: list[dict[str, Any]]) -> None:
    public = [public_task(task) for task in tasks]
    private = [private_task(task) for task in tasks]
    full_configs = [full_evaluator_config(task) for task in tasks]
    task_ids = [str(task.get("task_id")) for task in tasks]

    write_json(output_dir / "public_tasks" / f"{name}.json", public)
    write_json(output_dir / "private_eval" / f"{name}.json", private)
    write_json(output_dir / "private_webarena_configs" / name / "tasks.json", full_configs)

    ids_path = output_dir / "task_ids" / f"{name}.txt"
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    ids_path.write_text("\n".join(task_ids) + ("\n" if task_ids else ""), encoding="utf-8")

    config_dir = output_dir / "private_webarena_configs" / name / "config_files"
    config_dir.mkdir(parents=True, exist_ok=True)
    for index, task in enumerate(full_configs):
        task_id = task.get("task_id", index)
        write_json(config_dir / f"{task_id}.json", task)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="WebArena-style JSON task source.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT, help="Directory for generated splits.")
    parser.add_argument("--splits", default=DEFAULT_SPLITS, help="Comma-separated name=size list.")
    parser.add_argument("--seed", type=int, default=20240604, help="Random seed for split sampling.")
    parser.add_argument(
        "--no-heldout",
        action="store_true",
        help="Do not write a heldout split with remaining tasks.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = prepare_splits(
        source=Path(args.source),
        output_dir=Path(args.output_dir),
        splits=parse_split_sizes(args.splits),
        seed=args.seed,
        include_heldout=not args.no_heldout,
    )
    print(json.dumps({"output_dir": args.output_dir, "splits": manifest["splits"]}, indent=2))


if __name__ == "__main__":
    main()
