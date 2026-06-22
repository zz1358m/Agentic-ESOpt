#!/usr/bin/env python
"""Prepare a WebArena non-Lite train/val split for SkillOpt.

The source is VAB's full WebArena config directory. The held-out WebArena-Lite
tasks are removed by matching their ``old_task_id`` values against full
WebArena ``task_id`` values.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def stratum_key(task: dict[str, Any]) -> tuple[str, str]:
    sites = "|".join(str(site) for site in task.get("sites", ["unknown"]))
    eval_block = task.get("eval", {})
    eval_types = eval_block.get("eval_types", ["unknown"])
    return sites, "|".join(str(item) for item in eval_types)


def stratified_shuffle(tasks: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        strata[stratum_key(task)].append(task)
    for bucket in strata.values():
        rng.shuffle(bucket)
    keys = list(strata)
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


def normalize_item(task: dict[str, Any], source_path: Path) -> dict[str, Any]:
    task_id = int(task["task_id"])
    sites = [str(site) for site in task.get("sites", [])]
    return {
        "id": str(task_id),
        "task_id": task_id,
        "intent": task.get("intent", ""),
        "sites": sites,
        "task_type": sites[0] if sites else "unknown",
        "eval_types": task.get("eval", {}).get("eval_types", []),
        "config_path": str(source_path.resolve()),
    }


def site_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter((item.get("sites") or ["unknown"])[0] for item in items))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-dir", default="data/webarena/vab-lite/config_files/wa/test_webarena")
    parser.add_argument("--lite-dir", default="data/webarena/vab-lite/config_files/wa/test_webarena_lite")
    parser.add_argument("--output-dir", default="data/webarena/skillopt_nonlite_sft")
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.0, help="Deprecated; kept for compatibility and must stay 0.")
    args = parser.parse_args()

    full_dir = Path(args.full_dir)
    lite_dir = Path(args.lite_dir)
    output_dir = Path(args.output_dir)
    if not full_dir.is_dir():
        raise FileNotFoundError(full_dir)
    if not lite_dir.is_dir():
        raise FileNotFoundError(lite_dir)
    if not (0.0 <= args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be between 0 and 1.")
    if args.test_ratio != 0.0:
        raise ValueError("--test-ratio is deprecated for this setup. Use 0; official WebArena-Lite 165 is the test set.")

    lite_old_ids = set()
    for path in lite_dir.glob("*.json"):
        task = load_json(path)
        if "old_task_id" in task:
            lite_old_ids.add(int(task["old_task_id"]))

    selected: list[dict[str, Any]] = []
    excluded: list[int] = []
    for path in sorted(full_dir.glob("*.json"), key=lambda p: int(p.stem)):
        task = load_json(path)
        task_id = int(task["task_id"])
        if task_id in lite_old_ids:
            excluded.append(task_id)
            continue
        selected.append(normalize_item(task, path))

    ordered = stratified_shuffle(selected, args.seed)
    val_n = round(len(ordered) * args.val_ratio)
    val_items = ordered[:val_n]
    train_items = ordered[val_n:]
    test_items: list[dict[str, Any]] = []

    write_json(output_dir / "train" / "items.json", train_items)
    write_json(output_dir / "val" / "items.json", val_items)
    write_json(output_dir / "test" / "items.json", test_items)
    manifest = {
        "name": "webarena-nonlite-sft-skillopt",
        "source_full_dir": str(full_dir.resolve()),
        "source_lite_dir": str(lite_dir.resolve()),
        "excluded_lite_old_task_count": len(lite_old_ids),
        "excluded_lite_old_task_ids": sorted(lite_old_ids),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": 0.0,
        "counts": {
            "full": len(list(full_dir.glob("*.json"))),
            "selected_nonlite": len(selected),
            "train": len(train_items),
            "val": len(val_items),
            "test": len(test_items),
        },
        "site_counts": {
            "train": site_counts(train_items),
            "val": site_counts(val_items),
            "test": site_counts(test_items),
        },
        "note": "Non-Lite tasks are split only into train/val. test/items.json is intentionally empty; official WebArena-Lite 165 is the only test set.",
    }
    write_json(output_dir / "split_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
