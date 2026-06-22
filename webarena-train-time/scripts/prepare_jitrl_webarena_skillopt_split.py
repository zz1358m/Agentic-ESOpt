#!/usr/bin/env python
"""Prepare JitRL-aligned WebArena train/val/test splits.

JitRL WebArena-Lite is the first 165 tasks from the WebArena task file
(`task_id` 0..164). Training uses the remaining WebArena tasks from the same
source (`task_id` 165..811), with an optional validation split.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SITE_COLUMNS = {
    "shopping_admin": "Admin",
    "gitlab": "GitLab",
    "map": "Map",
    "reddit": "Reddit",
    "shopping": "Shopping",
}


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
    first_site = sites[0] if sites else "unknown"
    return {
        "id": str(task_id),
        "task_id": task_id,
        "intent": task.get("intent", ""),
        "sites": sites,
        "task_type": first_site,
        "jitrl_category": SITE_COLUMNS.get(first_site, "Other"),
        "eval_types": task.get("eval", {}).get("eval_types", []),
        "config_path": str(source_path.resolve()),
    }


def site_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter((item.get("sites") or ["unknown"])[0] for item in items))


def included_site_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for item in items:
        counter.update(str(site) for site in item.get("sites", []))
    return dict(counter)


def load_tasks(config_dir: Path) -> list[dict[str, Any]]:
    tasks = []
    for path in sorted(config_dir.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 10**9):
        if not path.stem.isdigit():
            continue
        task = load_json(path)
        task["task_id"] = int(task.get("task_id", path.stem))
        tasks.append(normalize_item(task, path))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="data/webarena/jitrl/config_files")
    parser.add_argument("--output-dir", default="data/webarena/jitrl_skillopt_splits")
    parser.add_argument("--lite-start", type=int, default=0)
    parser.add_argument("--lite-end", type=int, default=164)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    if not (0.0 <= args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be between 0 and 1.")

    all_items = load_tasks(source_dir)
    by_id = {int(item["task_id"]): item for item in all_items}
    lite_ids = set(range(args.lite_start, args.lite_end + 1))
    missing_lite = sorted(lite_ids - set(by_id))
    if missing_lite:
        raise RuntimeError(f"Missing JitRL lite task ids from {source_dir}: {missing_lite}")

    test_items = [by_id[task_id] for task_id in sorted(lite_ids)]
    train_val_items = [item for item in all_items if int(item["task_id"]) not in lite_ids]
    ordered = stratified_shuffle(train_val_items, args.seed)
    val_n = round(len(ordered) * args.val_ratio)
    val_items = ordered[:val_n]
    train_items = ordered[val_n:]

    write_json(output_dir / "train" / "items.json", train_items)
    write_json(output_dir / "val" / "items.json", val_items)
    write_json(output_dir / "train_val" / "items.json", train_val_items)
    write_json(output_dir / "test" / "items.json", test_items)

    manifest = {
        "name": "jitrl-webarena-skillopt-splits",
        "source_dir": str(source_dir.resolve()),
        "definition": {
            "test": f"JitRL WebArena-Lite task_id {args.lite_start}..{args.lite_end}",
            "train_val": "All WebArena tasks from the same source excluding JitRL WebArena-Lite ids.",
        },
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "counts": {
            "full": len(all_items),
            "test": len(test_items),
            "train_val": len(train_val_items),
            "train": len(train_items),
            "val": len(val_items),
        },
        "task_id_ranges": {
            "test": [args.lite_start, args.lite_end],
            "train_val_minmax": [
                min(int(item["task_id"]) for item in train_val_items),
                max(int(item["task_id"]) for item in train_val_items),
            ],
        },
        "site_counts": {
            "test": site_counts(test_items),
            "train_val": site_counts(train_val_items),
            "train": site_counts(train_items),
            "val": site_counts(val_items),
        },
        "included_site_counts": {
            "test": included_site_counts(test_items),
            "train_val": included_site_counts(train_val_items),
        },
        "excluded_test_task_ids": sorted(lite_ids),
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
