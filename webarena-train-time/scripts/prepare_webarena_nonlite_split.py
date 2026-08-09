#!/usr/bin/env python
"""Prepare the shared WebArena non-Lite train/validation split.

The source is VAB's full WebArena config directory. The held-out WebArena-Lite
tasks are removed by matching their ``old_task_id`` values against full
WebArena ``task_id`` values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


RELEASED_SEED = 20260605
RELEASED_VAL_RATIO = 0.1
RELEASED_COUNTS = {"full": 812, "lite": 165, "nonlite": 647, "train": 582, "val": 65}
RELEASED_ORDERED_TASK_ID_SHA256 = {
    "train": "c0a433f7ca57809442f97c1042f4d4154fd2cb26bd049e805ab567d28058c271",
    "val": "5b772371a33363601a6fe094208b39ad732c1824e3cebdb09fe02f3d9e12f49b",
}
RELEASED_RAW_CONFIG_SHA256 = {
    "full": "d35a86509d117021744a58c735eeb61e34356a42163475d8c2535f65ba9c0d33",
    "lite": "92cef9ca77065d28ad3cac19ccf7f27c2a3784a19bed14905467f71b003846bf",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def stratum_key(task: dict[str, Any]) -> tuple[str]:
    """Return the exact site stratum used by the released experiment split."""
    sites = "|".join(str(site) for site in task.get("sites", ["unknown"]))
    return (sites,)


def stratified_shuffle(tasks: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    strata: dict[tuple[str], list[dict[str, Any]]] = defaultdict(list)
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


def ordered_task_id_sha256(items: list[dict[str, Any]]) -> str:
    task_ids = [int(item["task_id"]) for item in items]
    payload = json.dumps(task_ids, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def canonical_json_sha256(path: Path) -> str:
    value = load_json(path)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-dir", default="data/webarena/vab-lite/config_files/wa/test_webarena")
    parser.add_argument("--lite-dir", default="data/webarena/vab-lite/config_files/wa/test_webarena_lite")
    parser.add_argument("--output-dir", default="data/webarena/vab_nonlite_split")
    parser.add_argument("--seed", type=int, default=RELEASED_SEED)
    parser.add_argument("--val-ratio", type=float, default=RELEASED_VAL_RATIO)
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.0,
        help="Deprecated; kept for compatibility and must stay 0.",
    )
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
        raise ValueError(
            "--test-ratio is deprecated for this setup. Use 0; "
            "official WebArena-Lite 165 is the test set."
        )

    full_paths = sorted(full_dir.glob("*.json"), key=lambda path: int(path.stem))
    lite_paths = sorted(lite_dir.glob("*.json"), key=lambda path: int(path.stem))
    raw_config_paths = {
        "full": full_dir.parent / f"{full_dir.name}.raw.json",
        "lite": lite_dir.parent / f"{lite_dir.name}.raw.json",
    }
    missing_raw_configs = [str(path) for path in raw_config_paths.values() if not path.is_file()]
    if missing_raw_configs:
        raise FileNotFoundError(
            "Missing raw WebArena config source needed to identify the released dataset: "
            + ", ".join(missing_raw_configs)
        )
    raw_config_hashes = {name: canonical_json_sha256(path) for name, path in raw_config_paths.items()}

    lite_old_ids = set()
    for path in lite_paths:
        task = load_json(path)
        if task.get("old_task_id") is None:
            raise RuntimeError(f"Missing old_task_id in WebArena-Lite config: {path}")
        lite_old_ids.add(int(task["old_task_id"]))

    selected: list[dict[str, Any]] = []
    excluded: list[int] = []
    for path in full_paths:
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

    ordered_hashes = {
        "train": ordered_task_id_sha256(train_items),
        "val": ordered_task_id_sha256(val_items),
    }
    is_released_protocol = args.seed == RELEASED_SEED and args.val_ratio == RELEASED_VAL_RATIO
    if is_released_protocol:
        observed_counts = {
            "full": len(full_paths),
            "lite": len(lite_old_ids),
            "nonlite": len(selected),
            "train": len(train_items),
            "val": len(val_items),
        }
        if (
            observed_counts != RELEASED_COUNTS
            or ordered_hashes != RELEASED_ORDERED_TASK_ID_SHA256
            or raw_config_hashes != RELEASED_RAW_CONFIG_SHA256
        ):
            raise RuntimeError(
                "The generated split does not match the released WebArena experiments. "
                f"Observed counts={observed_counts}, ordered_task_id_sha256={ordered_hashes}; "
                f"raw_config_sha256={raw_config_hashes}; expected counts={RELEASED_COUNTS}, "
                f"ordered hashes={RELEASED_ORDERED_TASK_ID_SHA256}, "
                f"raw config hashes={RELEASED_RAW_CONFIG_SHA256}."
            )

    write_json(output_dir / "train" / "items.json", train_items)
    write_json(output_dir / "val" / "items.json", val_items)
    write_json(output_dir / "test" / "items.json", test_items)
    manifest = {
        "name": "vab-webarena-nonlite-train-val",
        "source_full_dir": str(full_dir.resolve()),
        "source_lite_dir": str(lite_dir.resolve()),
        "source_raw_config_sha256": raw_config_hashes,
        "definition": {
            "test": "The 165 VAB/WebArena-Lite configs, addressed by their new task_id 0..164.",
            "train_val": "Original WebArena task_id 0..811 after excluding every VAB-Lite old_task_id.",
            "validation": (
                f"The first round({len(selected)} * {args.val_ratio}) = {val_n} "
                "items after site-stratified ordering."
            ),
            "train": f"The remaining {len(train_items)} ordered non-Lite items.",
        },
        "excluded_lite_old_task_count": len(lite_old_ids),
        "excluded_lite_old_task_ids": sorted(lite_old_ids),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": 0.0,
        "ordering": {
            "input": "full configs sorted by numeric original task_id",
            "stratum": "the complete ordered sites list",
            "algorithm": "seeded per-stratum shuffle followed by seeded shuffled round-robin interleaving",
        },
        "ordered_task_id_sha256": ordered_hashes,
        "released_experiment_reference": {
            "protocol": "vab-nonlite-site-stratified-20260605-v1",
            "counts": RELEASED_COUNTS,
            "ordered_task_id_sha256": RELEASED_ORDERED_TASK_ID_SHA256,
            "raw_config_sha256": RELEASED_RAW_CONFIG_SHA256,
            "matches": is_released_protocol,
        },
        "counts": {
            "full": len(full_paths),
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
        "note": (
            "Non-Lite tasks are split only into train/val. test/items.json is intentionally empty; "
            "data/webarena/vab_lite_split/items.json is the only final test set."
        ),
    }
    write_json(output_dir / "split_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
