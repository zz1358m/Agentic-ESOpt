#!/usr/bin/env python3
"""Check the on-disk data contract for the five maintained tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Check:
    task: str
    component: str
    path: str
    kind: str = "exists"
    expected_count: int | None = None
    expected_task_id_sha256: str | None = None
    expected_old_task_id_sha256: str | None = None
    expected_canonical_json_sha256: str | None = None


CHECKS = (
    Check("sudoku", "ES / multi-turn GRPO train", "data/sudoku/train.jsonl", "jsonl"),
    Check("sudoku", "ES / multi-turn GRPO eval", "data/sudoku/eval.jsonl", "jsonl"),
    Check("math", "ES / Trace2Skill / GRPO train", "data/trace2skill/math_reasoning/dapo_evolve.jsonl", "jsonl"),
    Check("math", "held-out DAPO eval", "data/trace2skill/math_reasoning/dapo_test.jsonl", "jsonl"),
    Check("math", "AIME eval", "data/trace2skill/math_reasoning/aime_2026.jsonl", "jsonl"),
    Check("docvqa", "ES / Trace2Skill / GRPO train", "data/trace2skill/docvqa/evolve.jsonl", "jsonl"),
    Check("docvqa", "held-out eval", "data/trace2skill/docvqa/test.jsonl", "jsonl"),
    Check("docvqa", "document images", "data/trace2skill/docvqa/images", "nonempty_dir"),
    Check("webarena", "VAB checkout", "data/webarena/vab-lite", "nonempty_dir"),
    Check(
        "webarena",
        "released original WebArena source",
        "data/webarena/vab-lite/config_files/wa/test_webarena.raw.json",
        "canonical_json",
        expected_canonical_json_sha256="d35a86509d117021744a58c735eeb61e34356a42163475d8c2535f65ba9c0d33",
    ),
    Check(
        "webarena",
        "released VAB/WebArena-Lite source",
        "data/webarena/vab-lite/config_files/wa/test_webarena_lite.raw.json",
        "canonical_json",
        expected_canonical_json_sha256="92cef9ca77065d28ad3cac19ccf7f27c2a3784a19bed14905467f71b003846bf",
    ),
    Check(
        "webarena",
        "released non-Lite train split",
        "data/webarena/vab_nonlite_split/train/items.json",
        "json",
        582,
        "c0a433f7ca57809442f97c1042f4d4154fd2cb26bd049e805ab567d28058c271",
    ),
    Check(
        "webarena",
        "released non-Lite validation split",
        "data/webarena/vab_nonlite_split/val/items.json",
        "json",
        65,
        "5b772371a33363601a6fe094208b39ad732c1824e3cebdb09fe02f3d9e12f49b",
    ),
    Check(
        "webarena",
        "released WebArena-Lite test split",
        "data/webarena/vab_lite_split/items.json",
        "json",
        165,
        "ccde5c90a96e54627d1b78c9a53d289c47b7671c9bc30862be93b0d027749339",
        "79e446fc5738d4a616d5b11f5d804e7d339c8b1932b9f09627394a0977bc7642",
    ),
    Check("webarena", "train/val/test disjointness", "data/webarena", "webarena_split_contract"),
    Check(
        "webarena",
        "Trace2Skill implementation",
        "webarena-train-time/methods/trace2skill/source/skill_evolver",
        "nonempty_dir",
    ),
    Check(
        "webarena",
        "WebArena rollout runtime",
        "webarena-train-time/third_party/skillopt/skillopt/envs/webarena_sft",
        "nonempty_dir",
    ),
    Check("ahd", "TSP constructive train", "data/ahd/datasets/tsp_constructive/train50_dataset.npy"),
    Check("ahd", "KP constructive train", "data/ahd/datasets/kp_constructive/train100_dataset.npy"),
    Check("ahd", "TSP ACO train", "data/ahd/datasets/tsp_aco/train50_dataset.npy"),
    Check("ahd", "CVRP ACO train", "data/ahd/datasets/cvrp_aco/train50_dataset.npy"),
    Check("ahd", "BPP ACO train", "data/ahd/datasets/bpp_offline_aco/train500_dataset.npz"),
)


def ordered_id_sha256(items: list[dict], field: str) -> str:
    values = [int(item[field]) for item in items]
    payload = json.dumps(values, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def inspect_webarena_split_contract() -> tuple[bool, str]:
    paths = {
        "train": ROOT / "data/webarena/vab_nonlite_split/train/items.json",
        "val": ROOT / "data/webarena/vab_nonlite_split/val/items.json",
        "test": ROOT / "data/webarena/vab_lite_split/items.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        return False, f"missing {', '.join(missing)} split"
    try:
        splits = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
        train_ids = [int(item["task_id"]) for item in splits["train"]]
        val_ids = [int(item["task_id"]) for item in splits["val"]]
        lite_ids = [int(item["task_id"]) for item in splits["test"]]
        held_out_old_ids = [int(item["old_task_id"]) for item in splits["test"]]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return False, f"invalid split metadata: {exc}"

    expected_counts = (582, 65, 165)
    if (len(train_ids), len(val_ids), len(lite_ids)) != expected_counts:
        return False, f"counts are {(len(train_ids), len(val_ids), len(lite_ids))}; expected {expected_counts}"
    if len(set(train_ids)) != len(train_ids) or len(set(val_ids)) != len(val_ids):
        return False, "duplicate original task_id in train or validation"
    if lite_ids != list(range(165)) or len(set(held_out_old_ids)) != 165:
        return False, "Lite task_id/old_task_id mapping is not the released 165-task mapping"

    train_set = set(train_ids)
    val_set = set(val_ids)
    held_out_set = set(held_out_old_ids)
    if train_set & val_set or train_set & held_out_set or val_set & held_out_set:
        return False, "train, validation, and held-out old_task_id sets overlap"
    if train_set | val_set | held_out_set != set(range(812)):
        return False, "the three partitions do not cover original WebArena task_id 0..811"
    return True, "582 train + 65 validation + 165 held-out; disjoint and covers original task_id 0..811"


def inspect(check: Check) -> tuple[bool, str]:
    if check.kind == "webarena_split_contract":
        return inspect_webarena_split_contract()
    path = ROOT / check.path
    if not path.exists():
        return False, "missing"
    if check.kind == "canonical_json":
        if not path.is_file():
            return False, "not a file"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"invalid JSON: {exc}"
        digest = canonical_json_sha256(payload)
        if digest != check.expected_canonical_json_sha256:
            return False, f"canonical JSON SHA-256 is {digest}; expected {check.expected_canonical_json_sha256}"
        size = len(payload) if hasattr(payload, "__len__") else 1
        return True, f"{size} entries; canonical source hash matches"
    if check.kind == "jsonl":
        if not path.is_file():
            return False, "not a file"
        rows = 0
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        return False, f"line {line_no} is not a JSON object"
                    rows += 1
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"invalid JSONL: {exc}"
        return rows > 0, f"{rows} rows"
    if check.kind == "json":
        if not path.is_file():
            return False, "not a file"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"invalid JSON: {exc}"
        size = len(payload) if hasattr(payload, "__len__") else 1
        if check.expected_count is not None:
            if size != check.expected_count:
                return False, f"{size} entries; expected {check.expected_count}"
        if check.expected_task_id_sha256 is not None:
            try:
                digest = ordered_id_sha256(payload, "task_id")
            except (KeyError, TypeError, ValueError) as exc:
                return False, f"cannot hash ordered task_id values: {exc}"
            if digest != check.expected_task_id_sha256:
                return False, f"ordered task_id SHA-256 is {digest}; expected {check.expected_task_id_sha256}"
        if check.expected_old_task_id_sha256 is not None:
            try:
                digest = ordered_id_sha256(payload, "old_task_id")
            except (KeyError, TypeError, ValueError) as exc:
                return False, f"cannot hash ordered old_task_id values: {exc}"
            if digest != check.expected_old_task_id_sha256:
                return False, f"ordered old_task_id SHA-256 is {digest}; expected {check.expected_old_task_id_sha256}"
        return size > 0, f"{size} entries"
    if check.kind == "nonempty_dir":
        if not path.is_dir():
            return False, "not a directory"
        count = sum(1 for item in path.iterdir() if item.is_file() or item.is_dir())
        return count > 0, f"{count} entries"
    if not path.is_file():
        return False, "not a file"
    return path.stat().st_size > 0, f"{path.stat().st_size} bytes"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["all", "sudoku", "math", "docvqa", "webarena", "ahd"], default="all")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when any selected item is missing or empty.",
    )
    args = parser.parse_args()

    selected = [check for check in CHECKS if args.task == "all" or check.task == args.task]
    failures = 0
    for check in selected:
        ok, detail = inspect(check)
        failures += int(not ok)
        status = "OK" if ok else "MISSING"
        print(f"{status:7} {check.task:8} {check.component}: {check.path} ({detail})")
    print(f"\n{len(selected) - failures}/{len(selected)} checks ready")
    if args.strict and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
