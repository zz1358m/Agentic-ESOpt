#!/usr/bin/env python3
"""Prepare the repository-standard WebArena data layout.

Standard layout:
- data/webarena/lite: WebArena-Lite 165-task test benchmark.
- data/webarena/webrl: WebRL SFT/experience data for SkillOpt and ES training.

This script can always prepare the 165-task lite test split from the vendored
JitRL WebArena configs. WebRL SFT preparation requires the original WebRL files
to be present under data/webarena/webrl/source or explicitly passed in.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "webarena"
DEFAULT_JITRL_CONFIGS = DATA_ROOT / "jitrl" / "config_files"
DEFAULT_LITE_ROOT = DATA_ROOT / "lite"
DEFAULT_WEBRL_ROOT = DATA_ROOT / "webrl"
DEFAULT_WEBRL_SOURCE = DEFAULT_WEBRL_ROOT / "source"
DEFAULT_SKILLOPT_SPLIT_ROOT = DATA_ROOT / "skillopt_splits"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_lite(*, source_config_dir: Path, output_root: Path, n_tasks: int = 165) -> dict[str, Any]:
    config_dir = output_root / "config_files"
    config_dir.mkdir(parents=True, exist_ok=True)
    items = []
    missing = []
    for task_id in range(n_tasks):
        src = source_config_dir / f"{task_id}.json"
        if not src.exists():
            missing.append(task_id)
            continue
        data = read_json(src)
        data["task_id"] = task_id
        dst = config_dir / f"{task_id}.json"
        write_json(dst, data)
        items.append(
            {
                "task_id": task_id,
                "sites": data.get("sites", []),
                "intent": data.get("intent", ""),
                "intent_template": data.get("intent_template", ""),
                "config_file": str(dst.relative_to(REPO_ROOT)),
            }
        )
    if missing:
        raise FileNotFoundError(f"Missing WebArena-Lite config ids under {source_config_dir}: {missing}")

    write_json(output_root / "items.json", items)
    manifest = {
        "name": "webarena-lite-test",
        "definition": "165-task WebArena-Lite test benchmark.",
        "source_config_dir": str(source_config_dir.relative_to(REPO_ROOT)),
        "config_dir": str(config_dir.relative_to(REPO_ROOT)),
        "items": str((output_root / "items.json").relative_to(REPO_ROOT)),
        "task_count": len(items),
        "task_ids": [item["task_id"] for item in items],
    }
    write_json(output_root / "manifest.json", manifest)
    return manifest


def split_rows(rows: list[Any], *, val_size: int, seed: int) -> tuple[list[Any], list[Any]]:
    import random

    rng = random.Random(seed)
    indexed = list(enumerate(rows))
    rng.shuffle(indexed)
    val_size = min(max(int(val_size), 0), len(indexed))
    val_indices = {idx for idx, _ in indexed[:val_size]}
    train = [row for idx, row in enumerate(rows) if idx not in val_indices]
    val = [row for idx, row in enumerate(rows) if idx in val_indices]
    return train, val


def maybe_prepare_webrl(
    *,
    source_root: Path,
    output_root: Path,
    val_size: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    sft = source_root / "scripts" / "webarena_lite_sft.pt"
    info = source_root / "WebArena-Lite_info.json"
    if not sft.exists():
        return {
            "prepared": False,
            "reason": f"WebRL SFT file not found: {sft}",
            "expected_sft": str(sft.relative_to(REPO_ROOT)) if sft.is_relative_to(REPO_ROOT) else str(sft),
            "expected_info": str(info.relative_to(REPO_ROOT)) if info.is_relative_to(REPO_ROOT) else str(info),
        }, {}

    import torch

    trajectories = torch.load(sft, map_location="cpu")
    if not isinstance(trajectories, list):
        raise TypeError(f"Expected list trajectories in {sft}, got {type(trajectories).__name__}")

    train_traj, val_traj = split_rows(trajectories, val_size=val_size, seed=seed)
    split_root = output_root / "skillopt_splits"
    split_root.mkdir(parents=True, exist_ok=True)

    def write_trajectories(split: str, rows: list[Any]) -> None:
        split_dir = split_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        items = []
        with (split_dir / "trajectories.jsonl").open("w", encoding="utf-8") as f:
            for idx, traj in enumerate(rows):
                item = {
                    "id": str(idx),
                    "task_id": idx,
                    "task_type": "webrl_lite_sft",
                    "n_steps": len(traj) if isinstance(traj, list) else 0,
                    "trajectory_reward": traj[0].get("trajectory_reward", 0) if traj and isinstance(traj[0], dict) else 0,
                    "final_reward": traj[-1].get("reward", 0) if traj and isinstance(traj[-1], dict) else 0,
                }
                items.append(item)
                f.write(json.dumps({"id": str(idx), "trajectory": traj}, ensure_ascii=True) + "\n")
        write_json(split_dir / "items.json", items)

    write_trajectories("train", train_traj)
    write_trajectories("val", val_traj)
    if info.exists():
        shutil.copy2(info, output_root / "WebArena-Lite_info.json")

    manifest = {
        "prepared": True,
        "name": "webrl-sft-skillopt-es-train",
        "source_sft": str(sft.relative_to(REPO_ROOT)) if sft.is_relative_to(REPO_ROOT) else str(sft),
        "source_info": str(info.relative_to(REPO_ROOT)) if info.exists() and info.is_relative_to(REPO_ROOT) else str(info),
        "split_root": str(split_root.relative_to(REPO_ROOT)),
        "train_count": len(train_traj),
        "val_count": len(val_traj),
        "seed": seed,
    }
    write_json(output_root / "manifest.json", manifest)
    return manifest, {"train": train_traj, "val": val_traj}


def prepare_skillopt_splits(
    *,
    split_root: Path,
    webrl_root: Path,
    lite_items: list[dict[str, Any]],
    webrl_manifest: dict[str, Any],
    webrl_rows: dict[str, list[Any]],
) -> dict[str, Any]:
    split_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (split_root / split).mkdir(parents=True, exist_ok=True)

    if webrl_manifest.get("prepared"):
        for split in ("train", "val"):
            rows = []
            for idx, traj in enumerate(webrl_rows[split]):
                rows.append(
                    {
                        "id": str(idx),
                        "task_id": idx,
                        "task_type": "webrl_lite_sft",
                        "source": "WebRL SFT",
                        "n_steps": len(traj) if isinstance(traj, list) else 0,
                        "trajectory_file": f"{split}/trajectories.jsonl",
                    }
                )
            write_json(split_root / split / "items.json", rows)
            shutil.copy2(webrl_root / "skillopt_splits" / split / "trajectories.jsonl", split_root / split / "trajectories.jsonl")
    else:
        for split in ("train", "val"):
            marker = {
                "prepared": False,
                "reason": webrl_manifest.get("reason", "WebRL SFT source is missing."),
                "expected_source": webrl_manifest.get("expected_sft", ""),
            }
            write_json(split_root / split / "MISSING_WEBRL_SFT.json", marker)

    write_json(split_root / "test" / "items.json", lite_items)
    manifest = {
        "name": "skillopt-webrl-train-val-webarena-lite-test",
        "train": {
            "source": "WebRL SFT",
            "path": str((split_root / "train" / "items.json").relative_to(REPO_ROOT)),
            "prepared": bool(webrl_manifest.get("prepared")),
        },
        "val": {
            "source": "WebRL SFT heldout validation",
            "path": str((split_root / "val" / "items.json").relative_to(REPO_ROOT)),
            "prepared": bool(webrl_manifest.get("prepared")),
        },
        "test": {
            "source": "WebArena-Lite 165",
            "path": str((split_root / "test" / "items.json").relative_to(REPO_ROOT)),
            "prepared": True,
            "task_count": len(lite_items),
        },
    }
    write_json(split_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jitrl-config-dir", default=str(DEFAULT_JITRL_CONFIGS))
    parser.add_argument("--lite-root", default=str(DEFAULT_LITE_ROOT))
    parser.add_argument("--webrl-source-root", default=str(DEFAULT_WEBRL_SOURCE))
    parser.add_argument("--webrl-root", default=str(DEFAULT_WEBRL_ROOT))
    parser.add_argument("--skillopt-split-root", default=str(DEFAULT_SKILLOPT_SPLIT_ROOT))
    parser.add_argument("--val-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260605)
    args = parser.parse_args()

    lite_manifest = prepare_lite(
        source_config_dir=Path(args.jitrl_config_dir),
        output_root=Path(args.lite_root),
    )
    lite_items = read_json(Path(args.lite_root) / "items.json")
    webrl_manifest, webrl_rows = maybe_prepare_webrl(
        source_root=Path(args.webrl_source_root),
        output_root=Path(args.webrl_root),
        val_size=args.val_size,
        seed=args.seed,
    )
    skillopt_manifest = prepare_skillopt_splits(
        split_root=Path(args.skillopt_split_root),
        webrl_root=Path(args.webrl_root),
        lite_items=lite_items,
        webrl_manifest=webrl_manifest,
        webrl_rows=webrl_rows,
    )
    combined = {
        "lite": lite_manifest,
        "webrl": webrl_manifest,
        "skillopt_splits": skillopt_manifest,
    }
    write_json(DATA_ROOT / "manifest.json", combined)
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
