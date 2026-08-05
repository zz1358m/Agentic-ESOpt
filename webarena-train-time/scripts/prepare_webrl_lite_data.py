#!/usr/bin/env python
"""Prepare WebRL/VAB WebArena-Lite train and test data.

This keeps the WebRL paper split semantics:
- train: public WebRL SFT trajectories from THUDM/WebRL
- test: the full 165-task VAB-WebArena-Lite benchmark
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch


DEFAULT_SFT = "data/webarena/webrl/scripts/webarena_lite_sft.pt"
DEFAULT_TRAIN_INFO = "data/webarena/webrl/WebArena-Lite_info.json"
DEFAULT_TEST_RAW = "data/webarena/vab-lite/new/test_webarena_lite.raw.json"
DEFAULT_OUTPUT = "data/webarena/webrl/prepared"

DEFAULT_URLS = {
    "__SHOPPING__": "http://127.0.0.1:7770",
    "__REDDIT__": "http://127.0.0.1:9999",
    "__SHOPPING_ADMIN__": "http://127.0.0.1:7780/admin",
    "__GITLAB__": "http://127.0.0.1:8023",
    "__MAP__": "http://127.0.0.1:3000",
    "__WIKIPEDIA__": "http://127.0.0.1:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing",
    "__HOMEPAGE__": "http://127.0.0.1:4399",
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def replace_urls(text: str, urls: dict[str, str]) -> str:
    for key, value in urls.items():
        text = text.replace(key, value)
    return text


def extract_task_id_from_observation(text: str, fallback: int) -> str:
    task = ""
    match = re.search(r"Task Instruction:\s*(.*?)\n\nRound\s+0", text, re.S)
    if match:
        task = match.group(1).strip()
    return str(abs(hash(task)) if task else fallback)


def trajectory_summary(traj: list[dict[str, Any]], index: int) -> dict[str, Any]:
    first = traj[0] if traj else {}
    task = str(first.get("task", ""))
    return {
        "id": str(index),
        "task_id": index,
        "task_description": task,
        "question": task,
        "task_type": "webrl_lite_sft",
        "n_steps": len(traj),
        "trajectory_reward": first.get("trajectory_reward", 0),
        "final_reward": traj[-1].get("reward", 0) if traj else 0,
    }


def site_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update(str(site) for site in row.get("sites", []))
    return dict(counter)


def eval_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update(str(kind) for kind in row.get("eval", {}).get("eval_types", []))
    return dict(counter)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", default=DEFAULT_SFT)
    parser.add_argument("--train-info", default=DEFAULT_TRAIN_INFO)
    parser.add_argument("--test-raw", default=DEFAULT_TEST_RAW)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = Path(args.output_dir)
    train_dir = output / "train"
    test_dir = output / "test"
    config_dir = test_dir / "config_files"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    trajectories = torch.load(args.sft, map_location="cpu")
    if not isinstance(trajectories, list):
        raise TypeError(f"Expected list trajectories in {args.sft}, got {type(trajectories).__name__}")

    train_items = [trajectory_summary(traj, idx) for idx, traj in enumerate(trajectories)]
    write_json(train_dir / "items.json", train_items)
    with (train_dir / "trajectories.jsonl").open("w", encoding="utf-8") as f:
        for idx, traj in enumerate(trajectories):
            row = {"id": str(idx), "trajectory": traj}
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    train_info_path = Path(args.train_info)
    train_info = json.loads(train_info_path.read_text()) if train_info_path.exists() else []
    write_json(train_dir / "webarena_lite_info.json", train_info)

    raw_text = replace_urls(Path(args.test_raw).read_text(), DEFAULT_URLS)
    test_rows = json.loads(raw_text)
    for idx, item in enumerate(test_rows):
        item["task_id"] = idx
        write_json(config_dir / f"{idx}.json", item)
    write_json(test_dir / "raw.json", test_rows)
    write_json(test_dir / "items.json", test_rows)

    step_count = sum(len(traj) for traj in trajectories if isinstance(traj, list))
    terminal_rewards = Counter((traj[-1].get("reward") if traj else None) for traj in trajectories)
    manifest = {
        "source": {
            "sft": str(Path(args.sft).resolve()),
            "train_info": str(train_info_path.resolve()) if train_info_path.exists() else "",
            "test_raw": str(Path(args.test_raw).resolve()),
        },
        "train": {
            "trajectory_count": len(trajectories),
            "step_count": step_count,
            "train_info_count": len(train_info) if isinstance(train_info, list) else None,
            "terminal_reward_counts": {str(k): v for k, v in terminal_rewards.items()},
        },
        "test": {
            "task_count": len(test_rows),
            "site_distribution": site_distribution(test_rows),
            "eval_distribution": eval_distribution(test_rows),
            "config_dir": str(config_dir.resolve()),
        },
        "setting": {
            "benchmark": "VAB-WebArena-Lite",
            "test_ids": "0-164",
            "observation_type": "webrl",
            "action_set_tag": "webrl_id",
            "viewport": "1280x720",
            "max_steps": 30,
            "max_tokens": 2048,
            "temperature": 0.0,
        },
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
