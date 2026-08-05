#!/usr/bin/env python
"""Summarize the WebRL-provided SFT experience data.

WebRL alignment uses ``data/webarena/webrl/scripts/webarena_lite_sft.pt`` and
``data/webarena/webrl/WebArena-Lite_info.json`` as the original training data
source. This is not the same as splitting WebArena raw task ids.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", default="data/webarena/webrl/scripts/webarena_lite_sft.pt")
    parser.add_argument("--info", default="data/webarena/webrl/WebArena-Lite_info.json")
    parser.add_argument("--output-dir", default="data/webarena/webrl/manifest")
    args = parser.parse_args()

    info = json.loads(Path(args.info).read_text())
    trajectories = torch.load(args.sft, map_location="cpu")
    if not isinstance(info, list):
        raise TypeError(f"Expected list in {args.info}")
    if not isinstance(trajectories, list):
        raise TypeError(f"Expected list in {args.sft}")

    site_counts = Counter((item.get("sites") or ["unknown"])[0] for item in info)
    included_site_counts = Counter(site for item in info for site in item.get("sites", []))
    step_counts = [len(traj) for traj in trajectories if isinstance(traj, list)]
    task_counts = Counter()
    for traj in trajectories:
        if not traj or not isinstance(traj, list) or not isinstance(traj[0], dict):
            continue
        task = traj[0].get("task")
        if task:
            task_counts[str(task)] += 1

    manifest = {
        "definition": {
            "name": "WebRL SFT experience data",
            "note": "Original WebRL training source; not a WebArena raw task-id split.",
        },
        "source": {
            "sft": str(Path(args.sft).resolve()),
            "info": str(Path(args.info).resolve()),
            "repo": "https://github.com/THUDM/WebRL",
        },
        "counts": {
            "task_info_count": len(info),
            "trajectory_count": len(trajectories),
            "unique_task_text_count": len(task_counts),
            "duplicated_task_text_count": sum(1 for count in task_counts.values() if count > 1),
            "step_total": sum(step_counts),
            "step_min": min(step_counts) if step_counts else 0,
            "step_max": max(step_counts) if step_counts else 0,
            "step_avg": (sum(step_counts) / len(step_counts)) if step_counts else 0.0,
        },
        "first_site_counts": dict(site_counts),
        "included_site_counts": dict(included_site_counts),
    }
    write_json(Path(args.output_dir) / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
