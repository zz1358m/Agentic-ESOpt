#!/usr/bin/env python
"""Export the VAB/WebRL WebArena-Lite 165-task split as items.json."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default="data/webarena/vab-lite/config_files/wa/test_webarena_lite")
    parser.add_argument("--output-dir", default="data/webarena/vab_lite_split")
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    output_dir = Path(args.output_dir)
    if not config_dir.is_dir():
        raise FileNotFoundError(config_dir)

    items = []
    for path in sorted(config_dir.glob("*.json"), key=lambda p: int(p.stem)):
        task = load_json(path)
        task_id = int(path.stem)
        sites = [str(site) for site in task.get("sites", [])]
        items.append(
            {
                "id": str(task_id),
                "task_id": task_id,
                "old_task_id": task.get("old_task_id"),
                "intent": task.get("intent", ""),
                "intent_template": task.get("intent_template", ""),
                "sites": sites,
                "task_type": sites[0] if sites else "unknown",
                "eval_types": task.get("eval", {}).get("eval_types", []),
                "config_path": str(path.resolve()),
            }
        )

    if len(items) != 165:
        raise RuntimeError(f"Expected 165 VAB/WebArena-Lite items, got {len(items)}")

    write_json(output_dir / "items.json", items)
    manifest = {
        "name": "vab-webarena-lite",
        "definition": "VAB/WebRL WebArena-Lite 165-task split.",
        "config_dir": str(config_dir.resolve()),
        "items": str((output_dir / "items.json").resolve()),
        "task_count": len(items),
        "site_counts": dict(Counter((item.get("sites") or ["unknown"])[0] for item in items)),
        "old_task_id_count": len({int(item["old_task_id"]) for item in items if item.get("old_task_id") is not None}),
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
