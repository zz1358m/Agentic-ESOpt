#!/usr/bin/env python
"""Export the VAB/WebRL WebArena-Lite 165-task split as items.json."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


RELEASED_TASK_COUNT = 165
RELEASED_OLD_TASK_ID_SHA256 = "79e446fc5738d4a616d5b11f5d804e7d339c8b1932b9f09627394a0977bc7642"
RELEASED_RAW_CONFIG_SHA256 = "92cef9ca77065d28ad3cac19ccf7f27c2a3784a19bed14905467f71b003846bf"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def ordered_id_sha256(items: list[dict[str, Any]], field: str) -> str:
    values = [int(item[field]) for item in items]
    payload = json.dumps(values, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def canonical_json_sha256(path: Path) -> str:
    value = load_json(path)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default="data/webarena/vab-lite/config_files/wa/test_webarena_lite")
    parser.add_argument("--output-dir", default="data/webarena/vab_lite_split")
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    output_dir = Path(args.output_dir)
    if not config_dir.is_dir():
        raise FileNotFoundError(config_dir)
    raw_config_path = config_dir.parent / f"{config_dir.name}.raw.json"
    if not raw_config_path.is_file():
        raise FileNotFoundError(
            f"Missing raw WebArena-Lite config source needed to identify the released dataset: {raw_config_path}"
        )
    raw_config_sha256 = canonical_json_sha256(raw_config_path)
    if raw_config_sha256 != RELEASED_RAW_CONFIG_SHA256:
        raise RuntimeError(
            "The raw VAB/WebArena-Lite config does not match the released experiments: "
            f"observed {raw_config_sha256}, expected {RELEASED_RAW_CONFIG_SHA256}."
        )

    items = []
    for path in sorted(config_dir.glob("*.json"), key=lambda p: int(p.stem)):
        task = load_json(path)
        task_id = int(path.stem)
        if int(task.get("task_id", task_id)) != task_id:
            raise RuntimeError(f"task_id does not match the config filename: {path}")
        if task.get("old_task_id") is None:
            raise RuntimeError(f"Missing old_task_id in WebArena-Lite config: {path}")
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

    if len(items) != RELEASED_TASK_COUNT:
        raise RuntimeError(f"Expected {RELEASED_TASK_COUNT} VAB/WebArena-Lite items, got {len(items)}")
    task_ids = [int(item["task_id"]) for item in items]
    if task_ids != list(range(RELEASED_TASK_COUNT)):
        raise RuntimeError("Expected the VAB/WebArena-Lite task_id sequence 0..164.")
    old_task_ids = [int(item["old_task_id"]) for item in items]
    if len(set(old_task_ids)) != RELEASED_TASK_COUNT:
        raise RuntimeError("Expected 165 unique VAB/WebArena-Lite old_task_id values.")
    old_task_id_sha256 = ordered_id_sha256(items, "old_task_id")
    if old_task_id_sha256 != RELEASED_OLD_TASK_ID_SHA256:
        raise RuntimeError(
            "The VAB/WebArena-Lite old_task_id mapping does not match the released experiments: "
            f"observed {old_task_id_sha256}, expected {RELEASED_OLD_TASK_ID_SHA256}."
        )

    write_json(output_dir / "items.json", items)
    manifest = {
        "name": "vab-webarena-lite",
        "definition": "VAB/WebRL WebArena-Lite 165-task split.",
        "config_dir": str(config_dir.resolve()),
        "items": str((output_dir / "items.json").resolve()),
        "task_count": len(items),
        "task_id_range": [0, RELEASED_TASK_COUNT - 1],
        "raw_config_sha256": raw_config_sha256,
        "ordered_old_task_id_sha256": old_task_id_sha256,
        "site_counts": dict(Counter((item.get("sites") or ["unknown"])[0] for item in items)),
        "old_task_id_count": len({int(item["old_task_id"]) for item in items if item.get("old_task_id") is not None}),
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
