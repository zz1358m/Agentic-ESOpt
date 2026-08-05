#!/usr/bin/env python
"""Export WebArena train configs as all installed tasks excluding JitRL-Lite.

The current evaluation split is JitRL WebArena-Lite task ids 0..164.  For
training, use the remaining WebArena tasks from the same installed
``webarena/test.raw.json`` source.
"""

from __future__ import annotations

import argparse
import importlib.resources
import json
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="cache/jitrl_webarena_train_excluding_lite")
    parser.add_argument("--lite-start", type=int, default=0)
    parser.add_argument("--lite-end", type=int, default=164)
    args = parser.parse_args()

    import webarena

    raw_path = importlib.resources.files(webarena).joinpath("test.raw.json")
    rows = json.loads(raw_path.read_text())
    lite_ids = set(range(args.lite_start, args.lite_end + 1))
    selected = []
    for row in rows:
        if "task_id" not in row:
            continue
        task_id = int(row["task_id"])
        if task_id in lite_ids:
            continue
        item = dict(row)
        item["jitrl_train_category"] = SITE_COLUMNS.get((item.get("sites") or ["unknown"])[0], "Other")
        selected.append(item)

    selected.sort(key=lambda item: int(item["task_id"]))
    output = Path(args.output_dir)
    write_json(output / "items.json", selected)
    config_dir = output / "config_files"
    for item in selected:
        write_json(config_dir / f"{item['task_id']}.json", item)

    first_site_counts = Counter((item.get("sites") or ["unknown"])[0] for item in selected)
    included_site_counts = Counter(site for item in selected for site in item.get("sites", []))
    task_ids_by_column: dict[str, list[int]] = defaultdict(list)
    for item in selected:
        task_ids_by_column[item["jitrl_train_category"]].append(int(item["task_id"]))

    manifest = {
        "source": {
            "webarena_package": getattr(webarena, "__file__", None),
            "raw": str(raw_path),
        },
        "definition": {
            "name": "WebArena train excluding JitRL WebArena-Lite",
            "excluded_eval_task_ids": f"{args.lite_start}-{args.lite_end}",
            "task_count": len(selected),
            "note": "Training split uses installed WebArena tasks except the 165 JitRL-Lite eval tasks.",
        },
        "columns": ["Admin", "GitLab", "Map", "Reddit", "Shopping", "Other"],
        "first_site_counts": dict(first_site_counts),
        "included_site_counts": dict(included_site_counts),
        "task_ids_by_column": {key: sorted(value) for key, value in sorted(task_ids_by_column.items())},
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
