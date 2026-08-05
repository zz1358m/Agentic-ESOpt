#!/usr/bin/env python
"""Export the JitRL WebArena-Lite split from the installed WebArena package.

JitRL defines WebArena-Lite as WebArena task ids 0..164 from
``webarena/test.raw.json``. This is different from the VAB-WebArena-Lite files.
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
    parser.add_argument("--output-dir", default="data/webarena/lite")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=164)
    args = parser.parse_args()

    import webarena

    raw_path = importlib.resources.files(webarena).joinpath("test.raw.json")
    rows = json.loads(raw_path.read_text())
    by_id = {int(row["task_id"]): row for row in rows if "task_id" in row}
    task_ids = list(range(args.start, args.end + 1))
    selected = []
    missing = []
    for task_id in task_ids:
        row = by_id.get(task_id)
        if row is None:
            missing.append(task_id)
            continue
        item = dict(row)
        item["jitrl_lite_category"] = SITE_COLUMNS.get((item.get("sites") or ["unknown"])[0], "Other")
        selected.append(item)

    if missing:
        raise RuntimeError(f"Missing WebArena task ids from installed package: {missing}")

    output = Path(args.output_dir)
    write_json(output / "items.json", selected)
    config_dir = output / "config_files"
    for item in selected:
        write_json(config_dir / f"{item['task_id']}.json", item)

    first_site_counts = Counter((item.get("sites") or ["unknown"])[0] for item in selected)
    included_site_counts = Counter(site for item in selected for site in item.get("sites", []))
    task_ids_by_column: dict[str, list[int]] = defaultdict(list)
    for item in selected:
        task_ids_by_column[item["jitrl_lite_category"]].append(int(item["task_id"]))

    manifest = {
        "source": {
            "repo": "https://github.com/liushiliushi/JitRL",
            "webarena_package": getattr(webarena, "__file__", None),
            "raw": str(raw_path),
        },
        "definition": {
            "name": "JitRL WebArena-Lite",
            "task_ids": f"{args.start}-{args.end}",
            "task_count": len(selected),
            "note": "JitRL uses the first 165 tasks from the installed WebArena package, not VAB-WebArena-Lite.",
        },
        "columns": ["Admin", "GitLab", "Map", "Reddit", "Shopping", "Avg"],
        "first_site_counts": dict(first_site_counts),
        "included_site_counts": dict(included_site_counts),
        "task_ids_by_column": {key: sorted(value) for key, value in sorted(task_ids_by_column.items())},
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
