#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/webarena/skillopt_splits")
    parser.add_argument("--output", default="data/webarena/skillopt_splits_available")
    parser.add_argument(
        "--allowed-sites",
        default="shopping,shopping_admin,reddit,gitlab,wikipedia",
    )
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    allowed = {site.strip() for site in args.allowed_sites.split(",") if site.strip()}
    manifest = {"source": str(source), "allowed_sites": sorted(allowed), "splits": {}}

    for split in ("train", "val", "test"):
        items_path = source / split / "items.json"
        if not items_path.exists():
            marker = source / split / "MISSING_WEBRL_SFT.json"
            detail = ""
            if marker.exists():
                detail = f" Details: {marker.read_text(encoding='utf-8').strip()}"
            raise FileNotFoundError(
                f"Missing {split} split items: {items_path}. "
                "Prepare WebRL SFT data with prepare_standard_webarena_data.py first."
                f"{detail}"
            )
        items = json.loads(items_path.read_text())
        kept = [
            item for item in items
            if set(item.get("sites") or []).issubset(allowed)
        ]
        split_dir = output / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "items.json").write_text(json.dumps(kept, indent=2) + "\n")
        manifest["splits"][split] = {
            "count": len(kept),
            "task_ids": [item.get("task_id") for item in kept],
        }

    output.mkdir(parents=True, exist_ok=True)
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
