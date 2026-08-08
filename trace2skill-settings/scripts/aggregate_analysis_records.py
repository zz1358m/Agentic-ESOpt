#!/usr/bin/env python3
"""Aggregate per-trajectory Trace2Skill analysis into one MAP unit per task."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--error-json", type=Path, required=True)
    parser.add_argument("--success-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--empty-success-json", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    analysis_to_task = {
        row["analysis_id"]: row["task_id"]
        for row in manifest["rows"]
        if row.get("selected_for_analysis") and row.get("analysis_id")
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    parsed_counts: dict[str, int] = {}
    missing_ids: list[str] = []
    for source, path in (("error", args.error_json), ("success", args.success_json)):
        records = read_json(path)
        parsed_counts[source] = len(records)
        for record in records:
            analysis_id = str(record.get("instance_id", ""))
            task_id = analysis_to_task.get(analysis_id)
            if task_id is None:
                missing_ids.append(analysis_id)
                continue
            for item in record.get("items", []):
                enriched = dict(item)
                enriched["trajectory_analysis_id"] = analysis_id
                enriched["analysis_source"] = source
                grouped[task_id].append(enriched)
    if missing_ids:
        raise ValueError(f"Parsed records missing from trajectory manifest: {sorted(set(missing_ids))[:20]}")

    evidence_records = [
        {
            "instance_id": task_id,
            "source_file": "aggregated_trajectory_analysis",
            "items": items,
        }
        for task_id, items in sorted(grouped.items())
        if items
    ]
    write_json(args.output_json, evidence_records)
    write_json(args.empty_success_json, [])
    summary = {
        "analysis_manifest_count": len(analysis_to_task),
        "parsed_trajectory_records": parsed_counts,
        "map_evidence_units": len(evidence_records),
        "evidence_items": sum(len(record["items"]) for record in evidence_records),
    }
    write_json(args.summary_json, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
