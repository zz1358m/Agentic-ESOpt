#!/usr/bin/env python
"""Summarize WebArena-Lite results by benchmark column.

The default mode keeps the original JitRL-style hard success summary. Use
``--mode webrl-soft`` to match the WebRL/VAB reporting style: group tasks by
site from ``test_webarena_lite.raw.json`` and report mean score * 100.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


COLUMNS = ["Admin", "GitLab", "Map", "Reddit", "Shopping"]
SITE_TO_COLUMN = {
    "shopping_admin": "Admin",
    "gitlab": "GitLab",
    "map": "Map",
    "reddit": "Reddit",
    "shopping": "Shopping",
}
WEBRL_COLUMNS = ["Reddit", "Gitlab", "CMS", "Map", "OSS"]
WEBRL_SITE_TO_COLUMN = {
    "reddit": "Reddit",
    "gitlab": "Gitlab",
    "shopping_admin": "CMS",
    "map": "Map",
    "shopping": "OSS",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_items(path: Path) -> dict[int, dict[str, Any]]:
    rows = load_json(path)
    return {int(row["task_id"]): row for row in rows}


def load_raw_config(path: Path) -> dict[int, dict[str, Any]]:
    rows = load_json(path)
    return {int(row["task_id"]): row for row in rows}


def result_success(row: dict[str, Any]) -> bool:
    if "success" in row:
        return bool(row["success"])
    if "success_count" in row:
        return int(row.get("success_count", 0)) > 0
    if "status" in row:
        return str(row["status"]).lower() in {"success", "succeeded", "passed"}
    return False


def normalize_results(data: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(data, dict) and isinstance(data.get("scores"), list):
        return data["scores"]
    rows = data.get("results", data if isinstance(data, list) else [])
    if not isinstance(rows, list):
        raise TypeError("Expected a JitRL results list or dict with a results field.")
    return rows


def result_score(row: dict[str, Any]) -> float | None:
    if row.get("skipped"):
        return None
    if "score" in row:
        try:
            score = float(row["score"])
        except (TypeError, ValueError):
            return None
        return score if score >= 0.0 else None
    return 1.0 if result_success(row) else 0.0


def summarize(method: str, results_paths: list[Path], items_by_id: dict[int, dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for results_path in results_paths:
        rows.extend(normalize_results(load_json(results_path)))
    task_success: dict[int, list[bool]] = defaultdict(list)
    for row in rows:
        if row.get("skipped"):
            continue
        if "task_id" not in row:
            continue
        task_success[int(row["task_id"])].append(result_success(row))

    by_col: dict[str, list[bool]] = {column: [] for column in COLUMNS}
    for task_id, attempts in task_success.items():
        item = items_by_id.get(task_id)
        if item is None:
            continue
        site = (item.get("sites") or ["unknown"])[0]
        column = SITE_TO_COLUMN.get(site)
        if column is None:
            continue
        by_col[column].append(any(attempts))

    values: dict[str, float | None] = {}
    all_values: list[bool] = []
    for column in COLUMNS:
        vals = by_col[column]
        values[column] = (sum(vals) / len(vals)) if vals else None
        all_values.extend(vals)
    values["Avg"] = (sum(all_values) / len(all_values)) if all_values else None
    values["Method"] = method
    values["Tasks"] = len(all_values)
    return values


def summarize_webrl_soft(
    method: str,
    results_paths: list[Path],
    raw_by_id: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for results_path in results_paths:
        rows.extend(normalize_results(load_json(results_path)))

    task_score: dict[int, float] = {}
    for row in rows:
        if "task_id" not in row:
            continue
        score = result_score(row)
        if score is None:
            continue
        task_id = int(row["task_id"])
        task_score[task_id] = max(score, task_score.get(task_id, 0.0))

    by_col: dict[str, list[float]] = {column: [] for column in WEBRL_COLUMNS}
    for task_id, config in raw_by_id.items():
        score = task_score.get(task_id, 0.0)
        for site in config.get("sites", []):
            column = WEBRL_SITE_TO_COLUMN.get(site)
            if column is not None:
                by_col[column].append(score)

    values: dict[str, float | int | str | None] = {"Method": method}
    all_values: list[float] = []
    for column in WEBRL_COLUMNS:
        vals = by_col[column]
        values[column] = (sum(vals) / len(vals)) if vals else None
        all_values.extend(vals)
    values["Avg"] = (sum(all_values) / len(all_values)) if all_values else None
    values["Tasks"] = len(task_score)
    return values


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{100.0 * value:.2f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", default="data/webarena/lite/items.json")
    parser.add_argument(
        "--raw-config",
        default="data/webarena/vab-lite/config_files/wa/test_webarena_lite.raw.json",
        help="VAB/WebRL raw config used by --mode webrl-soft.",
    )
    parser.add_argument(
        "--mode",
        choices=["jitrl-hard", "webrl-soft"],
        default="jitrl-hard",
        help="jitrl-hard: hard pass by primary column; webrl-soft: site mean score * 100.",
    )
    parser.add_argument(
        "--result",
        action="append",
        nargs=2,
        metavar=("METHOD", "PATH"),
        required=True,
        help="Method name and JitRL results JSON path. Can be repeated.",
    )
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    grouped: dict[str, list[Path]] = defaultdict(list)
    for method, path in args.result:
        grouped[method].append(Path(path))
    if args.mode == "webrl-soft":
        raw_by_id = load_raw_config(Path(args.raw_config))
        rows = [summarize_webrl_soft(method, paths, raw_by_id) for method, paths in grouped.items()]
        headers = ["Method", "Reddit", "Gitlab", "CMS", "Map", "OSS", "Avg", "Tasks"]
    else:
        items_by_id = load_items(Path(args.items))
        rows = [summarize(method, paths, items_by_id) for method, paths in grouped.items()]
        headers = ["Method", "Admin", "GitLab", "Map", "Reddit", "Shopping", "Avg", "Tasks"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(header)) for header in headers) + " |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.output_md:
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_md).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
