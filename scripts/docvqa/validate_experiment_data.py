#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(value)
    return rows


def _identity(row: dict[str, Any], *, historical: bool) -> tuple[str, list[str], str]:
    id_key = "task_id" if historical else "id"
    task_id = str(row.get(id_key, ""))
    answers = [str(answer) for answer in row.get("answers", [])]
    image_name = Path(str(row.get("image", "")).replace("\\", "/")).name
    return task_id, answers, image_name


def validate_alignment(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    historical_rows: list[dict[str, Any]],
    *,
    limit: int,
    samples: int,
) -> dict[str, Any]:
    if limit <= 0 or samples <= 0:
        raise ValueError("limit and samples must be positive")
    if len(test_rows) < limit:
        raise ValueError(f"test rows={len(test_rows)} is less than limit={limit}")

    train_ids = {str(row.get("id", "")) for row in train_rows}
    test_ids = {str(row.get("id", "")) for row in test_rows[:limit]}
    overlap = sorted(train_ids & test_ids)
    if overlap:
        raise ValueError(f"train/test overlap: {overlap[:10]}")

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in historical_rows:
        row_index = int(row.get("row_index", -1))
        if row_index >= 0:
            grouped.setdefault(row_index, []).append(row)

    reference_groups: list[tuple[int, list[dict[str, Any]]]] = []
    for source_row_index in sorted(grouped):
        group = grouped[source_row_index]
        task_id = _identity(group[0], historical=True)[0]
        if task_id not in train_ids:
            reference_groups.append((source_row_index, group))
        if len(reference_groups) == limit:
            break
    if len(reference_groups) < limit:
        raise ValueError(
            f"historical rows contain only {len(reference_groups)} non-training tasks, expected {limit}"
        )

    task_ids = []
    source_row_indexes = []
    for row_index, (source_row_index, historical_group) in enumerate(reference_groups):
        sample_indexes = {int(row.get("sample_index", -1)) for row in historical_group}
        if sample_indexes != set(range(samples)) or len(historical_group) != samples:
            raise ValueError(
                f"historical row {source_row_index} samples={sorted(sample_indexes)}, expected 0..{samples - 1}"
            )
        expected = _identity(historical_group[0], historical=True)
        for duplicate in historical_group[1:]:
            if _identity(duplicate, historical=True) != expected:
                raise ValueError(f"historical identity differs across samples at row {source_row_index}")
        test_row = test_rows[row_index]
        actual = _identity(test_row, historical=False)
        for label, actual_value, expected_value in zip(
            ("task_id", "answers", "image basename"),
            actual,
            expected,
            strict=True,
        ):
            if actual_value != expected_value:
                raise ValueError(
                    f"row {row_index}: {label} mismatch: actual={actual_value!r}, expected={expected_value!r}"
                )
        task_ids.append(actual[0])
        source_row_indexes.append(source_row_index)

    return {
        "train_records": len(train_rows),
        "test_records_available": len(test_rows),
        "test_records_checked": limit,
        "historical_records_checked": limit * samples,
        "samples_per_task": samples,
        "task_ids": task_ids,
        "historical_source_row_indexes": source_row_indexes,
        "status": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate GRPO DocVQA data against historical JSONL identities.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = validate_alignment(
        read_jsonl(args.train),
        read_jsonl(args.test),
        read_jsonl(args.historical),
        limit=args.limit,
        samples=args.samples,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
