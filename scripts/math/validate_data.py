#!/usr/bin/env python3
"""Validate and fingerprint the fixed DAPO-400 Math experiment splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_TRAIN_SHA256 = "859079023cab64c793383f93841659b3aaf698f9a36228a0474e2d0832086636"
DEFAULT_COUNTS = {"train": 400, "dapo100": 100, "aime2026": 30}
FILES = {
    "train": "dapo_evolve.jsonl",
    "dapo100": "dapo_test.jsonl",
    "aime2026": "aime_2026.jsonl",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"all rows in {path} must be JSON objects")
    return rows


def validate_math_data(
    data_root: Path,
    *,
    expected_counts: dict[str, int] = DEFAULT_COUNTS,
    expected_train_sha256: str = EXPECTED_TRAIN_SHA256,
) -> dict[str, Any]:
    split_sets: dict[str, tuple[set[str], set[str]]] = {}
    split_reports: dict[str, dict[str, Any]] = {}
    for split, filename in FILES.items():
        path = data_root / filename
        rows = _read_rows(path)
        expected = expected_counts[split]
        if len(rows) != expected:
            raise ValueError(f"{split} has {len(rows)} rows, expected {expected}")
        ids = [str(row.get("id", "")).strip() for row in rows]
        questions = [str(row.get("question", "")).strip() for row in rows]
        if any(not item for item in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"{split} ids must be non-empty and unique")
        if any(not item for item in questions):
            raise ValueError(f"{split} questions must be non-empty")
        sha256 = _sha256(path)
        split_sets[split] = (set(ids), set(questions))
        split_reports[split] = {
            "path": str(path.resolve()),
            "rows": len(rows),
            "unique_ids": len(set(ids)),
            "unique_questions": len(set(questions)),
            "duplicate_question_rows": len(questions) - len(set(questions)),
            "sha256": sha256,
        }

    actual_train_sha = split_reports["train"]["sha256"]
    if actual_train_sha != expected_train_sha256:
        raise ValueError(
            f"training SHA-256 mismatch: expected {expected_train_sha256}, got {actual_train_sha}"
        )

    intersections: dict[str, dict[str, int]] = {}
    for left, right in combinations(FILES, 2):
        id_overlap = len(split_sets[left][0] & split_sets[right][0])
        question_overlap = len(split_sets[left][1] & split_sets[right][1])
        if id_overlap:
            raise ValueError(f"id overlap between {left} and {right}: {id_overlap}")
        if question_overlap:
            raise ValueError(f"question overlap between {left} and {right}: {question_overlap}")
        intersections[f"{left}__{right}"] = {
            "ids": id_overlap,
            "questions": question_overlap,
        }
    return {
        "status": "PASS",
        "fixed_training_records_retained": True,
        "expected_train_sha256": expected_train_sha256,
        "splits": split_reports,
        "pairwise_intersections": intersections,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data/trace2skill/math_reasoning",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = validate_math_data(args.data_root)
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
