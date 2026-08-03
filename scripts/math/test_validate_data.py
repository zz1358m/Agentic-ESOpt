from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_data import validate_math_data  # noqa: E402


def _write(path: Path, rows: list[dict[str, str]]) -> str:
    content = "".join(json.dumps(row) + "\n" for row in rows)
    path.write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode()).hexdigest()


def test_validation_reports_internal_duplicates_but_rejects_no_fixed_rows(tmp_path: Path) -> None:
    train_sha = _write(
        tmp_path / "dapo_evolve.jsonl",
        [
            {"id": "t1", "question": "same", "answer": "1"},
            {"id": "t2", "question": "same", "answer": "1"},
        ],
    )
    _write(tmp_path / "dapo_test.jsonl", [{"id": "v1", "question": "held out", "answer": "2"}])
    _write(tmp_path / "aime_2026.jsonl", [{"id": "a1", "question": "aime", "answer": "3"}])

    report = validate_math_data(
        tmp_path,
        expected_counts={"train": 2, "dapo100": 1, "aime2026": 1},
        expected_train_sha256=train_sha,
    )
    assert report["status"] == "PASS"
    assert report["splits"]["train"]["unique_questions"] == 1
    assert report["splits"]["train"]["duplicate_question_rows"] == 1
    assert all(value == 0 for pair in report["pairwise_intersections"].values() for value in pair.values())


def test_validation_rejects_question_leakage(tmp_path: Path) -> None:
    train_sha = _write(tmp_path / "dapo_evolve.jsonl", [{"id": "t1", "question": "leak"}])
    _write(tmp_path / "dapo_test.jsonl", [{"id": "v1", "question": "leak"}])
    _write(tmp_path / "aime_2026.jsonl", [{"id": "a1", "question": "other"}])

    with pytest.raises(ValueError, match="question overlap"):
        validate_math_data(
            tmp_path,
            expected_counts={"train": 1, "dapo100": 1, "aime2026": 1},
            expected_train_sha256=train_sha,
        )
