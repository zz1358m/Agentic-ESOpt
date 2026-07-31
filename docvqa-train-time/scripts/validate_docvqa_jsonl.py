#!/usr/bin/env python3
"""Validate stored DocVQA scores by replaying ANLS offline."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DOCVQA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DOCVQA_ROOT))

from envs.docvqa import anls  # noqa: E402


LEGACY_ANLS_METHOD = "docvqa_paper_react_cli_anls"
ACCURACY_METHOD = "docvqa_paper_react_cli_anls_gt_0.5_acc"


@dataclass
class ValidationReport:
    records: int = 0
    anls_total: float = 0.0
    accuracy_total: float = 0.0
    error_sample_limit: int = 10
    errors: list[str] = field(default_factory=list)
    mismatch_counts: Counter[str] = field(default_factory=Counter)
    _anls_compensation: float = field(default=0.0, repr=False)

    @property
    def mean_anls(self) -> float:
        return self.anls_total / self.records if self.records else 0.0

    @property
    def mean_accuracy(self) -> float:
        return self.accuracy_total / self.records if self.records else 0.0

    @property
    def mismatches(self) -> int:
        return sum(self.mismatch_counts.values())

    def add_error(self, category: str, message: str) -> None:
        self.mismatch_counts[category] += 1
        if len(self.errors) < self.error_sample_limit:
            self.errors.append(message)

    def add_score(self, anls_score: float, accuracy: float) -> None:
        adjusted = anls_score - self._anls_compensation
        updated = self.anls_total + adjusted
        self._anls_compensation = (updated - self.anls_total) - adjusted
        self.anls_total = updated
        self.accuracy_total += accuracy


def _stored_number(row: dict[str, Any], field_name: str, line_no: int) -> float:
    value = row.get(field_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"line {line_no}: {field_name} must be a finite number")
    try:
        converted = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"line {line_no}: {field_name} must be a finite number") from None
    if not math.isfinite(converted):
        raise ValueError(f"line {line_no}: {field_name} must be a finite number")
    return converted


def validate_jsonl(path: Path, *, atol: float = 1e-12) -> ValidationReport:
    if not math.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be a finite non-negative number")
    report = ValidationReport()
    with path.open(encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                report.add_error("json", f"line {line_no}: invalid JSON: {exc.msg}")
                continue
            if not isinstance(row, dict):
                report.add_error("format", f"line {line_no}: expected a JSON object")
                continue
            report.records += 1
            prediction = row.get("prediction")
            answers = row.get("answers")
            if not isinstance(prediction, str):
                report.add_error("prediction", f"line {line_no}: prediction must be a string")
                continue
            valid_answers = (
                isinstance(answers, list)
                and bool(answers)
                and all(isinstance(answer, str) for answer in answers)
            )
            if not valid_answers:
                report.add_error(
                    "answers",
                    f"line {line_no}: answers must be a non-empty list of strings",
                )
                continue
            if "error" not in row:
                report.add_error("error", f"line {line_no}: missing error field")
                continue
            request_error = row["error"]
            if request_error is not None and not isinstance(request_error, str):
                report.add_error("error", f"line {line_no}: error must be null or a string")
                continue
            if request_error not in (None, ""):
                report.add_error("request_error", f"line {line_no}: request error: {request_error}")
                continue
            score_method = row.get("score_method")
            if score_method not in (LEGACY_ANLS_METHOD, ACCURACY_METHOD):
                report.add_error(
                    "score_method",
                    f"line {line_no}: unknown score_method: {score_method!r}",
                )
                continue

            recalculated_anls = anls(prediction, answers)
            recalculated_accuracy = 1.0 if recalculated_anls > 0.5 else 0.0
            expected_score = (
                recalculated_anls
                if score_method == LEGACY_ANLS_METHOD
                else recalculated_accuracy
            )
            expected = {
                "anls": recalculated_anls,
                "vlns": recalculated_anls,
                "acc": recalculated_accuracy,
                "score": expected_score,
            }
            for field_name, expected_value in expected.items():
                try:
                    actual = _stored_number(row, field_name, line_no)
                except ValueError as exc:
                    report.add_error(field_name, str(exc))
                    continue
                if not math.isclose(actual, expected_value, rel_tol=0.0, abs_tol=atol):
                    report.add_error(
                        field_name,
                        f"line {line_no}: {field_name}={actual!r}, expected {expected_value!r}"
                    )

            report.add_score(recalculated_anls, recalculated_accuracy)
    if report.records == 0:
        report.add_error("format", "no JSON records found")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="DocVQA result JSONL to validate")
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--expected-mean-anls", type=float)
    parser.add_argument("--expected-mean-accuracy", type=float)
    parser.add_argument("--atol", type=float, default=1e-12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = validate_jsonl(args.jsonl, atol=args.atol)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    if args.expected_count is not None and report.records != args.expected_count:
        report.add_error("expected_count", f"records={report.records}, expected {args.expected_count}")
    if args.expected_mean_anls is not None and not math.isclose(
        report.mean_anls,
        args.expected_mean_anls,
        rel_tol=0.0,
        abs_tol=args.atol,
    ):
        report.add_error(
            "expected_mean_anls",
            f"mean_anls={report.mean_anls!r}, expected {args.expected_mean_anls!r}"
        )
    if args.expected_mean_accuracy is not None and not math.isclose(
        report.mean_accuracy,
        args.expected_mean_accuracy,
        rel_tol=0.0,
        abs_tol=args.atol,
    ):
        report.add_error(
            "expected_mean_accuracy",
            f"mean_accuracy={report.mean_accuracy!r}, expected {args.expected_mean_accuracy!r}"
        )

    print(f"records={report.records}")
    print(f"mean_anls={report.mean_anls:.16f}")
    print(f"mean_accuracy={report.mean_accuracy:.16f}")
    for field_name in ("anls", "vlns", "acc", "score"):
        print(f"{field_name}_mismatches={report.mismatch_counts[field_name]}")
    for category in sorted(set(report.mismatch_counts) - {"anls", "vlns", "acc", "score"}):
        print(f"{category}_mismatches={report.mismatch_counts[category]}")
    print(f"mismatches={report.mismatches}")
    for error in report.errors:
        print(f"ERROR: {error}")
    print("PASS" if not report.mismatches else "FAIL")
    return 0 if not report.mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
