from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_docvqa_jsonl.py"
LEGACY_METHOD = "docvqa_paper_react_cli_anls"
ACCURACY_METHOD = "docvqa_paper_react_cli_anls_gt_0.5_acc"


def result_row(
    prediction: str,
    answers: list[str],
    anls: float,
    *,
    score_method: str = LEGACY_METHOD,
) -> dict:
    accuracy = 1.0 if anls > 0.5 else 0.0
    return {
        "prediction": prediction,
        "answers": answers,
        "anls": anls,
        "vlns": anls,
        "acc": accuracy,
        "score": anls if score_method == LEGACY_METHOD else accuracy,
        "score_method": score_method,
        "error": None,
    }


class ValidateDocVQAJsonlTests(unittest.TestCase):
    def test_valid_legacy_anls_output_passes(self) -> None:
        row = result_row("Invoice #123", ["invoice 123"], 1.0)

        result = self._run([row], "--expected-count", "1", "--expected-mean-anls", "1")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("records=1", result.stdout)
        self.assertIn("mean_anls=1.0000000000000000", result.stdout)
        self.assertIn("mean_accuracy=1.0000000000000000", result.stdout)
        self.assertIn("mismatches=0", result.stdout)
        self.assertIn("PASS", result.stdout)

    def test_current_accuracy_score_method_passes(self) -> None:
        row = result_row("ABCD", ["ABCE"], 0.75, score_method=ACCURACY_METHOD)

        result = self._run([row])

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("mean_anls=0.7500000000000000", result.stdout)
        self.assertIn("PASS", result.stdout)

    def test_anls_uses_best_normalized_answer(self) -> None:
        row = result_row("INVOICE #123", ["receipt 999", "invoice 123"], 1.0)

        result = self._run([row])

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("mean_anls=1.0000000000000000", result.stdout)

    def test_anls_threshold_and_empty_prediction_score_zero(self) -> None:
        rows = [
            result_row("ab", ["ac"], 0.0),
            result_row("", ["answer"], 0.0),
        ]

        result = self._run(rows)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("mean_anls=0.0000000000000000", result.stdout)
        self.assertIn("mean_accuracy=0.0000000000000000", result.stdout)

    def test_expected_mean_accuracy_is_enforced(self) -> None:
        row = result_row("ABCD", ["ABCE"], 0.75)

        passing = self._run([row], "--expected-mean-accuracy", "1")
        failing = self._run([row], "--expected-mean-accuracy", "0")

        self.assertEqual(passing.returncode, 0, passing.stderr)
        self.assertEqual(failing.returncode, 1, failing.stderr)
        self.assertIn("mean_accuracy=1.0, expected 0.0", failing.stdout)

    def test_score_mismatch_is_reported_by_field(self) -> None:
        row = result_row("invoice 123", ["invoice 123"], 1.0)
        row["score"] = 0.0

        result = self._run([row])

        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("score_mismatches=1", result.stdout)
        self.assertIn("mismatches=1", result.stdout)
        self.assertIn("line 1: score=0.0, expected 1.0", result.stdout)

    def test_missing_field_fails_without_stopping_later_records(self) -> None:
        missing_score = result_row("first", ["first"], 1.0)
        del missing_score["score"]
        valid = result_row("second", ["second"], 1.0)

        result = self._run([missing_score, valid])

        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("records=2", result.stdout)
        self.assertIn("score_mismatches=1", result.stdout)
        self.assertIn("line 1: score must be a finite number", result.stdout)

    def test_malformed_json_fails_without_stopping_later_records(self) -> None:
        valid = json.dumps(result_row("answer", ["answer"], 1.0))

        result = self._run_raw("{not json}\n" + valid + "\n")

        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("records=1", result.stdout)
        self.assertIn("json_mismatches=1", result.stdout)
        self.assertIn("line 1: invalid JSON", result.stdout)

    def test_unknown_score_method_is_a_validation_failure(self) -> None:
        unknown = result_row("answer", ["answer"], 1.0)
        unknown["score_method"] = "unknown"

        result = self._run([unknown])

        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("score_method_mismatches=1", result.stdout)
        self.assertIn("unknown score_method: 'unknown'", result.stdout)

    def test_request_error_is_a_validation_failure(self) -> None:
        failed_request = result_row("", ["answer"], 0.0)
        failed_request["error"] = "ReadTimeout"
        failed_request["score_method"] = "request_error"

        result = self._run([failed_request])

        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("request_error_mismatches=1", result.stdout)
        self.assertIn("line 1: request error: ReadTimeout", result.stdout)
        self.assertNotIn("score_method_mismatches", result.stdout)

    def test_invalid_prediction_and_answers_are_reported_by_field(self) -> None:
        invalid_prediction = result_row("answer", ["answer"], 1.0)
        invalid_prediction["prediction"] = None
        invalid_answers = result_row("answer", ["answer"], 1.0)
        invalid_answers["answers"] = "answer"
        valid = result_row("answer", ["answer"], 1.0)

        result = self._run([invalid_prediction, invalid_answers, valid])

        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("records=3", result.stdout)
        self.assertIn("prediction_mismatches=1", result.stdout)
        self.assertIn("answers_mismatches=1", result.stdout)

    def test_empty_file_is_a_validation_failure(self) -> None:
        result = self._run_raw("\n")

        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("records=0", result.stdout)
        self.assertIn("format_mismatches=1", result.stdout)
        self.assertIn("no JSON records found", result.stdout)

    def test_non_finite_stored_score_is_a_validation_failure(self) -> None:
        row = result_row("answer", ["answer"], 1.0)
        row["score"] = float("nan")

        result = self._run([row])

        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("score_mismatches=1", result.stdout)
        self.assertIn("score must be a finite number", result.stdout)

    def test_unrepresentable_stored_number_is_a_validation_failure(self) -> None:
        row = result_row("answer", ["answer"], 1.0)
        row["score"] = 10**4000

        result = self._run([row])

        self.assertEqual(result.returncode, 1)
        self.assertIn("score_mismatches=1", result.stdout)
        self.assertIn("score must be a finite number", result.stdout)
        self.assertNotIn("Traceback", result.stderr)

    def test_missing_error_field_is_a_validation_failure(self) -> None:
        row = result_row("answer", ["answer"], 1.0)
        del row["error"]

        result = self._run([row])

        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("error_mismatches=1", result.stdout)
        self.assertIn("line 1: missing error field", result.stdout)

    def test_tolerance_must_be_finite_and_non_negative(self) -> None:
        row = result_row("answer", ["answer"], 1.0)

        infinite = self._run([row], "--atol", "inf")
        negative = self._run([row], "--atol", "-1")

        self.assertEqual(infinite.returncode, 1)
        self.assertEqual(negative.returncode, 1)
        self.assertIn("atol must be a finite non-negative number", infinite.stderr)
        self.assertIn("atol must be a finite non-negative number", negative.stderr)

    def _run(self, rows: list[dict], *args: str) -> subprocess.CompletedProcess[str]:
        return self._run_raw("".join(json.dumps(row) + "\n" for row in rows), *args)

    def _run_raw(self, content: str, *args: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.jsonl"
            path.write_text(content, encoding="utf-8")
            return subprocess.run(
                [sys.executable, str(SCRIPT), str(path), *args],
                capture_output=True,
                text=True,
                check=False,
            )


if __name__ == "__main__":
    unittest.main()
