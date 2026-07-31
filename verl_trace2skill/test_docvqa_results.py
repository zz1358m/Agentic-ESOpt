from __future__ import annotations

import unittest

from verl_trace2skill.docvqa_results import compare_results, summarize_results


def row(task: str, sample: int, anls: float, *, error: str | None = None) -> dict:
    return {
        "key": f"docvqa:{task}:sample{sample:02d}",
        "task_id": task,
        "sample_index": sample,
        "anls": anls,
        "acc": 1.0 if anls > 0.5 else 0.0,
        "score": anls,
        "error": error,
        "react_error": None,
        "react_steps": [{"action": {"name": "bash"}, "observation": "ok"}],
        "latency_s": 2.0,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class DocVQAResultsTests(unittest.TestCase):
    def test_summary_reports_continuous_anls_and_tool_rate(self) -> None:
        summary = summarize_results([row("a", 0, 0.75), row("b", 0, 0.25)])

        self.assertEqual(summary["records"], 2)
        self.assertEqual(summary["valid_records"], 2)
        self.assertEqual(summary["mean_anls"], 0.5)
        self.assertEqual(summary["mean_accuracy"], 0.5)
        self.assertEqual(summary["tool_success_rate"], 1.0)
        self.assertEqual(summary["mean_turns"], 2.0)

    def test_summary_counts_format_retries_and_bash_timeouts(self) -> None:
        result = row("a", 0, 0.0)
        result["react_error"] = "max_react_turns_exceeded"
        result["react_steps"] = [
            {"observation": "No valid action was parsed."},
            {
                "action": {"name": "bash"},
                "observation": "Bash timed out after 20.0s.",
            },
        ]

        summary = summarize_results([result])

        self.assertEqual(summary["tool_call_rate"], 1.0)
        self.assertEqual(summary["tool_success_rate"], 0.0)
        self.assertEqual(summary["format_retries"], 1)
        self.assertEqual(summary["bash_timeouts"], 1)
        self.assertEqual(summary["mean_turns"], 2.0)

    def test_comparison_pairs_by_task_and_reports_delta(self) -> None:
        before = [row("a", 0, 0.25), row("a", 1, 0.75), row("b", 0, 0.0), row("b", 1, 0.0)]
        after = [row("a", 0, 0.75), row("a", 1, 0.75), row("b", 0, 0.5), row("b", 1, 0.5)]

        comparison = compare_results(before, after, bootstrap_samples=200, seed=42)

        self.assertEqual(comparison["paired_tasks"], 2)
        self.assertEqual(comparison["mean_anls_before"], 0.25)
        self.assertEqual(comparison["mean_anls_after"], 0.625)
        self.assertEqual(comparison["mean_anls_delta"], 0.375)
        self.assertEqual(len(comparison["bootstrap_95_ci"]), 2)

    def test_comparison_rejects_different_task_sets(self) -> None:
        with self.assertRaisesRegex(ValueError, "task sets differ"):
            compare_results([row("a", 0, 1.0)], [row("b", 0, 1.0)], bootstrap_samples=10)


if __name__ == "__main__":
    unittest.main()
