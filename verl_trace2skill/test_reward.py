from __future__ import annotations

import unittest

from verl_trace2skill.reward import _last_boxed_content, compute_score


class RewardTests(unittest.TestCase):
    def test_math_reward_requires_and_accepts_bash_trace(self) -> None:
        response = (
            'Action: {"name": "bash", "arguments": {"command": "python -c \'print(42)\'"}}\n'
            "Observation from bash:\n42\n"
            "Final answer: \\boxed{42}"
        )
        self.assertEqual(compute_score("trace2skill_math_dapo", response, "42")["score"], 1.0)
        self.assertEqual(
            compute_score("trace2skill_math_dapo", "Final answer: \\boxed{42}", "42")["score"],
            0.0,
        )

    def test_nested_boxed_answer(self) -> None:
        self.assertEqual(_last_boxed_content(r"work \\boxed{\\frac{1}{2}}"), r"\\frac{1}{2}")

    def test_docvqa_anls_reward(self) -> None:
        response = (
            '<tool_call>{"name": "bash", "arguments": {"command": "ocrmypdf --help"}}</tool_call>\n'
            "Final answer: invoice 123"
        )
        score = compute_score(
            "trace2skill_docvqa",
            response,
            ["Invoice 123", "invoice no. 123"],
        )
        self.assertEqual(score["score"], 1.0)
        self.assertEqual(score["anls"], 1.0)


if __name__ == "__main__":
    unittest.main()
