from __future__ import annotations

import unittest

from algorithms.verl_trace2skill.docvqa_protocol import (
    DOCVQA_IMAGE_PATH,
    FORMAT_WARNING,
    TOOL_REQUIRED_WARNING,
    bash_action_command,
    build_docvqa_messages,
    choose_endpoint,
    initial_system_prompt_tokens,
    incremental_message_token_count,
    parse_action,
    paper_react_sampling_params,
    react_step,
    response_budget_exceeded,
)


class DocVQAProtocolTests(unittest.TestCase):
    def test_only_nonempty_bash_commands_satisfy_the_tool_gate(self) -> None:
        command, error = bash_action_command(
            {"name": "bash", "arguments": {"command": "tesseract document.png stdout"}}
        )
        self.assertEqual(command, "tesseract document.png stdout")
        self.assertIsNone(error)

        command, error = bash_action_command({"name": "bash", "arguments": {"command": "  "}})
        self.assertIsNone(command)
        self.assertEqual(error, "No shell command was provided.")

        command, error = bash_action_command({"name": "python", "arguments": {}})
        self.assertIsNone(command)
        self.assertEqual(error, "Unknown action 'python'. Available action is bash.")

    def test_parses_only_historical_action_json(self) -> None:
        action = parse_action(
            'Action: {"name": "bash", "arguments": {"command": "tesseract document.png stdout"}}'
        )

        self.assertEqual(action, {"name": "bash", "arguments": {"command": "tesseract document.png stdout"}})
        self.assertIsNone(
            parse_action(
                "<tool_call><function=bash><parameter=command>pwd</parameter></function></tool_call>"
            )
        )

    def test_requires_tool_before_accepting_final_answer(self) -> None:
        decision = react_step("Final answer: guessed", used_tool=False)

        self.assertEqual(decision.kind, "retry")
        self.assertEqual(decision.message, TOOL_REQUIRED_WARNING)

    def test_accepts_final_answer_after_tool(self) -> None:
        decision = react_step("Final answer: invoice 123", used_tool=True)

        self.assertEqual(decision.kind, "final")
        self.assertEqual(decision.answer, "invoice 123")

    def test_invalid_output_returns_shared_format_warning(self) -> None:
        decision = react_step("I should inspect the image", used_tool=False)

        self.assertEqual(decision.kind, "retry")
        self.assertEqual(decision.message, FORMAT_WARNING)

    def test_messages_use_virtual_sandbox_image_path(self) -> None:
        messages = build_docvqa_messages("What is the invoice number?")

        self.assertEqual(messages[1]["role"], "user")
        self.assertIn(f"Image path: {DOCVQA_IMAGE_PATH}", messages[1]["content"])
        self.assertNotIn("<tool_call>", messages[0]["content"])

    def test_messages_include_distilled_skill(self) -> None:
        messages = build_docvqa_messages(
            "What is the invoice number?", "Anchor the label first."
        )

        self.assertIn("Anchor the label first.", messages[0]["content"])
        self.assertIn("bash/OCR", messages[0]["content"])

    def test_endpoint_assignment_round_robins_four_samples_across_four_replicas(self) -> None:
        endpoints = [f"http://127.0.0.1:{18080 + index}/v1" for index in range(4)]

        assignments = [
            choose_endpoint(endpoints, row_index=row, sample_index=sample, samples=4)
            for row in range(2)
            for sample in range(4)
        ]

        self.assertEqual(assignments, endpoints * 2)

    def test_32k_accumulated_response_budget(self) -> None:
        self.assertFalse(response_budget_exceeded({"completion_tokens": 32767}, 32768))
        self.assertTrue(response_budget_exceeded({"completion_tokens": 32768}, 32768))
        self.assertTrue(response_budget_exceeded({"output_tokens": 40000}, 32768))

    def test_incremental_observation_count_matches_rl_chat_rendering(self) -> None:
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.call = (messages, kwargs)
                return [1, 2, 3, 4, 5]

        tokenizer = Tokenizer()
        message = {"role": "user", "content": "Observation from bash:\nOCR"}
        self.assertEqual(
            incremental_message_token_count(
                tokenizer,
                message,
                apply_chat_template_kwargs={"enable_thinking": False},
            ),
            5,
        )
        self.assertEqual(tokenizer.call[0], [message])
        self.assertTrue(tokenizer.call[1]["add_generation_prompt"])

    def test_paper_react_does_not_render_an_empty_chat_template(self) -> None:
        class RejectEmptyMessagesTokenizer:
            def apply_chat_template(self, *args, **kwargs):
                raise AssertionError("paper ReAct must not render the empty-message template")

        self.assertEqual(
            initial_system_prompt_tokens(
                RejectEmptyMessagesTokenizer(),
                paper_react_cli=True,
                use_inference_chat_template=False,
                apply_chat_template_kwargs={},
            ),
            [],
        )

    def test_paper_react_uses_token_stop_for_tokenizer_free_sglang(self) -> None:
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                self.request = (text, add_special_tokens)
                return [36514, 362]

        tokenizer = Tokenizer()
        original = {"temperature": 1.0, "stop": ["existing"], "stop_token_ids": [7]}

        result = paper_react_sampling_params(original, tokenizer, max_new_tokens=512)

        self.assertNotIn("stop", result)
        self.assertEqual(result["stop_token_ids"], [7, 36514])
        self.assertEqual(result["max_new_tokens"], 512)
        self.assertEqual(tokenizer.request, ("Observation", False))
        self.assertEqual(original["stop"], ["existing"])


if __name__ == "__main__":
    unittest.main()
