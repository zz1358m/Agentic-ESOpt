from __future__ import annotations

import unittest

from verl.experimental.agent_loop.tool_agent_loop import (
    is_first_bash_action_only,
    is_text_react_format,
)
from verl.experimental.agent_loop.tool_parser import Trace2SkillToolParser
from verl_trace2skill.math_protocol import build_math_messages


class MathProtocolTests(unittest.TestCase):
    def test_messages_require_action_json_instead_of_xml_tool_calls(self) -> None:
        messages = build_math_messages("What is 20 + 22?")

        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn('Action:\n{"name": "bash"', messages[0]["content"])
        self.assertIn("What is 20 + 22?", messages[1]["content"])
        self.assertNotIn("<tool_call>", "\n".join(message["content"] for message in messages))

    def test_trace2skill_uses_text_react_without_native_tool_schema(self) -> None:
        self.assertTrue(is_text_react_format("trace2skill"))
        self.assertTrue(is_text_react_format("paper_react_cli"))
        self.assertFalse(is_text_react_format("hermes"))

    def test_trace2skill_parser_accepts_only_a_real_bash_action(self) -> None:
        parser = Trace2SkillToolParser(tokenizer=None)
        valid = parser._parse_action(
            'Action: {"name":"bash","arguments":{"command":"python -c \'print(42)\'"}}'
        )
        wrong_name = parser._parse_action(
            'Action: {"name":"python","arguments":{"command":"print(42)"}}'
        )
        missing_command = parser._parse_action(
            'Action: {"name":"bash","arguments":{"code":"print(42)"}}'
        )

        self.assertIsNotNone(valid)
        self.assertIsNone(wrong_name)
        self.assertIsNone(missing_command)

    def test_first_assistant_turn_must_only_be_a_bash_action(self) -> None:
        action = 'Action: {"name":"bash","arguments":{"command":"python -c \'print(42)\'"}}'

        self.assertTrue(is_first_bash_action_only(action))
        self.assertTrue(is_first_bash_action_only(f"<think>\n\n</think>\n{action}"))
        self.assertFalse(is_first_bash_action_only(f"Let me reason first.\n{action}"))
        self.assertFalse(
            is_first_bash_action_only(
                'Action: {"name":"python","arguments":{"command":"print(42)"}}'
            )
        )


if __name__ == "__main__":
    unittest.main()
