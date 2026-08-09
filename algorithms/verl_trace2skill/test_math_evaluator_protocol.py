from __future__ import annotations

import asyncio
import importlib.util
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


SCRIPT = Path(__file__).parents[2] / "scripts/trace2skill/run_trace2skill_vllm_eval16.py"
SPEC = importlib.util.spec_from_file_location("trace2skill_eval_protocol", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MathEvaluatorProtocolTests(unittest.TestCase):
    def test_no_skill_prompt_matches_qwen_math_benchmark_guidance(self) -> None:
        messages = MODULE.math_messages({"question": "What is 20+22?"})

        self.assertEqual([message["role"] for message in messages], ["user"])
        self.assertIn("What is 20+22?", messages[0]["content"])
        self.assertIn("put your final answer within \\boxed{}", messages[0]["content"])

    def test_legacy_no_skill_prompt_preserves_original_system_message(self) -> None:
        messages = MODULE.math_messages(
            {"question": "What is 20+22?"},
            "legacy-no-skill",
        )

        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("Solve the math problem carefully", messages[0]["content"])

    def test_evaluator_accepts_only_bash_with_nonempty_command(self) -> None:
        valid = MODULE.parse_react_action(
            'Action: {"name":"bash","arguments":{"command":"python -c \'print(42)\'"}}'
        )
        wrong_name = MODULE.parse_react_action(
            'Action: {"name":"python","arguments":{"command":"print(42)"}}'
        )
        missing_command = MODULE.parse_react_action(
            'Action: {"name":"bash","arguments":{"code":"print(42)"}}'
        )

        self.assertIsNotNone(valid)
        self.assertIsNone(wrong_name)
        self.assertIsNone(missing_command)

    def test_evaluator_timeout_kills_nested_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            started = time.monotonic()
            observation = MODULE.run_bash(
                "python -c \"import subprocess; subprocess.run(['sleep', '2'])\"",
                Path(directory),
                timeout=0.1,
                limit=1000,
            )

        self.assertLess(time.monotonic() - started, 1.0)
        self.assertIn("timed out", observation.lower())


class MathEvaluatorAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_direct_mode_uses_thinking_and_stable_sample_seed(self) -> None:
        args = SimpleNamespace(
            seed=7,
            math_max_tokens=81920,
            math_enable_thinking=True,
            math_direct_prompt="qwen-benchmark",
        )
        post_chat = AsyncMock(return_value=("Answer: \\boxed{42}", {"completion_tokens": 5}))

        with patch.object(MODULE, "post_chat", post_chat):
            completion, usage = await MODULE.run_math_direct(
                client=object(),
                chat_url="http://localhost/v1/chat/completions",
                model="model",
                row={"question": "20+22", "answer": "42"},
                row_index=3,
                sample_index=2,
                args=args,
            )

        self.assertEqual(completion, "Answer: \\boxed{42}")
        self.assertEqual(usage, {"completion_tokens": 5})
        kwargs = post_chat.await_args.kwargs
        self.assertTrue(kwargs["enable_thinking"])
        self.assertEqual(kwargs["max_tokens"], 81920)
        self.assertEqual(kwargs["seed"], 7 + 2 * 1_000_003 + 3)

    async def test_real_bash_does_not_block_other_async_requests(self) -> None:
        action = 'Action:\n{"name":"bash","arguments":{"command":"sleep 0.3; echo 42"}}'
        args = SimpleNamespace(
            seed=1,
            math_max_turns=2,
            math_max_tokens=512,
            math_tool_cwd=Path.cwd(),
            math_python_timeout=2.0,
            tool_observation_limit=1000,
        )
        post_chat = AsyncMock(
            side_effect=[
                (action, {"completion_tokens": 10}),
                ("Final answer: \\boxed{42}", {"completion_tokens": 5}),
            ]
        )

        started = time.monotonic()
        with patch.object(MODULE, "post_chat", post_chat):
            task = asyncio.create_task(
                MODULE.run_math_react(
                    client=object(),
                    chat_url="http://localhost/v1/chat/completions",
                    model="model",
                    row={"question": "20+22", "answer": "42"},
                    row_index=0,
                    sample_index=0,
                    args=args,
                )
            )
            await asyncio.sleep(0.05)
            scheduling_delay = time.monotonic() - started
            await task

        self.assertLess(scheduling_delay, 0.2)


if __name__ == "__main__":
    unittest.main()
