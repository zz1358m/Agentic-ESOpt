from __future__ import annotations

import unittest
import time
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

import yaml

from verl.tools.schemas import OpenAIFunctionToolSchema
from verl_trace2skill.local_bash_tool import LocalBashTool


ROOT = Path(__file__).resolve().parents[1]


class MathBashToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_math_config_executes_without_an_image(self) -> None:
        declaration = yaml.safe_load(
            (ROOT / "verl_trace2skill/math_bash_tool_config.yaml").read_text(encoding="utf-8")
        )["tools"][0]
        tool = LocalBashTool(
            declaration["config"],
            OpenAIFunctionToolSchema.model_validate(declaration["tool_schema"]),
        )

        instance_id, _ = await tool.create(cwd=str(ROOT))
        response, _, metrics = await tool.execute(
            instance_id,
            {"command": "python -c 'print(20 + 22)'"},
        )

        self.assertIn("Observation from bash:\n42", response.text or "")
        self.assertEqual(metrics["bash_returncode"], 0.0)

    async def test_math_config_uses_the_dedicated_tool_workspace(self) -> None:
        declaration = yaml.safe_load(
            (ROOT / "verl_trace2skill/math_bash_tool_config.yaml").read_text(encoding="utf-8")
        )["tools"][0]
        with TemporaryDirectory() as workspace, patch.dict(
            "os.environ", {"TRACE2SKILL_MATH_TOOL_CWD": workspace}
        ):
            tool = LocalBashTool(
                declaration["config"],
                OpenAIFunctionToolSchema.model_validate(declaration["tool_schema"]),
            )
            instance_id, _ = await tool.create()
            response, _, _ = await tool.execute(instance_id, {"command": "pwd"})

        self.assertIn(str(Path(workspace).resolve()), response.text or "")

    async def test_timeout_kills_nested_processes_without_waiting_for_their_pipes(self) -> None:
        declaration = yaml.safe_load(
            (ROOT / "verl_trace2skill/math_bash_tool_config.yaml").read_text(encoding="utf-8")
        )["tools"][0]
        declaration["config"]["timeout"] = 0.1
        tool = LocalBashTool(
            declaration["config"],
            OpenAIFunctionToolSchema.model_validate(declaration["tool_schema"]),
        )
        instance_id, _ = await tool.create(cwd=str(ROOT))

        started = time.monotonic()
        response, _, metrics = await tool.execute(
            instance_id,
            {"command": "python -c \"import subprocess; subprocess.run(['sleep', '2'])\""},
        )

        self.assertLess(time.monotonic() - started, 1.0)
        self.assertIn("timed out", (response.text or "").lower())
        self.assertEqual(metrics["bash_timeout"], 1.0)


if __name__ == "__main__":
    unittest.main()
