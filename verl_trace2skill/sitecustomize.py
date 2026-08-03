"""Runtime extension for trace2skill VERL jobs.

Python imports ``sitecustomize`` automatically at startup when it is on
``PYTHONPATH``. The GRPO launcher adds this package directory directly, so the
upstream VERL package keeps its normal imports while getting an extra tool
parser registered for trace2skill's text ReAct format.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re


logger = logging.getLogger(__name__)


if os.environ.get("TRACE2SKILL_EAGER_PATCH_DENSE_QWEN3NEXT") == "1":
    from verl_trace2skill.sglang_dense_qwen3next_compat import patch_sglang_dense_qwen3next

    patch_sglang_dense_qwen3next()


def _register_trace2skill_tool_parser() -> None:
    try:
        from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
        from verl.utils.rollout_trace import rollout_trace_op
    except Exception as exc:
        logger.debug("trace2skill parser registration skipped: %r", exc)
        return

    if "trace2skill" in ToolParser._registry:
        return

    @ToolParser.register("trace2skill")
    class Trace2SkillToolParser(ToolParser):
        def __init__(self, tokenizer) -> None:
            super().__init__(tokenizer)
            self.action_regex = re.compile(r"Action:\s*(\{.*?\})\s*$", re.DOTALL | re.IGNORECASE)
            self.fallback_regex = re.compile(r"Action:\s*(\{.*\})", re.DOTALL | re.IGNORECASE)

        @rollout_trace_op
        async def extract_tool_calls(self, responses_ids: list[int]) -> tuple[str, list[FunctionCall]]:
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(None, self.tokenizer.decode, responses_ids)
            action = self._parse_action(text)
            if action is None:
                return text, []
            return text, [action]

        def _parse_action(self, text: str) -> FunctionCall | None:
            match = self.action_regex.search(text) or self.fallback_regex.search(text)
            if not match:
                return None
            raw = match.group(1).strip()
            decoder = json.JSONDecoder()
            try:
                payload, _ = decoder.raw_decode(raw)
            except json.JSONDecodeError:
                return None
            if not isinstance(payload, dict):
                return None
            name = payload.get("name")
            arguments = payload.get("arguments", {})
            command = arguments.get("command") if isinstance(arguments, dict) else None
            if name != "bash" or not isinstance(command, str) or not command.strip():
                return None
            return FunctionCall(name=name, arguments=json.dumps(arguments, ensure_ascii=False))


# DocVQA's paper Action protocol is parsed directly by its agent loop. Avoid
# importing VERL (and therefore torch/CUDA) in the parent process merely to
# register the unrelated generic Trace2Skill parser; SGLang children apply the
# dense-model patch only after their individual GPU visibility is isolated.
if os.environ.get("TRACE2SKILL_PATCH_DENSE_QWEN3NEXT") != "1":
    _register_trace2skill_tool_parser()
