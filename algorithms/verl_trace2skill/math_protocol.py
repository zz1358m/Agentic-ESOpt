"""Shared prompt protocol for tool-using Math trajectories."""

from __future__ import annotations

import json
import re
from typing import Any


MATH_SYSTEM = """You are a math reasoning agent. Solve the problem using a command-line Python ReAct loop.

You are not allowed to answer from the problem alone. Your very first assistant turn must consist only of one bash action. Do not reason, solve, or write any text before that first action. After receiving its observation, continue solving and call bash again when useful.

Available action:

Action:
{"name": "bash", "arguments": {"command": "<shell command>"}}

Use command-line Python deliberately, for example python -c "...", for arithmetic, algebraic verification, brute force checks, or symbolic computation. When finished, output exactly:

Final answer: \\boxed{<answer>}

Do not include tool outputs in the final answer."""

_FIRST_BASH_ACTION = re.compile(
    r"^\s*(?:<think>\s*</think>\s*)?Action:\s*(\{.*\})\s*$",
    re.DOTALL | re.IGNORECASE,
)


def is_first_bash_action_only(text: str) -> bool:
    """Validate the no-reasoning-before-tool rule for the first Math turn."""
    match = _FIRST_BASH_ACTION.match(text)
    if match is None:
        return False
    try:
        action = json.loads(match.group(1))
    except json.JSONDecodeError:
        return False
    if not isinstance(action, dict):
        return False
    arguments = action.get("arguments")
    command = arguments.get("command") if isinstance(arguments, dict) else None
    return action.get("name") == "bash" and isinstance(command, str) and bool(command.strip())


def build_math_messages(question: str) -> list[dict[str, Any]]:
    """Build the canonical two-message Math ReAct prompt."""
    user = (
        "Task: Solve the following math problem.\n\n"
        f"{question}\n\n"
        "You must call the bash action at least once before giving the final answer."
    )
    return [
        {"role": "system", "content": MATH_SYSTEM},
        {"role": "user", "content": user},
    ]
