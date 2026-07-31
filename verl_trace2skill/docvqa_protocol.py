from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


DOCVQA_IMAGE_PATH = "/workspace/document.png"

DOCVQA_SYSTEM = """You are a DocVQA agent. You answer questions about document images using a command-line and Python ReAct loop.

You are not allowed to answer from the question alone. You must inspect or process the local image file using command-line tools and Python commands, then answer from the textual observations you produced.

Available action:

Action:
{"name": "bash", "arguments": {"command": "<shell command>"}}

The bash action runs in an isolated image workspace. Use shell commands and command-line Python, for example python -c "...", to inspect or process the provided image path.
Tool observations are text only. Do not expect the image to be displayed back to you.
When finished, output exactly:

Final answer: <short answer>

Return only the requested short answer after the Final answer prefix. Do not include reasoning in the final answer."""

FORMAT_WARNING = (
    "No valid action was parsed. Use exactly:\n"
    "Action:\n"
    '{"name": "bash", "arguments": {"command": "<shell command>"}}\n'
    "or finish after tool use with: Final answer: <short answer>"
)
TOOL_REQUIRED_WARNING = (
    "You must use a bash Action to inspect/process the image file before answering. "
    "Tool observations are text only; then provide Final answer."
)

_ACTION_PATTERN = re.compile(r"Action:\s*(\{.*?\})\s*$", flags=re.DOTALL | re.IGNORECASE)
_ACTION_FALLBACK = re.compile(r"Action:\s*(\{.*\})", flags=re.DOTALL | re.IGNORECASE)
_FINAL_PATTERN = re.compile(
    r"^\s*(?:final answer|answer)\s*[:：]\s*(.+?)\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class ReactDecision:
    kind: str
    action: dict[str, Any] | None = None
    answer: str | None = None
    message: str | None = None


def parse_action(text: str) -> dict[str, Any] | None:
    match = _ACTION_PATTERN.search(text) or _ACTION_FALLBACK.search(text)
    if match is None:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(match.group(1).strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    name = value.get("name")
    arguments = value.get("arguments")
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    return {"name": name, "arguments": arguments}


def extract_final_answer(text: str) -> str | None:
    matches = _FINAL_PATTERN.findall(text)
    return matches[-1].strip() if matches else None


def react_step(text: str, *, used_tool: bool) -> ReactDecision:
    action = parse_action(text)
    if action is not None:
        return ReactDecision(kind="action", action=action)
    answer = extract_final_answer(text)
    if answer is not None:
        if used_tool:
            return ReactDecision(kind="final", answer=answer)
        return ReactDecision(kind="retry", message=TOOL_REQUIRED_WARNING)
    return ReactDecision(kind="retry", message=FORMAT_WARNING)


def observation_message(name: str, text: str) -> str:
    return f"Observation from {name}:\n{text}"


def bash_action_command(action: dict[str, Any]) -> tuple[str | None, str | None]:
    """Validate the single action accepted by the DocVQA paper ReAct protocol."""
    name = str(action.get("name", ""))
    if name != "bash":
        return None, f"Unknown action '{name}'. Available action is bash."
    arguments = action.get("arguments")
    command = str(arguments.get("command", "")) if isinstance(arguments, dict) else ""
    if not command.strip():
        return None, "No shell command was provided."
    return command, None


def build_docvqa_messages(question: str) -> list[dict[str, str]]:
    user = (
        "Task: Answer the document visual question.\n"
        f"Image path: {DOCVQA_IMAGE_PATH}\n"
        f"Question: {question}\n"
        "You must call at least one bash action before giving the final answer."
    )
    return [
        {"role": "system", "content": DOCVQA_SYSTEM},
        {"role": "user", "content": user},
    ]


def initial_system_prompt_tokens(
    tokenizer: Any,
    *,
    paper_react_cli: bool,
    use_inference_chat_template: bool,
    apply_chat_template_kwargs: dict[str, Any],
) -> list[int]:
    """Return the prefix stripped from standalone incremental messages.

    Qwen3.5 requires a real user query in every rendered conversation, so its
    template rejects the historical ``[{}]`` probe. Paper ReAct appends only
    user observations and therefore has no synthetic system prefix to strip.
    """
    if paper_react_cli or use_inference_chat_template:
        return []
    return tokenizer.apply_chat_template(
        [{}],
        add_generation_prompt=False,
        tokenize=True,
        **apply_chat_template_kwargs,
    )


def paper_react_sampling_params(
    sampling_params: dict[str, Any],
    tokenizer: Any,
    *,
    max_new_tokens: int | None = None,
) -> dict[str, Any]:
    """Build tokenizer-free SGLang stops for one paper ReAct turn.

    VERL's async SGLang servers use ``skip_tokenizer_init=True``. A textual
    stop makes SGLang call ``tokenizer.decode`` and crashes those servers, so
    encode the first token of the historical ``Observation`` delimiter in the
    agent process and pass only token IDs across the boundary.
    """
    result = dict(sampling_params)
    result.pop("stop", None)
    observation_ids = tokenizer.encode("Observation", add_special_tokens=False)
    if not observation_ids:
        raise ValueError("tokenizer produced no tokens for the Observation delimiter")
    stop_token_ids = list(result.get("stop_token_ids") or [])
    if observation_ids[0] not in stop_token_ids:
        stop_token_ids.append(observation_ids[0])
    result["stop_token_ids"] = stop_token_ids
    if max_new_tokens is not None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        result["max_new_tokens"] = max_new_tokens
    return result


def choose_endpoint(
    endpoints: list[str],
    *,
    row_index: int,
    sample_index: int,
    samples: int,
) -> str:
    if not endpoints:
        raise ValueError("at least one model endpoint is required")
    if samples <= 0:
        raise ValueError("samples must be positive")
    return endpoints[(row_index * samples + sample_index) % len(endpoints)]


def response_budget_exceeded(usage: dict[str, Any] | None, limit: int) -> bool:
    """Return whether accumulated generated tokens reached the trajectory budget."""
    if limit <= 0 or not usage:
        return False
    value = usage.get("completion_tokens", usage.get("output_tokens", 0))
    try:
        return int(value) >= limit
    except (TypeError, ValueError):
        return False


def incremental_message_token_count(
    tokenizer: Any,
    message: dict[str, Any],
    *,
    apply_chat_template_kwargs: dict[str, Any],
) -> int:
    """Count an appended observation exactly as ``ToolAgentLoop`` renders it."""
    token_ids = tokenizer.apply_chat_template(
        [message],
        add_generation_prompt=True,
        tokenize=True,
        **apply_chat_template_kwargs,
    )
    return len(token_ids)
