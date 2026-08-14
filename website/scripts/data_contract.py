from __future__ import annotations

import re
from typing import Any


PRIVATE_PATH = re.compile(
    r"(?:/home/[^/\s]+|/mnt/(?:hdd|private)/[^\s]+|/workspace|/tmp)(?:/[^\s\"']*)?"
)
LOCAL_ENDPOINT = re.compile(
    r"(?:https?://)?(?:localhost|127(?:\.\d{1,3}){3}|0\.0\.0\.0)(?::\d+)?(?:/[^\s\"']*)?",
    re.IGNORECASE,
)


def redact_text(value: str) -> str:
    """Remove machine-local paths while leaving the surrounding trace readable."""
    return LOCAL_ENDPOINT.sub("[local endpoint]", PRIVATE_PATH.sub("[local path]", value))


def redact(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, dict):
        return {key: redact(item) for key, item in value.items()}
    return value


def compact_react_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for step in steps:
        item: dict[str, Any] = {
            "turn": int(step.get("turn", len(compacted) + 1)),
            "assistant": redact_text(str(step.get("assistant", ""))),
            "observation": redact_text(str(step.get("observation", ""))),
        }
        action = step.get("action")
        if isinstance(action, dict):
            item["action"] = redact(action)
        compacted.append(item)
    return compacted
