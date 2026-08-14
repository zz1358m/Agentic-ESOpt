from __future__ import annotations

import json
import re
from typing import Any


PRIVATE_PATH = re.compile(r"/(?:home|mnt|workspace|tmp)/")
LOCAL_ENDPOINT = re.compile(
    r"(?:https?://)?(?:localhost|127(?:\.\d{1,3}){3}|0\.0\.0\.0)(?::\d+)?",
    re.IGNORECASE,
)
EMAIL_ADDRESS = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_NUMBER = re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)")
ALLOWED_CURVE_KINDS = {"train", "eval", "final", "baseline"}


def validate_public_payload(payload: dict[str, Any]) -> None:
    required = {"metadata", "curves", "checkpoints", "cases", "finalResults"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")

    for curve in payload["curves"]:
        if curve.get("kind") not in ALLOWED_CURVE_KINDS:
            raise ValueError(f"unknown curve kind: {curve.get('kind')}")
        generations = [point.get("generation") for point in curve.get("points", [])]
        if generations != sorted(generations):
            raise ValueError(f"curve generations are not sorted: {curve.get('id')}")

    serialized = json.dumps(payload, ensure_ascii=False)
    if PRIVATE_PATH.search(serialized):
        raise ValueError("private path found in public payload")
    if LOCAL_ENDPOINT.search(serialized):
        raise ValueError("local endpoint found in public payload")
    if EMAIL_ADDRESS.search(serialized) or PHONE_NUMBER.search(serialized):
        raise ValueError("contact information found in public payload")
