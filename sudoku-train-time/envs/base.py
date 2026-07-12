from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationResult:
    task_id: str
    score: float
    response: str
    prediction: list[list[int]] | None
    error: str = ""
