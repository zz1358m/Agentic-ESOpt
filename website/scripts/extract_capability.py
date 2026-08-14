from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from scripts.data_contract import redact_text


WEBARENA_EVAL = re.compile(
    r"^\[eval\] eval_after_epoch_(?P<epoch>\d+).*\btask=(?P<task>\d+)\s+score=(?P<score>-?[\d.]+)"
)


def parse_webarena_task_scores(text: str) -> dict[int, dict[int, float]]:
    """Index every retained WebArena task score by task ID and evaluation epoch."""
    scores: dict[int, dict[int, float]] = {}
    for line in text.splitlines():
        match = WEBARENA_EVAL.match(line)
        if match:
            scores.setdefault(int(match.group("task")), {})[
                int(match.group("epoch"))
            ] = float(match.group("score"))
    return scores


def extract_sudoku_case_checkpoints(
    history: list[dict[str, Any]], *, case_id: str, run_index: int = 0
) -> list[dict[str, Any]]:
    """Return one fixed Sudoku case at every retained periodic-eval checkpoint."""
    checkpoints: list[dict[str, Any]] = []
    for row in history:
        evaluation = row.get("eval")
        if not isinstance(evaluation, dict):
            continue
        runs = evaluation.get("runs")
        if not isinstance(runs, list) or len(runs) <= run_index:
            continue
        scores = runs[run_index].get("scores", [])
        selected = next((score for score in scores if score.get("task_id") == case_id), None)
        if selected is None:
            continue
        checkpoints.append(
            {
                "optimizationStep": int(row["generation"]),
                "sourceStep": int(row["generation"]),
                "stepKind": "es_generation",
                "aggregateMetric": float(evaluation["average"]),
                "aggregateMetricLabel": "Periodic evaluation success",
                "score": float(selected["score"]),
                "prediction": selected.get("prediction", []),
                "feedback": redact_text(str(selected.get("feedback", ""))),
            }
        )
    return checkpoints


def parse_webarena_case_progression(
    text: str,
    *,
    task_id: int,
    aggregate_by_epoch: dict[int, float],
    selected_epochs: list[int],
    final_output: str,
    parsed_scores: dict[int, dict[int, float]] | None = None,
) -> list[dict[str, Any]]:
    score_index = (
        parsed_scores
        if parsed_scores is not None
        else parse_webarena_task_scores(text)
    )
    scores = score_index.get(task_id, {})
    checkpoints: list[dict[str, Any]] = []
    for epoch in selected_epochs:
        if epoch not in scores or epoch not in aggregate_by_epoch:
            raise ValueError(f"WebArena task {task_id} lacks a linked score at epoch {epoch}")
        item: dict[str, Any] = {
            "optimizationStep": epoch,
            "sourceStep": epoch,
            "stepKind": "evaluation_epoch",
            "modelCheckpointId": f"eval_after_epoch_{epoch:03d}",
            "aggregateMetric": aggregate_by_epoch[epoch],
            "aggregateMetricLabel": "Periodic WebArena-Lite success",
            "score": scores[epoch],
            "outputUnavailable": epoch != selected_epochs[-1],
        }
        if epoch == selected_epochs[-1]:
            item["output"] = redact_text(final_output)
            item["outputUnavailable"] = False
        checkpoints.append(item)
    return checkpoints


def extract_ahd_heuristic_checkpoints(
    directory: Path, *, generations: list[int]
) -> list[dict[str, Any]]:
    checkpoints: list[dict[str, Any]] = []
    for generation in generations:
        path = directory / f"population_generation_{generation}.json"
        if not path.is_file():
            raise ValueError(f"missing AHD best-population artifact: {path.name}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        code = redact_text(str(payload.get("code", "")))
        if not code.strip():
            raise ValueError(f"empty AHD heuristic at generation {generation}")
        checkpoints.append(
            {
                "optimizationStep": generation,
                "sourceStep": generation,
                "stepKind": "search_generation",
                "objective": float(payload["objective"]),
                "algorithm": redact_text(str(payload.get("algorithm", ""))),
                "heuristic": code,
                "isFinal": generation == generations[-1],
            }
        )
    return checkpoints
