from __future__ import annotations

from typing import Any

from scripts.data_contract import compact_react_steps, redact


def _selected_score(record: dict[str, Any], case_id: str) -> dict[str, Any] | None:
    for split_name in ("dapo_eval", "aime_eval"):
        split = record.get(split_name)
        if not isinstance(split, dict):
            continue
        matches = [
            score
            for score in split.get("scores", [])
            if score.get("task_id") == case_id and score.get("sample_index", 0) == 0
        ]
        if matches:
            return matches[0]
    return None


def _task_fields(score: dict[str, Any]) -> dict[str, Any]:
    task = score.get("task", {}) if isinstance(score.get("task"), dict) else {}
    fields: dict[str, Any] = {
        "id": score.get("task_id"),
        "question": task.get("question", ""),
        "source": task.get("source", ""),
    }
    answer = task.get("answer", score.get("answer"))
    answers = task.get("answers", score.get("answers"))
    if answer is not None:
        fields["answer"] = answer
    if answers is not None:
        fields["answers"] = answers
    return fields


def extract_history_case(
    history: list[dict[str, Any]],
    *,
    case_id: str,
    checkpoint_generations: list[int],
    task_name: str,
    metric_name: str,
) -> dict[str, Any]:
    records = [row for row in history if isinstance(row.get("generation"), int)]
    train_points = [
        {"generation": row["generation"], "value": float(row["reward_mean"])}
        for row in records
        if isinstance(row.get("reward_mean"), (int, float))
    ]

    checkpoint_items: list[dict[str, Any]] = []
    selected_task: dict[str, Any] | None = None
    for generation in checkpoint_generations:
        record = next((row for row in records if row["generation"] == generation), None)
        if record is None:
            continue
        score = _selected_score(record, case_id)
        if score is None:
            continue
        selected_task = selected_task or _task_fields(score)
        checkpoint: dict[str, Any] = {
            "generation": generation,
            "score": float(score.get("score", 0.0)),
            "prediction": score.get("prediction", ""),
            "terminationReason": score.get("termination_reason", ""),
            "steps": compact_react_steps(score.get("react_steps", [])),
        }
        for key in ("anls", "acc"):
            if isinstance(score.get(key), (int, float)):
                checkpoint[key] = float(score[key])
        checkpoint_items.append(checkpoint)

    case = selected_task or {"id": case_id, "question": "", "source": ""}
    case["checkpoints"] = checkpoint_items
    payload = {
        "metadata": {
            "task": task_name,
            "method": "Agentic ESOpt",
            "metric": metric_name,
            "sourceFiles": ["training history"],
        },
        "configurations": [],
        "curves": [{"id": "train", "kind": "train", "label": "Train reward", "points": train_points}],
        "checkpoints": [
            {
                "generation": generation,
                "trajectoryAvailable": any(item["generation"] == generation for item in checkpoint_items),
                "caseIds": [case_id] if any(item["generation"] == generation for item in checkpoint_items) else [],
            }
            for generation in sorted({row["generation"] for row in records})
        ],
        "cases": [case],
        "finalResults": [],
    }
    return redact(payload)
