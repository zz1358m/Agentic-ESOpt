from __future__ import annotations

import math
import random
from collections import Counter
from statistics import fmean
from typing import Any


def _valid_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        if row.get("error") not in (None, ""):
            continue
        value = row.get("anls")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if math.isfinite(float(value)) and float(value) >= 0.0:
            result.append(row)
    return result


def _mean_number(rows: list[dict[str, Any]], key: str) -> float | None:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if math.isfinite(float(value)):
            values.append(float(value))
    return fmean(values) if values else None


def _steps(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [step for step in (row.get("react_steps") or []) if isinstance(step, dict)]


def _bash_steps(row: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for step in _steps(row):
        if not isinstance(step, dict):
            continue
        action = step.get("action")
        if (
            isinstance(action, dict)
            and action.get("name") == "bash"
            and "observation" in step
        ):
            result.append(step)
    return result


def _timed_out(step: dict[str, Any]) -> bool:
    return "timed out after" in str(step.get("observation", "")).lower()


def _tool_succeeded(row: dict[str, Any]) -> bool:
    for step in _bash_steps(row):
        observation = str(step.get("observation", ""))
        if not _timed_out(step) and "[ERROR]" not in observation:
            return True
    return False


def _format_retries(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [step for step in _steps(row) if not isinstance(step.get("action"), dict)]


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = _valid_rows(rows)
    react_errors = Counter(str(row["react_error"]) for row in rows if row.get("react_error"))
    usage_keys = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "accumulated_response_tokens",
    )
    usage = {}
    for key in usage_keys:
        values = [
            float(row["usage"][key])
            for row in valid
            if isinstance(row.get("usage"), dict)
            and isinstance(row["usage"].get(key), (int, float))
            and not isinstance(row["usage"].get(key), bool)
        ]
        if values:
            usage[f"mean_{key}"] = fmean(values)
    format_retry_counts = [len(_format_retries(row)) for row in valid]
    timeout_counts = [sum(_timed_out(step) for step in _bash_steps(row)) for row in valid]
    tool_call_rate = fmean(float(bool(_bash_steps(row))) for row in valid) if valid else None
    return {
        "records": len(rows),
        "valid_records": len(valid),
        "request_errors": len(rows) - len(valid),
        "mean_anls": _mean_number(valid, "anls"),
        "mean_accuracy": _mean_number(valid, "acc"),
        "tool_call_rate": tool_call_rate,
        "tool_success_rate": fmean(float(_tool_succeeded(row)) for row in valid) if valid else None,
        "mean_turns": (
            fmean(len(_steps(row)) + (0 if row.get("react_error") else 1) for row in valid)
            if valid
            else None
        ),
        "mean_latency_s": _mean_number(valid, "latency_s"),
        "format_retry_rows": sum(count > 0 for count in format_retry_counts),
        "format_retries": sum(format_retry_counts),
        "bash_timeout_rows": sum(count > 0 for count in timeout_counts),
        "bash_timeouts": sum(timeout_counts),
        "react_errors": dict(sorted(react_errors.items())),
        "usage": usage,
    }


def _task_means(rows: list[dict[str, Any]]) -> dict[str, tuple[set[int], float]]:
    grouped: dict[str, list[tuple[int, float]]] = {}
    for row in _valid_rows(rows):
        task_id = str(row.get("task_id", ""))
        sample_index = int(row.get("sample_index", 0))
        grouped.setdefault(task_id, []).append((sample_index, float(row["anls"])))
    return {
        task_id: ({sample for sample, _ in values}, fmean(score for _, score in values))
        for task_id, values in grouped.items()
    }


def compare_results(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    *,
    bootstrap_samples: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    before_tasks = _task_means(before)
    after_tasks = _task_means(after)
    if set(before_tasks) != set(after_tasks):
        raise ValueError("before/after task sets differ")
    if not before_tasks:
        raise ValueError("no paired valid DocVQA tasks")
    for task_id in before_tasks:
        if before_tasks[task_id][0] != after_tasks[task_id][0]:
            raise ValueError(f"before/after sample sets differ for {task_id}")

    task_ids = sorted(before_tasks)
    before_values = [before_tasks[task_id][1] for task_id in task_ids]
    after_values = [after_tasks[task_id][1] for task_id in task_ids]
    differences = [right - left for left, right in zip(before_values, after_values, strict=True)]
    rng = random.Random(seed)
    estimates = []
    for _ in range(bootstrap_samples):
        estimates.append(fmean(differences[rng.randrange(len(differences))] for _ in differences))
    estimates.sort()
    lower = estimates[int(0.025 * (len(estimates) - 1))]
    upper = estimates[int(0.975 * (len(estimates) - 1))]
    return {
        "paired_tasks": len(task_ids),
        "mean_anls_before": fmean(before_values),
        "mean_anls_after": fmean(after_values),
        "mean_anls_delta": fmean(differences),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "bootstrap_95_ci": [lower, upper],
    }
