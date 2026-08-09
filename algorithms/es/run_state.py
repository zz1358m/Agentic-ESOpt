"""Shared scheduling and replay helpers for Agentic-ESOpt runners.

The task runners intentionally keep task-specific rollout code, but use this
module for the pieces that must have identical semantics across settings:

* an explicit ``sigma_start -> sigma_end`` schedule;
* durable ``history.json`` writes; and
* replaying completed seed/reward updates on a freshly initialized server.
"""

from __future__ import annotations

import json
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


SUPPORTED_SIGMA_SCHEDULES = ("constant", "linear", "cosine")


def validate_es_run_shape(
    *,
    generations: int,
    population: int,
    case_batch_size: int,
    allow_zero_generations: bool = False,
) -> None:
    minimum_generations = 0 if allow_zero_generations else 1
    if int(generations) < minimum_generations:
        qualifier = "non-negative" if allow_zero_generations else "positive"
        raise ValueError(f"generations must be {qualifier}.")
    if int(population) <= 0:
        raise ValueError("population must be positive.")
    if int(case_batch_size) <= 0:
        raise ValueError("case_batch_size must be positive.")


def normalize_sigma_schedule(schedule: str) -> str:
    """Normalize legacy spellings while keeping one public vocabulary."""

    value = str(schedule or "constant").strip().lower().replace("_", "-")
    if value not in SUPPORTED_SIGMA_SCHEDULES:
        raise ValueError(
            f"Unsupported sigma schedule {schedule!r}; choose one of "
            f"{', '.join(SUPPORTED_SIGMA_SCHEDULES)}."
        )
    return value


def resolve_warmup_steps(total_steps: int, warmup_steps: int) -> int:
    """Validate and clamp an explicit warmup."""

    total = max(0, int(total_steps))
    warmup = int(warmup_steps)
    if warmup < 0:
        raise ValueError("warmup_steps must be non-negative.")
    # For a decaying schedule, reserve the final step for sigma_end.
    return min(warmup, max(0, total - 1))


def sigma_at_step(
    *,
    sigma_start: float,
    sigma_end: float,
    step: int,
    total_steps: int,
    schedule: str,
    warmup_steps: int = 0,
) -> float:
    """Return sigma for a zero-indexed step, including both requested ends.

    When decay is active and there is at least one post-warmup step, the first
    decay step is ``sigma_start`` and the final step is exactly ``sigma_end``.
    A constant schedule always returns ``sigma_start``.
    """

    start = float(sigma_start)
    end = float(sigma_end)
    if not math.isfinite(start) or not math.isfinite(end) or start < 0.0 or end < 0.0:
        raise ValueError("sigma_start and sigma_end must be finite and non-negative.")
    normalized = normalize_sigma_schedule(schedule)
    total = max(0, int(total_steps))
    if normalized == "constant" or total <= 1:
        return start

    warmup = resolve_warmup_steps(total, warmup_steps)
    index = min(max(0, int(step)), total - 1)
    if index == total - 1:
        return end
    if index < warmup or warmup >= total:
        return start

    denominator = max(1, total - 1 - warmup)
    progress = min(1.0, max(0.0, (index - warmup) / denominator))
    if normalized == "linear":
        weight = progress
    else:
        weight = 0.5 * (1.0 - math.cos(math.pi * progress))
    return start + (end - start) * weight


def read_history(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"ES history must be a JSON list: {source}")
    return [row for row in payload if isinstance(row, dict)]


def atomic_write_history(path: str | Path, history: Sequence[dict[str, Any]]) -> Path:
    """Write history without exposing a partially written JSON file."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(list(history), indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return target


def completed_update_records(
    history: Iterable[dict[str, Any]],
    *,
    limit: int | None = None,
    require_contiguous_generations: bool = True,
) -> list[dict[str, Any]]:
    """Select replayable records and validate their generation sequence."""

    records = [
        row
        for row in history
        if row.get("update_applied", True) is not False
        if isinstance(row.get("seeds"), list)
        and isinstance(row.get("rewards"), list)
        and len(row["seeds"]) == len(row["rewards"])
        and len(row["seeds"]) > 0
    ]
    if limit is not None and int(limit) >= 0:
        records = records[: int(limit)]
    if require_contiguous_generations and records:
        generations = [int(row.get("generation", index)) for index, row in enumerate(records)]
        expected = list(range(generations[0], generations[0] + len(generations)))
        if generations != expected:
            raise ValueError(
                "Replay history generations are not contiguous: "
                f"found {generations}, expected {expected}."
            )
    return records


def history_prefix_through_updates(
    history: Sequence[dict[str, Any]],
    update_count: int,
) -> list[dict[str, Any]]:
    """Return a history prefix containing exactly ``update_count`` updates.

    This is used by partial replay. Keeping the full source history after only
    replaying its first updates would leave duplicate future generations in the
    new output and make the next resume ambiguous.
    """

    wanted = max(0, int(update_count))
    available = completed_update_records(history, require_contiguous_generations=False)
    if wanted >= len(available):
        return list(history)

    prefix: list[dict[str, Any]] = []
    seen = 0
    for row in history:
        is_update = (
            row.get("update_applied", True) is not False
            and isinstance(row.get("seeds"), list)
            and isinstance(row.get("rewards"), list)
            and len(row["seeds"]) == len(row["rewards"])
            and len(row["seeds"]) > 0
        )
        if is_update and seen >= wanted:
            break
        prefix.append(row)
        if is_update:
            seen += 1
    return prefix


def validate_seed_sequence(
    records: Sequence[dict[str, Any]],
    *,
    population: int,
    seed: int,
) -> int:
    """Validate a standard task runner's deterministic seed stream.

    Returns the next generation index. A mismatch is rejected instead of
    silently continuing from a different ES trajectory.
    """

    population_size = int(population)
    if population_size <= 0:
        raise ValueError("population must be positive.")
    rng = random.Random(int(seed))
    for index, record in enumerate(records):
        generation = int(record.get("generation", index))
        if generation != index:
            raise ValueError(
                f"Replay history must start at generation 0; record {index} "
                f"has generation {generation}."
            )
        actual = [int(value) for value in record["seeds"]]
        if len(actual) != population_size:
            raise ValueError(
                f"Replay history population mismatch at generation {generation}: "
                f"history has {len(actual)}, runner expects {population_size}."
            )
        expected = [rng.randrange(1, 2**31 - 1) for _ in range(population_size)]
        if actual != expected:
            raise ValueError(
                f"Replay history seed mismatch at generation {generation}; "
                "use the original --seed and --population."
            )
    return len(records)


def replay_http_updates(
    *,
    endpoints: Sequence[str],
    records: Sequence[dict[str, Any]],
    post_json: Callable[..., dict[str, Any]],
    timeout: float,
    default_alpha: float,
    default_reward_normalization: str,
    default_reward_normalization_ddof: int = 0,
    default_reward_normalization_eps: float = 1e-8,
) -> list[dict[str, Any]]:
    """Replay seed/reward updates on every initialized model endpoint."""

    replay_log: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        payload = {
            "seeds": [int(seed) for seed in record["seeds"]],
            "rewards": [float(reward) for reward in record["rewards"]],
            "alpha": float(record.get("alpha", default_alpha)),
            "reward_normalization": str(
                record.get("reward_normalization", default_reward_normalization)
            ),
            "reward_normalization_ddof": int(
                record.get("reward_normalization_ddof", default_reward_normalization_ddof)
            ),
            "reward_normalization_eps": float(
                record.get("reward_normalization_eps", default_reward_normalization_eps)
            ),
        }
        endpoint_results = []
        for endpoint in endpoints:
            response = post_json(
                f"{str(endpoint).rstrip('/')}/es/update",
                payload,
                timeout=timeout,
            )
            endpoint_results.append({"endpoint": endpoint, "update": response})
        replay_log.append(
            {
                "record_index": record_index,
                "generation": record.get("generation"),
                "alpha": payload["alpha"],
                "seeds": payload["seeds"],
                "endpoints": endpoint_results,
            }
        )
    return replay_log


def history_output_path(
    *,
    root: str | Path,
    task_dir: str,
    run_id: str,
    explicit_path: str = "",
) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()
    return Path(root).resolve() / "runs" / task_dir / run_id / "history.json"


def map_endpoint_serial(
    *,
    endpoints: Sequence[str],
    count: int,
    worker: Callable[[int, str], Any],
) -> list[Any]:
    """Map population work while allowing one active job per endpoint.

    ES servers hold a single mutable perturbation. A generic thread pool can
    accidentally schedule a second seed on an endpoint whose first seed is
    still active, so each endpoint gets its own serial index queue here.
    """

    normalized_endpoints = [str(endpoint) for endpoint in endpoints]
    item_count = int(count)
    if not normalized_endpoints:
        raise ValueError("At least one endpoint is required.")
    if item_count < 0:
        raise ValueError("count must be non-negative.")
    if item_count == 0:
        return []

    results: list[Any] = [None] * item_count
    with ThreadPoolExecutor(max_workers=min(len(normalized_endpoints), item_count)) as pool:
        queues = {
            endpoint_index: iter(range(endpoint_index, item_count, len(normalized_endpoints)))
            for endpoint_index in range(len(normalized_endpoints))
        }
        futures: dict[Any, int] = {}

        def submit_next(endpoint_index: int) -> None:
            try:
                item_index = next(queues[endpoint_index])
            except StopIteration:
                return
            future = pool.submit(worker, item_index, normalized_endpoints[endpoint_index])
            futures[future] = endpoint_index

        for endpoint_index in queues:
            submit_next(endpoint_index)
        while futures:
            future = next(as_completed(tuple(futures)))
            endpoint_index = futures.pop(future)
            item_index, value = future.result()
            if int(item_index) < 0 or int(item_index) >= item_count:
                raise ValueError(f"Worker returned invalid item index: {item_index}")
            results[int(item_index)] = value
            submit_next(endpoint_index)
    return results
