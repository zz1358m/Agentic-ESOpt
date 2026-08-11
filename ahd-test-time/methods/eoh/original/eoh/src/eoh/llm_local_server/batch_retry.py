"""Retry a failed inference batch as ordered smaller batches."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar


Item = TypeVar("Item")
Result = TypeVar("Result")


def run_with_oom_splitting(
    items: Sequence[Item],
    *,
    run_batch: Callable[[list[Item]], Result],
    is_oom: Callable[[Exception], bool],
    on_split: Callable[[], None] | None = None,
) -> list[Result]:
    batch = list(items)
    if not batch:
        return []
    try:
        return [run_batch(batch)]
    except Exception as exc:
        if not is_oom(exc) or len(batch) == 1:
            raise

    # The failed run_batch frame and its tensors are gone before cleanup and
    # recursion begin, so the smaller retry can actually reclaim GPU memory.
    if on_split is not None:
        on_split()
    midpoint = len(batch) // 2
    return run_with_oom_splitting(
        batch[:midpoint],
        run_batch=run_batch,
        is_oom=is_oom,
        on_split=on_split,
    ) + run_with_oom_splitting(
        batch[midpoint:],
        run_batch=run_batch,
        is_oom=is_oom,
        on_split=on_split,
    )
