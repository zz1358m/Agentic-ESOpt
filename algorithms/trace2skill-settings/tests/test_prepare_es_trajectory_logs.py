from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_es_trajectory_logs.py"
SPEC = importlib.util.spec_from_file_location("prepare_es_trajectory_logs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_history(path: Path, task_ids: list[str]) -> None:
    path.write_text(
        json.dumps([{"generation": 0, "case_batch": task_ids}]),
        encoding="utf-8",
    )


def touch(path: Path) -> None:
    path.write_text("trajectory\n", encoding="utf-8")


def test_math_exact_window_selects_failures_only(tmp_path: Path) -> None:
    task_ids = [
        "dapo_00000000-0000-0000-0000-000000000001",
        "dapo_00000000-0000-0000-0000-000000000002",
    ]
    history = tmp_path / "history.json"
    write_history(history, task_ids)
    for task_id in task_ids:
        touch(tmp_path / f"math_agent_gen000_candidate000_seed1_{task_id}_sample00_FAILED.md")
        touch(tmp_path / f"math_agent_gen000_candidate001_seed2_{task_id}_sample00_SUCCEED.md")

    selected, metadata = MODULE.select_math(
        SimpleNamespace(
            trace_roots=[tmp_path],
            history=history,
            checkpoint_step=1,
            task_count=2,
            population=2,
            case_batch_size=2,
            one_error_per_task=True,
            one_per_outcome_per_task=False,
            first_generation=0,
            last_generation=0,
        )
    )

    assert len(selected) == 2
    assert {row["task_id"] for row in selected} == set(task_ids)
    assert {row["outcome"] for row in selected} == {"FAILED"}
    assert metadata["selection"] == "one failed trajectory from each exact last task occurrence"


def test_docvqa_exact_window_selects_one_success_and_one_failure(tmp_path: Path) -> None:
    task_ids = ["doc_a", "doc_b"]
    history = tmp_path / "history.json"
    write_history(history, task_ids)
    for task_id in task_ids:
        touch(tmp_path / f"docvqa_agent_gen000_candidate000_seed1_{task_id}_sample00_FAILED.md")
        touch(tmp_path / f"docvqa_agent_gen000_candidate001_seed2_{task_id}_sample00_SUCCEED.md")

    selected, metadata = MODULE.select_docvqa(
        SimpleNamespace(
            trace_root=tmp_path,
            history=history,
            checkpoint_step=1,
            task_count=2,
            population=2,
            one_per_outcome_per_task=True,
        )
    )

    assert len(selected) == 4
    assert {row["task_id"] for row in selected} == set(task_ids)
    assert {row["outcome"] for row in selected} == {"FAILED", "SUCCEED"}
    for task_id in task_ids:
        assert [row["outcome"] for row in selected if row["task_id"] == task_id] == [
            "FAILED",
            "SUCCEED",
        ]
    assert metadata["selection"] == "one trajectory per outcome from each exact last task occurrence"


def test_canonical_launcher_uses_task_specific_selection_flags() -> None:
    launcher = (Path(__file__).resolve().parents[3] / "scripts" / "es_skill_workflow.sh").read_text(
        encoding="utf-8"
    )
    math_block, docvqa_block = launcher.split('if [[ "$TASK" == "math" ]]', 1)[1].split(
        "else", 1
    )

    assert "--one-error-per-task" in math_block
    assert "--one-per-outcome-per-task" not in math_block
    assert "--one-per-outcome-per-task" in docvqa_block
