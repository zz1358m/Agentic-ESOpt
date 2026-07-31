from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "verl"))
from verl.utils.checkpoint import fsdp_checkpoint_manager as MODULE  # noqa: E402


def test_checkpoint_step_parses_only_numeric_global_step_components() -> None:
    assert MODULE.checkpoint_step("/tmp/run/global_step_120/actor") == 120
    assert MODULE.checkpoint_step("/tmp/run/global_step_latest/actor") is None
    assert MODULE.checkpoint_step("/tmp/run/actor") is None


def test_protected_checkpoint_steps_validate_environment() -> None:
    with mock.patch.dict(os.environ, {"VERL_PROTECTED_CHECKPOINT_STEPS": "60,120,180"}):
        assert MODULE.protected_checkpoint_steps() == {60, 120, 180}
    with mock.patch.dict(os.environ, {"VERL_PROTECTED_CHECKPOINT_STEPS": "0"}):
        try:
            MODULE.protected_checkpoint_steps()
        except ValueError as exc:
            assert "must be positive" in str(exc)
        else:
            raise AssertionError("zero protected checkpoint step was accepted")


def test_resume_discovers_existing_same_component_checkpoints() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        expected = []
        for step in (60, 119, 120):
            actor = root / f"global_step_{step}" / "actor"
            actor.mkdir(parents=True)
            expected.append(str(actor))
        (root / "global_step_invalid" / "actor").mkdir(parents=True)

        assert MODULE.sibling_checkpoint_paths(expected[-1]) == expected


def test_rolling_retention_keeps_milestones_and_only_latest_non_milestone() -> None:
    paths = [f"/tmp/run/global_step_{step}/actor" for step in (60, 61, 120, 121, 122)]

    removed, retained = MODULE.rolling_checkpoint_partition(
        paths,
        max_ckpt_to_keep=1,
        protected_steps={60, 120, 180},
    )

    assert retained == [paths[0], paths[2], paths[4]]
    assert removed == [paths[1], paths[3]]
