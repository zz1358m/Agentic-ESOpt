from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_four_gpu_eval import (  # noqa: E402
    LEGACY_NO_SKILL_PROFILE,
    REPO_REACT_V1_50X4096_PROFILE,
    REPORT_NONTHINKING_PROFILE,
    REPORT_PROFILE,
    SERVED_MODEL,
    evaluator_command,
    expected_result_keys,
    resolve_eval_gpus,
)


GPU_QUERY = "\n".join(
    f"{index}, GPU-uuid-{index}, NVIDIA A100-SXM4-80GB" for index in range(7)
)


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_math_eval_requires_exact_physical_gpus_3_through_6() -> None:
    physical, identities = resolve_eval_gpus("3,4,5,6", "", query_output=GPU_QUERY)
    assert physical == ("3", "4", "5", "6")
    assert [identity.uuid for identity in identities] == [f"GPU-uuid-{index}" for index in range(3, 7)]

    with pytest.raises(ValueError, match="3,4,5,6"):
        resolve_eval_gpus("0,1,2,3", "", query_output=GPU_QUERY)


def test_expected_keys_cover_dapo100_and_aime30_at_16_samples(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "dapo_test.jsonl",
        [{"id": f"dapo-{index}", "question": f"dapo q{index}"} for index in range(100)],
    )
    _write_jsonl(
        tmp_path / "aime_2026.jsonl",
        [{"id": f"aime-{index}", "question": f"aime q{index}"} for index in range(30)],
    )

    keys = expected_result_keys(tmp_path, samples=16)
    assert set(keys) == {"dapo100", "aime2026"}
    assert len(keys["dapo100"]) == 1600
    assert len(keys["aime2026"]) == 480
    assert "dapo100:dapo-0:sample00" in keys["dapo100"]
    assert "aime2026:aime-29:sample15" in keys["aime2026"]


def test_evaluator_command_fixes_sampling_and_protocol(tmp_path: Path) -> None:
    endpoints = [f"http://127.0.0.1:{18180 + index}/v1" for index in range(4)]
    command = evaluator_command(
        python=sys.executable,
        evaluator=tmp_path / "eval.py",
        endpoints=endpoints,
        model_path=tmp_path / "model",
        math_root=tmp_path / "math",
        out_dir=tmp_path / "out",
        samples=16,
        concurrency=8,
        seed=20260629,
        resume=True,
    )
    joined = " ".join(command)
    assert SERVED_MODEL in command
    assert "--datasets dapo100,aime2026" in joined
    assert "--samples 16" in joined
    assert "--temperature 1.0" in joined
    assert "--top-p 1.0" in joined
    assert "--top-k 40" in joined
    assert "--math-max-turns 50" in joined
    assert "--math-max-tokens 4096" in joined
    assert command[-1] == "--resume"


def test_report_profile_uses_four_4096_token_thinking_samples(tmp_path: Path) -> None:
    command = evaluator_command(
        python=sys.executable,
        evaluator=tmp_path / "eval.py",
        endpoints=["http://127.0.0.1:18180/v1"],
        model_path=tmp_path / "model",
        math_root=tmp_path / "math",
        out_dir=tmp_path / "out",
        samples=4,
        concurrency=4,
        seed=20260629,
        resume=True,
        profile=REPORT_PROFILE,
    )
    joined = " ".join(command)
    assert "--samples 4" in joined
    assert "--top-p 0.95" in joined
    assert "--top-k 20" in joined
    assert "--presence-penalty 1.5" in joined
    assert "--math-max-tokens 4096" in joined
    assert "--math-mode direct" in joined
    assert "--math-enable-thinking" in command


def test_report_nonthinking_candidate_keeps_reasoning_sampling(tmp_path: Path) -> None:
    command = evaluator_command(
        python=sys.executable,
        evaluator=tmp_path / "eval.py",
        endpoints=["http://127.0.0.1:18180/v1"],
        model_path=tmp_path / "model",
        math_root=tmp_path / "math",
        out_dir=tmp_path / "out",
        samples=4,
        concurrency=4,
        seed=20260629,
        resume=True,
        profile=REPORT_NONTHINKING_PROFILE,
    )
    joined = " ".join(command)
    assert "--top-p 1.0" in joined
    assert "--top-k 40" in joined
    assert "--presence-penalty 2.0" in joined
    assert "--math-max-tokens 4096" in joined
    assert "--math-mode direct" in joined
    assert "--math-enable-thinking" not in command


def test_legacy_candidate_selects_original_no_skill_prompt(tmp_path: Path) -> None:
    command = evaluator_command(
        python=sys.executable,
        evaluator=tmp_path / "eval.py",
        endpoints=["http://127.0.0.1:18180/v1"],
        model_path=tmp_path / "model",
        math_root=tmp_path / "math",
        out_dir=tmp_path / "out",
        samples=4,
        concurrency=4,
        seed=20260629,
        resume=True,
        profile=LEGACY_NO_SKILL_PROFILE,
    )
    joined = " ".join(command)
    assert "--math-mode direct" in joined
    assert "--math-direct-prompt legacy-no-skill" in joined
    assert "--math-enable-thinking" not in command


def test_repo_react_alignment_profile_is_fixed_to_50_turns_and_4096_tokens(
    tmp_path: Path,
) -> None:
    command = evaluator_command(
        python=sys.executable,
        evaluator=tmp_path / "eval.py",
        endpoints=["http://127.0.0.1:18180/v1"],
        model_path=tmp_path / "model",
        math_root=tmp_path / "math",
        out_dir=tmp_path / "out",
        samples=4,
        concurrency=8,
        seed=20260629,
        resume=True,
        profile=REPO_REACT_V1_50X4096_PROFILE,
    )
    joined = " ".join(command)
    assert "--samples 4" in joined
    assert "--math-max-turns 50" in joined
    assert "--math-max-tokens 4096" in joined
    assert "--math-mode react" in joined
    assert "--math-react-prompt repo-react-v1" in joined
    assert "--retry-react-errors" in command
    assert "--math-enable-thinking" not in command
