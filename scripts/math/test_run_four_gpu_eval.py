from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_four_gpu_eval import (  # noqa: E402
    REPO_REACT_V1_50X4096_PROFILE,
    SERVED_MODEL,
    build_endpoint_plan,
    evaluator_command,
    expected_result_keys,
    parse_datasets,
    resolve_eval_gpus,
    server_command,
)


GPU_QUERY = "\n".join(
    f"{index}, GPU-uuid-{index}, NVIDIA A100-SXM4-80GB" for index in range(7)
)


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_math_eval_requires_four_unique_physical_gpus() -> None:
    physical, identities = resolve_eval_gpus("0,1,2,3", "", query_output=GPU_QUERY)
    assert physical == ("0", "1", "2", "3")
    assert [identity.uuid for identity in identities] == [f"GPU-uuid-{index}" for index in range(4)]

    with pytest.raises(ValueError, match="four unique"):
        resolve_eval_gpus("0,1,2", "", query_output=GPU_QUERY)


def test_stage2_endpoint_plan_places_two_engines_on_each_gpu() -> None:
    _, identities = resolve_eval_gpus("0,1,2,3", "", query_output=GPU_QUERY)

    plan = build_endpoint_plan(identities, port_base=18180)

    assert [row.port for row in plan] == list(range(18180, 18188))
    assert [row.gpu_index for row in plan] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert len({row.endpoint for row in plan}) == 8


def test_math_server_disables_unsafe_overlapped_cuda_graph_execution(tmp_path: Path) -> None:
    command = server_command(
        sys.executable,
        tmp_path / "model",
        18180,
        context_length=131072,
        memory_fraction=0.3,
    )

    assert "--disable-cuda-graph" in command
    assert "--disable-overlap-schedule" in command


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


def test_dataset_selection_supports_independent_aime_run(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "aime_2026.jsonl",
        [{"id": f"aime-{index}", "question": f"aime q{index}"} for index in range(30)],
    )

    datasets = parse_datasets("aime2026")
    keys = expected_result_keys(tmp_path, samples=4, datasets=datasets)

    assert datasets == ("aime2026",)
    assert set(keys) == {"aime2026"}
    assert len(keys["aime2026"]) == 120


def test_dataset_selection_rejects_unknown_or_empty_values() -> None:
    with pytest.raises(ValueError, match="must select"):
        parse_datasets("")
    with pytest.raises(ValueError, match="must select"):
        parse_datasets("aime2026,unknown")


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


def test_evaluator_command_can_run_aime_independently(tmp_path: Path) -> None:
    command = evaluator_command(
        python=sys.executable,
        evaluator=tmp_path / "eval.py",
        endpoints=["http://127.0.0.1:19180/v1"],
        model_path=tmp_path / "model",
        math_root=tmp_path / "math",
        out_dir=tmp_path / "out",
        samples=4,
        concurrency=8,
        seed=20260627,
        resume=False,
        profile=REPO_REACT_V1_50X4096_PROFILE,
        datasets=("aime2026",),
    )

    assert command[command.index("--datasets") + 1] == "aime2026"


def test_evaluator_command_passes_the_recorded_math_skill(tmp_path: Path) -> None:
    skill_file = tmp_path / "evidence" / "math_skill" / "SKILL.md"
    command = evaluator_command(
        python=sys.executable,
        evaluator=tmp_path / "eval.py",
        endpoints=["http://127.0.0.1:18180/v1"],
        model_path=tmp_path / "model",
        math_root=tmp_path / "math",
        out_dir=tmp_path / "out",
        samples=4,
        concurrency=8,
        seed=20260627,
        resume=False,
        profile=REPO_REACT_V1_50X4096_PROFILE,
        math_skill_file=skill_file,
    )

    assert command[command.index("--math-skill-file") + 1] == str(skill_file)


def test_evaluator_command_keeps_checkpoint_only_path_skill_free(tmp_path: Path) -> None:
    command = evaluator_command(
        python=sys.executable,
        evaluator=tmp_path / "eval.py",
        endpoints=["http://127.0.0.1:18180/v1"],
        model_path=tmp_path / "model",
        math_root=tmp_path / "math",
        out_dir=tmp_path / "out",
        samples=4,
        concurrency=8,
        seed=20260627,
        resume=False,
        profile=REPO_REACT_V1_50X4096_PROFILE,
    )

    assert "--math-skill-file" not in command


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
        dapo_seed=20270652,
        aime_seed=20280652,
        resume=True,
        profile=REPO_REACT_V1_50X4096_PROFILE,
    )
    joined = " ".join(command)
    assert "--samples 4" in joined
    assert "--math-max-turns 50" in joined
    assert "--math-max-tokens 4096" in joined
    assert "--math-react-prompt repo-react-v1" in joined
    assert "--dapo-seed 20270652" in joined
    assert "--aime-seed 20280652" in joined
    assert "--retry-react-errors" in command
    assert "--math-enable-thinking" not in command
