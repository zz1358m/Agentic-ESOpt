from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "scripts/docvqa/run_four_gpu_experiment_pipeline.sh"
TRAIN_LAUNCHER = ROOT / "scripts/trace2skill/run_verl_agentic_rl.sh"


def test_four_gpu_pipeline_is_valid_bash_and_contains_all_acceptance_stages() -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(PIPELINE)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stderr

    script = PIPELINE.read_text(encoding="utf-8")
    required_fragments = (
        'DOCVQA_PHYSICAL_GPU_IDS="${DOCVQA_PHYSICAL_GPU_IDS:-auto}"',
        "run_four_gpu_eval.py",
        "--resume",
        "DOCVQA_TRAIN_LIMIT=50",
        "DOCVQA_VAL_LIMIT=100",
        "TRAIN_BATCH_SIZE=4",
        "PPO_MINI_BATCH_SIZE=4",
        "ROLLOUT_N=8",
        "TOTAL_EPOCHS=15",
        "DATA_SEED=42",
        "SAVE_FREQ=1",
        "VERL_PROTECTED_CHECKPOINT_STEPS=60,120,180",
        "TEST_FREQ=-1",
        "VAL_BEFORE_TRAIN=False",
        "global_step_60",
        "global_step_120",
        "global_step_180",
        "--concurrency 8",
        "--fallback-concurrency 4",
        "--strict-concurrency",
        "verify_text_checkpoint.py",
        "report_grpo_experiment.py",
    )
    for fragment in required_fragments:
        assert fragment in script


def test_pipeline_is_portable_and_has_no_six_gpu_or_trajectory_collector() -> None:
    script = PIPELINE.read_text(encoding="utf-8")
    assert "/data/external/" not in script
    assert "run_six_gpu" not in script
    assert "paired_trajectory" not in script
    assert "sample_only" not in script
    assert "rollout_data_dir" not in script


def test_training_launcher_exposes_chunked_fused_logprob_backend() -> None:
    script = TRAIN_LAUNCHER.read_text(encoding="utf-8")
    assert 'actor_rollout_ref.model.use_fused_kernels="${USE_FUSED_KERNELS:-False}"' in script
    assert (
        'actor_rollout_ref.model.fused_kernel_options.impl_backend="${FUSED_KERNELS_BACKEND:-torch}"'
        in script
    )


def test_training_launcher_does_not_enable_rollout_dumping_or_paired_seeds() -> None:
    script = TRAIN_LAUNCHER.read_text(encoding="utf-8")
    assert "rollout_data_dir" not in script
    assert "paired_trajectory_seeds" not in script
