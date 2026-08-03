#!/usr/bin/env python3
"""Persistently execute every stage of the fixed DAPO-400 experiment."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from report_grpo_experiment import code_version  # noqa: E402
from run_training_until_complete import (  # noqa: E402
    LIMIT_TIERS,
    classify_failure,
    fixed_environment,
    inspect_completion,
    selected_gpu_occupancy,
)


RUN_TAG = "qwen35-4b-math-grpo-dapo400-e15-seed1"
ALIGNED_EVAL_PROFILE = "repo-react-v1-50x4096"
EXPECTED_EVAL = {"dapo100": 1600, "aime2026": 480}
EXPECTED_REPORT_EVAL = {"dapo100": 400, "aime2026": 120}
EXPECTED_PHASES = {"train": 48000, "validation": 6100, "baseline": 2080, "post": 2080}
SMOKE_SUFFIX = "-smoke-step1"
SMOKE_DUMP = re.compile(r"^1(?:\.attempt(\d+))?\.jsonl$")


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def select_runtime_python(
    *,
    requested: str | None,
    current: str,
    conda_envs: list[str],
    usable: Callable[[str], bool],
) -> str:
    """Choose one interpreter containing the complete eval/train runtime."""
    if requested:
        if not usable(requested):
            raise RuntimeError(f"requested Python lacks the Math runtime dependencies: {requested}")
        return requested
    candidates = [current]
    candidates.extend(
        str(Path(environment) / "bin" / "python")
        for environment in conda_envs
        if Path(environment).name == "grpo"
    )
    for candidate in dict.fromkeys(candidates):
        if usable(candidate):
            return candidate
    raise RuntimeError("could not find a Python with torch/transformers/datasets/sglang/ray")


def _runtime_python_usable(python: str) -> bool:
    command = [
        python,
        "-c",
        (
            "import importlib.util; import sys; "
            "sys.exit(0 if all(importlib.util.find_spec(n) for n in "
            "('torch','transformers','datasets','sglang','ray')) else 1)"
        ),
    ]
    try:
        return subprocess.run(command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    except OSError:
        return False


def resolve_runtime_python(requested: str | None) -> str:
    conda_envs: list[str] = []
    try:
        result = subprocess.run(
            ["conda", "env", "list", "--json"],
            check=True,
            text=True,
            capture_output=True,
        )
        conda_envs = [str(item) for item in json.loads(result.stdout).get("envs", [])]
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        pass
    return select_runtime_python(
        requested=requested,
        current=sys.executable,
        conda_envs=conda_envs,
        usable=_runtime_python_usable,
    )


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def inspect_eval_completion(path: Path, *, expected: dict[str, int] = EXPECTED_EVAL) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    complete = True
    for dataset, expected_records in expected.items():
        source = path / "outputs" / f"{dataset}.jsonl"
        rows: list[dict[str, Any]] = []
        if source.is_file():
            try:
                with source.open(encoding="utf-8", errors="replace") as handle:
                    rows = [json.loads(line) for line in handle if line.strip()]
            except (json.JSONDecodeError, OSError):
                rows = []
        keys = [str(row.get("key", "")) for row in rows]
        errors = sum(bool(row.get("error")) for row in rows)
        dataset_complete = (
            len(rows) == expected_records
            and len(set(keys)) == expected_records
            and all(key.startswith(dataset + ":") for key in keys)
            and errors == 0
        )
        datasets[dataset] = {
            "records": len(rows),
            "expected_records": expected_records,
            "unique_keys": len(set(keys)),
            "request_errors": errors,
            "complete": dataset_complete,
        }
        complete = complete and dataset_complete
    return {"complete": complete, "datasets": datasets}


def trajectory_summary_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    phases = summary.get("by_phase") or {}
    return (
        int(summary.get("records", -1)) == sum(EXPECTED_PHASES.values())
        and all(int(phases.get(phase, -1)) == count for phase, count in EXPECTED_PHASES.items())
    )


def data_validation_command(*, python: str, root: Path, out: Path) -> list[str]:
    return [
        python,
        str(root / "scripts" / "math" / "validate_data.py"),
        "--data-root",
        str(root / "data" / "trace2skill" / "math_reasoning"),
        "--out",
        str(out),
    ]


def smoke_environment(root: Path, run_tag: str, *, tier: int) -> dict[str, str]:
    """Build an isolated one-step training environment at a capacity tier."""
    env = fixed_environment(root, run_tag + SMOKE_SUFFIX, tier)
    env.update(
        {
            "TOTAL_TRAINING_STEPS": "1",
            "SAVE_FREQ": "1",
            "TEST_FREQ": "1",
            "VAL_BEFORE_TRAIN": "False",
        }
    )
    return env


def inspect_smoke_completion(root: Path, run_tag: str) -> dict[str, Any]:
    smoke_tag = run_tag + SMOKE_SUFFIX
    trajectory_dir = root / "runs" / "multiturn_grpo" / "trajectories" / smoke_tag / "train_raw"
    candidates: list[tuple[int, Path]] = []
    if trajectory_dir.is_dir():
        for candidate in trajectory_dir.glob("1*.jsonl"):
            match = SMOKE_DUMP.match(candidate.name)
            if match:
                candidates.append((int(match.group(1) or 1), candidate))
    records: list[dict[str, Any]] = []
    if candidates:
        latest = max(candidates)[1]
        try:
            with latest.open(encoding="utf-8", errors="replace") as handle:
                records = [json.loads(line) for line in handle if line.strip()]
        except (json.JSONDecodeError, OSError):
            records = []
    tool_records = sum(
        bool(row.get("steps")) or float(row.get("tool_used", 0.0)) > 0.0 for row in records
    )
    reward_records = sum("score" in row for row in records)
    checkpoint = (
        root
        / "runs"
        / "multiturn_grpo"
        / "checkpoints"
        / smoke_tag
        / "global_step_1"
        / "actor"
        / "huggingface"
        / "config.json"
    )
    train_log = root / "runs" / "multiturn_grpo" / "logs" / smoke_tag / "train.log"
    log_text = train_log.read_text(encoding="utf-8", errors="replace") if train_log.is_file() else ""
    actor_update = any(
        "step:1 -" in line and "timing_s/update_actor:" in line for line in log_text.splitlines()
    )
    details = {
        "records": len(records),
        "expected_records": 160,
        "bash_tool_records": tool_records,
        "reward_records": reward_records,
        "actor_update_logged": actor_update,
        "hf_checkpoint": checkpoint.is_file(),
    }
    complete = all(
        (
            len(records) == 160,
            tool_records == 160,
            reward_records == 160,
            actor_update,
            checkpoint.is_file(),
        )
    )
    return {"complete": complete, **details}


def eval_command(
    *,
    python: str,
    model_path: Path,
    out_dir: Path,
    profile: str = "matched-agentic",
) -> list[str]:
    table_aligned = profile == ALIGNED_EVAL_PROFILE
    if profile not in ("matched-agentic", ALIGNED_EVAL_PROFILE):
        raise ValueError(f"unknown evaluation profile: {profile}")
    command = [
        python,
        str(ROOT / "scripts" / "math" / "run_four_gpu_eval.py"),
        "--model-path",
        str(model_path),
        "--out-dir",
        str(out_dir),
        "--samples",
        "4" if table_aligned else "16",
        "--seed",
        "20260629",
        "--concurrency",
        "8",
        "--fallback-concurrency",
        "4",
        "--context-length",
        str(256 * 1024),
        "--resume",
    ]
    if table_aligned:
        command.extend(["--profile", ALIGNED_EVAL_PROFILE])
    return command


def _record_state(path: Path, stage: str, status: str, **details: Any) -> None:
    _write_json_atomic(
        path,
        {"run_tag": RUN_TAG, "updated_at": now(), "stage": stage, "status": status, **details},
    )


def _log(handle: Any, message: str) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    handle.write(line + "\n")
    handle.flush()


def _run_logged(command: list[str], *, root: Path, handle: Any, env: dict[str, str]) -> int:
    _log(handle, "exec " + json.dumps(command))
    process = subprocess.run(
        command,
        cwd=root,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        check=False,
    )
    _log(handle, f"command exit rc={process.returncode}")
    return process.returncode


def _wait_until_gpus_free(
    *, stage: str, state_path: Path, handle: Any, poll_seconds: float
) -> None:
    while True:
        rows = selected_gpu_occupancy()
        occupied = [row for row in rows if row["occupied"]]
        if not occupied:
            return
        compact = [
            {
                "index": row["index"],
                "memory_mib": row["memory_mib"],
                "pids": [item["pid"] for item in row["compute_processes"]],
            }
            for row in occupied
        ]
        _record_state(state_path, stage, "waiting_for_gpus", occupied=compact)
        _log(handle, f"{stage}: waiting for physical GPUs 3,4,5,6; occupied={compact}")
        time.sleep(poll_seconds)


def _complete_eval_stage(
    *,
    stage: str,
    path: Path,
    model_path: Path,
    python: str,
    root: Path,
    state_path: Path,
    handle: Any,
    env: dict[str, str],
    poll_seconds: float,
    retry_delay: float,
    expected: dict[str, int] = EXPECTED_EVAL,
    profile: str = "matched-agentic",
) -> None:
    while True:
        completion = inspect_eval_completion(path, expected=expected)
        if completion["complete"]:
            _record_state(state_path, stage, "complete", completion=completion)
            _log(handle, f"{stage}: exact {sum(expected.values())}-record acceptance complete")
            return
        _record_state(state_path, stage, "waiting", completion=completion)
        _wait_until_gpus_free(
            stage=stage,
            state_path=state_path,
            handle=handle,
            poll_seconds=poll_seconds,
        )
        # Another persistent evaluator may have completed while we waited.
        completion = inspect_eval_completion(path, expected=expected)
        if completion["complete"]:
            continue
        _record_state(state_path, stage, "running", completion=completion)
        _run_logged(
            eval_command(
                python=python,
                model_path=model_path,
                out_dir=path,
                profile=profile,
            ),
            root=root,
            handle=handle,
            env=env,
        )
        if not inspect_eval_completion(path, expected=expected)["complete"]:
            _log(handle, f"{stage}: incomplete after evaluator exit; resume in {retry_delay}s")
            time.sleep(retry_delay)


def _complete_smoke_stage(
    *,
    run_tag: str,
    python: str,
    root: Path,
    state_path: Path,
    handle: Any,
    poll_seconds: float,
    retry_delay: float,
) -> int:
    """Prove rollout, bash, reward, actor update, and checkpoint before the long run."""
    history_path = state_path.parent / "smoke_history.json"
    history: list[dict[str, Any]] = []
    if history_path.is_file():
        try:
            loaded = json.loads(history_path.read_text(encoding="utf-8"))
            history = loaded if isinstance(loaded, list) else []
        except (json.JSONDecodeError, OSError):
            history = []
    tier = int(history[-1].get("tier", 0)) if history else 0
    while True:
        completion = inspect_smoke_completion(root, run_tag)
        if completion["complete"]:
            _record_state(state_path, "one_step_smoke", "complete", completion=completion)
            _log(handle, "one_step_smoke: 160 trajectories + bash + reward + actor update + HF checkpoint PASS")
            return 0
        _wait_until_gpus_free(
            stage="one_step_smoke",
            state_path=state_path,
            handle=handle,
            poll_seconds=poll_seconds,
        )
        env = smoke_environment(root, run_tag, tier=tier)
        env["PY"] = python
        env["CONDA_ENV"] = ""
        train_log = Path(env["LOG_DIR"]) / "train.log"
        offset = train_log.stat().st_size if train_log.is_file() else 0
        attempt = {
            "attempt": len(history) + 1,
            "started_at": now(),
            "tier": tier,
            "max_turns": LIMIT_TIERS[tier][0],
            "max_response_tokens": LIMIT_TIERS[tier][1],
            "status": "running",
        }
        history.append(attempt)
        _write_json_atomic(history_path, history)
        _record_state(state_path, "one_step_smoke", "running", completion=completion, attempt=attempt)
        rc = _run_logged(
            ["bash", str(root / "scripts" / "trace2skill" / "run_verl_agentic_rl.sh")],
            root=root,
            handle=handle,
            env=env,
        )
        completion = inspect_smoke_completion(root, run_tag)
        failure_text = ""
        if train_log.is_file():
            with train_log.open("rb") as source:
                source.seek(min(offset, train_log.stat().st_size))
                failure_text = source.read().decode("utf-8", errors="replace")
        failure_class = "complete" if completion["complete"] else classify_failure(failure_text)
        history[-1].update(
            {
                "finished_at": now(),
                "return_code": rc,
                "failure_class": failure_class,
                "completion": completion,
                "status": "complete" if completion["complete"] else "failed",
            }
        )
        _write_json_atomic(history_path, history)
        if completion["complete"]:
            continue
        if failure_class == "capacity":
            if tier + 1 >= len(LIMIT_TIERS):
                _record_state(
                    state_path,
                    "one_step_smoke",
                    "lowest_tier_capacity_failure",
                    completion=completion,
                )
                _log(handle, "one_step_smoke: lowest capacity tier failed")
                return 2
            tier += 1
            _log(handle, f"one_step_smoke: confirmed capacity failure; fallback to {LIMIT_TIERS[tier]}")
        else:
            _log(handle, f"one_step_smoke: incomplete rc={rc}; retry same tier={LIMIT_TIERS[tier]}")
        time.sleep(retry_delay)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--python")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--retry-delay-seconds", type=float, default=60.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    python = resolve_runtime_python(args.python)
    run_tag = RUN_TAG
    model = root / "runs" / "docvqa_grpo" / "assets" / "Qwen3.5-4B-text"
    base = root / "runs" / "multiturn_grpo"
    baseline = base / "eval" / run_tag / "baseline_50x4096"
    baseline_report = base / "eval" / run_tag / "baseline_table_50x4096"
    post = base / "eval" / run_tag / "post_50x4096"
    trajectories = base / "trajectories" / run_tag
    checkpoint_dir = base / "checkpoints" / run_tag
    final_model = checkpoint_dir / "global_step_300" / "actor" / "huggingface"
    log_dir = base / "logs" / run_tag
    report_dir = base / "reports" / run_tag
    normalized_dir = trajectories / "normalized"
    state_path = report_dir / "pipeline_state.json"
    pipeline_log = report_dir / "pipeline.log"
    data_manifest = report_dir / "data_manifest.json"
    start_code_version = report_dir / "code_version_at_start.json"
    report_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["MATH_PHYSICAL_GPU_IDS"] = "3,4,5,6"
    env["PYTHONUNBUFFERED"] = "1"
    env["PY"] = python
    env["CONDA_ENV"] = ""
    with pipeline_log.open("a", encoding="utf-8", buffering=1) as handle:
        _log(handle, f"pipeline start run_tag={run_tag}")
        if not start_code_version.is_file():
            _write_json_atomic(start_code_version, {**code_version(root), "captured_at": now()})
        _record_state(state_path, "data_validation", "running")
        rc = _run_logged(
            data_validation_command(python=python, root=root, out=data_manifest),
            root=root,
            handle=handle,
            env=env,
        )
        try:
            manifest_ok = rc == 0 and json.loads(data_manifest.read_text(encoding="utf-8"))["status"] == "PASS"
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            manifest_ok = False
        if not manifest_ok:
            _record_state(state_path, "data_validation", "failed", return_code=rc)
            _log(handle, "data_validation: hard gate FAILED")
            return 1
        _record_state(state_path, "data_validation", "complete", manifest=str(data_manifest))
        _log(handle, "data_validation: 400/100/30, split isolation, and fixed SHA-256 PASS")
        _complete_eval_stage(
            stage="baseline_table_50x4096",
            path=baseline_report,
            model_path=model,
            python=python,
            root=root,
            state_path=state_path,
            handle=handle,
            env=env,
            poll_seconds=args.poll_seconds,
            retry_delay=args.retry_delay_seconds,
            expected=EXPECTED_REPORT_EVAL,
            profile=ALIGNED_EVAL_PROFILE,
        )
        _complete_eval_stage(
            stage="baseline_eval",
            path=baseline,
            model_path=model,
            python=python,
            root=root,
            state_path=state_path,
            handle=handle,
            env=env,
            poll_seconds=args.poll_seconds,
            retry_delay=args.retry_delay_seconds,
        )

        smoke_rc = _complete_smoke_stage(
            run_tag=run_tag,
            python=python,
            root=root,
            state_path=state_path,
            handle=handle,
            poll_seconds=args.poll_seconds,
            retry_delay=args.retry_delay_seconds,
        )
        if smoke_rc != 0:
            return smoke_rc

        while True:
            training = inspect_completion(
                trajectories / "train_raw",
                trajectories / "validation_raw",
                checkpoint_dir,
            )
            if training["complete"]:
                break
            _record_state(state_path, "training", "running", completion=training)
            rc = _run_logged(
                [
                    python,
                    str(root / "scripts" / "math" / "run_training_until_complete.py"),
                    "--root",
                    str(root),
                    "--run-tag",
                    run_tag,
                    "--poll-seconds",
                    str(args.poll_seconds),
                    "--restart-delay-seconds",
                    str(args.retry_delay_seconds),
                ],
                root=root,
                handle=handle,
                env=env,
            )
            if rc == 2:
                _record_state(state_path, "training", "lowest_tier_capacity_failure", completion=training)
                return 2
            if not inspect_completion(
                trajectories / "train_raw", trajectories / "validation_raw", checkpoint_dir
            )["complete"]:
                _log(handle, f"training watchdog exited rc={rc}; restart in {args.retry_delay_seconds}s")
                time.sleep(args.retry_delay_seconds)
        _record_state(state_path, "training", "complete", completion=training)

        _complete_eval_stage(
            stage="post_eval",
            path=post,
            model_path=final_model,
            python=python,
            root=root,
            state_path=state_path,
            handle=handle,
            env=env,
            poll_seconds=args.poll_seconds,
            retry_delay=args.retry_delay_seconds,
        )

        summary_path = normalized_dir / "summary.json"
        while not trajectory_summary_complete(summary_path):
            _record_state(state_path, "trajectory_export", "running")
            _run_logged(
                [
                    python,
                    str(root / "scripts" / "trace2skill" / "export_math_trajectories.py"),
                    "--train-dir",
                    str(trajectories / "train_raw"),
                    "--validation-dir",
                    str(trajectories / "validation_raw"),
                    "--baseline-dir",
                    str(baseline),
                    "--post-dir",
                    str(post),
                    "--out-dir",
                    str(normalized_dir),
                ],
                root=root,
                handle=handle,
                env=env,
            )
            if not trajectory_summary_complete(summary_path):
                _log(handle, f"trajectory export incomplete; retry in {args.retry_delay_seconds}s")
                time.sleep(args.retry_delay_seconds)
        _record_state(state_path, "trajectory_export", "complete")

        final_json = report_dir / "final_report.json"
        final_markdown = report_dir / "final_report.md"
        while True:
            _record_state(state_path, "final_report", "running")
            rc = _run_logged(
                [
                    python,
                    str(root / "scripts" / "math" / "report_grpo_experiment.py"),
                    "--experiment-config",
                    str(log_dir / "experiment_config.json"),
                    "--gpu-resources",
                    str(log_dir / "gpu_resources.json"),
                    "--data-manifest",
                    str(data_manifest),
                    "--train-log",
                    str(log_dir / "train.log"),
                    "--checkpoint-dir",
                    str(checkpoint_dir),
                    "--trajectory-summary",
                    str(summary_path),
                    "--before-eval-dir",
                    str(baseline),
                    "--table-alignment-eval-dir",
                    str(baseline_report),
                    "--after-eval-dir",
                    str(post),
                    "--code-version-at-start",
                    str(start_code_version),
                    "--run-history",
                    str(report_dir / "run_history.json"),
                    "--json-out",
                    str(final_json),
                    "--markdown-out",
                    str(final_markdown),
                ],
                root=root,
                handle=handle,
                env=env,
            )
            try:
                accepted = json.loads(final_json.read_text(encoding="utf-8"))["acceptance"]["status"] == "PASS"
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                accepted = False
            if rc == 0 and accepted:
                break
            _log(handle, f"final report incomplete rc={rc}; retry in {args.retry_delay_seconds}s")
            time.sleep(args.retry_delay_seconds)
        _record_state(
            state_path,
            "complete",
            "accepted",
            final_report_json=str(final_json),
            final_report_markdown=str(final_markdown),
        )
        _log(handle, "full DAPO-400 experiment acceptance PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
