#!/usr/bin/env python3
"""Persistently run the fixed DAPO-400 GRPO job on physical GPUs 3--6."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
APPROVED_GPUS = (3, 4, 5, 6)
LIMIT_TIERS = (
    (100, 8192),
    (30, 4096),
    (15, 3072),
    (10, 2048),
    (8, 1536),
    (5, 1024),
    (3, 768),
    (2, 512),
)
RAW_NAME = re.compile(r"^(\d+)(?:\.attempt(\d+))?\.jsonl$")
CAPACITY_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (
        r"CUDA out of memory",
        r"torch\.OutOfMemoryError",
        r"(?:^|\W)out of memory(?:\W|$)",
        r"(?:maximum|max) sequence length.{0,160}(?:exceed|larger|capacity)",
        r"context length.{0,160}(?:exceed|larger|capacity)",
        r"(?:exceed|larger).{0,160}max_model_len",
        r"KV cache.{0,160}(?:insufficient|not enough|capacity|exceed)",
        r"memory pool.{0,80}(?:<=\s*0|insufficient|not enough)",
    )
)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def classify_failure(text: str) -> str:
    """Return capacity only for direct OOM/context-capacity evidence."""
    return "capacity" if any(pattern.search(text) for pattern in CAPACITY_PATTERNS) else "transient"


def _latest_attempts(path: Path) -> dict[int, tuple[int, Path]]:
    latest: dict[int, tuple[int, Path]] = {}
    if not path.is_dir():
        return latest
    for candidate in path.glob("*.jsonl"):
        match = RAW_NAME.match(candidate.name)
        if match is None:
            continue
        step = int(match.group(1))
        attempt = int(match.group(2) or 1)
        if step not in latest or attempt > latest[step][0]:
            latest[step] = (attempt, candidate)
    return latest


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(bool(line.strip()) for line in handle)


def inspect_completion(train_dir: Path, validation_dir: Path, checkpoint_dir: Path) -> dict[str, Any]:
    """Check exact logical counts while leaving every raw retry on disk."""
    train = _latest_attempts(train_dir)
    validation = _latest_attempts(validation_dir)
    expected_train_steps = set(range(1, 301))
    expected_validation_steps = {0, *range(5, 301, 5)}
    train_counts = {step: _line_count(item[1]) for step, item in train.items()}
    validation_counts = {step: _line_count(item[1]) for step, item in validation.items()}

    expected_checkpoint_steps = list(range(20, 301, 20))
    actual_checkpoint_steps = sorted(
        int(candidate.name.removeprefix("global_step_"))
        for candidate in checkpoint_dir.glob("global_step_*")
        if candidate.name.removeprefix("global_step_").isdigit()
    ) if checkpoint_dir.is_dir() else []
    hf_complete = all(
        (checkpoint_dir / f"global_step_{step}" / "actor" / "huggingface" / "config.json").is_file()
        for step in expected_checkpoint_steps
    )
    train_complete = set(train) == expected_train_steps and all(
        train_counts.get(step) == 160 for step in expected_train_steps
    )
    validation_complete = set(validation) == expected_validation_steps and all(
        validation_counts.get(step) == 100 for step in expected_validation_steps
    )
    checkpoints_complete = actual_checkpoint_steps == expected_checkpoint_steps and hf_complete
    return {
        "complete": train_complete and validation_complete and checkpoints_complete,
        "train_complete": train_complete,
        "validation_complete": validation_complete,
        "checkpoints_complete": checkpoints_complete,
        "train_steps": len(train),
        "train_records": sum(train_counts.values()),
        "validation_rounds": len(validation),
        "validation_records": sum(validation_counts.values()),
        "checkpoint_steps": actual_checkpoint_steps,
        "hf_models_complete": hf_complete,
    }


def _gpu_rows() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        index, uuid, memory = (part.strip() for part in line.split(",", 2))
        rows.append({"index": int(index), "uuid": uuid, "memory_mib": int(memory)})
    return rows


def selected_gpu_occupancy() -> list[dict[str, Any]]:
    selected = [row for row in _gpu_rows() if row["index"] in APPROVED_GPUS]
    if tuple(row["index"] for row in selected) != APPROVED_GPUS:
        raise RuntimeError(f"could not resolve physical GPUs {APPROVED_GPUS}: {selected}")
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    selected_uuids = {row["uuid"] for row in selected}
    compute: dict[str, list[dict[str, int]]] = {uuid: [] for uuid in selected_uuids}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3 or parts[0] not in selected_uuids:
            continue
        try:
            compute[parts[0]].append({"pid": int(parts[1]), "memory_mib": int(parts[2])})
        except ValueError:
            continue
    for row in selected:
        row["compute_processes"] = compute[row["uuid"]]
        row["occupied"] = bool(row["compute_processes"]) or row["memory_mib"] > 1024
    return selected


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"run history must be a JSON list: {path}")
    return value


def _log(handle: Any, message: str) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    handle.write(line + "\n")
    handle.flush()


def wait_for_selected_gpus(handle: Any, poll_seconds: float) -> list[dict[str, Any]]:
    while True:
        rows = selected_gpu_occupancy()
        occupied = [row for row in rows if row["occupied"]]
        if not occupied:
            _log(handle, "physical GPUs 3,4,5,6 are free")
            return rows
        compact = [
            {
                "index": row["index"],
                "memory_mib": row["memory_mib"],
                "pids": [item["pid"] for item in row["compute_processes"]],
            }
            for row in occupied
        ]
        _log(handle, f"waiting for approved GPUs; occupied={json.dumps(compact, sort_keys=True)}")
        time.sleep(poll_seconds)


def terminate_own_process_group(group_id: int, grace_seconds: float = 30.0) -> None:
    """Clean up only the process group created for one launcher attempt."""
    try:
        os.killpg(group_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        try:
            os.killpg(group_id, 0)
        except ProcessLookupError:
            return
        time.sleep(0.5)
    try:
        os.killpg(group_id, signal.SIGKILL)
    except ProcessLookupError:
        pass


def fixed_environment(root: Path, run_tag: str, tier: int) -> dict[str, str]:
    turns, tokens = LIMIT_TIERS[tier]
    trajectory_root = root / "runs" / "multiturn_grpo" / "trajectories" / run_tag
    env = os.environ.copy()
    env.pop("TOTAL_TRAINING_STEPS", None)
    env.update(
        {
            "TASK": "math",
            "RUN_TAG": run_tag,
            "ROOT": str(root),
            "MODEL_PATH": str(root / "runs" / "docvqa_grpo" / "assets" / "Qwen3.5-4B-text"),
            "REF_MODEL_PATH": str(root / "runs" / "docvqa_grpo" / "assets" / "Qwen3.5-4B-text"),
            "MATH_PHYSICAL_GPU_IDS": "3,4,5,6",
            "N_GPUS_PER_NODE": "4",
            "TRAIN_BATCH_SIZE": "20",
            "PPO_MINI_BATCH_SIZE": "20",
            "ROLLOUT_N": "8",
            "TOTAL_EPOCHS": "15",
            "TEST_FREQ": "5",
            "SAVE_FREQ": "20",
            "VAL_BEFORE_TRAIN": "True",
            "MAX_USER_TURNS": str(turns),
            "MAX_ASSISTANT_TURNS": str(turns),
            "MAX_RESPONSE_LENGTH": str(tokens),
            "MAX_TURN_RESPONSE_LENGTH": "512",
            "LR": "1e-6",
            "USE_KL_LOSS": "True",
            "KL_LOSS_COEF": "0.001",
            "TEMPERATURE": "1.0",
            "TOP_P": "1.0",
            "TOP_K": "40",
            "DATA_SHUFFLE": "True",
            "DATA_SEED": "1",
            "RAY_NUM_CPUS": "32",
            "GPU_MEMORY_UTILIZATION": "0.50",
            "MAX_NUM_SEQS": "16",
            "TRACE2SKILL_PATCH_DENSE_QWEN3NEXT": "1",
            "TRACE2SKILL_REGISTER_TOOL_PARSER": "1",
            "ROLLOUT_DATA_DIR": str(trajectory_root / "train_raw"),
            "VALIDATION_DATA_DIR": str(trajectory_root / "validation_raw"),
            "TRACE2SKILL_MATH_TOOL_CWD": str(trajectory_root / "tool_workspace"),
            "CKPT_DIR": str(root / "runs" / "multiturn_grpo" / "checkpoints" / run_tag),
            "LOG_DIR": str(root / "runs" / "multiturn_grpo" / "logs" / run_tag),
            "MAX_ACTOR_CKPT_TO_KEEP": "null",
            "MAX_CRITIC_CKPT_TO_KEEP": "null",
        }
    )
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run-tag", default="qwen35-4b-math-grpo-dapo400-e15-seed1")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--restart-delay-seconds", type=float, default=60.0)
    parser.add_argument("--initial-tier", type=int, choices=range(len(LIMIT_TIERS)), default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    launcher = root / "scripts" / "trace2skill" / "run_verl_agentic_rl.sh"
    run_tag = args.run_tag
    train_dir = root / "runs" / "multiturn_grpo" / "trajectories" / run_tag / "train_raw"
    validation_dir = root / "runs" / "multiturn_grpo" / "trajectories" / run_tag / "validation_raw"
    checkpoint_dir = root / "runs" / "multiturn_grpo" / "checkpoints" / run_tag
    train_log = root / "runs" / "multiturn_grpo" / "logs" / run_tag / "train.log"
    report_dir = root / "runs" / "multiturn_grpo" / "reports" / run_tag
    history_path = report_dir / "run_history.json"
    control_path = report_dir / "training_watchdog.log"
    report_dir.mkdir(parents=True, exist_ok=True)
    tier = args.initial_tier

    with control_path.open("a", encoding="utf-8", buffering=1) as control:
        _log(control, f"watchdog start run_tag={run_tag} approved_gpus=3,4,5,6")
        while True:
            completion = inspect_completion(train_dir, validation_dir, checkpoint_dir)
            if completion["complete"]:
                _write_json_atomic(report_dir / "training_completion.json", completion)
                _log(control, f"training acceptance complete: {json.dumps(completion, sort_keys=True)}")
                return 0

            gpu_rows = wait_for_selected_gpus(control, args.poll_seconds)
            history = _load_history(history_path)
            attempt = {
                "attempt": len(history) + 1,
                "started_at": now(),
                "tier": tier,
                "max_turns": LIMIT_TIERS[tier][0],
                "max_response_tokens": LIMIT_TIERS[tier][1],
                "reason": "initial" if not history else "resume_latest_checkpoint",
                "gpu_preflight": gpu_rows,
                "status": "running",
            }
            history.append(attempt)
            _write_json_atomic(history_path, history)
            offset = train_log.stat().st_size if train_log.is_file() else 0
            _log(
                control,
                f"launch attempt={attempt['attempt']} tier={tier} limits={LIMIT_TIERS[tier]}",
            )

            process: subprocess.Popen[bytes] | None = None
            return_code = -1
            try:
                process = subprocess.Popen(
                    ["bash", str(launcher)],
                    cwd=root,
                    env=fixed_environment(root, run_tag, tier),
                    stdout=control.buffer,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                return_code = process.wait()
            finally:
                if process is not None and process.poll() is None:
                    terminate_own_process_group(process.pid)
                    process.wait()

            completion = inspect_completion(train_dir, validation_dir, checkpoint_dir)
            if process is not None and not completion["complete"]:
                terminate_own_process_group(process.pid)
            failure_text = ""
            if train_log.is_file():
                with train_log.open("rb") as handle:
                    handle.seek(min(offset, train_log.stat().st_size))
                    failure_text = handle.read().decode("utf-8", errors="replace")
            failure_class = "complete" if completion["complete"] else classify_failure(failure_text)
            history = _load_history(history_path)
            history[-1].update(
                {
                    "finished_at": now(),
                    "return_code": return_code,
                    "failure_class": failure_class,
                    "completion": completion,
                    "status": "complete" if completion["complete"] else "failed",
                }
            )
            _write_json_atomic(history_path, history)
            if completion["complete"]:
                _write_json_atomic(report_dir / "training_completion.json", completion)
                _log(control, "launcher exited with full training acceptance")
                return 0
            if failure_class == "capacity":
                if tier + 1 >= len(LIMIT_TIERS):
                    evidence_path = report_dir / "lowest_tier_capacity_failure.log"
                    evidence_path.write_text(failure_text, encoding="utf-8")
                    _log(control, f"lowest approved tier failed capacity; evidence={evidence_path}")
                    return 2
                tier += 1
                _log(control, f"confirmed capacity failure; fallback to tier={tier} limits={LIMIT_TIERS[tier]}")
            else:
                _log(control, f"non-capacity exit rc={return_code}; resume same tier={tier}")
            time.sleep(args.restart_delay_seconds)


if __name__ == "__main__":
    sys.exit(main())
