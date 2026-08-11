#!/usr/bin/env python3
"""Run direct Sudoku checkpoint evaluation on 16 endpoints / 4 or 8 GPUs."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "sudoku-train-time/scripts/run_sudoku_es_train.py"
MODEL_SERVER = (
    ROOT
    / "ahd-test-time/methods/eoh/original/eoh/src/eoh/llm_local_server/llama31_instruct_server.py"
)
DEFAULT_SUDOKU_GPU_COUNT = 8
SUDOKU_ENDPOINT_COUNT = 16
SUPPORTED_PHYSICAL_GPU_COUNTS = (4, 8)
SUDOKU_RELEASE_ENDPOINT_BATCH_SIZE = 32


@dataclass(frozen=True)
class GpuIdentity:
    index: int
    uuid: str
    name: str


@dataclass(frozen=True)
class EndpointAssignment:
    port: int
    endpoint: str
    gpu_index: int
    gpu_uuid: str
    gpu_name: str
    inference_seed: int | None = None


def validate_physical_gpu_ids(value: str) -> tuple[int, ...]:
    try:
        ids = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"physical GPU ids must be numeric, got {value!r}") from exc
    if len(ids) not in SUPPORTED_PHYSICAL_GPU_COUNTS or len(set(ids)) != len(ids):
        raise ValueError(f"Sudoku requires four or eight unique physical GPU ids, got {value!r}")
    if any(index < 0 for index in ids):
        raise ValueError("physical GPU ids must be non-negative")
    return ids


def parse_nvidia_smi(output: str) -> dict[int, GpuIdentity]:
    result: dict[int, GpuIdentity] = {}
    for line_number, raw_line in enumerate(output.splitlines(), 1):
        if not raw_line.strip():
            continue
        fields = [field.strip() for field in raw_line.split(",", 2)]
        if len(fields) != 3:
            raise ValueError(f"invalid nvidia-smi row {line_number}: {raw_line!r}")
        try:
            index = int(fields[0])
        except ValueError as exc:
            raise ValueError(f"invalid GPU index on row {line_number}: {fields[0]!r}") from exc
        if index in result:
            raise ValueError(f"duplicate physical GPU index {index}")
        if not fields[1].startswith("GPU-"):
            raise ValueError(f"invalid GPU UUID on row {line_number}: {fields[1]!r}")
        result[index] = GpuIdentity(index=index, uuid=fields[1], name=fields[2])
    return result


def resolve_gpus(value: str) -> list[GpuIdentity]:
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    available = parse_nvidia_smi(output)
    if value.strip().lower() == "auto":
        if len(available) < DEFAULT_SUDOKU_GPU_COUNT:
            raise ValueError(f"Sudoku auto mode requires eight GPUs, but nvidia-smi reported {len(available)}")
        ids = tuple(sorted(available)[:DEFAULT_SUDOKU_GPU_COUNT])
    else:
        ids = validate_physical_gpu_ids(value)
    missing = [index for index in ids if index not in available]
    if missing:
        raise ValueError(f"requested physical GPUs were not reported by nvidia-smi: {missing}")
    gpus = [available[index] for index in ids]
    if len({gpu.uuid for gpu in gpus}) != len(gpus):
        raise ValueError("nvidia-smi returned duplicate UUIDs for the requested GPUs")
    return gpus


def build_endpoint_plan(
    gpus: list[GpuIdentity],
    *,
    first_port: int = 12100,
    inference_seed: int | None = None,
) -> list[EndpointAssignment]:
    gpu_count = len(gpus)
    if gpu_count not in SUPPORTED_PHYSICAL_GPU_COUNTS or len({gpu.uuid for gpu in gpus}) != gpu_count:
        raise ValueError("Sudoku endpoint planning requires four or eight unique GPUs")
    endpoints_per_gpu = SUDOKU_ENDPOINT_COUNT // gpu_count
    plan = []
    for replica in range(endpoints_per_gpu):
        for gpu in gpus:
            port = first_port + replica * len(gpus) + len(plan) % len(gpus)
            plan.append(
                EndpointAssignment(
                    port=port,
                    endpoint=f"http://127.0.0.1:{port}/completions",
                    gpu_index=gpu.index,
                    gpu_uuid=gpu.uuid,
                    gpu_name=gpu.name,
                    inference_seed=(
                        None if inference_seed is None else inference_seed + len(plan)
                    ),
                )
            )
    counts = Counter(row.gpu_uuid for row in plan)
    if len(plan) != SUDOKU_ENDPOINT_COUNT or set(counts.values()) != {endpoints_per_gpu}:
        raise AssertionError(f"invalid 16-endpoint topology: {counts}")
    return plan


def build_startup_waves(plan: list[EndpointAssignment]) -> list[list[EndpointAssignment]]:
    gpu_count = len({row.gpu_uuid for row in plan})
    if len(plan) != SUDOKU_ENDPOINT_COUNT or gpu_count not in SUPPORTED_PHYSICAL_GPU_COUNTS:
        raise ValueError("Sudoku startup requires 16 endpoints across four or eight GPUs")
    waves = [
        plan[start : start + gpu_count]
        for start in range(0, len(plan), gpu_count)
    ]
    if any(len({row.gpu_uuid for row in wave}) != gpu_count for wave in waves):
        raise ValueError("each startup wave must contain one endpoint per physical GPU")
    return waves


def build_server_command(
    *,
    python: str,
    model_path: Path,
    assignment: EndpointAssignment,
) -> list[str]:
    command = [
        python,
        str(MODEL_SERVER),
        "--path",
        str(model_path),
        "--d",
        str(assignment.gpu_index),
        "--host",
        "127.0.0.1",
        "--port",
        str(assignment.port),
        "--dtype",
        "bfloat16",
        "--chat-template-enable-thinking",
        "false",
    ]
    if assignment.inference_seed is not None:
        command.extend(["--seed", str(assignment.inference_seed)])
    return command


def build_trainer_command(
    *,
    python: str,
    endpoints: list[str],
    eval_data: Path,
    eval_limit: int,
    eval_repeats: int,
    result_root: Path,
    served_model_name: str,
) -> list[str]:
    return [
        python,
        str(TRAINER),
        "--eval-only",
        "--endpoints",
        ",".join(endpoints),
        "--eval-data",
        str(eval_data),
        "--mask-count",
        "15",
        "--eval-limit",
        str(eval_limit),
        "--eval-repeats",
        str(eval_repeats),
        "--result-root",
        str(result_root),
        "--model",
        served_model_name,
        "--max-turns",
        "45",
        "--max-tokens",
        "64",
        "--temperature",
        "0.7",
        "--top-p",
        "0.8",
        "--top-k",
        "20",
        "--min-p",
        "0.0",
        "--presence-penalty",
        "1.5",
        "--repetition-penalty",
        "1.0",
        "--batched-eval",
        "--endpoint-batch-size",
        str(SUDOKU_RELEASE_ENDPOINT_BATCH_SIZE),
    ]


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def health_is_ready(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_for_wave(
    rows: list[EndpointAssignment],
    processes: dict[int, subprocess.Popen[str]],
    *,
    timeout: int,
) -> None:
    pending = {row.port for row in rows}
    deadline = time.monotonic() + timeout
    while pending:
        for port in list(pending):
            process = processes[port]
            if process.poll() is not None:
                raise RuntimeError(f"model endpoint on port {port} exited with code {process.returncode}")
            if health_is_ready(port):
                pending.remove(port)
                print(f"[ready] port={port}", flush=True)
        if not pending:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for model-server ports: {sorted(pending)}")
        time.sleep(2)


def gpu_snapshot(path: Path) -> None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    path.write_text(result.stdout + result.stderr, encoding="utf-8")


def validate_eval_history(
    history: object,
    *,
    expected_count: int,
    expected_repeats: int,
) -> dict:
    if not isinstance(history, list) or len(history) != 2:
        raise RuntimeError("checkpoint history must contain config and one evaluation record")
    config_record = history[0]
    config = config_record.get("config") if isinstance(config_record, dict) else None
    if not isinstance(config, dict) or config.get("batched_eval") is not True:
        raise RuntimeError("checkpoint history does not record batched_eval=true")
    if config.get("endpoint_batch_size") != SUDOKU_RELEASE_ENDPOINT_BATCH_SIZE:
        raise RuntimeError(
            "checkpoint history endpoint_batch_size is "
            f"{config.get('endpoint_batch_size')!r}, expected {SUDOKU_RELEASE_ENDPOINT_BATCH_SIZE}"
        )
    evaluation_record = history[-1]
    if not isinstance(evaluation_record, dict) or evaluation_record.get("generation") != -1:
        raise RuntimeError("checkpoint history must contain only generation=-1 evaluation evidence")
    evaluation = evaluation_record.get("eval")
    if not isinstance(evaluation, dict):
        raise RuntimeError("checkpoint history is missing the eval summary")
    if evaluation.get("repeat_count") != expected_repeats:
        raise RuntimeError(
            f"eval repeat_count is {evaluation.get('repeat_count')!r}, expected {expected_repeats}"
        )
    if evaluation.get("count") != expected_count:
        raise RuntimeError(f"eval count is {evaluation.get('count')!r}, expected {expected_count}")
    runs = evaluation.get("runs")
    if not isinstance(runs, list) or len(runs) != expected_repeats:
        raise RuntimeError(f"eval contains {len(runs) if isinstance(runs, list) else 0} runs")

    completed = 0
    legal_action_turns = 0
    invalid_action_turns = 0
    reference_task_ids: set[str] | None = None
    for repeat, run in enumerate(runs):
        if not isinstance(run, dict):
            raise RuntimeError(f"repeat {repeat} is not a result object")
        if run.get("repeat") != repeat:
            raise RuntimeError(f"repeat label is {run.get('repeat')!r}, expected {repeat}")
        if run.get("count") != expected_count:
            raise RuntimeError(f"repeat {repeat} count is {run.get('count')!r}, expected {expected_count}")
        if run.get("valid_count") != expected_count:
            raise RuntimeError(
                f"repeat {repeat} valid_count is {run.get('valid_count')!r}, expected {expected_count}"
            )
        scores = run.get("scores")
        if not isinstance(scores, list) or len(scores) != expected_count:
            raise RuntimeError(f"repeat {repeat} does not contain {expected_count} score rows")
        task_ids: set[str] = set()
        for row in scores:
            if not isinstance(row, dict):
                raise RuntimeError(f"repeat {repeat} contains a non-object score row")
            task_id = str(row.get("task_id", ""))
            score = row.get("score")
            turns = row.get("turns")
            if not task_id or task_id in task_ids:
                raise RuntimeError(f"repeat {repeat} contains a missing or duplicate task_id")
            if not isinstance(score, (int, float)) or not math.isfinite(float(score)) or score < 0:
                raise RuntimeError(f"repeat {repeat} task {task_id} has invalid score {score!r}")
            if not isinstance(turns, list) or not turns:
                raise RuntimeError(f"repeat {repeat} task {task_id} has no inspectable trajectory")
            if "error" in row or any(isinstance(turn, dict) and "error" in turn for turn in turns):
                raise RuntimeError(f"repeat {repeat} task {task_id} contains an evaluation error")
            row_legal_turns = sum(
                isinstance(turn, dict) and turn.get("valid") is True for turn in turns
            )
            row_invalid_turns = sum(
                isinstance(turn, dict) and turn.get("valid") is False for turn in turns
            )
            if row_legal_turns == 0:
                raise RuntimeError(f"repeat {repeat} task {task_id} contains no legal actions")
            if not row.get("endpoint"):
                raise RuntimeError(f"repeat {repeat} task {task_id} is missing endpoint evidence")
            legal_action_turns += row_legal_turns
            invalid_action_turns += row_invalid_turns
            task_ids.add(task_id)
            completed += 1
        if reference_task_ids is None:
            reference_task_ids = task_ids
        elif task_ids != reference_task_ids:
            raise RuntimeError(f"repeat {repeat} evaluated a different task set")

    average = evaluation.get("average")
    if not isinstance(average, (int, float)) or not math.isfinite(float(average)):
        raise RuntimeError(f"eval average is invalid: {average!r}")
    return {
        "expected_samples_per_repeat": expected_count,
        "repeat_count": expected_repeats,
        "batched_eval": True,
        "endpoint_batch_size": SUDOKU_RELEASE_ENDPOINT_BATCH_SIZE,
        "completed_trajectories": completed,
        "valid_trajectories": completed,
        "legal_action_turns": legal_action_turns,
        "invalid_action_turns": invalid_action_turns,
        "average": float(average),
    }


def terminate_servers(processes: dict[int, subprocess.Popen[str]]) -> None:
    for process in processes.values():
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 30
    for process in processes.values():
        if process.poll() is not None:
            continue
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--physical-devices", default="auto")
    parser.add_argument("--first-port", type=int, default=12100)
    parser.add_argument("--stage", choices=("stage1", "stage2"), default="stage2")
    parser.add_argument("--eval-data", type=Path, default=ROOT / "data/sudoku/eval.jsonl")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--served-model-name", default="Qwen3.5-4B")
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument(
        "--inference-seed",
        type=int,
        default=None,
        help="Master sampled-generation seed; endpoint index is added to derive unique process seeds.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_path = args.model_path.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()
    eval_data = args.eval_data.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {model_path}")
    if not eval_data.is_file():
        raise FileNotFoundError(f"Sudoku eval data does not exist: {eval_data}")
    run_root.mkdir(parents=True, exist_ok=False)
    logs = run_root / "server_logs"
    logs.mkdir()

    gpus = resolve_gpus(args.physical_devices)
    plan = build_endpoint_plan(
        gpus,
        first_port=args.first_port,
        inference_seed=args.inference_seed,
    )
    busy = [row.port for row in plan if not port_is_free(row.port)]
    if busy:
        raise RuntimeError(f"required ports are already in use: {busy}")

    eval_limit, eval_repeats = (1, 1) if args.stage == "stage1" else (32, 3)
    manifest = {
        "mode": "direct_checkpoint_zero_history_replay",
        "stage": args.stage,
        "model_path": str(model_path),
        "eval_data": str(eval_data),
        "eval_limit": eval_limit,
        "eval_repeats": eval_repeats,
        "engine_count": len(plan),
        "physical_gpu_count": len(gpus),
        "tensor_parallel_size": 1,
        "server": "transformers_flask",
        "batched_eval": True,
        "endpoint_batch_size": SUDOKU_RELEASE_ENDPOINT_BATCH_SIZE,
        "inference_seed": args.inference_seed,
        "gpus": [asdict(gpu) for gpu in gpus],
        "endpoints": [asdict(row) for row in plan],
        "status": "STARTING",
    }
    manifest_path = run_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (run_root / "command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + shlex.join([sys.executable, *sys.argv])
        + "\n",
        encoding="utf-8",
    )
    gpu_snapshot(run_root / "gpu_before.txt")

    processes: dict[int, subprocess.Popen[str]] = {}
    log_handles = []
    try:
        for wave in build_startup_waves(plan):
            for row in wave:
                log_path = logs / f"port_{row.port}_gpu_{row.gpu_index}.log"
                handle = log_path.open("w", encoding="utf-8")
                log_handles.append(handle)
                env = os.environ.copy()
                env["PYTHONPATH"] = os.pathsep.join(
                    [str(ROOT), env.get("PYTHONPATH", "")]
                ).rstrip(os.pathsep)
                command = build_server_command(
                    python=args.python,
                    model_path=model_path,
                    assignment=row,
                )
                processes[row.port] = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                print(f"[start] port={row.port} gpu={row.gpu_index} uuid={row.gpu_uuid}", flush=True)
            wait_for_wave(wave, processes, timeout=args.startup_timeout)

        gpu_snapshot(run_root / "gpu_ready.txt")
        trainer_command = build_trainer_command(
            python=args.python,
            endpoints=[row.endpoint for row in plan],
            eval_data=eval_data,
            eval_limit=eval_limit,
            eval_repeats=eval_repeats,
            result_root=run_root,
            served_model_name=args.served_model_name,
        )
        (run_root / "trainer_command.sh").write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join(trainer_command) + "\n",
            encoding="utf-8",
        )
        with (run_root / "stdout.log").open("w", encoding="utf-8") as output:
            completed = subprocess.run(
                trainer_command,
                cwd=ROOT,
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"Sudoku evaluator exited with code {completed.returncode}")
        history = json.loads((run_root / "history.json").read_text(encoding="utf-8"))
        manifest["validation"] = validate_eval_history(
            history,
            expected_count=eval_limit,
            expected_repeats=eval_repeats,
        )
        manifest["status"] = "COMPLETED"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    except BaseException:
        manifest["status"] = "FAILED"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        terminate_servers(processes)
        for handle in log_handles:
            handle.close()
        gpu_snapshot(run_root / "gpu_after.txt")


if __name__ == "__main__":
    main()
