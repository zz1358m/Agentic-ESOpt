#!/usr/bin/env python3
"""Run the fixed DAPO-100 and AIME 2026 evaluation on four TP=1 replicas."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DOCVQA_SCRIPT_DIR = ROOT / "scripts" / "docvqa"
if str(DOCVQA_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(DOCVQA_SCRIPT_DIR))

from gpu_visibility import (  # noqa: E402
    cuda_visible_devices,
    manifest as gpu_manifest,
    resolve_physical_gpus,
)
_COMMON_SPEC = importlib.util.spec_from_file_location(
    "_docvqa_four_gpu_eval",
    DOCVQA_SCRIPT_DIR / "run_four_gpu_eval.py",
)
if _COMMON_SPEC is None or _COMMON_SPEC.loader is None:
    raise ImportError("could not load shared four-GPU evaluation helpers")
_COMMON = importlib.util.module_from_spec(_COMMON_SPEC)
_COMMON_SPEC.loader.exec_module(_COMMON)
preflight = _COMMON.preflight
results_complete = _COMMON.results_complete
server_command = _COMMON.server_command
validate_manifest_fingerprint = _COMMON.validate_manifest_fingerprint
validate_results = _COMMON.validate_results
wait_for_servers = _COMMON.wait_for_servers


SERVED_MODEL = "qwen35-4b-math"
APPROVED_PHYSICAL_GPUS = ("3", "4", "5", "6")
DATASETS = ("dapo100", "aime2026")
MATCHED_PROFILE = "matched-agentic"
REPO_REACT_V1_50X4096_PROFILE = "repo-react-v1-50x4096"
REPO_REACT_V1_TURN100_PROFILE = "repo-react-v1-turn100"
EVAL_MAX_TURNS = 50
EVAL_MAX_TOKENS = 4096
EVAL_CONTEXT_LENGTH = 262144


def resolve_eval_gpus(
    physical_gpu_value: str,
    gpu_uuids: str,
    *,
    query_output: str | None = None,
) -> tuple[tuple[str, ...], list[Any]]:
    requested = tuple(part.strip() for part in physical_gpu_value.split(",") if part.strip())
    if requested != APPROVED_PHYSICAL_GPUS:
        raise ValueError(
            "Math evaluation is fixed to physical GPUs 3,4,5,6; "
            f"got {physical_gpu_value!r}"
        )
    identities = resolve_physical_gpus(
        physical_gpu_value,
        expected_uuids=gpu_uuids or None,
        query_output=query_output,
    )
    return requested, identities


def _read_eval_rows(path: Path, expected: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected:
        raise RuntimeError(f"{path} contains {len(rows)} rows, expected {expected}")
    ids = [str(row.get("id", "")).strip() for row in rows]
    if any(not task_id for task_id in ids) or len(set(ids)) != len(ids):
        raise RuntimeError(f"{path} evaluation ids must be non-empty and unique")
    return rows


def expected_result_keys(math_root: Path, *, samples: int) -> dict[str, set[str]]:
    specs = {
        "dapo100": (math_root / "dapo_test.jsonl", 100),
        "aime2026": (math_root / "aime_2026.jsonl", 30),
    }
    result: dict[str, set[str]] = {}
    for dataset, (path, expected) in specs.items():
        rows = _read_eval_rows(path, expected)
        result[dataset] = {
            f"{dataset}:{row['id']}:sample{sample_index:02d}"
            for row in rows
            for sample_index in range(samples)
        }
    return result


def evaluator_command(
    *,
    python: str,
    evaluator: Path,
    endpoints: list[str],
    model_path: Path,
    math_root: Path,
    out_dir: Path,
    samples: int,
    concurrency: int,
    seed: int,
    resume: bool,
    profile: str = MATCHED_PROFILE,
) -> list[str]:
    if profile not in (
        MATCHED_PROFILE,
        REPO_REACT_V1_50X4096_PROFILE,
        REPO_REACT_V1_TURN100_PROFILE,
    ):
        raise ValueError(f"unknown evaluation profile: {profile}")
    repo_react_v1 = profile in (
        REPO_REACT_V1_50X4096_PROFILE,
        REPO_REACT_V1_TURN100_PROFILE,
    )
    command = [
        python,
        str(evaluator),
        "--base-urls",
        ",".join(endpoints),
        "--model",
        SERVED_MODEL,
        "--tokenizer-path",
        str(model_path),
        "--math-root",
        str(math_root),
        "--out-dir",
        str(out_dir),
        "--datasets",
        ",".join(DATASETS),
        "--samples",
        str(samples),
        "--concurrency",
        str(concurrency),
        "--seed",
        str(seed),
        "--temperature",
        "1.0",
        "--top-p",
        "1.0",
        "--top-k",
        "40",
        "--presence-penalty",
        "2.0",
        "--repetition-penalty",
        "1.0",
        "--math-max-turns",
        str(EVAL_MAX_TURNS),
        "--math-max-tokens",
        str(EVAL_MAX_TOKENS),
        "--max-errors",
        "1",
    ]
    if repo_react_v1:
        command.extend(
            [
                "--math-react-prompt",
                "repo-react-v1",
                "--retry-react-errors",
            ]
        )
    if resume:
        command.append("--resume")
    return command


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--math-root", type=Path, default=ROOT / "data/trace2skill/math_reasoning")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--port-base", type=int, default=18180)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--fallback-concurrency", type=int, default=4)
    parser.add_argument("--startup-timeout", type=float, default=1800.0)
    parser.add_argument("--context-length", type=int, default=EVAL_CONTEXT_LENGTH)
    parser.add_argument("--memory-fraction", type=float, default=0.82)
    parser.add_argument("--physical-gpus", default=os.environ.get("MATH_PHYSICAL_GPU_IDS", "3,4,5,6"))
    parser.add_argument("--gpu-uuids", default=os.environ.get("MATH_GPU_UUIDS", ""))
    parser.add_argument("--strict-concurrency", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry-source",
        type=Path,
        help="Seed a turn-limit retry run from a completed 30-turn result directory.",
    )
    parser.add_argument(
        "--profile",
        choices=(
            MATCHED_PROFILE,
            REPO_REACT_V1_50X4096_PROFILE,
            REPO_REACT_V1_TURN100_PROFILE,
        ),
        default=MATCHED_PROFILE,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    physical_gpus, identities = resolve_eval_gpus(args.physical_gpus, args.gpu_uuids)
    required_samples = 16 if args.profile == MATCHED_PROFILE else 4
    if args.samples != required_samples:
        raise ValueError(f"{args.profile} evaluation requires --samples {required_samples}")
    if args.profile != MATCHED_PROFILE and args.context_length < 128 * 1024:
        raise ValueError("four-sample ReAct evaluation requires a context length of at least 128K")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")
    if args.fallback_concurrency <= 0 or args.fallback_concurrency > args.concurrency:
        raise ValueError("--fallback-concurrency must be positive and no greater than --concurrency")

    resolved_visibility = cuda_visible_devices(identities)
    caller_visibility = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    os.environ["CUDA_VISIBLE_DEVICES"] = resolved_visibility
    model_path = args.model_path.expanduser().resolve()
    math_root = args.math_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    expected_keys = expected_result_keys(math_root, samples=args.samples)
    out_dir.mkdir(parents=True, exist_ok=True)

    ports = [args.port_base + offset for offset in range(4)]
    endpoints = [f"http://127.0.0.1:{port}/v1" for port in ports]
    output_paths = {dataset: out_dir / "outputs" / f"{dataset}.jsonl" for dataset in DATASETS}
    manifest_path = out_dir / "four_gpu_manifest.json"
    data_hashes = {
        "dapo100": _sha256(math_root / "dapo_test.jsonl"),
        "aime2026": _sha256(math_root / "aime_2026.jsonl"),
    }
    retry_source = args.retry_source.expanduser().resolve() if args.retry_source else None
    if retry_source is not None:
        if args.profile != REPO_REACT_V1_TURN100_PROFILE or not args.resume:
            raise ValueError("--retry-source requires --profile repo-react-v1-turn100 and --resume")
        for dataset, destination in output_paths.items():
            source = retry_source / "outputs" / f"{dataset}.jsonl"
            if not source.is_file():
                raise FileNotFoundError(source)
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
    fingerprint = {
        "physical_gpus": list(physical_gpus),
        "visible_devices": resolved_visibility,
        "replicas": 4,
        "tensor_parallel_size": 1,
        "endpoints": endpoints,
        "model_path": str(model_path),
        "context_length": args.context_length,
        "samples": args.samples,
        "seed": args.seed,
        "profile": args.profile,
        "data_hashes": data_hashes,
        "retry_source": str(retry_source) if retry_source else None,
    }
    if args.resume and manifest_path.exists():
        validate_manifest_fingerprint(manifest_path, fingerprint)
    elif args.resume and any(path.exists() for path in output_paths.values()) and retry_source is None:
        validate_manifest_fingerprint(manifest_path, fingerprint)
    if args.resume and all(
        results_complete(path, len(expected_keys[dataset]), expected_keys=expected_keys[dataset])
        for dataset, path in output_paths.items()
    ) and retry_source is None:
        total_records = sum(len(keys) for keys in expected_keys.values())
        print(f"Math evaluation already complete: {out_dir} ({total_records} records)", flush=True)
        return

    logs_dir = out_dir / "server_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen[bytes]] = []
    handles: list[Any] = []
    log_paths: list[Path] = []
    selected_concurrency = args.concurrency
    preflight_errors: list[str] = []
    try:
        for identity, port in zip(identities, ports, strict=True):
            gpu = str(identity.index)
            log_path = logs_dir / f"gpu{gpu}.log"
            handle = log_path.open("ab", buffering=0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = identity.uuid
            env["TRACE2SKILL_EAGER_PATCH_DENSE_QWEN3NEXT"] = "1"
            env["TRACE2SKILL_REGISTER_TOOL_PARSER"] = "1"
            env["TRITON_CACHE_DIR"] = str((logs_dir / "triton_cache" / f"gpu{gpu}").resolve())
            Path(env["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
            python_paths = [str(ROOT / "algorithms" / "verl_trace2skill"), str(ROOT), str(ROOT / "algorithms" / "verl")]
            if env.get("PYTHONPATH"):
                python_paths.append(env["PYTHONPATH"])
            env["PYTHONPATH"] = os.pathsep.join(python_paths)
            command = server_command(
                args.python,
                model_path,
                port,
                context_length=args.context_length,
                memory_fraction=args.memory_fraction,
                served_model=SERVED_MODEL,
            )
            processes.append(
                subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            )
            handles.append(handle)
            log_paths.append(log_path)

        wait_for_servers(processes, log_paths, ports, physical_gpus, args.startup_timeout)
        ok, preflight_errors = preflight(
            endpoints,
            selected_concurrency,
            served_model=SERVED_MODEL,
        )
        if not ok and not args.strict_concurrency and selected_concurrency != args.fallback_concurrency:
            selected_concurrency = args.fallback_concurrency
            ok, fallback_errors = preflight(
                endpoints,
                selected_concurrency,
                served_model=SERVED_MODEL,
            )
            preflight_errors.extend(fallback_errors)
        if not ok:
            raise RuntimeError("model-service preflight failed:\n" + "\n".join(preflight_errors))

        repo_react_v1 = args.profile in (
            REPO_REACT_V1_50X4096_PROFILE,
            REPO_REACT_V1_TURN100_PROFILE,
        )
        manifest = {
            **fingerprint,
            "caller_visible_devices": caller_visibility,
            "gpu_resources": gpu_manifest(identities),
            "requested_concurrency": args.concurrency,
            "fallback_concurrency": args.fallback_concurrency,
            "concurrency": selected_concurrency,
            "strict_concurrency": args.strict_concurrency,
            "preflight_errors": preflight_errors,
            "datasets": list(DATASETS),
            "sampling": {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 40,
                "presence_penalty": 2.0,
                "repetition_penalty": 1.0,
            },
            "protocol": (
                "Repository Math ReAct v1 + Action JSON + bash observation"
                if repo_react_v1
                else "Action JSON + bash observation"
            ),
            "evaluation_turn_limit": EVAL_MAX_TURNS,
            "max_react_turns": EVAL_MAX_TURNS,
            "max_turn_tokens": EVAL_MAX_TOKENS,
            "max_output_tokens": EVAL_MAX_TOKENS,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        command = evaluator_command(
            python=args.python,
            evaluator=ROOT / "scripts/trace2skill/run_trace2skill_vllm_eval16.py",
            endpoints=endpoints,
            model_path=model_path,
            math_root=math_root,
            out_dir=out_dir,
            samples=args.samples,
            concurrency=selected_concurrency,
            seed=args.seed,
            resume=args.resume,
            profile=args.profile,
        )
        subprocess.run(command, check=True, env=os.environ.copy())
        for dataset, path in output_paths.items():
            validate_results(path, len(expected_keys[dataset]), expected_keys=expected_keys[dataset])
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + 30.0
        for process in processes:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
