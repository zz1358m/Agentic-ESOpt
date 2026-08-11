#!/usr/bin/env python3
"""Run fixed Math evaluation on eight TP=1 replicas across four GPUs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
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
    sglang_visible_device,
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
common_server_command = _COMMON.server_command
validate_manifest_fingerprint = _COMMON.validate_manifest_fingerprint
validate_results = _COMMON.validate_results
wait_for_servers = _COMMON.wait_for_servers
terminate_server_process_groups = _COMMON.terminate_server_process_groups
prepare_skill_evidence = _COMMON.prepare_skill_evidence


SERVED_MODEL = "qwen35-4b-math"
PHYSICAL_GPU_COUNT = 4
ENGINE_COUNT = 8
DATASETS = ("dapo100", "aime2026")
MATCHED_PROFILE = "matched-agentic"
REPO_REACT_V1_50X4096_PROFILE = "repo-react-v1-50x4096"
REPO_REACT_V1_TURN100_PROFILE = "repo-react-v1-turn100"
EVAL_MAX_TURNS = 50
EVAL_MAX_TOKENS = 4096
EVAL_CONTEXT_LENGTH = 262144


@dataclass(frozen=True)
class EndpointAssignment:
    port: int
    endpoint: str
    gpu_index: int
    gpu_uuid: str
    gpu_name: str


def server_command(
    python: str,
    model_path: Path,
    port: int,
    *,
    context_length: int,
    memory_fraction: float,
) -> list[str]:
    """Build the stable two-engines-per-GPU SGLang command used by Math."""
    return common_server_command(
        python,
        model_path,
        port,
        context_length=context_length,
        memory_fraction=memory_fraction,
        served_model=SERVED_MODEL,
        disable_cuda_graph=True,
        disable_overlap_schedule=True,
    )


def resolve_eval_gpus(
    physical_gpu_value: str,
    gpu_uuids: str,
    *,
    query_output: str | None = None,
) -> tuple[tuple[str, ...], list[Any]]:
    requested = tuple(part.strip() for part in physical_gpu_value.split(",") if part.strip())
    if len(requested) != PHYSICAL_GPU_COUNT or len(set(requested)) != PHYSICAL_GPU_COUNT:
        raise ValueError(f"Math evaluation requires four unique physical GPUs, got {physical_gpu_value!r}")
    identities = resolve_physical_gpus(
        physical_gpu_value,
        expected_uuids=gpu_uuids or None,
        query_output=query_output,
    )
    return requested, identities


def build_endpoint_plan(identities: list[Any], *, port_base: int) -> list[EndpointAssignment]:
    if len(identities) != PHYSICAL_GPU_COUNT or len({row.uuid for row in identities}) != PHYSICAL_GPU_COUNT:
        raise ValueError("Math endpoint planning requires four unique physical GPUs")
    return [
        EndpointAssignment(
            port=port_base + engine_index,
            endpoint=f"http://127.0.0.1:{port_base + engine_index}/v1",
            gpu_index=int(identity.index),
            gpu_uuid=str(identity.uuid),
            gpu_name=str(identity.name),
        )
        for engine_index in range(ENGINE_COUNT)
        for identity in [identities[engine_index % PHYSICAL_GPU_COUNT]]
    ]


def _read_eval_rows(path: Path, expected: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected:
        raise RuntimeError(f"{path} contains {len(rows)} rows, expected {expected}")
    ids = [str(row.get("id", "")).strip() for row in rows]
    if any(not task_id for task_id in ids) or len(set(ids)) != len(ids):
        raise RuntimeError(f"{path} evaluation ids must be non-empty and unique")
    return rows


def parse_datasets(value: str) -> tuple[str, ...]:
    requested = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = set(requested) - set(DATASETS)
    if not requested or unknown:
        raise ValueError(
            f"--datasets must select from {','.join(DATASETS)}; got {value!r}"
        )
    return tuple(dataset for dataset in DATASETS if dataset in requested)


def expected_result_keys(
    math_root: Path,
    *,
    samples: int,
    datasets: tuple[str, ...] = DATASETS,
) -> dict[str, set[str]]:
    specs = {
        "dapo100": (math_root / "dapo_test.jsonl", 100),
        "aime2026": (math_root / "aime_2026.jsonl", 30),
    }
    result: dict[str, set[str]] = {}
    for dataset in datasets:
        path, expected = specs[dataset]
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
    dapo_seed: int | None = None,
    aime_seed: int | None = None,
    resume: bool,
    math_skill_file: Path | None = None,
    profile: str = MATCHED_PROFILE,
    datasets: tuple[str, ...] = DATASETS,
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
        ",".join(datasets),
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
    if dapo_seed is not None:
        command.extend(["--dapo-seed", str(dapo_seed)])
    if aime_seed is not None:
        command.extend(["--aime-seed", str(aime_seed)])
    if math_skill_file is not None:
        command.extend(["--math-skill-file", str(math_skill_file)])
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
    parser.add_argument("--math-skill-file", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--port-base", type=int, default=18180)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--dapo-seed", type=int, default=20270652)
    parser.add_argument("--aime-seed", type=int, default=20280652)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--fallback-concurrency", type=int, default=4)
    parser.add_argument("--startup-timeout", type=float, default=1800.0)
    parser.add_argument("--context-length", type=int, default=EVAL_CONTEXT_LENGTH)
    parser.add_argument("--memory-fraction", type=float, default=0.4)
    parser.add_argument("--physical-gpus", default=os.environ.get("MATH_PHYSICAL_GPU_IDS", "0,1,2,3"))
    parser.add_argument("--datasets", default=",".join(DATASETS))
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
    datasets = parse_datasets(args.datasets)
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
    expected_keys = expected_result_keys(math_root, samples=args.samples, datasets=datasets)
    out_dir.mkdir(parents=True, exist_ok=True)
    recorded_skill, skill_metadata = prepare_skill_evidence(
        args.math_skill_file,
        out_dir,
        task_name="math",
    )

    plan = build_endpoint_plan(identities, port_base=args.port_base)
    identities_by_index = {identity.index: identity for identity in identities}
    ports = [row.port for row in plan]
    endpoints = [row.endpoint for row in plan]
    output_paths = {dataset: out_dir / "outputs" / f"{dataset}.jsonl" for dataset in datasets}
    manifest_path = out_dir / "four_gpu_manifest.json"
    data_paths = {
        "dapo100": math_root / "dapo_test.jsonl",
        "aime2026": math_root / "aime_2026.jsonl",
    }
    data_hashes = {dataset: _sha256(data_paths[dataset]) for dataset in datasets}
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
        "replicas": ENGINE_COUNT,
        "tensor_parallel_size": 1,
        "endpoints": endpoints,
        "engine_mapping": [row.__dict__ for row in plan],
        "model_path": str(model_path),
        "context_length": args.context_length,
        "samples": args.samples,
        "seed": args.seed,
        "dataset_seeds": {
            "dapo100": args.dapo_seed,
            "aime2026": args.aime_seed,
        },
        "datasets": list(datasets),
        "profile": args.profile,
        "data_hashes": data_hashes,
        "retry_source": str(retry_source) if retry_source else None,
        "skill": skill_metadata,
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
        for wave_start in range(0, ENGINE_COUNT, PHYSICAL_GPU_COUNT):
            wave = plan[wave_start : wave_start + PHYSICAL_GPU_COUNT]
            wave_processes: list[subprocess.Popen[bytes]] = []
            wave_logs: list[Path] = []
            for row in wave:
                gpu = str(row.gpu_index)
                log_path = logs_dir / f"port{row.port}_gpu{gpu}.log"
                handle = log_path.open("ab", buffering=0)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = sglang_visible_device(
                    identities_by_index[row.gpu_index]
                )
                env["TRACE2SKILL_EAGER_PATCH_DENSE_QWEN3NEXT"] = "1"
                env["TRACE2SKILL_REGISTER_TOOL_PARSER"] = "1"
                env["TRITON_CACHE_DIR"] = str((logs_dir / "triton_cache" / f"port{row.port}").resolve())
                Path(env["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
                python_paths = [str(ROOT / "algorithms" / "verl_trace2skill"), str(ROOT), str(ROOT / "algorithms" / "verl")]
                if env.get("PYTHONPATH"):
                    python_paths.append(env["PYTHONPATH"])
                env["PYTHONPATH"] = os.pathsep.join(python_paths)
                command = server_command(
                    args.python,
                    model_path,
                    row.port,
                    context_length=args.context_length,
                    memory_fraction=args.memory_fraction,
                )
                process = subprocess.Popen(
                    command,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
                processes.append(process)
                wave_processes.append(process)
                handles.append(handle)
                log_paths.append(log_path)
                wave_logs.append(log_path)

            wait_for_servers(
                wave_processes,
                wave_logs,
                [row.port for row in wave],
                tuple(str(row.gpu_index) for row in wave),
                args.startup_timeout,
            )
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
            "datasets": list(datasets),
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
            dapo_seed=args.dapo_seed,
            aime_seed=args.aime_seed,
            resume=args.resume,
            profile=args.profile,
            datasets=datasets,
            math_skill_file=recorded_skill,
        )
        subprocess.run(command, check=True, env=os.environ.copy())
        for dataset, path in output_paths.items():
            validate_results(path, len(expected_keys[dataset]), expected_keys=expected_keys[dataset])
    finally:
        terminate_server_process_groups(processes)
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
