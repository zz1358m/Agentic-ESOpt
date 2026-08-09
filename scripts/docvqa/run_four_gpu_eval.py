#!/usr/bin/env python3
"""Start four TP=1 SGLang replicas and run the fixed DocVQA evaluation."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from gpu_visibility import (  # noqa: E402
    cuda_visible_devices,
    manifest as gpu_manifest,
    resolve_physical_gpus,
)

SERVED_MODEL = "qwen35-4b-docvqa"


def resolve_eval_gpus(
    physical_gpu_value: str,
    gpu_uuids: str,
    *,
    query_output: str | None = None,
) -> tuple[tuple[str, ...], list[Any]]:
    identities = resolve_physical_gpus(
        physical_gpu_value,
        expected_uuids=gpu_uuids or None,
        query_output=query_output,
    )
    if len(identities) != 4:
        raise ValueError(f"DocVQA evaluation requires four GPUs, got {len(identities)}")
    physical_gpus = tuple(str(identity.index) for identity in identities)
    return physical_gpus, identities


def server_command(
    python: str,
    model_path: Path,
    port: int,
    *,
    context_length: int = 131072,
    memory_fraction: float = 0.82,
    served_model: str = SERVED_MODEL,
) -> list[str]:
    return [
        python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model_path),
        "--served-model-name",
        served_model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tp-size",
        "1",
        "--context-length",
        str(context_length),
        "--mem-fraction-static",
        str(memory_fraction),
        "--trust-remote-code",
    ]


def _request_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _tail(path: Path, lines: int = 50) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def wait_for_servers(
    processes: list[subprocess.Popen[bytes]],
    log_paths: list[Path],
    ports: list[int],
    physical_gpus: tuple[str, ...],
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    pending = set(range(len(ports)))
    while pending and time.monotonic() < deadline:
        for index in list(pending):
            code = processes[index].poll()
            if code is not None:
                raise RuntimeError(
                    f"SGLang replica on GPU {physical_gpus[index]} exited with {code}:\n{_tail(log_paths[index])}"
                )
            try:
                _request_json(f"http://127.0.0.1:{ports[index]}/v1/models", timeout=2.0)
            except (OSError, urllib.error.URLError, ValueError):
                continue
            pending.remove(index)
        if pending:
            time.sleep(2.0)
    if pending:
        details = "\n".join(
            f"GPU {physical_gpus[index]}:\n{_tail(log_paths[index], 20)}" for index in sorted(pending)
        )
        raise TimeoutError(f"SGLang replicas did not become ready: {sorted(pending)}\n{details}")


def preflight(
    endpoints: list[str],
    concurrency: int,
    *,
    served_model: str = SERVED_MODEL,
) -> tuple[bool, list[str]]:
    payload = {
        "model": served_model,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "temperature": 0.0,
        "max_tokens": 8,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    errors: list[str] = []

    def one(index: int) -> None:
        endpoint = endpoints[index % len(endpoints)]
        try:
            _request_json(endpoint + "/chat/completions", payload, timeout=120.0)
        except Exception as exc:  # noqa: BLE001 - collect every endpoint failure for the manifest
            errors.append(f"{endpoint}: {type(exc).__name__}: {exc}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(one, range(concurrency)))
    return not errors, errors


def expected_result_keys(data_path: Path, *, limit: int, samples: int) -> set[str]:
    rows = [
        json.loads(line)
        for line in data_path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ][:limit]
    if len(rows) != limit:
        raise RuntimeError(f"DocVQA data has {len(rows)} rows, expected at least {limit}")
    task_ids = [str(row.get("id", row_index)).strip() for row_index, row in enumerate(rows)]
    if any(not task_id for task_id in task_ids) or len(set(task_ids)) != len(task_ids):
        raise RuntimeError("DocVQA evaluation task_ids must be non-empty and unique")
    return {
        f"docvqa:{task_id}:sample{sample_index:02d}"
        for task_id in task_ids
        for sample_index in range(samples)
    }


def validate_manifest_fingerprint(path: Path, expected_fields: dict[str, Any]) -> None:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"evaluation resume manifest is unavailable or invalid: {path}") from exc
    mismatches = {
        key: {"expected": expected, "actual": manifest.get(key)}
        for key, expected in expected_fields.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            "evaluation resume manifest fingerprint mismatch: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )


def validate_results(
    path: Path,
    expected: int,
    *,
    expected_keys: set[str] | None = None,
    attempts: int = 5,
    retry_delay: float = 1.0,
) -> None:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
            keys = [str(row.get("key", "")) for row in rows]
            errors = [row for row in rows if row.get("error")]
            key_set_matches = expected_keys is None or set(keys) == expected_keys
            if len(rows) == expected and len(set(keys)) == expected and not errors and key_set_matches:
                return
            last_error = RuntimeError(
                f"evaluation acceptance failed: records={len(rows)}, unique={len(set(keys))}, "
                f"request_errors={len(errors)}, key_set_matches={key_set_matches}, expected={expected}"
            )
        except (OSError, json.JSONDecodeError) as exc:
            last_error = exc
        if attempt + 1 < attempts:
            time.sleep(retry_delay)
    raise RuntimeError(f"evaluation acceptance failed after {attempts} attempts: {last_error}") from last_error


def results_complete(
    path: Path,
    expected: int,
    *,
    expected_keys: set[str] | None = None,
) -> bool:
    try:
        validate_results(
            path,
            expected,
            expected_keys=expected_keys,
            attempts=2,
            retry_delay=0.1,
        )
    except RuntimeError:
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--docvqa-root", type=Path, required=True)
    parser.add_argument("--docvqa-data", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--port-base", type=int, default=18080)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--fallback-concurrency", type=int, default=4)
    parser.add_argument("--startup-timeout", type=float, default=1800.0)
    parser.add_argument("--context-length", type=int, default=131072)
    parser.add_argument("--memory-fraction", type=float, default=0.82)
    parser.add_argument("--physical-gpus", default=os.environ.get("DOCVQA_PHYSICAL_GPU_IDS", "auto"))
    parser.add_argument("--gpu-uuids", default=os.environ.get("DOCVQA_GPU_UUIDS", ""))
    parser.add_argument("--strict-concurrency", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    physical_gpus, identities = resolve_eval_gpus(
        args.physical_gpus,
        args.gpu_uuids,
    )
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")
    if args.fallback_concurrency <= 0 or args.fallback_concurrency > args.concurrency:
        raise ValueError("--fallback-concurrency must be positive and no greater than --concurrency")
    resolved_visibility = cuda_visible_devices(identities)
    caller_visibility = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    os.environ["CUDA_VISIBLE_DEVICES"] = resolved_visibility
    model_path = args.model_path.expanduser().resolve()
    docvqa_root = args.docvqa_root.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    data_path = (
        args.docvqa_data.expanduser().resolve()
        if args.docvqa_data is not None
        else docvqa_root / "data/trace2skill/docvqa/test.jsonl"
    )
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    expected_keys = expected_result_keys(
        data_path,
        limit=args.limit,
        samples=args.samples,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.out_dir / "outputs/docvqa.jsonl"
    expected_results = args.limit * args.samples
    ports = [args.port_base + offset for offset in range(len(identities))]
    endpoints = [f"http://127.0.0.1:{port}/v1" for port in ports]
    manifest_path = args.out_dir / "four_gpu_manifest.json"
    manifest_fingerprint = {
        "physical_gpus": list(physical_gpus),
        "visible_devices": resolved_visibility,
        "replicas": len(identities),
        "tensor_parallel_size": 1,
        "endpoints": endpoints,
        "requested_concurrency": args.concurrency,
        "fallback_concurrency": args.fallback_concurrency,
        "model_path": str(model_path),
        "context_length": args.context_length,
        "samples": args.samples,
        "limit": args.limit,
        "seed": args.seed,
        "data_path": str(data_path),
    }
    if args.resume and (result_path.exists() or manifest_path.exists()):
        validate_manifest_fingerprint(manifest_path, manifest_fingerprint)
    if args.resume and results_complete(
        result_path,
        expected_results,
        expected_keys=expected_keys,
    ):
        print(f"DocVQA evaluation already complete: {result_path} ({expected_results} records)", flush=True)
        return
    logs_dir = args.out_dir / "server_logs"
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
            env["TRITON_CACHE_DIR"] = str((logs_dir / "triton_cache" / f"gpu{gpu}").resolve())
            Path(env["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
            root = Path(__file__).resolve().parents[2]
            python_paths = [str(root / "verl_trace2skill"), str(root), str(root / "verl")]
            if env.get("PYTHONPATH"):
                python_paths.append(env["PYTHONPATH"])
            env["PYTHONPATH"] = os.pathsep.join(python_paths)
            command = server_command(
                args.python,
                model_path,
                port,
                context_length=args.context_length,
                memory_fraction=args.memory_fraction,
            )
            processes.append(
                subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            )
            handles.append(handle)
            log_paths.append(log_path)

        wait_for_servers(processes, log_paths, ports, physical_gpus, args.startup_timeout)
        ok, preflight_errors = preflight(endpoints, selected_concurrency)
        if not ok and not args.strict_concurrency and selected_concurrency != args.fallback_concurrency:
            selected_concurrency = args.fallback_concurrency
            ok, fallback_errors = preflight(endpoints, selected_concurrency)
            preflight_errors.extend(fallback_errors)
        if not ok:
            raise RuntimeError("model-service preflight failed:\n" + "\n".join(preflight_errors))

        manifest = {
            "physical_gpus": list(physical_gpus),
            "caller_visible_devices": caller_visibility,
            "visible_devices": resolved_visibility,
            "gpu_resources": gpu_manifest(identities),
            "replicas": len(identities),
            "tensor_parallel_size": 1,
            "endpoints": endpoints,
            "requested_concurrency": args.concurrency,
            "fallback_concurrency": args.fallback_concurrency,
            "concurrency": selected_concurrency,
            "strict_concurrency": args.strict_concurrency,
            "preflight_errors": preflight_errors,
            "model_path": str(model_path),
            "context_length": args.context_length,
            "samples": args.samples,
            "limit": args.limit,
            "seed": args.seed,
            "data_path": str(data_path),
            "protocol": "paper_react_cli",
            "max_react_turns": 50,
            "max_total_tokens": 32768,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        evaluator = Path(__file__).resolve().parents[1] / "trace2skill" / "run_trace2skill_vllm_eval16.py"
        command = [
            args.python,
            str(evaluator),
            "--base-urls",
            ",".join(endpoints),
            "--model",
            SERVED_MODEL,
            "--tokenizer-path",
            str(model_path),
            "--docvqa-root",
            str(docvqa_root),
            "--docvqa-data",
            str(data_path),
            "--out-dir",
            str(args.out_dir),
            "--datasets",
            "docvqa",
            "--samples",
            str(args.samples),
            "--docvqa-limit",
            str(args.limit),
            "--concurrency",
            str(selected_concurrency),
            "--seed",
            str(args.seed),
            "--docvqa-max-turns",
            "50",
            "--docvqa-max-total-tokens",
            "32768",
            "--max-errors",
            "1",
        ]
        if args.resume:
            command.append("--resume")
        subprocess.run(command, check=True, env=os.environ.copy())
        validate_results(
            result_path,
            expected_results,
            expected_keys=expected_keys,
        )
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
