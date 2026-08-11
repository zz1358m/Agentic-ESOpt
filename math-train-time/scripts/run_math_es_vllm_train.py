#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("ROOT", Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "math-train-time"))

from envs.math_reasoning import (  # noqa: E402
    MathReasoningEnv,
    MathRolloutJob,
    MathTask,
    extract_math_answer,
    final_answer_line,
    math_react_messages,
    math_score,
    parse_react_action,
    react_observation_text,
    run_bash,
    safe_workdir_name,
    strip_think,
    trim_oldest_react_exchange,
    trace_markdown,
)
from algorithms.es.run_state import (  # noqa: E402
    atomic_write_history,
    completed_update_records,
    history_prefix_through_updates,
    read_history,
    resolve_warmup_steps,
    sigma_at_step,
    validate_es_run_shape,
    validate_seed_sequence,
)


DEFAULT_TRAIN = ROOT / "data/trace2skill/math_reasoning/dapo_evolve.jsonl"
DEFAULT_EVAL = ROOT / "data/trace2skill/math_reasoning/dapo_test.jsonl"
DEFAULT_AIME = ROOT / "data/trace2skill/math_reasoning/aime_2026.jsonl"
EVAL_DATASETS = ("dapo", "aime")
FORMAL_EVAL_ITEMS = {"dapo": 100, "aime": 30}
FORMAL_EVAL_KEY_SEED_SHA256 = "26f2b5ee15bb079d1d16feb2261fec24a5af9e41b33bcea7d14ac5259ce6c7c5"
FORMAL_EVAL_DATA_SHA256 = {
    "dapo": "a0e64c93e7801957f0949ab80f5a26233ecd87a02ad5c4628de2da0692b5c4a2",
    "aime": "abc8651f3af75ff59341b9de986fef39b1e909aa1466e3b73ee20ec9b6f7242e",
}
FORMAL_EVAL_SKILL_SHA256 = "481e7a67d10bac9575786f166b47f4d64f306fde0107b19c12aa1b2e7b53b275"


def parse_eval_datasets(value: str) -> tuple[str, ...]:
    requested = {item.strip() for item in value.split(",") if item.strip()}
    unknown = requested - set(EVAL_DATASETS)
    if not requested or unknown:
        raise ValueError(
            f"eval datasets must select from {','.join(EVAL_DATASETS)}; got {value!r}"
        )
    return tuple(dataset for dataset in EVAL_DATASETS if dataset in requested)


def trajectory_seed(base_seed: int, *, row_index: int, sample_index: int) -> int:
    return int(base_seed) + int(sample_index) * 1_000_003 + int(row_index)


def load_eval_key_seeds(
    path: str | Path, *, expected_sha256: str | None = None
) -> dict[str, int]:
    seed_path = Path(path).expanduser().resolve()
    content = seed_path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"eval key/seed SHA256 mismatch for {seed_path}: "
            f"expected {expected_sha256}, got {digest}"
        )

    result: dict[str, int] = {}
    for line_number, raw_line in enumerate(content.decode("utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        if not isinstance(row, dict) or "key" not in row or "seed" not in row:
            raise ValueError(f"invalid eval key/seed row at {seed_path}:{line_number}")
        key = str(row["key"])
        if key in result:
            raise ValueError(f"duplicate key in eval key/seed file: {key}")
        result[key] = int(row["seed"])
    if not result:
        raise ValueError(f"eval key/seed file is empty: {seed_path}")
    return result


def require_file_sha256(path: str | Path, expected_sha256: str, *, label: str) -> str:
    resolved = Path(path).expanduser().resolve()
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"{label} SHA256 mismatch for {resolved}: expected {expected_sha256}, got {digest}"
        )
    return digest


def validate_formal_eval_artifacts(
    args: argparse.Namespace, eval_datasets: tuple[str, ...]
) -> None:
    data_paths = {"dapo": args.eval_data, "aime": args.aime_data}
    for dataset in eval_datasets:
        require_file_sha256(
            data_paths[dataset],
            FORMAL_EVAL_DATA_SHA256[dataset],
            label=f"formal {dataset} data",
        )
    if args.skill_file:
        require_file_sha256(
            args.skill_file,
            FORMAL_EVAL_SKILL_SHA256,
            label="formal Math skill",
        )


def validate_eval_key_seed_coverage(
    dataset: str,
    key_seeds: dict[str, int],
    *,
    tasks: list[MathTask],
    samples: int,
) -> None:
    expected_keys = {
        f"{task.id}:sample{sample_index:02d}"
        for task in tasks
        for sample_index in range(samples)
    }
    missing = sorted(expected_keys - set(key_seeds))
    if missing:
        raise ValueError(f"{dataset}: eval key/seed file is missing keys {missing[:3]}")


def validate_formal_eval_summary(
    dataset: str,
    summary: dict[str, Any],
    *,
    tasks: list[MathTask],
    samples: int,
    base_seed: int | None = None,
    expected_seeds: dict[str, int] | None = None,
) -> None:
    if dataset not in FORMAL_EVAL_ITEMS:
        raise ValueError(f"unknown formal eval dataset: {dataset}")
    expected_items = FORMAL_EVAL_ITEMS[dataset]
    if len(tasks) != expected_items:
        raise ValueError(f"{dataset}: expected {expected_items} tasks, got {len(tasks)}")
    if samples != 4:
        raise ValueError(f"{dataset}: formal evaluation requires 4 samples, got {samples}")

    rows = summary.get("scores")
    if not isinstance(rows, list):
        raise ValueError(f"{dataset}: formal evaluation summary is missing scores")
    expected_count = expected_items * samples
    if len(rows) != expected_count:
        raise ValueError(f"{dataset}: expected {expected_count} rows, got {len(rows)}")

    expected_keys = {
        f"{task.id}:sample{sample_index:02d}"
        for task in tasks
        for sample_index in range(samples)
    }
    actual_keys = [str(row.get("key", "")) for row in rows]
    if len(set(actual_keys)) != len(actual_keys):
        raise ValueError(f"{dataset}: duplicate result keys")
    if set(actual_keys) != expected_keys:
        missing = sorted(expected_keys - set(actual_keys))[:3]
        extra = sorted(set(actual_keys) - expected_keys)[:3]
        raise ValueError(f"{dataset}: result key mismatch missing={missing} extra={extra}")

    for row in rows:
        for field in ("seed", "engine_index"):
            if field not in row:
                raise ValueError(f"{dataset}: result row {row.get('key')} is missing {field}")
        if expected_seeds is not None:
            expected_seed = expected_seeds[str(row["key"])]
        elif base_seed is not None:
            expected_seed = trajectory_seed(
                base_seed,
                row_index=int(row["row_index"]),
                sample_index=int(row["sample_index"]),
            )
        else:
            expected_seed = None
        if expected_seed is not None:
            if int(row["seed"]) != expected_seed:
                raise ValueError(
                    f"{dataset}: result row {row.get('key')} has seed={row['seed']}, "
                    f"expected {expected_seed}"
                )


def validate_formal_eval_options(args: argparse.Namespace, eval_datasets: tuple[str, ...]) -> None:
    if not args.formal_eval:
        return
    shortcut_flags = {
        "--skip-initial-eval": bool(args.skip_initial_eval),
        "--reuse-initial-eval-history": bool(args.reuse_initial_eval_history),
        "--resume-history": bool(args.resume_history),
    }
    for flag, enabled in shortcut_flags.items():
        if enabled:
            raise ValueError(f"{flag} is incompatible with --formal-eval")
    if not args.eval_key_seed_file:
        raise ValueError("--eval-key-seed-file is required with --formal-eval")

    expected_options = {
        "eval_only": True,
        "generations": 0,
        "num_engines": 4,
        "gpu_fraction": 1.0,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.85,
        "max_model_len": 131072,
        "gdn_prefill_backend": "triton",
        "enforce_eager": True,
        "inference_batch_size": 16,
        "rollout_token_budget": 131072,
        "max_total_tokens": 0,
        "eval_samples": 4,
        "eval_limit": FORMAL_EVAL_ITEMS["dapo"],
        "aime_limit": FORMAL_EVAL_ITEMS["aime"],
        "max_turns": 50,
        "max_tokens": 4096,
        "vllm_default_max_tokens": 4096,
        "trim_context": False,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 40,
        "min_p": 0.0,
        "presence_penalty": 2.0,
        "repetition_penalty": 1.0,
        "parameter_scope": "full",
        "seed": 20260627,
    }
    for name, expected in expected_options.items():
        actual = getattr(args, name)
        if actual != expected:
            raise ValueError(f"formal evaluation requires {name}={expected!r}, got {actual!r}")
    effective_eval_seed = args.seed if args.eval_seed is None else args.eval_seed
    if effective_eval_seed != 20260627:
        raise ValueError(
            f"formal evaluation requires eval_seed=20260627, got {effective_eval_seed!r}"
        )

    canonical_data = {"dapo": DEFAULT_EVAL, "aime": DEFAULT_AIME}
    configured_data = {"dapo": Path(args.eval_data), "aime": Path(args.aime_data)}
    for dataset in eval_datasets:
        if configured_data[dataset].expanduser().resolve() != canonical_data[dataset].resolve():
            raise ValueError(
                f"formal evaluation requires canonical {dataset} data {canonical_data[dataset]}, "
                f"got {configured_data[dataset]}"
            )


def validate_engine_topologies(
    topologies: list[dict[str, Any]], *, expected_engines: int
) -> None:
    if len(topologies) != expected_engines:
        raise ValueError(
            f"engine topology expected {expected_engines} actors, got {len(topologies)}"
        )
    expected_indices = list(range(expected_engines))
    actual_indices = sorted(int(row.get("engine_index", -1)) for row in topologies)
    if actual_indices != expected_indices:
        raise ValueError(
            f"engine topology indices expected {expected_indices}, got {actual_indices}"
        )

    actor_visible_devices = []
    ray_gpu_ids_seen = []
    device_uuids = []
    for row in topologies:
        visible = str(row.get("actor_cuda_visible_devices", "")).strip()
        if not visible or "," in visible:
            raise ValueError(
                "engine topology requires exactly one CUDA_VISIBLE_DEVICES entry per actor; "
                f"engine={row.get('engine_index')} visible={visible!r}"
            )
        if int(row.get("worker_torch_device_count", -1)) != 1:
            raise ValueError(
                "engine topology requires one CUDA device in each vLLM worker; "
                f"engine={row.get('engine_index')} "
                f"count={row.get('worker_torch_device_count')!r}"
            )
        ray_gpu_ids = row.get("ray_gpu_ids")
        if not isinstance(ray_gpu_ids, list) or len(ray_gpu_ids) != 1:
            raise ValueError(
                "engine topology requires exactly one Ray GPU ID per actor; "
                f"engine={row.get('engine_index')} ray_gpu_ids={ray_gpu_ids!r}"
            )
        device_uuid = str(row.get("worker_torch_device_uuid", "")).strip()
        if not device_uuid:
            raise ValueError(
                "engine topology requires a physical GPU UUID per actor; "
                f"engine={row.get('engine_index')}"
            )
        actor_visible_devices.append(visible)
        ray_gpu_ids_seen.append(str(ray_gpu_ids[0]))
        device_uuids.append(device_uuid)

    if len(set(actor_visible_devices)) != expected_engines:
        raise ValueError(
            "engine topology requires one distinct visible GPU per actor; "
            f"got {actor_visible_devices}"
        )
    if len(set(ray_gpu_ids_seen)) != expected_engines:
        raise ValueError(
            "engine topology requires one distinct Ray GPU ID per actor; "
            f"got {ray_gpu_ids_seen}"
        )
    if len(set(device_uuids)) != expected_engines:
        raise ValueError(
            "engine topology requires one distinct physical GPU UUID per actor; "
            f"got {device_uuids}"
        )


def mean_valid(scores: list[float]) -> float:
    """Average every rollout, counting request failures as zero reward."""
    scored = [max(0.0, float(score)) for score in scores]
    return sum(scored) / len(scored) if scored else 0.0


def choose_batch(tasks: list[MathTask], generation: int, batch_size: int) -> list[MathTask]:
    start = (generation * batch_size) % len(tasks)
    return [tasks[(start + offset) % len(tasks)] for offset in range(batch_size)]


def normalize_rewards(
    rewards: list[float],
    mode: str,
    *,
    ddof: int = 0,
    eps: float = 1e-8,
) -> list[float]:
    import torch

    tensor = torch.tensor(rewards, dtype=torch.float32)
    normalized_mode = str(mode or "none").strip().lower()
    if normalized_mode in {"none", "identity", "off"}:
        return tensor.tolist()
    if normalized_mode == "zscore":
        if tensor.numel() <= int(ddof):
            return torch.zeros_like(tensor).tolist()
        std = torch.std(tensor, unbiased=bool(ddof))
        return ((tensor - torch.mean(tensor)) / (std + float(eps))).tolist()
    if normalized_mode == "centered_rank":
        order = torch.argsort(torch.argsort(tensor))
        if tensor.numel() == 1:
            return [0.0]
        return (order.float() / (tensor.numel() - 1) - 0.5).tolist()
    raise ValueError(f"Unsupported reward_normalization: {mode}")


def chunks(items: list[Any], size: int) -> list[list[Any]]:
    size = max(1, int(size))
    return [items[i : i + size] for i in range(0, len(items), size)]


def token_budget_batches(request_tokens: list[int], token_budget: int) -> list[list[int]]:
    """Pack requests without letting one vLLM call overcommit live KV tokens."""
    if not request_tokens:
        return []
    budget = int(token_budget)
    if budget <= 0:
        return [list(range(len(request_tokens)))]

    batches: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    for index, tokens in enumerate(request_tokens):
        request_size = max(1, int(tokens))
        if current and current_tokens + request_size > budget:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(index)
        current_tokens += request_size
        if current_tokens >= budget:
            batches.append(current)
            current = []
            current_tokens = 0
    if current:
        batches.append(current)
    return batches


def compact_rollout_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Keep resume/accounting data while dropping large training transcripts."""
    compact = {key: value for key, value in summary.items() if key != "scores"}
    keep = {
        "key",
        "task_id",
        "row_index",
        "sample_index",
        "score",
        "prediction",
        "latency_s",
        "used_bash",
        "termination_reason",
        "answer_status",
        "context_trims",
        "max_context_tokens",
        "generated_tokens",
        "seed",
        "engine_index",
        "trajectory_tokens",
        "trace_rounds",
        "score_method",
    }
    compact["scores"] = [
        {key: value for key, value in row.items() if key in keep}
        for row in summary.get("scores", [])
    ]
    return compact


def task_to_payload(task: MathTask) -> dict[str, str]:
    return {
        "id": task.id,
        "question": task.question,
        "answer": task.answer,
        "source": task.source,
    }


def job_to_payload(job: MathRolloutJob) -> dict[str, Any]:
    return {
        "task": task_to_payload(job.task),
        "row_index": int(job.row_index),
        "sample_index": int(job.sample_index),
    }


def replay_history_updates(
    *,
    ray,
    engines: list[Any],
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    if not records:
        return
    for record in records:
        generation = int(record["generation"])
        history_sigma = record.get("sigma")
        seeds = [int(seed) for seed in record["seeds"]]
        rewards = [float(reward) for reward in record["rewards"]]
        reward_normalization = str(record.get("reward_normalization", args.reward_normalization))
        reward_normalization_ddof = int(
            record.get("reward_normalization_ddof", args.reward_normalization_ddof)
        )
        reward_normalization_eps = float(
            record.get("reward_normalization_eps", args.reward_normalization_eps)
        )
        alpha = float(record.get("alpha", args.alpha))
        weights = normalize_rewards(
            rewards,
            reward_normalization,
            ddof=reward_normalization_ddof,
            eps=reward_normalization_eps,
        )
        ray.get([engine.update_es.remote(seeds=seeds, weights=weights, alpha=alpha) for engine in engines])
        print(
            f"[resume_replay] generation={generation} sigma={history_sigma} "
            f"reward_mean={mean_valid(rewards):.4f}",
            flush=True,
        )


def summarize_rows(rows: list[dict[str, Any]], *, items: int, samples: int) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: (int(row.get("row_index", 0)), int(row.get("sample_index", 0))))
    scores = [max(0.0, float(row.get("score", 0.0))) for row in rows]
    by_task: dict[str, list[float]] = {}
    by_sample: dict[int, list[float]] = {}
    score_methods: dict[str, int] = {}
    termination_reasons: dict[str, int] = {}
    answer_statuses: dict[str, int] = {}
    for row, score in zip(rows, scores):
        by_task.setdefault(str(row.get("task_id", row.get("row_index", ""))), []).append(score)
        by_sample.setdefault(int(row.get("sample_index", -1)), []).append(score)
        method = row.get("score_method")
        if method:
            score_methods[str(method)] = score_methods.get(str(method), 0) + 1
        termination = str(row.get("termination_reason") or "unknown")
        termination_reasons[termination] = termination_reasons.get(termination, 0) + 1
        answer_status = str(row.get("answer_status") or "unknown")
        answer_statuses[answer_status] = answer_statuses.get(answer_status, 0) + 1
    max_at_n = sum(max(task_scores) for task_scores in by_task.values()) / len(by_task) if by_task else 0.0
    pass_at_n_count = sum(max(task_scores) >= 1.0 for task_scores in by_task.values())
    pass_at_n = pass_at_n_count / len(by_task) if by_task else 0.0
    average = sum(scores) / len(scores) if scores else 0.0
    return {
        "count": len(rows),
        "valid_count": len(rows),
        "error_count": sum(reason in {"context_length_exceeded", "request_error"} for reason in (
            str(row.get("termination_reason") or "") for row in rows
        )),
        "items": items,
        "samples": samples,
        "expected_count": items * samples,
        "average": average,
        "mean_score": average,
        f"max@{samples}": max_at_n,
        f"pass@{samples}": pass_at_n,
        f"pass@{samples}_count": pass_at_n_count,
        "max": max(scores) if scores else 0.0,
        "score_methods": score_methods,
        "termination_reasons": termination_reasons,
        "answer_statuses": answer_statuses,
        "by_sample": {
            str(idx): {
                "count": len(sample_scores),
                "mean_score": sum(sample_scores) / len(sample_scores) if sample_scores else -1.0,
            }
            for idx, sample_scores in sorted(by_sample.items())
        },
        "scores": rows,
    }


def write_trace_logs(
    trace_dir: Path | None,
    rows: list[dict[str, Any]],
    filename_prefix: str = "",
) -> None:
    if trace_dir is None:
        return
    trace_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        task_payload = row.get("task") or {}
        task = MathTask(
            id=str(task_payload.get("id", row.get("task_id", ""))),
            question=str(task_payload.get("question", "")),
            answer=str(task_payload.get("answer", row.get("answer", ""))),
            source=str(task_payload.get("source", "")),
        )
        outcome = "SUCCEED" if float(row.get("score", -1.0)) >= 1.0 else "FAILED"
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", task.id)
        sample_index = int(row.get("sample_index", 0))
        path = trace_dir / f"math_agent_{filename_prefix}{safe_id}_sample{sample_index:02d}_{outcome}.md"
        path.write_text(
            trace_markdown(task=task, row=row, transcript=row.get("react_steps", [])),
            encoding="utf-8",
            errors="replace",
        )
        row["trace_log"] = str(path)


class MathVllmActor:
    def __init__(
        self,
        *,
        root: str,
        math_path: str,
        model_path: str,
        tokenizer_path: str,
        dtype: str,
        trust_remote_code: bool,
        gpu_memory_utilization: float,
        max_model_len: int,
        gdn_prefill_backend: str,
        enforce_eager: bool,
        seed: int,
        default_max_tokens: int,
        skill: str,
        tool_work_root: str,
    ) -> None:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        if root not in sys.path:
            sys.path.insert(0, root)
        if math_path not in sys.path:
            sys.path.insert(0, math_path)

        from transformers import AutoTokenizer
        from vllm import LLM

        self.root = Path(root)
        self.tool_work_root = Path(tool_work_root)
        self.tool_work_root.mkdir(parents=True, exist_ok=True)
        self.skill = skill
        self.default_max_tokens = max(1, int(default_max_tokens))
        self.max_model_len = int(max_model_len)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.llm = LLM(
            model=model_path,
            tokenizer=tokenizer_path,
            tensor_parallel_size=1,
            distributed_executor_backend="mp",
            worker_extension_cls="vllm_math_es_worker.WorkerExtension",
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            gpu_memory_utilization=float(gpu_memory_utilization),
            max_model_len=int(max_model_len),
            additional_config={"gdn_prefill_backend": str(gdn_prefill_backend)},
            enforce_eager=bool(enforce_eager),
            enable_prefix_caching=False,
            disable_log_stats=True,
            seed=int(seed),
        )

    def ready(self) -> dict[str, Any]:
        import ray

        worker_topologies = self.llm.collective_rpc("topology_math_es")
        if len(worker_topologies) != 1:
            raise RuntimeError(
                f"TP=1 Math engine expected one vLLM worker, got {len(worker_topologies)}"
            )
        worker = worker_topologies[0]
        topology = {
            "pid": os.getpid(),
            "ray_gpu_ids": [str(gpu_id) for gpu_id in ray.get_gpu_ids()],
            "actor_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        }
        topology.update({f"worker_{key}": value for key, value in worker.items()})
        return topology

    def init_es(self, *, parameter_scope: str, target_modules: list[str] | None, verbose: bool) -> list[dict]:
        return self.llm.collective_rpc(
            "init_math_es",
            kwargs={
                "parameter_scope": parameter_scope,
                "target_modules": target_modules,
                "verbose": verbose,
            },
        )

    def status_es(self) -> list[dict]:
        return self.llm.collective_rpc("status_math_es")

    def apply_perturbation(self, *, seed: int, sigma: float) -> list[dict]:
        return self.llm.collective_rpc("apply_math_es", args=(int(seed), float(sigma)))

    def revert_perturbation(self, *, seed: int, sigma: float) -> list[dict]:
        return self.llm.collective_rpc("revert_math_es", args=(int(seed), float(sigma)))

    def update_es(self, *, seeds: list[int], weights: list[float], alpha: float) -> list[dict]:
        return self.llm.collective_rpc("dipu", args=(seeds, weights, float(alpha)))

    def evaluate_perturbation(
        self,
        *,
        jobs: list[dict[str, Any]],
        seed: int,
        sigma: float,
        batch_size: int,
        rollout_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        self.apply_perturbation(seed=seed, sigma=sigma)
        try:
            rows = []
            for chunk in chunks(jobs, batch_size):
                rows.extend(self.rollout_batch(jobs=chunk, seed=seed, **rollout_kwargs))
        finally:
            self.revert_perturbation(seed=seed, sigma=sigma)
        return {"seed": int(seed), "sigma": float(sigma), "rows": rows}

    def rollout_dataset(
        self,
        *,
        jobs: list[dict[str, Any]],
        batch_size: int,
        seed: int,
        rollout_kwargs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        rows = []
        done = 0
        for chunk in chunks(jobs, batch_size):
            chunk_rows = self.rollout_batch(jobs=chunk, seed=seed, **rollout_kwargs)
            rows.extend(chunk_rows)
            done += len(chunk)
            mean_score = (
                sum(float(row.get("score", 0.0)) for row in chunk_rows) / max(1, len(chunk_rows))
            )
            print(
                f"[engine_batch] done={done}/{len(jobs)} "
                f"rows={len(chunk_rows)} mean={mean_score:.4f}",
                flush=True,
            )
        return rows

    def _render_messages(self, messages: list[dict[str, Any]]) -> str:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            signature = inspect.signature(self.tokenizer.apply_chat_template)
            supports_enable_thinking = "enable_thinking" in signature.parameters or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
        except (TypeError, ValueError):
            supports_enable_thinking = False
        if supports_enable_thinking:
            kwargs["enable_thinking"] = False
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    @staticmethod
    def _output_text(output: Any) -> str:
        candidates = getattr(output, "outputs", None) or []
        if not candidates:
            return ""
        return str(getattr(candidates[0], "text", ""))

    def _sampling_params(
        self,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        presence_penalty: float,
        repetition_penalty: float,
        seed: int,
    ):
        from vllm import SamplingParams

        return SamplingParams(
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            min_p=float(min_p),
            presence_penalty=float(presence_penalty),
            repetition_penalty=float(repetition_penalty),
            max_tokens=int(max_tokens) if int(max_tokens) > 0 else self.default_max_tokens,
            seed=int(seed),
            stop=["Observation:"],
        )

    def _prompt_token_count(self, prompt: str) -> int:
        encoded = self.tokenizer(prompt, add_special_tokens=False, return_attention_mask=False)
        return len(encoded["input_ids"])

    def _fit_prompt_to_context(
        self,
        messages: list[dict[str, Any]],
        *,
        reserve_tokens: int,
        trim_context: bool,
    ) -> tuple[str | None, int, int]:
        trims = 0
        while True:
            prompt = self._render_messages(messages)
            prompt_tokens = self._prompt_token_count(prompt)
            if prompt_tokens + max(1, int(reserve_tokens)) <= self.max_model_len:
                return prompt, trims, prompt_tokens
            if not trim_context or not trim_oldest_react_exchange(messages):
                return None, trims, prompt_tokens
            trims += 1

    def rollout_batch(
        self,
        *,
        jobs: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        presence_penalty: float,
        repetition_penalty: float,
        max_turns: int,
        python_timeout: float,
        tool_observation_limit: int,
        trim_context: bool,
        rollout_token_budget: int,
        max_total_tokens: int,
        seed: int,
    ) -> list[dict[str, Any]]:
        del max_total_tokens  # The standalone Math protocol has no total trajectory cap.
        states = []
        for job in jobs:
            task_payload = job["task"]
            task = MathTask(
                id=str(task_payload["id"]),
                question=str(task_payload["question"]),
                answer=str(task_payload["answer"]),
                source=str(task_payload.get("source", "")),
            )
            row_index = int(job["row_index"])
            sample_index = int(job.get("sample_index", 0))
            seed_base = int(
                job.get(
                    "seed",
                    trajectory_seed(seed, row_index=row_index, sample_index=sample_index),
                )
            )
            tool_workdir = (
                self.tool_work_root
                / safe_workdir_name(task.id, fallback=f"row{row_index:05d}")
                / f"row{row_index:05d}_sample{sample_index:02d}_seed{seed_base}"
            )
            tool_workdir.mkdir(parents=True, exist_ok=True)
            states.append(
                {
                    "task": task,
                    "task_payload": task_payload,
                    "row_index": row_index,
                    "sample_index": sample_index,
                    "messages": math_react_messages(task, self.skill),
                    "steps": [],
                    "used_bash": False,
                    "completion": "",
                    "react_error": None,
                    "termination_reason": None,
                    "context_trims": 0,
                    "max_context_tokens": 0,
                    "generated_tokens": 0,
                    "done": False,
                    "started_at": time.time(),
                    "seed_base": seed_base,
                    "tool_workdir": tool_workdir,
                }
            )

        turn = 0
        while int(max_turns) <= 0 or turn < int(max_turns):
            active = [state for state in states if not state["done"]]
            if not active:
                break

            generation_states = []
            prompts = []
            sampling_params = []
            for state in active:
                output_tokens = int(max_tokens) if int(max_tokens) > 0 else self.default_max_tokens
                prompt, trims, prompt_tokens = self._fit_prompt_to_context(
                    state["messages"],
                    reserve_tokens=output_tokens,
                    trim_context=trim_context,
                )
                state["context_trims"] += trims
                if prompt is None:
                    state["done"] = True
                    state["termination_reason"] = "context_length_exceeded"
                    state["react_error"] = "context_length_exceeded"
                    continue
                generation_states.append(state)
                state["max_context_tokens"] = max(int(state["max_context_tokens"]), int(prompt_tokens))
                state["current_prompt_tokens"] = int(prompt_tokens)
                prompts.append(prompt)
                sampling_params.append(
                    self._sampling_params(
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        min_p=min_p,
                        presence_penalty=presence_penalty,
                        repetition_penalty=repetition_penalty,
                        seed=state["seed_base"] + turn * 97,
                    )
                )

            if not generation_states:
                continue

            current_turn = turn + 1
            turn += 1

            output_tokens = int(max_tokens) if int(max_tokens) > 0 else self.default_max_tokens
            request_tokens = [
                int(state["current_prompt_tokens"]) + output_tokens
                for state in generation_states
            ]
            request_batches = token_budget_batches(request_tokens, rollout_token_budget)
            if len(request_batches) > 1 and (current_turn == 1 or current_turn % 10 == 0):
                print(
                    f"[rollout_microbatch] turn={current_turn} requests={len(request_tokens)} "
                    f"groups={len(request_batches)} token_budget={rollout_token_budget}",
                    flush=True,
                )

            generated = []
            for request_batch in request_batches:
                batch_states = [generation_states[index] for index in request_batch]
                batch_prompts = [prompts[index] for index in request_batch]
                batch_params = [sampling_params[index] for index in request_batch]
                try:
                    outputs = self.llm.generate(batch_prompts, batch_params, use_tqdm=False)
                    generated.extend(zip(batch_states, outputs))
                    continue
                except Exception:
                    pass

                # A single invalid prompt must not invalidate every active rollout.
                for state, prompt, params in zip(batch_states, batch_prompts, batch_params):
                    try:
                        output = self.llm.generate([prompt], [params], use_tqdm=False)[0]
                        generated.append((state, output))
                    except Exception as exc:
                        state["done"] = True
                        state["termination_reason"] = "request_error"
                        state["react_error"] = f"{type(exc).__name__}: {exc}"

            for state, output in generated:
                try:
                    completion = self._output_text(output)
                    candidates = getattr(output, "outputs", None) or []
                    token_ids = getattr(candidates[0], "token_ids", None) if candidates else None
                    completion_tokens = len(token_ids) if token_ids is not None else self._prompt_token_count(completion)
                    state["generated_tokens"] += int(completion_tokens)
                    state["max_context_tokens"] = max(
                        int(state["max_context_tokens"]),
                        int(state["current_prompt_tokens"]) + int(completion_tokens),
                    )
                    state["completion"] = completion
                    state["messages"].append({"role": "assistant", "content": completion})

                    cleaned = strip_think(completion)
                    final_match = final_answer_line(cleaned) is not None
                    action = parse_react_action(cleaned)
                    if final_match and not action:
                        if state["used_bash"]:
                            state["done"] = True
                            state["termination_reason"] = "final_answer"
                            continue
                        warning = (
                            "You must call the bash Action before answering. "
                            "Use command-line Python to compute or verify the solution, then provide Final answer."
                        )
                        state["messages"].append({"role": "user", "content": react_observation_text("format_check", warning)})
                        state["steps"].append({"turn": current_turn, "assistant": completion, "observation": warning})
                        continue

                    if not action:
                        warning = (
                            'No valid action was parsed. Use exactly:\n'
                            'Action:\n{"name": "bash", "arguments": {"command": "<shell command>"}}\n'
                            'or finish after bash use with: Final answer: \\boxed{<answer>}'
                        )
                        state["messages"].append({"role": "user", "content": react_observation_text("format_check", warning)})
                        state["steps"].append({"turn": current_turn, "assistant": completion, "observation": warning})
                        continue

                    name = action["name"]
                    arguments = action["arguments"]
                    if name == "bash":
                        command = str(arguments.get("command", ""))
                        if not command.strip():
                            observation = "No shell command was provided."
                        else:
                            state["used_bash"] = True
                            observation = run_bash(
                                command,
                                state["tool_workdir"],
                                timeout=float(python_timeout),
                                limit=int(tool_observation_limit),
                            )
                    else:
                        observation = f"Unknown action '{name}'. Available action is bash."

                    state["messages"].append({"role": "user", "content": react_observation_text(name, observation)})
                    state["steps"].append(
                        {
                            "turn": current_turn,
                            "assistant": completion,
                            "action": action,
                            "observation": observation,
                        }
                    )
                except Exception as exc:
                    state["done"] = True
                    state["termination_reason"] = "request_error"
                    state["react_error"] = f"{type(exc).__name__}: {exc}"

        for state in states:
            if not state["done"]:
                state["done"] = True
                state["termination_reason"] = "max_turns"
                state["react_error"] = "max_react_turns_exceeded"

        rows = []
        for state in states:
            task = state["task"]
            completion = state["completion"]
            answer_status = "answered" if final_answer_line(strip_think(completion)) is not None else "missing_final_answer"
            termination_reason = str(state["termination_reason"] or "request_error")
            if termination_reason in {"context_length_exceeded", "request_error"}:
                score = 0.0
                score_method = termination_reason
            else:
                score, score_method = math_score(completion, task.answer)
                if not state["used_bash"]:
                    score = 0.0
                    score_method = "no_bash_tool_use"
            rows.append(
                {
                    "key": f"{task.id}:sample{int(state['sample_index']):02d}",
                    "task_id": task.id,
                    "task": state["task_payload"],
                    "row_index": int(state["row_index"]),
                    "sample_index": int(state["sample_index"]),
                    "seed": int(state["seed_base"]),
                    "score": score,
                    "answer": task.answer,
                    "prediction": extract_math_answer(completion),
                    "response": completion,
                    "completion": completion,
                    "latency_s": time.time() - float(state["started_at"]),
                    "usage": None,
                    "mode": "paper_react_cli_vllm",
                    "tool_workdir": str(state["tool_workdir"]),
                    "used_bash": bool(state["used_bash"]),
                    "termination_reason": termination_reason,
                    "answer_status": answer_status,
                    "context_trims": int(state["context_trims"]),
                    "max_context_tokens": int(state["max_context_tokens"]),
                    "generated_tokens": int(state["generated_tokens"]),
                    "react_error": state["react_error"],
                    "react_steps": state["steps"],
                    "trace_rounds": len(state["steps"]),
                    "score_method": f"math_paper_react_cli_{score_method}",
                }
            )
        rows.sort(key=lambda row: (int(row.get("row_index", 0)), int(row.get("sample_index", 0))))
        return rows


def rollout_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "max_turns": args.max_turns,
        "python_timeout": args.python_timeout,
        "tool_observation_limit": args.tool_observation_limit,
        "trim_context": args.trim_context,
        "rollout_token_budget": args.rollout_token_budget,
        "max_total_tokens": args.max_total_tokens,
    }


def eval_tasks_vllm(
    *,
    ray,
    engines: list[Any],
    tasks: list[MathTask],
    batch_size: int,
    samples: int,
    seed: int,
    trace_dir: Path | None,
    args: argparse.Namespace,
    label: str,
    checkpoint_path: Path | None = None,
    key_seeds: dict[str, int] | None = None,
) -> dict[str, Any]:
    jobs = [
        MathRolloutJob(task=task, row_index=row_index, sample_index=sample_index)
        for row_index, task in enumerate(tasks)
        for sample_index in range(samples)
    ]
    assignments = [[] for _ in engines]
    for idx, job in enumerate(jobs):
        payload = job_to_payload(job)
        if key_seeds is not None:
            key = f"{job.task.id}:sample{job.sample_index:02d}"
            payload["seed"] = key_seeds[key]
        assignments[idx % len(engines)].append(payload)

    queues = [chunks(assigned, batch_size) for assigned in assignments]
    cursors = [0 for _ in engines]
    total_jobs = len(jobs)
    rows: list[dict[str, Any]] = []
    started_at = time.time()
    refs: list[Any] = []
    meta: dict[Any, tuple[int, int, int]] = {}

    def submit_next(engine_index: int) -> None:
        if cursors[engine_index] >= len(queues[engine_index]):
            return
        chunk_index = cursors[engine_index]
        cursors[engine_index] += 1
        chunk = queues[engine_index][chunk_index]
        ref = engines[engine_index].rollout_dataset.remote(
            jobs=chunk,
            batch_size=batch_size,
            seed=seed,
            rollout_kwargs=rollout_kwargs(args),
        )
        refs.append(ref)
        meta[ref] = (engine_index, chunk_index, len(queues[engine_index]))

    for engine_index, assigned in enumerate(assignments):
        if assigned:
            submit_next(engine_index)

    while refs:
        ready, refs = ray.wait(refs, num_returns=1, timeout=args.ray_result_timeout)
        if not ready:
            raise TimeoutError(
                f"No {label} vLLM batch completed for {args.ray_result_timeout:.0f}s; "
                "aborting before a stuck engine can hold the server indefinitely."
            )
        ref = ready[0]
        engine_index, chunk_index, engine_chunks = meta.pop(ref)
        batch_rows = ray.get(ref)
        batch_scores = [float(row.get("score", 0.0)) for row in batch_rows]
        for row in batch_rows:
            row["engine_index"] = engine_index
            row["batch_size"] = batch_size
            rows.append(row)
        current = summarize_rows(rows, items=len(tasks), samples=samples)
        elapsed = time.time() - started_at
        print(
            f"[eval_batch] label={label} engine={engine_index} "
            f"chunk={chunk_index + 1}/{engine_chunks} rows={len(batch_rows)} "
            f"done={len(rows)}/{total_jobs} batch_mean={mean_valid(batch_scores):.4f} "
            f"mean={current['mean_score']:.4f} max@{samples}={current[f'max@{samples}']:.4f} "
            f"pass@{samples}_count={current[f'pass@{samples}_count']}/{current['items']} "
            f"elapsed_s={elapsed:.1f}",
            flush=True,
        )
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        submit_next(engine_index)

    write_trace_logs(trace_dir, rows)
    return summarize_rows(rows, items=len(tasks), samples=samples)


def init_ray(args: argparse.Namespace):
    import ray
    import torch

    os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
    os.environ.setdefault("RAY_DISABLE_DOCKER_CPU_WARNING", "1")
    py_path = os.pathsep.join(
        [
            str(ROOT),
            str(ROOT / "math-train-time"),
            str(ROOT / "math-train-time" / "scripts"),
            str(ROOT / "docvqa-train-time" / "scripts"),
            str(ROOT / "docvqa-train-time" / "envs"),
            os.environ.get("PYTHONPATH", ""),
        ]
    )
    worker_env = {
        "PYTHONPATH": py_path,
        "ROOT": str(ROOT),
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "PATH": os.environ.get("PATH", ""),
    }
    if os.environ.get("DOCVQA_TOOL_PREFIX"):
        worker_env["DOCVQA_TOOL_PREFIX"] = os.environ["DOCVQA_TOOL_PREFIX"]
    if not ray.is_initialized():
        num_engines = int(args.num_engines)
        ray_cpu_count = num_engines if num_engines > 0 else max(1, torch.cuda.device_count())
        ray.init(
            num_cpus=ray_cpu_count,
            ignore_reinit_error=True,
            include_dashboard=False,
            _node_ip_address=os.environ.get("RAY_NODE_IP_ADDRESS", "127.0.0.1"),
            _temp_dir=os.environ.get("RAY_TMPDIR", None),
            runtime_env={"env_vars": worker_env},
        )
    return ray


def build_engines(ray, args: argparse.Namespace, skill: str, result_root: Path):
    import torch

    gpu_count = torch.cuda.device_count()
    num_engines = int(args.num_engines if args.num_engines > 0 else max(1, gpu_count))
    Actor = ray.remote(num_cpus=1, num_gpus=float(args.gpu_fraction))(MathVllmActor)
    engines = []
    topologies = []
    for idx in range(num_engines):
        engine = Actor.remote(
            root=str(ROOT),
            math_path=str(ROOT / "math-train-time"),
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path or args.model_path,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            gdn_prefill_backend=args.gdn_prefill_backend,
            enforce_eager=args.enforce_eager,
            seed=args.seed + idx,
            default_max_tokens=args.vllm_default_max_tokens,
            skill=skill,
            tool_work_root=str(result_root / "tool_workdirs" / f"engine_{idx:02d}"),
        )
        engines.append(engine)
        topology = ray.get(engine.ready.remote())
        topology["engine_index"] = idx
        topologies.append(topology)
        print(
            f"[vllm_engine_ready] index={idx + 1}/{num_engines} "
            f"topology={json.dumps(topology, sort_keys=True)}",
            flush=True,
        )

    if args.formal_eval:
        validate_engine_topologies(topologies, expected_engines=num_engines)
    topology_path = result_root / "engine_topology.json"
    topology_path.write_text(
        json.dumps(topologies, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[vllm_engine_topology] path={topology_path}", flush=True)

    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()] or None
    init_refs = [
        engine.init_es.remote(
            parameter_scope=args.parameter_scope,
            target_modules=target_modules,
            verbose=True,
        )
        for engine in engines
    ]
    init_results = ray.get(init_refs)
    print(f"[vllm_es_init] engines={len(engines)} result0={init_results[0][0]}", flush=True)
    return engines


def eval_population_sample(
    *,
    ray,
    engine,
    engine_index: int,
    seed: int,
    sigma: float,
    jobs: list[dict[str, Any]],
    args: argparse.Namespace,
):
    ref = engine.evaluate_perturbation.remote(
        jobs=jobs,
        seed=seed,
        sigma=sigma,
        batch_size=args.inference_batch_size,
        rollout_kwargs=rollout_kwargs(args),
    )
    return ref, engine_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID", "math_reasoning_es_vllm"))
    parser.add_argument("--train-data", default=str(DEFAULT_TRAIN))
    parser.add_argument("--eval-data", default=str(DEFAULT_EVAL))
    parser.add_argument("--aime-data", default=str(DEFAULT_AIME))
    parser.add_argument("--skill-file", default=os.environ.get("MATH_SKILL_FILE", ""))
    parser.add_argument("--model-path", default=os.environ.get("MATH_MODEL_PATH", "Qwen/Qwen3.5-4B"))
    parser.add_argument("--tokenizer-path", default=os.environ.get("MATH_TOKENIZER_PATH", ""))
    parser.add_argument("--model", default=os.environ.get("MATH_MODEL_NAME", "Qwen3.5-4B"))
    parser.add_argument("--num-engines", type=int, default=int(os.environ.get("MATH_VLLM_NUM_ENGINES", "4")))
    parser.add_argument("--gpu-fraction", type=float, default=float(os.environ.get("MATH_VLLM_GPU_FRACTION", "1.0")))
    parser.add_argument("--dtype", default=os.environ.get("MATH_VLLM_DTYPE", "bfloat16"))
    parser.add_argument("--gpu-memory-utilization", type=float, default=float(os.environ.get("MATH_VLLM_GPU_MEMORY_UTILIZATION", "0.85")))
    parser.add_argument("--max-model-len", type=int, default=int(os.environ.get("MATH_VLLM_MAX_MODEL_LEN", "32768")))
    parser.add_argument("--gdn-prefill-backend", default=os.environ.get("MATH_VLLM_GDN_PREFILL_BACKEND", "triton"), choices=["flashinfer", "triton"])
    parser.add_argument("--vllm-default-max-tokens", type=int, default=int(os.environ.get("MATH_VLLM_DEFAULT_MAX_TOKENS", "4096")))
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--generations", type=int, default=int(os.environ.get("MATH_ES_GENERATIONS", "1")))
    parser.add_argument("--population", type=int, default=int(os.environ.get("MATH_ES_POPULATION", "8")))
    parser.add_argument("--case-batch-size", type=int, default=int(os.environ.get("MATH_ES_CASE_BATCH", "8")))
    parser.add_argument("--inference-batch-size", type=int, default=int(os.environ.get("MATH_INFERENCE_BATCH_SIZE", "16")))
    parser.add_argument(
        "--rollout-token-budget",
        type=int,
        default=int(os.environ.get("MATH_ROLLOUT_TOKEN_BUDGET", "0")),
        help="Maximum summed prompt+output tokens per vLLM generate call; 0 disables microbatching.",
    )
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=int(os.environ.get("MATH_MAX_TOTAL_TOKENS", "0")),
        help="Maximum accumulated generated+observation tokens per trajectory; 0 disables the cap.",
    )
    parser.add_argument(
        "--ray-result-timeout",
        type=float,
        default=float(os.environ.get("MATH_RAY_RESULT_TIMEOUT", "3600")),
        help="Abort after this many seconds without a completed Ray rollout batch.",
    )
    parser.add_argument("--train-samples", type=int, default=int(os.environ.get("MATH_TRAIN_SAMPLES", "1")))
    parser.add_argument("--eval-samples", type=int, default=int(os.environ.get("MATH_EVAL_SAMPLES", "16")))
    parser.add_argument("--max-react-rounds", "--max-turns", dest="max_turns", type=int, default=int(os.environ.get("MATH_MAX_REACT_ROUNDS", os.environ.get("MATH_MAX_TURNS", "0"))))
    parser.add_argument("--python-timeout", type=float, default=float(os.environ.get("MATH_PYTHON_TIMEOUT", "20.0")))
    parser.add_argument("--tool-observation-limit", type=int, default=int(os.environ.get("MATH_TOOL_OBSERVATION_LIMIT", "6000")))
    parser.add_argument(
        "--trim-context",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("MATH_TRIM_CONTEXT", "1").lower() in {"1", "true", "yes"},
        help="Trim oldest ReAct exchanges to stay in context. Disable to cap the complete trajectory.",
    )
    parser.add_argument("--write-trace-logs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=int(os.environ.get("MATH_EVAL_LIMIT", str(FORMAL_EVAL_ITEMS["dapo"]))),
    )
    parser.add_argument(
        "--aime-limit",
        type=int,
        default=int(os.environ.get("MATH_AIME_LIMIT", str(FORMAL_EVAL_ITEMS["aime"]))),
    )
    parser.add_argument(
        "--eval-datasets",
        default=os.environ.get("MATH_EVAL_DATASETS", "dapo,aime"),
        help="Comma-separated initial-evaluation datasets: dapo, aime, or both.",
    )
    parser.add_argument(
        "--formal-eval",
        action="store_true",
        help="Require the exact Stage 2 four-sample dataset matrix and per-row audit fields.",
    )
    parser.add_argument(
        "--eval-key-seed-file",
        default=os.environ.get("MATH_EVAL_KEY_SEED_FILE", ""),
        help="Frozen JSONL mapping from trajectory key to generation seed.",
    )
    parser.add_argument("--sigma-start", type=float, default=float(os.environ.get("MATH_ES_SIGMA_START", "5e-4")))
    parser.add_argument("--sigma-end", type=float, default=float(os.environ.get("MATH_ES_SIGMA_END", os.environ.get("MATH_ES_SIGMA_START", "5e-4"))))
    parser.add_argument(
        "--sigma-schedule",
        default=os.environ.get("MATH_ES_SIGMA_SCHEDULE", "constant"),
        choices=["constant", "linear", "cosine"],
    )
    parser.add_argument("--sigma-warmup-steps", type=int, default=int(os.environ.get("MATH_ES_SIGMA_WARMUP_STEPS", "0")))
    parser.add_argument("--alpha", type=float, default=float(os.environ.get("MATH_ES_ALPHA", "5e-4")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("MATH_ES_SEED", "20260627")))
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help="Optional rollout seed for initial DAPO/AIME evaluation; --seed remains the ES replay seed.",
    )
    parser.add_argument("--parameter-scope", default=os.environ.get("MATH_ES_SCOPE", "full"))
    parser.add_argument("--target-modules", default=os.environ.get("MATH_ES_TARGET_MODULES", ""))
    parser.add_argument("--reward-normalization", default=os.environ.get("MATH_ES_REWARD_NORMALIZATION", "zscore"))
    parser.add_argument("--reward-normalization-ddof", type=int, default=int(os.environ.get("MATH_ES_REWARD_NORMALIZATION_DDOF", "0")))
    parser.add_argument("--reward-normalization-eps", type=float, default=float(os.environ.get("MATH_ES_REWARD_NORMALIZATION_EPS", "1e-8")))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("MATH_MAX_TOKENS", "0")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("MATH_TEMPERATURE", "1.0")))
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("MATH_TOP_P", "1.0")))
    parser.add_argument("--top-k", type=int, default=int(os.environ.get("MATH_TOP_K", "40")))
    parser.add_argument("--min-p", type=float, default=float(os.environ.get("MATH_MIN_P", "0.0")))
    parser.add_argument("--presence-penalty", type=float, default=float(os.environ.get("MATH_PRESENCE_PENALTY", "2.0")))
    parser.add_argument("--repetition-penalty", type=float, default=float(os.environ.get("MATH_REPETITION_PENALTY", "1.0")))
    parser.add_argument("--skip-initial-eval", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-interval", type=int, default=int(os.environ.get("MATH_ES_EVAL_INTERVAL", "1")))
    parser.add_argument("--skip-final-interval-eval", action=argparse.BooleanOptionalAction, default=os.environ.get("MATH_ES_SKIP_FINAL_INTERVAL_EVAL", "0").lower() in {"1", "true", "yes"})
    parser.add_argument("--final-eval", action=argparse.BooleanOptionalAction, default=os.environ.get("MATH_FINAL_EVAL", "0").lower() in {"1", "true", "yes"})
    parser.add_argument("--final-eval-samples", type=int, default=int(os.environ.get("MATH_FINAL_EVAL_SAMPLES", "4")))
    parser.add_argument("--final-eval-max-turns", type=int, default=int(os.environ.get("MATH_FINAL_EVAL_MAX_TURNS", "0")))
    parser.add_argument("--final-eval-max-tokens", type=int, default=int(os.environ.get("MATH_FINAL_EVAL_MAX_TOKENS", "0")))
    parser.add_argument("--final-eval-vllm-default-max-tokens", type=int, default=int(os.environ.get("MATH_FINAL_EVAL_VLLM_DEFAULT_MAX_TOKENS", "4096")))
    parser.add_argument("--resume-history", default=os.environ.get("MATH_ES_RESUME_HISTORY", ""))
    parser.add_argument(
        "--reuse-initial-eval-history",
        default=os.environ.get("MATH_ES_REUSE_INITIAL_EVAL_HISTORY", ""),
        help="Reuse one completed generation=-1 eval record and start training at generation 0.",
    )
    parser.add_argument(
        "--resume-generations",
        type=int,
        default=int(os.environ.get("MATH_ES_RESUME_GENERATIONS", "-1")),
        help="Replay at most this many completed updates; -1 replays all.",
    )
    parser.add_argument("--history-file", default=os.environ.get("MATH_ES_HISTORY_FILE", ""))
    args = parser.parse_args()
    validate_es_run_shape(
        generations=args.generations,
        population=args.population,
        case_batch_size=args.case_batch_size,
        allow_zero_generations=args.eval_only,
    )
    eval_datasets = parse_eval_datasets(args.eval_datasets)
    if not args.eval_only and eval_datasets != EVAL_DATASETS:
        raise ValueError("independent --eval-datasets selection is supported only with --eval-only")
    validate_formal_eval_options(args, eval_datasets)
    if args.formal_eval:
        validate_formal_eval_artifacts(args, eval_datasets)
    formal_key_seeds = (
        load_eval_key_seeds(
            args.eval_key_seed_file,
            expected_sha256=FORMAL_EVAL_KEY_SEED_SHA256,
        )
        if args.formal_eval
        else None
    )

    sigma_warmup_steps = resolve_warmup_steps(args.generations, args.sigma_warmup_steps)
    skill = ""
    if args.skill_file:
        skill_path = Path(args.skill_file)
        if skill_path.exists():
            skill = skill_path.read_text(encoding="utf-8")

    train_env = None if args.eval_only else MathReasoningEnv(args.train_data, skill_file=args.skill_file)
    eval_env = (
        MathReasoningEnv(args.eval_data, limit=args.eval_limit, skill_file=args.skill_file)
        if "dapo" in eval_datasets
        else None
    )
    aime_env = (
        MathReasoningEnv(args.aime_data, limit=args.aime_limit, skill_file=args.skill_file)
        if "aime" in eval_datasets
        else None
    )
    if formal_key_seeds is not None:
        if eval_env is not None:
            validate_eval_key_seed_coverage(
                "dapo",
                formal_key_seeds,
                tasks=eval_env.tasks,
                samples=args.eval_samples,
            )
        if aime_env is not None:
            validate_eval_key_seed_coverage(
                "aime",
                formal_key_seeds,
                tasks=aime_env.tasks,
                samples=args.eval_samples,
            )
    result_subdir = os.environ.get("MATH_ES_RESULT_SUBDIR", "runs/math_es_vllm")
    result_root = ROOT / result_subdir / args.run_id
    result_root.mkdir(parents=True, exist_ok=True)
    history_path = Path(args.history_file).expanduser().resolve() if args.history_file else result_root / "history.json"

    print(
        f"[setting] backend=vllm model_path={args.model_path} model={args.model} "
        f"num_engines={args.num_engines} batch={args.inference_batch_size} "
        f"rollout_token_budget={args.rollout_token_budget} "
        f"max_total_tokens={args.max_total_tokens} "
        f"gpu_fraction={args.gpu_fraction} "
        f"population={args.population} train_samples={args.train_samples} "
        f"eval_samples={args.eval_samples} eval_datasets={','.join(eval_datasets)} "
        f"formal_eval={args.formal_eval} max_turns={args.max_turns} "
        f"max_model_len={args.max_model_len} trim_context={args.trim_context} "
        f"max_tokens={args.max_tokens} vllm_default_max_tokens={args.vllm_default_max_tokens} "
        f"sampling=(temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}, "
        f"min_p={args.min_p}, presence_penalty={args.presence_penalty}, "
        f"repetition_penalty={args.repetition_penalty}) sigma_start={args.sigma_start} sigma_end={args.sigma_end} "
        f"sigma_schedule={args.sigma_schedule} sigma_warmup_steps={sigma_warmup_steps} "
        f"alpha={args.alpha} scope={args.parameter_scope} "
        f"reward_normalization={args.reward_normalization}",
        flush=True,
    )

    ray = init_ray(args)
    engines = build_engines(ray, args, skill, result_root)
    history = [
        {
            "config": {
                "sigma_start": args.sigma_start,
                "sigma_end": args.sigma_end,
                "sigma_schedule": args.sigma_schedule,
                "sigma_warmup_steps": sigma_warmup_steps,
                "alpha": args.alpha,
                "population": args.population,
                "case_batch_size": args.case_batch_size,
                "inference_batch_size": args.inference_batch_size,
                "rollout_token_budget": args.rollout_token_budget,
                "max_total_tokens": args.max_total_tokens,
                "ray_result_timeout": args.ray_result_timeout,
                "max_turns": args.max_turns,
                "max_model_len": args.max_model_len,
                "trim_context": args.trim_context,
                "eval_samples": args.eval_samples,
                "eval_datasets": list(eval_datasets),
                "formal_eval": args.formal_eval,
                "final_eval_samples": args.final_eval_samples,
                "seed": args.seed,
                "eval_seed": args.seed if args.eval_seed is None else args.eval_seed,
                "eval_key_seed_file": (
                    str(Path(args.eval_key_seed_file).expanduser().resolve())
                    if args.eval_key_seed_file
                    else ""
                ),
                "eval_key_seed_sha256": (
                    FORMAL_EVAL_KEY_SEED_SHA256 if args.formal_eval else ""
                ),
                "history_file": str(history_path),
                "backend": "vllm",
            }
        }
    ]
    try:
        start_generation = 0
        if args.reuse_initial_eval_history:
            if args.resume_history:
                raise ValueError(
                    "--reuse-initial-eval-history and --resume-history are mutually exclusive"
                )
            source_history = read_history(args.reuse_initial_eval_history)
            initial_eval_records = [
                record
                for record in source_history
                if isinstance(record, dict) and record.get("generation") == -1
            ]
            if len(initial_eval_records) != 1:
                raise ValueError(
                    "Expected exactly one generation=-1 record in "
                    f"{args.reuse_initial_eval_history}, found {len(initial_eval_records)}"
                )
            history.append(initial_eval_records[0])
            args.skip_initial_eval = True
            print(
                f"[initial_eval_reused] history={args.reuse_initial_eval_history}",
                flush=True,
            )
        if args.resume_history:
            source_history = read_history(args.resume_history)
            replay_limit = None if args.resume_generations < 0 else args.resume_generations
            resume_records = completed_update_records(source_history, limit=replay_limit)
            if not resume_records:
                raise ValueError(f"No completed generations found in resume history: {args.resume_history}")
            start_generation = validate_seed_sequence(
                resume_records,
                population=args.population,
                seed=args.seed,
            )
            replay_history_updates(
                ray=ray,
                engines=engines,
                records=resume_records,
                args=args,
            )
            if start_generation > args.generations:
                raise ValueError(
                    f"Resume history has {start_generation} generations, "
                    f"but --generations={args.generations}."
                )
            history = history_prefix_through_updates(source_history, len(resume_records))
            history.append(
                {
                    "resume": {
                        "source": str(Path(args.resume_history).expanduser().resolve()),
                        "replayed_generations": start_generation,
                    }
                }
            )
            atomic_write_history(history_path, history)
            print(
                f"[resume_ready] history={args.resume_history} "
                f"replayed_generations={len(resume_records)} start_generation={start_generation}",
                flush=True,
            )

        if not args.skip_initial_eval:
            eval_seed = args.seed if args.eval_seed is None else args.eval_seed
            initial_eval: dict[str, Any] = {"generation": -1}
            if eval_env is not None:
                dapo_eval = eval_tasks_vllm(
                    ray=ray,
                    engines=engines,
                    tasks=eval_env.tasks,
                    batch_size=args.inference_batch_size,
                    samples=args.eval_samples,
                    seed=eval_seed,
                    trace_dir=(result_root / "trace_logs" / "dapo_eval") if args.write_trace_logs else None,
                    args=args,
                    label="dapo_initial",
                    checkpoint_path=result_root / "partial_dapo_initial.json",
                    key_seeds=formal_key_seeds,
                )
                if args.formal_eval:
                    validate_formal_eval_summary(
                        "dapo",
                        dapo_eval,
                        tasks=eval_env.tasks,
                        samples=args.eval_samples,
                        expected_seeds=formal_key_seeds,
                    )
                    print(
                        f"[formal_eval_validated] dataset=dapo "
                        f"rows={FORMAL_EVAL_ITEMS['dapo'] * args.eval_samples}",
                        flush=True,
                    )
                initial_eval["dapo_eval"] = dapo_eval
            if aime_env is not None:
                aime_eval = eval_tasks_vllm(
                    ray=ray,
                    engines=engines,
                    tasks=aime_env.tasks,
                    batch_size=args.inference_batch_size,
                    samples=args.eval_samples,
                    seed=eval_seed,
                    trace_dir=(result_root / "trace_logs" / "aime_eval") if args.write_trace_logs else None,
                    args=args,
                    label="aime_initial",
                    checkpoint_path=result_root / "partial_aime_initial.json",
                    key_seeds=formal_key_seeds,
                )
                if args.formal_eval:
                    validate_formal_eval_summary(
                        "aime",
                        aime_eval,
                        tasks=aime_env.tasks,
                        samples=args.eval_samples,
                        expected_seeds=formal_key_seeds,
                    )
                    print(
                        f"[formal_eval_validated] dataset=aime "
                        f"rows={FORMAL_EVAL_ITEMS['aime'] * args.eval_samples}",
                        flush=True,
                    )
                initial_eval["aime_eval"] = aime_eval
            history.append(initial_eval)
            atomic_write_history(history_path, history)
        if args.eval_only:
            return

        rng = random.Random(args.seed)
        for _ in range(start_generation * args.population):
            rng.randrange(1, 2**31 - 1)
        for generation in range(start_generation, args.generations):
            sigma_t = sigma_at_step(
                sigma_start=args.sigma_start,
                sigma_end=args.sigma_end,
                step=generation,
                total_steps=args.generations,
                schedule=args.sigma_schedule,
                warmup_steps=sigma_warmup_steps,
            )
            if train_env is None:
                raise RuntimeError("train_env unexpectedly missing outside eval-only mode")
            batch = choose_batch(train_env.tasks, generation, args.case_batch_size)
            train_jobs = [
                job_to_payload(MathRolloutJob(task=task, row_index=row_index, sample_index=sample_index))
                for row_index, task in enumerate(batch)
                for sample_index in range(args.train_samples)
            ]
            seeds = [rng.randrange(1, 2**31 - 1) for _ in range(args.population)]
            print(
                f"[generation {generation}] sigma={sigma_t:.12g} "
                f"case_batch={[task.id for task in batch]}",
                flush=True,
            )

            sample_refs = []
            for idx, seed in enumerate(seeds):
                engine_index = idx % len(engines)
                ref, engine_index = eval_population_sample(
                    ray=ray,
                    engine=engines[engine_index],
                    engine_index=engine_index,
                    seed=seed,
                    sigma=sigma_t,
                    jobs=train_jobs,
                    args=args,
                )
                sample_refs.append((idx, seed, engine_index, ref))

            samples_by_idx: dict[int, dict[str, Any]] = {}
            refs = [item[3] for item in sample_refs]
            meta = {item[3]: item[:3] for item in sample_refs}
            while refs:
                ready, refs = ray.wait(refs, num_returns=1, timeout=args.ray_result_timeout)
                if not ready:
                    pending = [meta[ref] for ref in refs]
                    raise TimeoutError(
                        f"No generation {generation} population sample completed for "
                        f"{args.ray_result_timeout:.0f}s; pending={pending}"
                    )
                ref = ready[0]
                idx, seed, engine_index = meta[ref]
                payload = ray.get(ref)
                result = summarize_rows(payload["rows"], items=len(batch), samples=args.train_samples)
                if args.write_trace_logs:
                    write_trace_logs(
                        result_root / "trace_logs" / "train",
                        payload["rows"],
                        filename_prefix=f"gen{generation:03d}_candidate{idx:02d}_seed{seed}_",
                    )
                samples_by_idx[idx] = {
                    "engine_index": engine_index,
                    "seed": seed,
                    "reward": result["average"],
                    "result": compact_rollout_summary(result),
                }
                print(
                    f"[sample] gen={generation} idx={idx} engine={engine_index} "
                    f"reward={samples_by_idx[idx]['reward']}",
                    flush=True,
                )

            sample_records = [samples_by_idx[idx] for idx in range(args.population)]
            rewards = [float(row["reward"]) for row in sample_records]
            weights = normalize_rewards(
                rewards,
                args.reward_normalization,
                ddof=args.reward_normalization_ddof,
                eps=args.reward_normalization_eps,
            )
            update_results = ray.get(
                [
                    engine.update_es.remote(seeds=seeds, weights=weights, alpha=args.alpha)
                    for engine in engines
                ]
            )
            scored_rewards = [max(0.0, reward) for reward in rewards]
            record: dict[str, Any] = {
                "generation": generation,
                "case_batch": [task.id for task in batch],
                "sigma": sigma_t,
                "sigma_start": args.sigma_start,
                "sigma_end": args.sigma_end,
                "sigma_schedule": args.sigma_schedule,
                "sigma_warmup_steps": sigma_warmup_steps,
                "seeds": seeds,
                "rewards": rewards,
                "weights": weights,
                "alpha": args.alpha,
                "reward_normalization": args.reward_normalization,
                "reward_normalization_ddof": args.reward_normalization_ddof,
                "reward_normalization_eps": args.reward_normalization_eps,
                "reward_mean": mean_valid(rewards),
                "reward_std": statistics.pstdev(scored_rewards) if len(scored_rewards) > 1 else 0.0,
                "samples": sample_records,
                "updates": update_results,
            }
            final_generation = generation + 1 == args.generations
            interval_eval_due = args.eval_interval > 0 and ((generation + 1) % args.eval_interval == 0 or final_generation)
            if interval_eval_due and not (args.skip_final_interval_eval and final_generation):
                record["dapo_eval"] = eval_tasks_vllm(
                    ray=ray,
                    engines=engines,
                    tasks=eval_env.tasks,
                    batch_size=args.inference_batch_size,
                    samples=args.eval_samples,
                    seed=args.seed + generation + 1,
                    trace_dir=(result_root / "trace_logs" / f"dapo_eval_after_{generation + 1:03d}") if args.write_trace_logs else None,
                    args=args,
                    label=f"dapo_after_{generation + 1:03d}",
                    checkpoint_path=result_root / f"partial_dapo_after_{generation + 1:03d}.json",
                )
                record["aime_eval"] = eval_tasks_vllm(
                    ray=ray,
                    engines=engines,
                    tasks=aime_env.tasks,
                    batch_size=args.inference_batch_size,
                    samples=args.eval_samples,
                    seed=args.seed + generation + 1,
                    trace_dir=(result_root / "trace_logs" / f"aime_eval_after_{generation + 1:03d}") if args.write_trace_logs else None,
                    args=args,
                    label=f"aime_after_{generation + 1:03d}",
                    checkpoint_path=result_root / f"partial_aime_after_{generation + 1:03d}.json",
                )
            history.append(record)
            atomic_write_history(history_path, history)
        if args.final_eval:
            final_args = argparse.Namespace(**vars(args))
            final_args.max_turns = args.final_eval_max_turns
            final_args.max_tokens = args.final_eval_max_tokens
            final_args.vllm_default_max_tokens = args.final_eval_vllm_default_max_tokens
            final_record = {
                "generation": args.generations,
                "final_eval": True,
                "eval_samples": args.final_eval_samples,
                "max_turns": final_args.max_turns,
                "max_tokens": final_args.max_tokens,
                "vllm_default_max_tokens": final_args.vllm_default_max_tokens,
                "dapo_eval": eval_tasks_vllm(
                    ray=ray,
                    engines=engines,
                    tasks=eval_env.tasks,
                    batch_size=args.inference_batch_size,
                    samples=args.final_eval_samples,
                    seed=args.seed + args.generations + 10_000,
                    trace_dir=(result_root / "trace_logs" / "dapo_final_eval") if args.write_trace_logs else None,
                    args=final_args,
                    label="dapo_final",
                    checkpoint_path=result_root / "partial_dapo_final.json",
                ),
                "aime_eval": eval_tasks_vllm(
                    ray=ray,
                    engines=engines,
                    tasks=aime_env.tasks,
                    batch_size=args.inference_batch_size,
                    samples=args.final_eval_samples,
                    seed=args.seed + args.generations + 20_000,
                    trace_dir=(result_root / "trace_logs" / "aime_final_eval") if args.write_trace_logs else None,
                    args=final_args,
                    label="aime_final",
                    checkpoint_path=result_root / "partial_aime_final.json",
                ),
            }
            history.append(final_record)
            atomic_write_history(history_path, history)
    finally:
        for engine in engines:
            try:
                ray.kill(engine)
            except Exception:
                pass


if __name__ == "__main__":
    main()
