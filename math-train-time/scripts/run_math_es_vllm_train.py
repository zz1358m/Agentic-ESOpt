#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from run_math_es_train import (  # noqa: E402
    DEFAULT_AIME,
    DEFAULT_EVAL,
    DEFAULT_TRAIN,
    choose_batch,
    mean_valid,
)
from es.run_state import (  # noqa: E402
    atomic_write_history,
    completed_update_records,
    history_prefix_through_updates,
    read_history,
    resolve_warmup_steps,
    sigma_at_step,
    validate_es_run_shape,
    validate_seed_sequence,
)


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


def write_trace_logs(trace_dir: Path | None, rows: list[dict[str, Any]]) -> None:
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
        path = trace_dir / f"math_agent_{safe_id}_sample{sample_index:02d}_{outcome}.md"
        path.write_text(trace_markdown(task=task, row=row, transcript=row.get("react_steps", [])), encoding="utf-8")
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

    def ready(self) -> bool:
        return True

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
        return self.llm.collective_rpc("apply_perturbation", args=(int(seed), float(sigma)))

    def revert_perturbation(self, *, seed: int, sigma: float) -> list[dict]:
        return self.llm.collective_rpc("revert_perturbation", args=(int(seed), float(sigma)))

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

    def _fit_prompt_to_context(self, messages: list[dict[str, Any]], *, reserve_tokens: int) -> tuple[str | None, int]:
        trims = 0
        while True:
            prompt = self._render_messages(messages)
            if self._prompt_token_count(prompt) + max(1, int(reserve_tokens)) <= self.max_model_len:
                return prompt, trims
            if not trim_oldest_react_exchange(messages):
                return None, trims
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
        seed: int,
    ) -> list[dict[str, Any]]:
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
            seed_base = int(seed) + sample_index * 1_000_003 + row_index
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
                prompt, trims = self._fit_prompt_to_context(state["messages"], reserve_tokens=output_tokens)
                state["context_trims"] += trims
                if prompt is None:
                    state["done"] = True
                    state["termination_reason"] = "context_length_exceeded"
                    state["react_error"] = "context_length_exceeded"
                    continue
                generation_states.append(state)
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

            generated = []
            try:
                outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
                generated = list(zip(generation_states, outputs))
            except Exception:
                # A single invalid prompt must not invalidate every active rollout.
                for state, prompt, params in zip(generation_states, prompts, sampling_params):
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
) -> dict[str, Any]:
    jobs = [
        MathRolloutJob(task=task, row_index=row_index, sample_index=sample_index)
        for row_index, task in enumerate(tasks)
        for sample_index in range(samples)
    ]
    assignments = [[] for _ in engines]
    for idx, job in enumerate(jobs):
        assignments[idx % len(engines)].append(job_to_payload(job))

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
        ready, refs = ray.wait(refs, num_returns=1)
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

    os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
    os.environ.setdefault("RAY_DISABLE_DOCKER_CPU_WARNING", "1")
    py_path = os.pathsep.join(
        [
            str(ROOT),
            str(ROOT / "math-train-time"),
            os.environ.get("PYTHONPATH", ""),
        ]
    )
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            _node_ip_address=os.environ.get("RAY_NODE_IP_ADDRESS", "127.0.0.1"),
            _temp_dir=os.environ.get("RAY_TMPDIR", None),
            runtime_env={
                "env_vars": {
                    "PYTHONPATH": py_path,
                    "ROOT": str(ROOT),
                    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
                }
            },
        )
    return ray


def build_engines(ray, args: argparse.Namespace, skill: str, result_root: Path):
    import torch

    gpu_count = torch.cuda.device_count()
    num_engines = int(args.num_engines if args.num_engines > 0 else max(1, gpu_count))
    Actor = ray.remote(num_cpus=1, num_gpus=float(args.gpu_fraction))(MathVllmActor)
    engines = []
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
        ray.get(engine.ready.remote())
        print(f"[vllm_engine_ready] index={idx + 1}/{num_engines}", flush=True)

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
    parser.add_argument("--train-samples", type=int, default=int(os.environ.get("MATH_TRAIN_SAMPLES", "1")))
    parser.add_argument("--eval-samples", type=int, default=int(os.environ.get("MATH_EVAL_SAMPLES", "16")))
    parser.add_argument("--max-react-rounds", "--max-turns", dest="max_turns", type=int, default=int(os.environ.get("MATH_MAX_REACT_ROUNDS", os.environ.get("MATH_MAX_TURNS", "0"))))
    parser.add_argument("--python-timeout", type=float, default=float(os.environ.get("MATH_PYTHON_TIMEOUT", "20.0")))
    parser.add_argument("--tool-observation-limit", type=int, default=int(os.environ.get("MATH_TOOL_OBSERVATION_LIMIT", "6000")))
    parser.add_argument("--write-trace-logs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-limit", type=int, default=int(os.environ.get("MATH_EVAL_LIMIT", "100")))
    parser.add_argument("--aime-limit", type=int, default=int(os.environ.get("MATH_AIME_LIMIT", "30")))
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

    sigma_warmup_steps = resolve_warmup_steps(args.generations, args.sigma_warmup_steps)
    skill = ""
    if args.skill_file:
        skill_path = Path(args.skill_file)
        if skill_path.exists():
            skill = skill_path.read_text(encoding="utf-8")

    train_env = None if args.eval_only else MathReasoningEnv(args.train_data, skill_file=args.skill_file)
    eval_env = MathReasoningEnv(args.eval_data, limit=args.eval_limit, skill_file=args.skill_file)
    aime_env = MathReasoningEnv(args.aime_data, limit=args.aime_limit, skill_file=args.skill_file)
    result_root = ROOT / "runs/math_es_vllm" / args.run_id
    result_root.mkdir(parents=True, exist_ok=True)
    history_path = Path(args.history_file).expanduser().resolve() if args.history_file else result_root / "history.json"

    print(
        f"[setting] backend=vllm model_path={args.model_path} model={args.model} "
        f"num_engines={args.num_engines} batch={args.inference_batch_size} "
        f"gpu_fraction={args.gpu_fraction} "
        f"population={args.population} train_samples={args.train_samples} "
        f"eval_samples={args.eval_samples} max_turns={args.max_turns} "
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
                "seed": args.seed,
                "history_file": str(history_path),
                "backend": "vllm",
            }
        }
    ]
    try:
        start_generation = 0
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
            dapo_eval = eval_tasks_vllm(
                ray=ray,
                engines=engines,
                tasks=eval_env.tasks,
                batch_size=args.inference_batch_size,
                samples=args.eval_samples,
                seed=args.seed,
                trace_dir=(result_root / "trace_logs" / "dapo_eval") if args.write_trace_logs else None,
                args=args,
                label="dapo_initial",
                checkpoint_path=result_root / "partial_dapo_initial.json",
            )
            aime_eval = eval_tasks_vllm(
                ray=ray,
                engines=engines,
                tasks=aime_env.tasks,
                batch_size=args.inference_batch_size,
                samples=args.eval_samples,
                seed=args.seed,
                trace_dir=(result_root / "trace_logs" / "aime_eval") if args.write_trace_logs else None,
                args=args,
                label="aime_initial",
                checkpoint_path=result_root / "partial_aime_initial.json",
            )
            history.append({"generation": -1, "dapo_eval": dapo_eval, "aime_eval": aime_eval})
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
                ready, refs = ray.wait(refs, num_returns=1)
                ref = ready[0]
                idx, seed, engine_index = meta[ref]
                payload = ray.get(ref)
                result = summarize_rows(payload["rows"], items=len(batch), samples=args.train_samples)
                samples_by_idx[idx] = {
                    "engine_index": engine_index,
                    "seed": seed,
                    "reward": result["average"],
                    "result": result,
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
