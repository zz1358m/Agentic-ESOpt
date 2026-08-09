#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(os.environ.get("ROOT", Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "math-train-time"))

from envs.math_reasoning import MathReasoningEnv, MathRolloutJob, MathTask, post_json  # noqa: E402
from algorithms.es.run_state import (  # noqa: E402
    atomic_write_history,
    completed_update_records,
    history_prefix_through_updates,
    map_endpoint_serial,
    read_history,
    replay_http_updates,
    resolve_warmup_steps,
    sigma_at_step,
    validate_es_run_shape,
    validate_seed_sequence,
)


DEFAULT_TRAIN = ROOT / "data/trace2skill/math_reasoning/dapo_evolve.jsonl"
DEFAULT_EVAL = ROOT / "data/trace2skill/math_reasoning/dapo_test.jsonl"
DEFAULT_AIME = ROOT / "data/trace2skill/math_reasoning/aime_2026.jsonl"


def mean_valid(scores: list[float]) -> float:
    # Failures are part of the objective and must never disappear from the mean.
    scored = [max(0.0, float(score)) for score in scores]
    return sum(scored) / len(scored) if scored else 0.0


def chunks(items: list[MathTask], size: int) -> list[list[MathTask]]:
    size = max(1, int(size))
    return [items[i : i + size] for i in range(0, len(items), size)]


def choose_batch(tasks: list[MathTask], generation: int, batch_size: int) -> list[MathTask]:
    start = (generation * batch_size) % len(tasks)
    return [tasks[(start + i) % len(tasks)] for i in range(batch_size)]


def eval_tasks(
    *,
    env: MathReasoningEnv,
    endpoints: list[str],
    tasks: list[MathTask],
    workers: int,
    batch_size: int,
    samples: int,
    max_turns: int,
    python_timeout: float,
    tool_observation_limit: int,
    request_retries: int,
    seed: int,
    trace_dir: Path | None,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    presence_penalty: float,
    repetition_penalty: float,
    timeout: int,
) -> dict:
    rows = []
    jobs = [
        MathRolloutJob(task=task, row_index=row_index, sample_index=sample_index)
        for row_index, task in enumerate(tasks)
        for sample_index in range(samples)
    ]
    endpoint_jobs: dict[str, list[MathRolloutJob]] = {endpoint: [] for endpoint in endpoints}
    for idx, job in enumerate(jobs):
        endpoint_jobs[endpoints[idx % len(endpoints)]].append(job)

    def eval_endpoint(endpoint: str, assigned: list[MathRolloutJob]) -> list[dict]:
        endpoint_rows = []
        for batch in chunks(assigned, batch_size):
            batch_rows = env.rollout_batch(
                endpoint=endpoint,
                jobs=batch,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                timeout=timeout,
                request_retries=request_retries,
                max_turns=max_turns,
                python_timeout=python_timeout,
                tool_observation_limit=tool_observation_limit,
                seed=seed,
                concurrency=max(1, min(len(batch), workers)),
                trace_dir=trace_dir,
            )
            for row in batch_rows:
                row["endpoint"] = endpoint
                row["batch_size"] = len(batch)
                endpoint_rows.append(row)
                print(
                    f"[case] task={row.get('task_id')} endpoint={endpoint} "
                    f"batch={len(batch)} score={row.get('score')}",
                    flush=True,
                )
        return endpoint_rows

    max_workers = max(1, min(len(endpoints), workers, len(jobs)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(eval_endpoint, endpoint, assigned): endpoint
            for endpoint, assigned in endpoint_jobs.items()
            if assigned
        }
        for future in as_completed(futures):
            rows.extend(future.result())
    rows.sort(key=lambda row: (int(row.get("row_index", 0)), int(row.get("sample_index", 0))))
    scores = [max(0.0, float(row.get("score", 0.0))) for row in rows]
    by_task: dict[str, list[float]] = {}
    by_sample: dict[int, list[float]] = {}
    score_methods: dict[str, int] = {}
    for row in rows:
        score = max(0.0, float(row.get("score", 0.0)))
        by_task.setdefault(str(row.get("task_id", row.get("row_index", ""))), []).append(score)
        by_sample.setdefault(int(row.get("sample_index", -1)), []).append(score)
        method = row.get("score_method")
        if method:
            score_methods[str(method)] = score_methods.get(str(method), 0) + 1
    max_at_n = sum(max(task_scores) for task_scores in by_task.values()) / len(by_task) if by_task else 0.0
    return {
        "count": len(rows),
        "valid_count": len(rows),
        "items": len(tasks),
        "samples": samples,
        "expected_count": len(tasks) * samples,
        "average": mean_valid(scores),
        "mean_score": mean_valid(scores),
        f"max@{samples}": max_at_n,
        "max": max(scores) if scores else 0.0,
        "score_methods": score_methods,
        "by_sample": {
            str(idx): {
                "count": len(sample_scores),
                "mean_score": sum(sample_scores) / len(sample_scores) if sample_scores else -1.0,
            }
            for idx, sample_scores in sorted(by_sample.items())
        },
        "scores": rows,
    }


def eval_sample(
    *,
    endpoint: str,
    seed: int,
    sigma: float,
    env: MathReasoningEnv,
    tasks: list[MathTask],
    workers: int,
    batch_size: int,
    samples: int,
    max_turns: int,
    python_timeout: float,
    tool_observation_limit: int,
    request_retries: int,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    presence_penalty: float,
    repetition_penalty: float,
    timeout: int,
) -> dict:
    post_json(f"{endpoint}/es/apply", {"seed": seed, "sigma": sigma}, timeout=timeout)
    try:
        result = eval_tasks(
            env=env,
            endpoints=[endpoint],
            tasks=tasks,
            workers=workers,
            batch_size=batch_size,
            samples=samples,
            max_turns=max_turns,
            python_timeout=python_timeout,
            tool_observation_limit=tool_observation_limit,
            request_retries=request_retries,
            seed=seed,
            trace_dir=None,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            timeout=timeout,
        )
    finally:
        post_json(f"{endpoint}/es/revert", {"seed": seed, "sigma": sigma}, timeout=timeout)
    return {"endpoint": endpoint, "seed": seed, "reward": result["average"], "result": result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoints", default=os.environ.get("MATH_ES_ENDPOINTS", "http://127.0.0.1:11013"))
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID", "math_reasoning_es"))
    parser.add_argument("--train-data", default=str(DEFAULT_TRAIN))
    parser.add_argument("--eval-data", default=str(DEFAULT_EVAL))
    parser.add_argument("--aime-data", default=str(DEFAULT_AIME))
    parser.add_argument("--skill-file", default=os.environ.get("MATH_SKILL_FILE", ""))
    parser.add_argument("--generations", type=int, default=int(os.environ.get("MATH_ES_GENERATIONS", "1")))
    parser.add_argument("--population", type=int, default=int(os.environ.get("MATH_ES_POPULATION", "8")))
    parser.add_argument("--case-batch-size", type=int, default=int(os.environ.get("MATH_ES_CASE_BATCH", "8")))
    parser.add_argument("--case-workers", type=int, default=int(os.environ.get("MATH_ES_CASE_WORKERS", "4")))
    parser.add_argument("--inference-batch-size", type=int, default=int(os.environ.get("MATH_INFERENCE_BATCH_SIZE", "16")))
    parser.add_argument("--train-samples", type=int, default=int(os.environ.get("MATH_TRAIN_SAMPLES", "1")))
    parser.add_argument("--eval-samples", type=int, default=int(os.environ.get("MATH_EVAL_SAMPLES", "16")))
    parser.add_argument("--max-react-rounds", "--max-turns", dest="max_turns", type=int, default=int(os.environ.get("MATH_MAX_REACT_ROUNDS", os.environ.get("MATH_MAX_TURNS", "0"))))
    parser.add_argument("--python-timeout", type=float, default=float(os.environ.get("MATH_PYTHON_TIMEOUT", "20.0")))
    parser.add_argument("--tool-observation-limit", type=int, default=int(os.environ.get("MATH_TOOL_OBSERVATION_LIMIT", "6000")))
    parser.add_argument("--request-retries", type=int, default=int(os.environ.get("MATH_REQUEST_RETRIES", "3")))
    parser.add_argument("--write-trace-logs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-limit", type=int, default=int(os.environ.get("MATH_EVAL_LIMIT", "100")))
    parser.add_argument("--aime-limit", type=int, default=int(os.environ.get("MATH_AIME_LIMIT", "30")))
    parser.add_argument(
        "--sigma-start",
        type=float,
        default=float(os.environ.get("MATH_ES_SIGMA_START", "5e-4")),
    )
    parser.add_argument(
        "--sigma-end",
        type=float,
        default=float(os.environ.get("MATH_ES_SIGMA_END", os.environ.get("MATH_ES_SIGMA_START", "5e-4"))),
    )
    parser.add_argument(
        "--sigma-schedule",
        default=os.environ.get("MATH_ES_SIGMA_SCHEDULE", "constant"),
        choices=["constant", "linear", "cosine"],
    )
    parser.add_argument(
        "--sigma-warmup-steps",
        type=int,
        default=int(os.environ.get("MATH_ES_SIGMA_WARMUP_STEPS", "0")),
        help="Number of initial generations to keep sigma fixed.",
    )
    parser.add_argument("--alpha", type=float, default=float(os.environ.get("MATH_ES_ALPHA", "5e-4")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("MATH_ES_SEED", "20260627")))
    parser.add_argument("--parameter-scope", default=os.environ.get("MATH_ES_SCOPE", "full"))
    parser.add_argument("--reward-normalization", default=os.environ.get("MATH_ES_REWARD_NORMALIZATION", "zscore"))
    parser.add_argument("--model", default=os.environ.get("MATH_MODEL_NAME", "Qwen3.5-4B"))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("MATH_MAX_TOKENS", "0")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("MATH_TEMPERATURE", "1.0")))
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("MATH_TOP_P", "1.0")))
    parser.add_argument("--top-k", type=int, default=int(os.environ.get("MATH_TOP_K", "40")))
    parser.add_argument("--min-p", type=float, default=float(os.environ.get("MATH_MIN_P", "0.0")))
    parser.add_argument("--presence-penalty", type=float, default=float(os.environ.get("MATH_PRESENCE_PENALTY", "2.0")))
    parser.add_argument("--repetition-penalty", type=float, default=float(os.environ.get("MATH_REPETITION_PENALTY", "1.0")))
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("MATH_TIMEOUT", "1800")))
    parser.add_argument("--skip-initial-eval", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--history-file", default=os.environ.get("MATH_ES_HISTORY_FILE", ""))
    parser.add_argument("--resume-history", default=os.environ.get("MATH_ES_RESUME_HISTORY", ""))
    parser.add_argument("--resume-generations", type=int, default=int(os.environ.get("MATH_ES_RESUME_GENERATIONS", "-1")))
    args = parser.parse_args()
    validate_es_run_shape(
        generations=args.generations,
        population=args.population,
        case_batch_size=args.case_batch_size,
        allow_zero_generations=args.eval_only,
    )
    sigma_warmup_steps = resolve_warmup_steps(args.generations, args.sigma_warmup_steps)

    endpoints = [item.strip().rstrip("/") for item in args.endpoints.split(",") if item.strip()]
    if not endpoints:
        raise ValueError("No ES endpoints provided.")
    result_root = ROOT / "runs/math_es" / args.run_id
    result_root.mkdir(parents=True, exist_ok=True)
    history_path = Path(args.history_file).expanduser().resolve() if args.history_file else result_root / "history.json"
    train_env = None if args.eval_only else MathReasoningEnv(
        args.train_data,
        skill_file=args.skill_file,
        tool_work_root=result_root / "tool_workdirs" / "train",
    )
    eval_env = MathReasoningEnv(
        args.eval_data,
        limit=args.eval_limit,
        skill_file=args.skill_file,
        tool_work_root=result_root / "tool_workdirs" / "eval_dapo",
    )
    aime_env = MathReasoningEnv(
        args.aime_data,
        limit=args.aime_limit,
        skill_file=args.skill_file,
        tool_work_root=result_root / "tool_workdirs" / "eval_aime",
    )

    print(
        f"[setting] endpoints={endpoints} population={args.population} "
        f"case_batch_size={args.case_batch_size} case_workers={args.case_workers} "
        f"inference_batch_size={args.inference_batch_size} train_samples={args.train_samples} "
        f"eval_samples={args.eval_samples} max_turns={args.max_turns} "
        f"python_timeout={args.python_timeout} tool_observation_limit={args.tool_observation_limit} "
        f"request_retries={args.request_retries} sampling=(temperature={args.temperature}, "
        f"top_p={args.top_p}, top_k={args.top_k}, min_p={args.min_p}, "
        f"presence_penalty={args.presence_penalty}, repetition_penalty={args.repetition_penalty}) "
        f"sigma_start={args.sigma_start} sigma_end={args.sigma_end} sigma_schedule={args.sigma_schedule} "
        f"sigma_warmup_steps={sigma_warmup_steps} alpha={args.alpha} "
        f"parameter_scope={args.parameter_scope} reward_normalization={args.reward_normalization}",
        flush=True,
    )

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
            }
        }
    ]
    start_generation = 0
    if not args.eval_only or args.resume_history:
        for endpoint in endpoints:
            init = post_json(f"{endpoint}/es/init", {"parameter_scope": args.parameter_scope, "verbose": True}, timeout=args.timeout)
            print(f"[es_init] endpoint={endpoint} {init}", flush=True)
    if args.resume_history:
        source_history = read_history(args.resume_history)
        replay_limit = None if args.resume_generations < 0 else args.resume_generations
        resume_records = completed_update_records(source_history, limit=replay_limit)
        start_generation = validate_seed_sequence(resume_records, population=args.population, seed=args.seed)
        replay_log = replay_http_updates(
            endpoints=endpoints,
            records=resume_records,
            post_json=post_json,
            timeout=args.timeout,
            default_alpha=args.alpha,
            default_reward_normalization=args.reward_normalization,
        )
        history = history_prefix_through_updates(source_history, len(resume_records))
        history.append({"resume": {"source": str(Path(args.resume_history).expanduser().resolve()), "replayed_generations": start_generation, "replay_log": replay_log}})
        print(f"[resume] replayed={start_generation} next_generation={start_generation}", flush=True)
        if start_generation > args.generations:
            raise ValueError(
                f"Resume history has {start_generation} generations, but --generations={args.generations}."
            )
    atomic_write_history(history_path, history)
    if not args.skip_initial_eval:
        dapo_eval = eval_tasks(
            env=eval_env,
            endpoints=endpoints,
            tasks=eval_env.tasks,
            workers=args.case_workers * len(endpoints),
            batch_size=args.inference_batch_size,
            samples=args.eval_samples,
            max_turns=args.max_turns,
            python_timeout=args.python_timeout,
            tool_observation_limit=args.tool_observation_limit,
            request_retries=args.request_retries,
            seed=args.seed,
            trace_dir=(result_root / "trace_logs" / "dapo_eval") if args.write_trace_logs else None,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            presence_penalty=args.presence_penalty,
            repetition_penalty=args.repetition_penalty,
            timeout=args.timeout,
        )
        aime_eval = eval_tasks(
            env=aime_env,
            endpoints=endpoints,
            tasks=aime_env.tasks,
            workers=args.case_workers * len(endpoints),
            batch_size=args.inference_batch_size,
            samples=args.eval_samples,
            max_turns=args.max_turns,
            python_timeout=args.python_timeout,
            tool_observation_limit=args.tool_observation_limit,
            request_retries=args.request_retries,
            seed=args.seed,
            trace_dir=(result_root / "trace_logs" / "aime_eval") if args.write_trace_logs else None,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            presence_penalty=args.presence_penalty,
            repetition_penalty=args.repetition_penalty,
            timeout=args.timeout,
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
        seeds = [rng.randrange(1, 2**31 - 1) for _ in range(args.population)]
        print(
            f"[generation {generation}] sigma={sigma_t:.12g} "
            f"case_batch={[task.id for task in batch]}",
            flush=True,
        )
        def eval_direction(idx: int, endpoint: str):
            return idx, eval_sample(
                    endpoint=endpoint,
                    seed=seeds[idx],
                    sigma=sigma_t,
                    env=train_env,
                    tasks=batch,
                    workers=args.case_workers,
                    batch_size=args.inference_batch_size,
                    samples=args.train_samples,
                    max_turns=args.max_turns,
                    python_timeout=args.python_timeout,
                    tool_observation_limit=args.tool_observation_limit,
                    request_retries=args.request_retries,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    min_p=args.min_p,
                    presence_penalty=args.presence_penalty,
                    repetition_penalty=args.repetition_penalty,
                    timeout=args.timeout,
                )

        sample_records = map_endpoint_serial(
            endpoints=endpoints,
            count=args.population,
            worker=eval_direction,
        )
        for idx, sample in enumerate(sample_records):
            print(f"[sample] gen={generation} idx={idx} reward={sample['reward']}", flush=True)
        rewards = [float(row["reward"]) for row in sample_records]
        updates = []
        for endpoint in endpoints:
            updates.append(
                {
                    "endpoint": endpoint,
                    "update": post_json(
                        f"{endpoint}/es/update",
                        {
                            "seeds": seeds,
                            "rewards": rewards,
                            "alpha": args.alpha,
                            "reward_normalization": args.reward_normalization,
                        },
                        timeout=args.timeout,
                    ),
                }
            )
        record = {
            "generation": generation,
            "case_batch": [task.id for task in batch],
            "sigma": sigma_t,
            "sigma_start": args.sigma_start,
            "sigma_end": args.sigma_end,
            "sigma_schedule": args.sigma_schedule,
            "sigma_warmup_steps": sigma_warmup_steps,
            "seeds": seeds,
            "rewards": rewards,
            "alpha": args.alpha,
            "reward_normalization": args.reward_normalization,
            "reward_mean": mean_valid(rewards),
            "reward_std": statistics.pstdev([r for r in rewards if r >= 0.0]) if len([r for r in rewards if r >= 0.0]) > 1 else 0.0,
            "samples": sample_records,
            "updates": updates,
        }
        if args.eval_interval > 0 and ((generation + 1) % args.eval_interval == 0 or generation + 1 == args.generations):
            record["dapo_eval"] = eval_tasks(
                env=eval_env,
                endpoints=endpoints,
                tasks=eval_env.tasks,
                workers=args.case_workers * len(endpoints),
                batch_size=args.inference_batch_size,
                samples=args.eval_samples,
                max_turns=args.max_turns,
                python_timeout=args.python_timeout,
                tool_observation_limit=args.tool_observation_limit,
                request_retries=args.request_retries,
                seed=args.seed + generation + 1,
                trace_dir=(result_root / "trace_logs" / f"dapo_eval_after_{generation + 1:03d}") if args.write_trace_logs else None,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                presence_penalty=args.presence_penalty,
                repetition_penalty=args.repetition_penalty,
                timeout=args.timeout,
            )
            record["aime_eval"] = eval_tasks(
                env=aime_env,
                endpoints=endpoints,
                tasks=aime_env.tasks,
                workers=args.case_workers * len(endpoints),
                batch_size=args.inference_batch_size,
                samples=args.eval_samples,
                max_turns=args.max_turns,
                python_timeout=args.python_timeout,
                tool_observation_limit=args.tool_observation_limit,
                request_retries=args.request_retries,
                seed=args.seed + generation + 1,
                trace_dir=(result_root / "trace_logs" / f"aime_eval_after_{generation + 1:03d}") if args.write_trace_logs else None,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                presence_penalty=args.presence_penalty,
                repetition_penalty=args.repetition_penalty,
                timeout=args.timeout,
            )
        history.append(record)
        atomic_write_history(history_path, history)


if __name__ == "__main__":
    main()
