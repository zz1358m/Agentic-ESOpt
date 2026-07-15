#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(os.environ.get("ROOT", Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docvqa-train-time"))

from envs.docvqa import DocVQAEnv, DocVQATask, post_json  # noqa: E402
from es.run_state import (  # noqa: E402
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


DEFAULT_TRAIN = ROOT / "data/trace2skill/docvqa/evolve.jsonl"
DEFAULT_EVAL = ROOT / "data/trace2skill/docvqa/test.jsonl"


def mean_valid(scores: list[float]) -> float:
    # Endpoint failures count as zero-reward rollouts so they cannot inflate ES.
    scored = [max(0.0, float(score)) for score in scores]
    return sum(scored) / len(scored) if scored else 0.0


def choose_batch(tasks: list[DocVQATask], generation: int, batch_size: int) -> list[DocVQATask]:
    start = (generation * batch_size) % len(tasks)
    return [tasks[(start + i) % len(tasks)] for i in range(batch_size)]


def write_trace_logs(rows: list[dict], tasks: list[DocVQATask], trace_dir: Path | None) -> None:
    if trace_dir is None:
        return
    trace_dir.mkdir(parents=True, exist_ok=True)
    by_id = {str(task.id): task for task in tasks}
    for row in rows:
        task = by_id.get(str(row.get("task_id")))
        if task is None:
            continue
        score = max(0.0, float(row.get("score", 0.0)))
        outcome = "SUCCEED" if score > 0.5 else "FAILED"
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task.id))
        text = "\n".join(
            [
                f"# Chat History docvqa_{task.id}",
                "",
                f"Task ID: {task.id}",
                f"Image: {task.image}",
                f"Question: {task.question}",
                f"Expected answers: {task.answers}",
                f"Score: {score}",
                f"Outcome: {outcome}",
                "",
                "## Assistant response",
                "",
                str(row.get("response", "")),
                "",
                "---",
                "",
                "## RESULT",
                outcome,
                "",
            ]
        )
        (trace_dir / f"docvqa_agent_{safe_id}_{outcome}.md").write_text(text, encoding="utf-8")


def eval_tasks(
    *,
    env: DocVQAEnv,
    endpoints: list[str],
    tasks: list[DocVQATask],
    workers: int,
    model: str,
    endpoint_mode: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: int,
    trace_dir: Path | None = None,
) -> dict:
    rows = []
    max_workers = max(1, min(workers, len(tasks)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for idx, task in enumerate(tasks):
            endpoint = endpoints[idx % len(endpoints)]
            future = pool.submit(
                env.evaluate_task,
                endpoint=endpoint,
                task=task,
                model=model,
                endpoint_mode=endpoint_mode,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                timeout=timeout,
            )
            futures[future] = (task, endpoint)
        for future in as_completed(futures):
            task, endpoint = futures[future]
            row = future.result()
            row["endpoint"] = endpoint
            rows.append(row)
            print(f"[case] task={task.id} endpoint={endpoint} score={row.get('score')}", flush=True)
    rows.sort(key=lambda row: str(row["task_id"]))
    scores = [float(row.get("score", -1.0)) for row in rows]
    valid = [score for score in scores if score >= 0.0]
    write_trace_logs(rows, tasks, trace_dir)
    return {
        "count": len(rows),
        "valid_count": len(valid),
        "average": mean_valid(scores),
        "max": max(valid) if valid else -1.0,
        "scores": rows,
    }


def eval_sample(
    *,
    endpoint: str,
    seed: int,
    sigma: float,
    env: DocVQAEnv,
    tasks: list[DocVQATask],
    workers: int,
    model: str,
    endpoint_mode: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: int,
    trace_dir: Path | None = None,
) -> dict:
    post_json(f"{endpoint}/es/apply", {"seed": seed, "sigma": sigma}, timeout=timeout)
    try:
        result = eval_tasks(
            env=env,
            endpoints=[endpoint],
            tasks=tasks,
            workers=workers,
            model=model,
            endpoint_mode=endpoint_mode,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
            trace_dir=trace_dir,
        )
    finally:
        post_json(f"{endpoint}/es/revert", {"seed": seed, "sigma": sigma}, timeout=timeout)
    return {"endpoint": endpoint, "seed": seed, "reward": result["average"], "result": result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoints", default=os.environ.get("DOCVQA_ES_ENDPOINTS", "http://127.0.0.1:11013"))
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID", "docvqa_es"))
    parser.add_argument("--train-data", default=str(DEFAULT_TRAIN))
    parser.add_argument("--eval-data", default=str(DEFAULT_EVAL))
    parser.add_argument("--skill-file", default=os.environ.get("DOCVQA_SKILL_FILE", ""))
    parser.add_argument("--endpoint-mode", choices=["completion", "openai_chat", "openai_vision_chat"], default=os.environ.get("DOCVQA_ENDPOINT_MODE", "openai_vision_chat"))
    parser.add_argument("--generations", type=int, default=int(os.environ.get("DOCVQA_ES_GENERATIONS", "1")))
    parser.add_argument("--population", type=int, default=int(os.environ.get("DOCVQA_ES_POPULATION", "8")))
    parser.add_argument("--case-batch-size", type=int, default=int(os.environ.get("DOCVQA_ES_CASE_BATCH", "8")))
    parser.add_argument("--case-workers", type=int, default=int(os.environ.get("DOCVQA_ES_CASE_WORKERS", "4")))
    parser.add_argument("--eval-limit", type=int, default=int(os.environ.get("DOCVQA_EVAL_LIMIT", "200")))
    parser.add_argument("--sigma-start", type=float, default=float(os.environ.get("DOCVQA_ES_SIGMA_START", "5e-4")))
    parser.add_argument("--sigma-end", type=float, default=float(os.environ.get("DOCVQA_ES_SIGMA_END", os.environ.get("DOCVQA_ES_SIGMA_START", "5e-4"))))
    parser.add_argument("--sigma-schedule", default=os.environ.get("DOCVQA_ES_SIGMA_SCHEDULE", "constant"), choices=["constant", "linear", "cosine"])
    parser.add_argument("--sigma-warmup-steps", type=int, default=int(os.environ.get("DOCVQA_ES_SIGMA_WARMUP_STEPS", "0")))
    parser.add_argument("--alpha", type=float, default=float(os.environ.get("DOCVQA_ES_ALPHA", "5e-4")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("DOCVQA_ES_SEED", "20260627")))
    parser.add_argument("--parameter-scope", default=os.environ.get("DOCVQA_ES_SCOPE", "full"))
    parser.add_argument("--reward-normalization", default=os.environ.get("DOCVQA_ES_REWARD_NORMALIZATION", "zscore"))
    parser.add_argument("--model", default=os.environ.get("DOCVQA_MODEL_NAME", "Qwen3.5-27B"))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("DOCVQA_MAX_TOKENS", "512")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("DOCVQA_TEMPERATURE", "0.0")))
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("DOCVQA_TOP_P", "0.9")))
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("DOCVQA_TIMEOUT", "900")))
    parser.add_argument("--skip-initial-eval", action="store_true")
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--write-trace-logs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--history-file", default=os.environ.get("DOCVQA_ES_HISTORY_FILE", ""))
    parser.add_argument("--resume-history", default=os.environ.get("DOCVQA_ES_RESUME_HISTORY", ""))
    parser.add_argument("--resume-generations", type=int, default=int(os.environ.get("DOCVQA_ES_RESUME_GENERATIONS", "-1")))
    args = parser.parse_args()
    validate_es_run_shape(
        generations=args.generations,
        population=args.population,
        case_batch_size=args.case_batch_size,
    )
    sigma_warmup_steps = resolve_warmup_steps(args.generations, args.sigma_warmup_steps)

    endpoints = [item.strip().rstrip("/") for item in args.endpoints.split(",") if item.strip()]
    if not endpoints:
        raise ValueError("No ES endpoints provided.")
    train_env = DocVQAEnv(args.train_data, skill_file=args.skill_file)
    eval_env = DocVQAEnv(args.eval_data, limit=args.eval_limit, skill_file=args.skill_file)
    result_root = ROOT / "runs/docvqa_es" / args.run_id
    result_root.mkdir(parents=True, exist_ok=True)
    history_path = Path(args.history_file).expanduser().resolve() if args.history_file else result_root / "history.json"

    for endpoint in endpoints:
        init = post_json(f"{endpoint}/es/init", {"parameter_scope": args.parameter_scope, "verbose": True}, timeout=args.timeout)
        print(f"[es_init] endpoint={endpoint} {init}", flush=True)

    history = [{"config": {"sigma_start": args.sigma_start, "sigma_end": args.sigma_end, "sigma_schedule": args.sigma_schedule, "sigma_warmup_steps": sigma_warmup_steps, "alpha": args.alpha, "population": args.population, "seed": args.seed, "history_file": str(history_path)}}]
    start_generation = 0
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
        if start_generation > args.generations:
            raise ValueError(
                f"Resume history has {start_generation} generations, but --generations={args.generations}."
            )
        print(f"[resume] replayed={start_generation} next_generation={start_generation}", flush=True)
    atomic_write_history(history_path, history)
    if not args.skip_initial_eval and eval_env.tasks:
        eval_rec = eval_tasks(
            env=eval_env,
            endpoints=endpoints,
            tasks=eval_env.tasks,
            workers=args.case_workers * len(endpoints),
            model=args.model,
            endpoint_mode=args.endpoint_mode,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
            trace_dir=(result_root / "trace_logs" / "eval_initial") if args.write_trace_logs else None,
        )
        history.append({"generation": -1, "eval": eval_rec})
        atomic_write_history(history_path, history)

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
        batch = choose_batch(train_env.tasks, generation, args.case_batch_size)
        seeds = [rng.randrange(1, 2**31 - 1) for _ in range(args.population)]
        def eval_direction(idx: int, endpoint: str):
            return idx, eval_sample(
                    endpoint=endpoint,
                    seed=seeds[idx],
                    sigma=sigma_t,
                    env=train_env,
                    tasks=batch,
                    workers=args.case_workers,
                    model=args.model,
                    endpoint_mode=args.endpoint_mode,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    timeout=args.timeout,
                    trace_dir=(result_root / "trace_logs" / "train" / f"generation_{generation:03d}" / f"sample_{idx:03d}") if args.write_trace_logs else None,
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
        scored_rewards = [max(0.0, reward) for reward in rewards]
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
            "reward_std": statistics.pstdev(scored_rewards) if len(scored_rewards) > 1 else 0.0,
            "samples": sample_records,
            "updates": updates,
        }
        if args.eval_interval > 0 and eval_env.tasks and ((generation + 1) % args.eval_interval == 0 or generation + 1 == args.generations):
            record["eval"] = eval_tasks(
                env=eval_env,
                endpoints=endpoints,
                tasks=eval_env.tasks,
                workers=args.case_workers * len(endpoints),
                model=args.model,
                endpoint_mode=args.endpoint_mode,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout=args.timeout,
                trace_dir=(result_root / "trace_logs" / f"eval_after_{generation + 1:03d}") if args.write_trace_logs else None,
            )
        history.append(record)
        atomic_write_history(history_path, history)


if __name__ == "__main__":
    main()
