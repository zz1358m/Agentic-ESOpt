#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(os.environ.get("ROOT", Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sudoku-train-time"))

from envs.sudoku import (  # noqa: E402
    SudokuEnv,
    SudokuTask,
    apply_action,
    build_action_prompt,
    call_completion_batch,
    clone_board,
    empty_count,
    feedback_for_board,
    is_full,
    post_json,
    score_board,
)
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


DEFAULT_TRAIN = ROOT / "data/sudoku/train.jsonl"
DEFAULT_EVAL = ROOT / "data/sudoku/eval.jsonl"


def mean_valid(scores: list[float]) -> float:
    # Transport/tool failures are failed rollouts, not missing observations.
    scored = [max(0.0, float(score)) for score in scores]
    return sum(scored) / len(scored) if scored else 0.0


def choose_batch(tasks: list[SudokuTask], generation: int, batch_size: int) -> list[SudokuTask]:
    start = (generation * batch_size) % len(tasks)
    return [tasks[(start + idx) % len(tasks)] for idx in range(batch_size)]


def eval_tasks(
    *,
    env: SudokuEnv,
    endpoints: list[str],
    tasks: list[SudokuTask],
    workers: int,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    min_p: float | None,
    presence_penalty: float,
    repetition_penalty: float,
    timeout: int,
    max_turns: int,
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
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                timeout=timeout,
                max_turns=max_turns,
            )
            futures[future] = (task, endpoint)
        for future in as_completed(futures):
            task, endpoint = futures[future]
            row = future.result()
            row["endpoint"] = endpoint
            rows.append(row)
            print(
                f"[case] task={task.id} mask={task.mask_count} endpoint={endpoint} score={row.get('score')}",
                flush=True,
            )
    rows.sort(key=lambda row: str(row["task_id"]))
    scores = [float(row.get("score", -1.0)) for row in rows]
    valid = [score for score in scores if score >= 0.0]
    return {
        "count": len(rows),
        "valid_count": len(valid),
        "average": mean_valid(scores),
        "solved": sum(1 for score in valid if score >= 1.0),
        "scores": rows,
    }


def _chunked(items: list[dict], size: int) -> list[list[dict]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def eval_endpoint_batched(
    *,
    env: SudokuEnv,
    endpoint: str,
    tasks: list[SudokuTask],
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    min_p: float | None,
    presence_penalty: float,
    repetition_penalty: float,
    timeout: int,
    max_turns: int,
    endpoint_batch_size: int,
) -> list[dict]:
    states = [
        {
            "task": task,
            "board": clone_board(task.puzzle),
            "turns": [],
            "feedback": "",
            "turn_index": 0,
        }
        for task in tasks
    ]
    active = list(states)
    batch_size = max(1, endpoint_batch_size)

    while active:
        next_active = []
        for batch in _chunked(active, batch_size):
            prompts = [
                build_action_prompt(
                    state["task"],
                    state["board"],
                    turn_index=int(state["turn_index"]),
                    feedback=str(state["feedback"]),
                )
                for state in batch
            ]
            try:
                responses = call_completion_batch(
                    endpoint,
                    prompts,
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
            except Exception as exc:
                responses = ["" for _ in batch]
                for state in batch:
                    state["turns"].append(
                        {
                            "task_id": state["task"].id,
                            "score": -1.0,
                            "mask_count": state["task"].mask_count,
                            "error": repr(exc),
                            "turn": int(state["turn_index"]),
                        }
                    )
                    state["turn_index"] = max_turns

            for state, response in zip(batch, responses):
                task = state["task"]
                if int(state["turn_index"]) >= max_turns:
                    continue
                try:
                    board, info = apply_action(task.puzzle, state["board"], response)
                    state["board"] = board
                    row = {
                        "task_id": task.id,
                        "mask_count": task.mask_count,
                        "response": response,
                        "board": board,
                        "remaining_empty": empty_count(board),
                        **info,
                    }
                except Exception as exc:
                    row = {"task_id": task.id, "score": -1.0, "mask_count": task.mask_count, "error": repr(exc)}
                row["turn"] = int(state["turn_index"])
                state["turns"].append(row)
                state["feedback"] = str(row.get("message", row.get("error", "")))
                state["turn_index"] = int(state["turn_index"]) + 1
                if not is_full(state["board"]) and int(state["turn_index"]) < max_turns:
                    next_active.append(state)
        active = next_active

    rows = []
    for state in states:
        task = state["task"]
        board = state["board"]
        rows.append(
            {
                "task_id": task.id,
                "score": score_board(task.puzzle, board),
                "mask_count": task.mask_count,
                "prediction": board,
                "filled": 81 - empty_count(board),
                "remaining_empty": empty_count(board),
                "done": is_full(board),
                "feedback": feedback_for_board(task.puzzle, board),
                "turns": state["turns"],
                "endpoint": endpoint,
            }
        )
    return rows


def eval_tasks_batched(
    *,
    env: SudokuEnv,
    endpoints: list[str],
    tasks: list[SudokuTask],
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    min_p: float | None,
    presence_penalty: float,
    repetition_penalty: float,
    timeout: int,
    max_turns: int,
    endpoint_batch_size: int,
) -> dict:
    rows = []
    endpoint_tasks = {endpoint: [] for endpoint in endpoints}
    for idx, task in enumerate(tasks):
        endpoint_tasks[endpoints[idx % len(endpoints)]].append(task)

    with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
        futures = {
            pool.submit(
                eval_endpoint_batched,
                env=env,
                endpoint=endpoint,
                tasks=assigned,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                timeout=timeout,
                max_turns=max_turns,
                endpoint_batch_size=endpoint_batch_size,
            ): endpoint
            for endpoint, assigned in endpoint_tasks.items()
            if assigned
        }
        for future in as_completed(futures):
            for row in future.result():
                rows.append(row)
                print(
                    f"[case] task={row['task_id']} mask={row['mask_count']} endpoint={row['endpoint']} score={row.get('score')}",
                    flush=True,
                )

    rows.sort(key=lambda row: str(row["task_id"]))
    scores = [float(row.get("score", -1.0)) for row in rows]
    valid = [score for score in scores if score >= 0.0]
    return {
        "count": len(rows),
        "valid_count": len(valid),
        "average": mean_valid(scores),
        "solved": sum(1 for score in valid if score >= 1.0),
        "scores": rows,
    }


def summarize_eval_runs(runs: list[dict]) -> dict:
    averages = [float(run["average"]) for run in runs]
    solved = [float(run["solved"]) for run in runs]
    return {
        "repeat_count": len(runs),
        "average": sum(averages) / max(1, len(averages)),
        "average_std": statistics.pstdev(averages) if len(averages) > 1 else 0.0,
        "solved_average": sum(solved) / max(1, len(solved)),
        "solved_std": statistics.pstdev(solved) if len(solved) > 1 else 0.0,
        "count": runs[0]["count"] if runs else 0,
        "runs": runs,
    }


def run_eval_once(
    *,
    args: argparse.Namespace,
    env: SudokuEnv,
    endpoints: list[str],
    top_k: int | None,
    min_p: float | None,
) -> dict:
    if args.batched_eval:
        return eval_tasks_batched(
            env=env,
            endpoints=endpoints,
            tasks=env.tasks,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=args.presence_penalty,
            repetition_penalty=args.repetition_penalty,
            timeout=args.timeout,
            max_turns=args.max_turns,
            endpoint_batch_size=args.endpoint_batch_size,
        )
    return eval_tasks(
        env=env,
        endpoints=endpoints,
        tasks=env.tasks,
        workers=args.case_workers * len(endpoints),
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=top_k,
        min_p=min_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        timeout=args.timeout,
        max_turns=args.max_turns,
    )


def run_eval_repeated(
    *,
    args: argparse.Namespace,
    env: SudokuEnv,
    endpoints: list[str],
    top_k: int | None,
    min_p: float | None,
) -> dict:
    runs = []
    for repeat_idx in range(max(1, args.eval_repeats)):
        row = run_eval_once(args=args, env=env, endpoints=endpoints, top_k=top_k, min_p=min_p)
        row["repeat"] = repeat_idx
        runs.append(row)
    return summarize_eval_runs(runs)


def eval_sample(
    *,
    endpoint: str,
    seed: int,
    sigma: float,
    env: SudokuEnv,
    tasks: list[SudokuTask],
    workers: int,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    min_p: float | None,
    presence_penalty: float,
    repetition_penalty: float,
    timeout: int,
    max_turns: int,
    batched_eval: bool,
    endpoint_batch_size: int,
) -> dict:
    post_json(f"{endpoint}/es/apply", {"seed": seed, "sigma": sigma}, timeout=timeout)
    try:
        if batched_eval:
            result = eval_tasks_batched(
                env=env,
                endpoints=[endpoint],
                tasks=tasks,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                timeout=timeout,
                max_turns=max_turns,
                endpoint_batch_size=endpoint_batch_size,
            )
        else:
            result = eval_tasks(
                env=env,
                endpoints=[endpoint],
                tasks=tasks,
                workers=workers,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                timeout=timeout,
                max_turns=max_turns,
            )
    finally:
        post_json(f"{endpoint}/es/revert", {"seed": seed, "sigma": sigma}, timeout=timeout)
    return {"endpoint": endpoint, "seed": seed, "reward": result["average"], "result": result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoints", default=os.environ.get("SUDOKU_ES_ENDPOINTS", "http://127.0.0.1:11013"))
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID", "sudoku_es"))
    parser.add_argument("--train-data", default=str(DEFAULT_TRAIN))
    parser.add_argument("--eval-data", default=str(DEFAULT_EVAL))
    parser.add_argument("--mask-count", type=int, default=int(os.environ.get("SUDOKU_TARGET_MASK_COUNT", "15")))
    parser.add_argument("--generations", type=int, default=int(os.environ.get("SUDOKU_ES_GENERATIONS", "1")))
    parser.add_argument("--population", type=int, default=int(os.environ.get("SUDOKU_ES_POPULATION", "8")))
    parser.add_argument("--case-batch-size", type=int, default=int(os.environ.get("SUDOKU_ES_CASE_BATCH", "8")))
    parser.add_argument("--case-workers", type=int, default=int(os.environ.get("SUDOKU_ES_CASE_WORKERS", "4")))
    parser.add_argument("--eval-limit", type=int, default=int(os.environ.get("SUDOKU_EVAL_LIMIT", "100")))
    parser.add_argument(
        "--sigma-start",
        type=float,
        default=float(os.environ.get("SUDOKU_ES_SIGMA_START", "5e-4")),
        help="Perturbation scale at the first ES generation.",
    )
    parser.add_argument(
        "--sigma-end",
        type=float,
        default=float(os.environ.get("SUDOKU_ES_SIGMA_END", os.environ.get("SUDOKU_ES_SIGMA_START", "5e-4"))),
        help="Perturbation scale at the final ES generation.",
    )
    parser.add_argument(
        "--sigma-schedule",
        default=os.environ.get("SUDOKU_ES_SIGMA_SCHEDULE", "constant"),
        choices=["constant", "linear", "cosine"],
    )
    parser.add_argument(
        "--sigma-warmup-steps",
        type=int,
        default=int(os.environ.get("SUDOKU_ES_SIGMA_WARMUP_STEPS", "0")),
        help="Number of initial generations to keep sigma fixed.",
    )
    parser.add_argument("--alpha", type=float, default=float(os.environ.get("SUDOKU_ES_ALPHA", "5e-4")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SUDOKU_ES_SEED", "20260701")))
    parser.add_argument("--parameter-scope", default=os.environ.get("SUDOKU_ES_SCOPE", "full"))
    parser.add_argument("--reward-normalization", default=os.environ.get("SUDOKU_ES_REWARD_NORMALIZATION", "zscore"))
    parser.add_argument("--model", default=os.environ.get("SUDOKU_MODEL_NAME", "Llama-3.1-8B-Instruct"))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("SUDOKU_MAX_TOKENS", "64")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("SUDOKU_TEMPERATURE", "0.0")))
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("SUDOKU_TOP_P", "0.9")))
    parser.add_argument("--top-k", type=int, default=int(os.environ.get("SUDOKU_TOP_K", "0")))
    parser.add_argument("--min-p", type=float, default=float(os.environ.get("SUDOKU_MIN_P", "-1")))
    parser.add_argument("--presence-penalty", type=float, default=float(os.environ.get("SUDOKU_PRESENCE_PENALTY", "0.0")))
    parser.add_argument("--repetition-penalty", type=float, default=float(os.environ.get("SUDOKU_REPETITION_PENALTY", "1.0")))
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("SUDOKU_TIMEOUT", "900")))
    parser.add_argument("--max-turns", type=int, default=int(os.environ.get("SUDOKU_MAX_TURNS", "0")))
    parser.add_argument("--batched-eval", action=argparse.BooleanOptionalAction, default=os.environ.get("SUDOKU_BATCHED_EVAL", "0") == "1")
    parser.add_argument("--endpoint-batch-size", type=int, default=int(os.environ.get("SUDOKU_ENDPOINT_BATCH_SIZE", "8")))
    parser.add_argument("--skip-initial-eval", action="store_true")
    parser.add_argument("--eval-interval", type=int, default=int(os.environ.get("SUDOKU_ES_EVAL_INTERVAL", "10")))
    parser.add_argument("--eval-repeats", type=int, default=int(os.environ.get("SUDOKU_EVAL_REPEATS", "3")))
    parser.add_argument("--history-file", default=os.environ.get("SUDOKU_ES_HISTORY_FILE", ""))
    parser.add_argument("--resume-history", default=os.environ.get("SUDOKU_ES_RESUME_HISTORY", ""))
    parser.add_argument("--resume-generations", type=int, default=int(os.environ.get("SUDOKU_ES_RESUME_GENERATIONS", "-1")))
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate the loaded checkpoint directly without initializing or updating ES state.",
    )
    parser.add_argument(
        "--result-root",
        default="",
        help="Explicit output directory. Defaults to runs/sudoku_es/<run-id>.",
    )
    args = parser.parse_args()
    validate_es_run_shape(
        generations=args.generations,
        population=args.population,
        case_batch_size=args.case_batch_size,
        allow_zero_generations=args.eval_only,
    )
    if args.max_turns <= 0:
        args.max_turns = args.mask_count * 3
    sigma_warmup_steps = resolve_warmup_steps(args.generations, args.sigma_warmup_steps)

    endpoints = [item.strip().rstrip("/") for item in args.endpoints.split(",") if item.strip()]
    if not endpoints:
        raise ValueError("No ES endpoints provided.")
    top_k = args.top_k if args.top_k > 0 else None
    min_p = args.min_p if args.min_p >= 0 else None
    eval_env = SudokuEnv(args.eval_data, limit=args.eval_limit, mask_count=args.mask_count)
    train_env = None if args.eval_only else SudokuEnv(args.train_data, mask_count=args.mask_count)
    result_root = (
        Path(args.result_root).expanduser().resolve()
        if args.result_root
        else ROOT / "runs/sudoku_es" / args.run_id
    )
    result_root.mkdir(parents=True, exist_ok=True)
    history_path = Path(args.history_file).expanduser().resolve() if args.history_file else result_root / "history.json"

    if args.eval_only:
        history = [
            {
                "config": {
                    "mode": "checkpoint_eval_only",
                    "mask_count": args.mask_count,
                    "max_turns": args.max_turns,
                    "eval_count": len(eval_env.tasks),
                    "eval_repeats": args.eval_repeats,
                    "batched_eval": args.batched_eval,
                    "endpoint_batch_size": args.endpoint_batch_size,
                    "model": args.model,
                    "endpoints": endpoints,
                }
            }
        ]
        eval_result = run_eval_repeated(
            args=args,
            env=eval_env,
            endpoints=endpoints,
            top_k=top_k,
            min_p=min_p,
        )
        history.append({"generation": -1, "eval": eval_result})
        atomic_write_history(history_path, history)
        print(
            f"[eval] generation=-1 split=eval repeats={eval_result['repeat_count']} "
            f"solved_avg={eval_result['solved_average']:.2f}/{eval_result['count']} "
            f"average={eval_result['average']:.6f} std={eval_result['average_std']:.6f}",
            flush=True,
        )
        return

    for endpoint in endpoints:
        init = post_json(f"{endpoint}/es/init", {"parameter_scope": args.parameter_scope, "verbose": True}, timeout=args.timeout)
        print(f"[es_init] endpoint={endpoint} {init}", flush=True)

    history = [
        {
            "config": {
                "mask_count": args.mask_count,
                "max_turns": args.max_turns,
                "train_count": len(train_env.tasks),
                "eval_count": len(eval_env.tasks),
                "eval_interval": args.eval_interval,
                "eval_repeats": args.eval_repeats,
                "sigma_start": args.sigma_start,
                "sigma_end": args.sigma_end,
                "sigma_schedule": args.sigma_schedule,
                "sigma_warmup_steps": sigma_warmup_steps,
                "alpha": args.alpha,
                "population": args.population,
                "case_batch_size": args.case_batch_size,
                "reward_normalization": args.reward_normalization,
                "parameter_scope": args.parameter_scope,
            }
        }
    ]
    start_generation = 0
    if args.resume_history:
        source_history = read_history(args.resume_history)
        replay_limit = None if args.resume_generations < 0 else args.resume_generations
        resume_records = completed_update_records(source_history, limit=replay_limit)
        start_generation = validate_seed_sequence(
            resume_records,
            population=args.population,
            seed=args.seed,
        )
        replay_log = replay_http_updates(
            endpoints=endpoints,
            records=resume_records,
            post_json=post_json,
            timeout=args.timeout,
            default_alpha=args.alpha,
            default_reward_normalization=args.reward_normalization,
        )
        history = history_prefix_through_updates(source_history, len(resume_records))
        history.append(
            {
                "resume": {
                    "source": str(Path(args.resume_history).expanduser().resolve()),
                    "replayed_generations": start_generation,
                    "replay_log": replay_log,
                }
            }
        )
        print(
            f"[resume] history={args.resume_history} replayed={start_generation} "
            f"next_generation={start_generation}",
            flush=True,
        )
        if start_generation > args.generations:
            raise ValueError(
                f"Resume history has {start_generation} generations, but --generations={args.generations}."
            )
    atomic_write_history(history_path, history)
    if not args.skip_initial_eval:
        train_eval_result = run_eval_repeated(args=args, env=train_env, endpoints=endpoints, top_k=top_k, min_p=min_p)
        eval_result = run_eval_repeated(args=args, env=eval_env, endpoints=endpoints, top_k=top_k, min_p=min_p)
        history.append({"generation": -1, "train_eval": train_eval_result, "eval": eval_result})
        print(
            f"[eval] generation=-1 split=train repeats={train_eval_result['repeat_count']} "
            f"solved_avg={train_eval_result['solved_average']:.2f}/{train_eval_result['count']} "
            f"average={train_eval_result['average']:.6f} std={train_eval_result['average_std']:.6f}",
            flush=True,
        )
        print(
            f"[eval] generation=-1 split=eval repeats={eval_result['repeat_count']} "
            f"solved_avg={eval_result['solved_average']:.2f}/{eval_result['count']} "
            f"average={eval_result['average']:.6f} std={eval_result['average_std']:.6f}",
            flush=True,
        )
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
        print(f"[sigma] gen={generation} sigma={sigma_t:.12g}", flush=True)
        def eval_direction(idx: int, endpoint: str):
            return idx, eval_sample(
                    endpoint=endpoint,
                    seed=seeds[idx],
                    sigma=sigma_t,
                    env=train_env,
                    tasks=batch,
                    workers=args.case_workers,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=top_k,
                    min_p=min_p,
                    presence_penalty=args.presence_penalty,
                    repetition_penalty=args.repetition_penalty,
                    timeout=args.timeout,
                    max_turns=args.max_turns,
                    batched_eval=args.batched_eval,
                    endpoint_batch_size=args.endpoint_batch_size,
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
            "alpha": args.alpha,
            "reward_normalization": args.reward_normalization,
            "seeds": seeds,
            "rewards": rewards,
            "reward_mean": mean_valid(rewards),
            "reward_std": statistics.pstdev(scored_rewards) if len(scored_rewards) > 1 else 0.0,
            "samples": sample_records,
            "updates": updates,
        }
        if args.eval_interval > 0 and ((generation + 1) % args.eval_interval == 0 or generation + 1 == args.generations):
            record["train_eval"] = run_eval_repeated(args=args, env=train_env, endpoints=endpoints, top_k=top_k, min_p=min_p)
            record["eval"] = run_eval_repeated(args=args, env=eval_env, endpoints=endpoints, top_k=top_k, min_p=min_p)
            print(
                f"[eval] generation={generation} split=train repeats={record['train_eval']['repeat_count']} "
                f"solved_avg={record['train_eval']['solved_average']:.2f}/{record['train_eval']['count']} "
                f"average={record['train_eval']['average']:.6f} std={record['train_eval']['average_std']:.6f}",
                flush=True,
            )
            print(
                f"[eval] generation={generation} split=eval repeats={record['eval']['repeat_count']} "
                f"solved_avg={record['eval']['solved_average']:.2f}/{record['eval']['count']} "
                f"average={record['eval']['average']:.6f} std={record['eval']['average_std']:.6f}",
                flush=True,
            )
        history.append(record)
        atomic_write_history(history_path, history)


if __name__ == "__main__":
    main()
