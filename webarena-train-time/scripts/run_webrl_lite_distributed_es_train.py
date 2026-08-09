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

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional runtime dependency
    SummaryWriter = None


ROOT = Path(os.environ.get("ROOT", Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_webrl_lite_full_es_train import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_SPLIT,
    load_tasks,
    post_json,
    post_json_retry,
    run_episode,
    resolve_sigma_warmup_steps,
    sigma_for_generation,
    validate_config_alignment,
)  # noqa: E402


def write_eval_scalars(writer: object | None, prefix: str, eval_rec: dict, step: int) -> None:
    if writer is None:
        return
    writer.add_scalar(f"{prefix}/average", eval_rec.get("average", -1.0), step)
    writer.add_scalar(f"{prefix}/valid_count", eval_rec.get("valid_count", 0), step)
    writer.add_scalar(f"{prefix}/count", eval_rec.get("count", 0), step)
    writer.add_scalar(f"{prefix}/max", eval_rec.get("max", -1.0), step)
    writer.flush()


def write_generation_scalars(
    writer: object | None,
    *,
    generation: int,
    rewards: list[float],
    sigma: float,
    alpha: float,
    eval_rec: dict | None,
) -> None:
    if writer is None:
        return
    valid_rewards = [reward for reward in rewards if reward >= 0.0]
    writer.add_scalar("es/sigma", sigma, generation)
    writer.add_scalar("es/alpha", alpha, generation)
    writer.add_scalar("train/valid_sample_count", len(valid_rewards), generation)
    writer.add_scalar("train/sample_count", len(rewards), generation)
    if valid_rewards:
        writer.add_scalar("train/reward_mean", sum(valid_rewards) / len(valid_rewards), generation)
        writer.add_scalar("train/reward_max", max(valid_rewards), generation)
        writer.add_scalar("train/reward_min", min(valid_rewards), generation)
        writer.add_scalar(
            "train/reward_std",
            statistics.pstdev(valid_rewards) if len(valid_rewards) > 1 else 0.0,
            generation,
        )
    else:
        writer.add_scalar("train/reward_mean", -1.0, generation)
    if eval_rec is not None:
        write_eval_scalars(writer, "eval", eval_rec, generation)
    writer.flush()


def eval_tasks_distributed(
    *,
    endpoints: list[str],
    task_ids: list[int],
    config_dir: Path,
    result_root: Path,
    run_name: str,
    skill_file: Path | None,
    workers_per_endpoint: int,
    instruction_path: str,
    model_name: str,
    mode: str,
    stop_token: str,
    temperature: float,
    top_p: float,
    top_k: int | None,
    min_p: float | None,
    presence_penalty: float,
    repetition_penalty: float,
) -> dict:
    scores_by_task: dict[int, float] = {}
    max_workers = max(1, len(endpoints) * int(workers_per_endpoint))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for i, task_id in enumerate(task_ids):
            endpoint = endpoints[i % len(endpoints)]
            future = pool.submit(
                run_episode,
                endpoint=endpoint,
                task_id=task_id,
                config_dir=config_dir,
                result_root=result_root,
                run_name=run_name,
                skill_file=skill_file,
                instruction_path=instruction_path,
                model_name=model_name,
                mode=mode,
                stop_token=stop_token,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
            )
            futures[future] = (task_id, endpoint)
        for future in as_completed(futures):
            task_id, endpoint = futures[future]
            score = future.result()
            scores_by_task[task_id] = score
            print(f"[eval] {run_name} endpoint={endpoint} task={task_id} score={score}", flush=True)

    scores = [{"task_id": task_id, "score": scores_by_task[task_id]} for task_id in task_ids]
    valid = [row["score"] for row in scores if row["score"] >= 0.0]
    return {
        "run_name": run_name,
        "count": len(scores),
        "valid_count": len(valid),
        "average": sum(valid) / len(valid) if valid else -1.0,
        "max": max(valid) if valid else -1.0,
        "scores": scores,
    }


def eval_population_sample(
    *,
    endpoint: str,
    seed: int,
    sigma: float,
    task_ids: list[int],
    config_dir: Path,
    result_root: Path,
    run_name: str,
    case_workers: int,
    instruction_path: str,
    model_name: str,
    mode: str,
    stop_token: str,
    temperature: float,
    top_p: float,
    top_k: int | None,
    min_p: float | None,
    presence_penalty: float,
    repetition_penalty: float,
) -> dict:
    post_json(f"{endpoint}/es/apply", {"seed": seed, "sigma": sigma})
    case_scores_by_task: dict[int, float] = {}
    try:
        with ThreadPoolExecutor(max_workers=max(1, int(case_workers))) as pool:
            futures = {}
            for task_id in task_ids:
                future = pool.submit(
                    run_episode,
                    endpoint=endpoint,
                    task_id=task_id,
                    config_dir=config_dir,
                    result_root=result_root,
                    run_name=run_name,
                    instruction_path=instruction_path,
                    model_name=model_name,
                    mode=mode,
                    stop_token=stop_token,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    presence_penalty=presence_penalty,
                    repetition_penalty=repetition_penalty,
                )
                futures[future] = task_id
            for future in as_completed(futures):
                task_id = futures[future]
                score = future.result()
                case_scores_by_task[task_id] = score
                print(
                    f"[case] {run_name} endpoint={endpoint} task={task_id} score={score}",
                    flush=True,
                )
    finally:
        post_json_retry(f"{endpoint}/es/revert", {"seed": seed, "sigma": sigma})
    case_scores = [case_scores_by_task[task_id] for task_id in task_ids]
    reward = sum(case_scores) / len(case_scores) if case_scores else 0.0
    return {
        "endpoint": endpoint,
        "seed": seed,
        "case_scores": case_scores,
        "reward": reward,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoints", required=True, help="Comma-separated model server base URLs.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", default="")
    parser.add_argument("--eval-split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--train-config-dir", default="")
    parser.add_argument("--sites", default="shopping,shopping_admin,reddit,gitlab,wikipedia,map")
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--case-batch-size", type=int, default=16)
    parser.add_argument("--case-workers-per-sample", type=int, default=4)
    parser.add_argument("--eval-workers-per-endpoint", type=int, default=4)
    parser.add_argument("--sigma-start", type=float, default=1e-3)
    parser.add_argument(
        "--sigma-end",
        type=float,
        default=None,
        help="Final cosine sigma. Defaults to --sigma-start.",
    )
    parser.add_argument(
        "--sigma-schedule",
        default=os.environ.get("WEBRL_ES_SIGMA_SCHEDULE", "cosine"),
        choices=["cosine"],
    )
    parser.add_argument(
        "--sigma-warmup-steps",
        type=int,
        default=int(os.environ.get("WEBRL_ES_SIGMA_WARMUP_STEPS", "-1")),
        help="Number of initial generations to keep sigma fixed. Defaults to generations // 4.",
    )
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--reward-normalization", default="zscore")
    parser.add_argument("--parameter-scope", default="full", choices=["full", "all_linear", "lora"])
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1,
        help="Run eval every N generations/ES updates. Use 0 to disable periodic eval.",
    )
    parser.add_argument("--skip-initial-eval", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--instruction-path", default="agent/prompts/jsons/p_webrl.json")
    parser.add_argument("--model-name", default="Llama-3.1-8B-Instruct")
    parser.add_argument("--mode", default="completion", choices=["completion", "chat"])
    parser.add_argument("--stop-token", default="<|eot_id|>")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument(
        "--replay-history",
        default="",
        help="Historical ES run directory or history.json used to replay recorded seed/reward updates.",
    )
    parser.add_argument(
        "--replay-generations",
        type=int,
        default=0,
        help=(
            "Number of initial generations to replay from --replay-history. "
            "Use -1 to replay every recorded ES generation."
        ),
    )
    parser.add_argument(
        "--tensorboard-dir",
        default="",
        help="TensorBoard log directory. Defaults to <result_root>/tensorboard.",
    )
    args = parser.parse_args()
    sigma_end = args.sigma_start if args.sigma_end is None else args.sigma_end
    if args.sigma_start < 0 or sigma_end < 0:
        raise ValueError("Sigma values must be non-negative.")
    if args.replay_generations < -1:
        raise ValueError("--replay-generations must be -1 or non-negative.")
    if args.replay_generations != 0 and not args.replay_history:
        raise ValueError("--replay-generations requires --replay-history.")
    sigma_warmup_steps = resolve_sigma_warmup_steps(args.generations, args.sigma_warmup_steps)

    endpoints = [item.strip().rstrip("/") for item in args.endpoints.split(",") if item.strip()]
    if not endpoints:
        raise RuntimeError("No endpoints provided.")
    result_root = ROOT / "runs/webrl_lite_full_es" / args.run_id
    result_root.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = Path(args.tensorboard_dir) if args.tensorboard_dir else result_root / "tensorboard"
    writer = SummaryWriter(str(tensorboard_dir)) if SummaryWriter is not None else None
    allowed_sites = {site.strip() for site in args.sites.split(",") if site.strip()}
    eval_task_ids = load_tasks(Path(args.eval_split), allowed_sites, args.eval_limit)
    config_dir = Path(args.config_dir)
    if not config_dir.exists():
        raise FileNotFoundError(config_dir)
    if not args.eval_only and not args.split:
        raise ValueError(
            "--split is required for ES training. "
            "Use data/webarena/vab_lite_split/items.json only for VAB/WebRL WebArena-Lite evaluation; "
            "training needs a config-backed split plus --train-config-dir."
        )
    train_task_ids = load_tasks(Path(args.split), allowed_sites, args.episodes) if args.split else []
    train_config_dir = Path(args.train_config_dir) if args.train_config_dir else None
    if train_task_ids and train_config_dir is None:
        raise ValueError("--train-config-dir is required when --split is set.")
    if train_config_dir is not None and not train_config_dir.exists():
        raise FileNotFoundError(train_config_dir)
    if train_task_ids and train_config_dir is not None:
        validate_config_alignment(Path(args.split), train_task_ids, train_config_dir)

    replay_source_root = None
    replay_by_generation: dict[int, dict] = {}
    if args.replay_history:
        replay_path = Path(args.replay_history).resolve()
        replay_history_path = replay_path if replay_path.is_file() else replay_path / "history.json"
        if not replay_history_path.is_file():
            raise FileNotFoundError(replay_history_path)
        replay_source_root = replay_history_path.parent
        replay_records = json.loads(replay_history_path.read_text(encoding="utf-8"))
        replay_by_generation = {
            int(record["generation"]): record
            for record in replay_records
            if isinstance(record.get("generation"), int) and record["generation"] >= 0
        }
        if args.replay_generations == -1:
            args.replay_generations = len(replay_by_generation)
        if args.replay_generations > args.generations:
            raise ValueError(
                "--replay-generations cannot exceed --generations: "
                f"{args.replay_generations} > {args.generations}."
            )
        missing = [
            generation
            for generation in range(args.replay_generations)
            if generation not in replay_by_generation
        ]
        if missing:
            raise ValueError(f"Replay history is missing generations: {missing}")

    print(
        f"[setting] endpoints={endpoints} population={args.population} "
        f"case_batch_size={args.case_batch_size} case_workers_per_sample={args.case_workers_per_sample} "
        f"eval_workers_per_endpoint={args.eval_workers_per_endpoint} "
        f"sigma_start={args.sigma_start} sigma_end={sigma_end} sigma_schedule={args.sigma_schedule} "
        f"sigma_warmup_steps={sigma_warmup_steps} "
        f"alpha={args.alpha} "
        f"parameter_scope={args.parameter_scope} "
        f"replay_generations={args.replay_generations} "
        f"replay_source={replay_source_root or ''} "
        f"train_config_dir={train_config_dir or ''} eval_config_dir={config_dir} "
        f"tensorboard_dir={tensorboard_dir if writer is not None else ''}",
        flush=True,
    )

    history = []
    manifest = {
        "run_id": args.run_id,
        "generations": args.generations,
        "population": args.population,
        "case_batch_size": args.case_batch_size,
        "sigma": args.sigma_start,
        "sigma_start": args.sigma_start,
        "sigma_end": sigma_end,
        "sigma_schedule": args.sigma_schedule,
        "alpha": args.alpha,
        "reward_normalization": args.reward_normalization,
        "seed": args.seed,
        "replay_generations": args.replay_generations,
        "replay_source": str(replay_source_root) if replay_source_root else "",
    }
    (result_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.eval_only:
        eval_rec = eval_tasks_distributed(
            endpoints=endpoints,
            task_ids=eval_task_ids,
            config_dir=config_dir,
            result_root=result_root,
            run_name="eval_only",
            skill_file=None,
            workers_per_endpoint=args.eval_workers_per_endpoint,
            instruction_path=args.instruction_path,
            model_name=args.model_name,
            mode=args.mode,
            stop_token=args.stop_token,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            presence_penalty=args.presence_penalty,
            repetition_penalty=args.repetition_penalty,
        )
        history.append({"generation": None, "kind": "eval_only", "eval": eval_rec})
        (result_root / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        write_eval_scalars(writer, "eval_only", eval_rec, 0)
        if writer is not None:
            writer.close()
        print(f"[eval_only] {eval_rec}", flush=True)
        return

    init_records = []
    for endpoint in endpoints:
        init = post_json(f"{endpoint}/es/init", {"parameter_scope": args.parameter_scope, "verbose": True})
        init_records.append({"endpoint": endpoint, "init": init})
    print(f"[es_init] {init_records}", flush=True)

    if not args.skip_initial_eval:
        initial_eval = eval_tasks_distributed(
            endpoints=endpoints,
            task_ids=eval_task_ids,
            config_dir=config_dir,
            result_root=result_root,
            run_name="initial_base_eval",
            skill_file=None,
            workers_per_endpoint=args.eval_workers_per_endpoint,
            instruction_path=args.instruction_path,
            model_name=args.model_name,
            mode=args.mode,
            stop_token=args.stop_token,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            presence_penalty=args.presence_penalty,
            repetition_penalty=args.repetition_penalty,
        )
        history.append({"generation": -1, "kind": "initial_base_eval", "eval": initial_eval})
        (result_root / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        write_eval_scalars(writer, "initial_eval", initial_eval, 0)
        print(f"[initial_eval] {initial_eval}", flush=True)

    rng = random.Random(args.seed)
    for generation in range(args.generations):
        sigma_t = sigma_for_generation(
            sigma_start=args.sigma_start,
            sigma_end=sigma_end,
            generation=generation,
            generations=args.generations,
            schedule=args.sigma_schedule,
            warmup_steps=sigma_warmup_steps,
        )
        batch_start = (generation * args.case_batch_size) % len(train_task_ids)
        selected = [train_task_ids[(batch_start + i) % len(train_task_ids)] for i in range(args.case_batch_size)]
        seeds = [rng.randrange(1, 2**31 - 1) for _ in range(args.population)]
        print(f"[generation {generation}] sigma={sigma_t:.12g} case_batch={selected}", flush=True)

        replayed = generation < args.replay_generations
        if replayed:
            replay_record = replay_by_generation[generation]
            if replay_record.get("case_batch") != selected:
                raise ValueError(
                    f"Replay generation {generation} case batch differs: "
                    f"{replay_record.get('case_batch')} != {selected}"
                )
            if replay_record.get("seeds") != seeds:
                raise ValueError(
                    f"Replay generation {generation} seeds differ: "
                    f"{replay_record.get('seeds')} != {seeds}"
                )
            replay_sigma = float(replay_record.get("sigma", sigma_t))
            if abs(replay_sigma - sigma_t) > 1e-12:
                raise ValueError(
                    f"Replay generation {generation} sigma differs: "
                    f"{replay_sigma} != {sigma_t}"
                )
            rewards = [float(value) for value in replay_record["rewards"]]
            if len(rewards) != args.population:
                raise ValueError(
                    f"Replay generation {generation} has {len(rewards)} rewards; "
                    f"expected {args.population}."
                )
            sample_records = replay_record.get("samples", [])
            print(
                f"[replay] gen={generation} source={replay_source_root} rewards={rewards}",
                flush=True,
            )
        else:
            samples_by_index: dict[int, dict] = {}
            with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
                endpoint_queues = {
                    endpoint: iter(range(endpoint_index, len(seeds), len(endpoints)))
                    for endpoint_index, endpoint in enumerate(endpoints)
                }
                futures: dict[object, tuple[int, str]] = {}

                def submit_next(endpoint: str) -> None:
                    try:
                        i = next(endpoint_queues[endpoint])
                    except StopIteration:
                        return
                    future = pool.submit(
                        eval_population_sample,
                        endpoint=endpoint,
                        seed=seeds[i],
                        sigma=sigma_t,
                        task_ids=selected,
                        config_dir=train_config_dir,
                        result_root=result_root,
                        run_name=f"gen_{generation:03d}_sample_{i:02d}_seed_{seeds[i]}",
                        case_workers=args.case_workers_per_sample,
                        instruction_path=args.instruction_path,
                        model_name=args.model_name,
                        mode=args.mode,
                        stop_token=args.stop_token,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        min_p=args.min_p,
                        presence_penalty=args.presence_penalty,
                        repetition_penalty=args.repetition_penalty,
                    )
                    futures[future] = (i, endpoint)

                for endpoint in endpoints:
                    submit_next(endpoint)

                while futures:
                    future = next(as_completed(tuple(futures)))
                    i, endpoint = futures.pop(future)
                    rec = future.result()
                    samples_by_index[i] = rec
                    print(
                        f"[sample] gen={generation} sample={i} endpoint={rec['endpoint']} "
                        f"seed={rec['seed']} case_scores={rec['case_scores']} reward={rec['reward']}",
                        flush=True,
                    )
                    submit_next(endpoint)

            sample_records = [samples_by_index[i] for i in range(args.population)]
            rewards = [rec["reward"] for rec in sample_records]
        update_payload = {
            "seeds": seeds,
            "rewards": rewards,
            "alpha": args.alpha,
            "reward_normalization": args.reward_normalization,
        }
        with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
            update_futures = {
                endpoint: pool.submit(
                    post_json,
                    f"{endpoint}/es/update",
                    update_payload,
                )
                for endpoint in endpoints
            }
            update_records = [
                {"endpoint": endpoint, "update": update_futures[endpoint].result()}
                for endpoint in endpoints
            ]
        print(f"[update] gen={generation} {update_records}", flush=True)

        eval_rec = None
        should_eval = args.eval_interval > 0 and (
            (generation + 1) % args.eval_interval == 0
            or (generation + 1) == args.generations
        )
        if should_eval:
            eval_rec = eval_tasks_distributed(
                endpoints=endpoints,
                task_ids=eval_task_ids,
                config_dir=config_dir,
                result_root=result_root,
                run_name=f"eval_after_epoch_{generation + 1:03d}",
                skill_file=None,
                workers_per_endpoint=args.eval_workers_per_endpoint,
                instruction_path=args.instruction_path,
                model_name=args.model_name,
                mode=args.mode,
                stop_token=args.stop_token,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                presence_penalty=args.presence_penalty,
                repetition_penalty=args.repetition_penalty,
            )
        record = {
            "generation": generation,
            "case_batch": selected,
            "sigma": sigma_t,
            "seeds": seeds,
            "rewards": rewards,
            "samples": sample_records,
            "updates": update_records,
            "replayed": replayed,
        }
        if replayed:
            record["replay_source"] = str(replay_source_root)
        if eval_rec is not None:
            record["eval"] = eval_rec
        history.append(record)
        (result_root / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        write_generation_scalars(
            writer,
            generation=generation + 1,
            rewards=rewards,
            sigma=sigma_t,
            alpha=args.alpha,
            eval_rec=eval_rec,
        )

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
