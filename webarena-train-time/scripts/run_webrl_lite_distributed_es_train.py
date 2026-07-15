#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import shutil
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional runtime dependency
    SummaryWriter = None


ROOT = Path(os.environ.get("ROOT", Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
TRACE_WRAPPER_PATH = ROOT / "webarena-train-time" / "scripts" / "run_trace2skill_webarena_sft.py"
EMPTY_WEB_ARENA_SKILL = """---
name: webarena-sft-trace-skill
description: Skill instructions for WebArena agents using WebRL id actions.
---

# WebArena Skill

"""

_TRACE2SKILL_ADAPTER = None


def load_trace2skill_adapter():
    """Load optional Trace2Skill/SkillOpt dependencies only when requested."""

    global _TRACE2SKILL_ADAPTER
    if _TRACE2SKILL_ADAPTER is not None:
        return _TRACE2SKILL_ADAPTER
    spec = importlib.util.spec_from_file_location("trace2skill_webarena_sft", TRACE_WRAPPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {TRACE_WRAPPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _TRACE2SKILL_ADAPTER = module
    return module

from run_webrl_lite_full_es_train import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_SPLIT,
    DEFAULT_TRAIN_CONFIG_DIR,
    DEFAULT_TRAIN_SPLIT,
    load_tasks,
    post_json,
    post_json_retry,
    run_episode,
    validate_config_alignment,
)  # noqa: E402
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


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def task_score(task_dir: Path) -> float:
    for path in sorted((task_dir / "actions").glob("*.json")):
        try:
            return float(json.loads(path.read_text(encoding="utf-8")).get("score", 0.0))
        except Exception:
            pass
    return -1.0


def truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def write_trace2skill_log(task_dir: Path, output_path: Path, html_limit: int) -> dict | None:
    trace_paths = sorted((task_dir / "traces").glob("*.jsonl"))
    if not trace_paths:
        return None
    rows = read_jsonl(trace_paths[0])
    if not rows:
        return None
    score = task_score(task_dir)
    if score < 0:
        return None
    outcome = "SUCCEED" if score >= 1.0 else "FAILED"
    task_id = str(rows[0].get("trace_id") or task_dir.name.removeprefix("task_"))
    target = str(rows[0].get("target") or rows[0].get("prompt") or "")
    lines = [
        f"# Chat History {task_dir.parent.name}_{task_id}",
        "",
        f"Task ID: {task_id}",
        "Site: unknown",
        f"Task: {target}",
        f"Score: {score}",
        f"Outcome: {outcome}",
        "Failure reason: WebArena evaluator score was 0." if score < 1.0 else "Failure reason: ",
        "",
        "## WebArena Execution Trace",
        "",
    ]
    for row in rows:
        idx = row.get("index", 0)
        html = truncate_text(str(row.get("html", "")), html_limit)
        prompt = str(row.get("prompt", ""))
        response = str(row.get("response", ""))
        lines.extend(
            [
                f"## Round {idx}",
                "",
                "### User",
                "",
                f"Task Instruction: {target}" if idx == 0 else (prompt or "** Simplified html **"),
                "",
                "Simplified HTML:",
                "",
                "```html",
                html,
                "```",
                "",
                "### Assistant",
                "",
                response,
                "",
            ]
        )
    lines.extend(["", "---", "", "## RESULT", outcome, ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return {"task_id": task_id, "score": score, "outcome": outcome, "source": str(task_dir)}


def evolve_skill_from_es_generation(
    *,
    result_root: Path,
    generation: int,
    skill_file: Path,
    optimizer_model: str,
    analysis_workers: int,
    seed: int,
    html_limit: int,
    official_prompts: bool,
    optimizer_generation_config: str,
) -> dict:
    trace2skill = load_trace2skill_adapter()
    skill_dir = skill_file.parent
    update_dir = result_root / "trace2skill_updates" / f"generation_{generation + 1:03d}"
    logs_dir = update_dir / "trace_logs"
    if logs_dir.exists():
        shutil.rmtree(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    records = []
    sample_dirs = sorted(result_root.glob(f"gen_{generation:03d}_sample_*"))
    for sample_dir in sample_dirs:
        for task_dir in sorted(sample_dir.glob("task_*")):
            record = write_trace2skill_log(
                task_dir,
                logs_dir
                / (
                    f"webarena_agent_{sample_dir.name}_{task_dir.name}_"
                    f"{'SUCCEED' if task_score(task_dir) >= 1.0 else 'FAILED'}.md"
                ),
                html_limit,
            )
            if record:
                records.append(record)
    trace2skill.write_json(update_dir / "source_traces.json", records)
    if not records:
        return {"generation": generation, "trace_count": 0, "updated": False}
    trace2skill.run_analysis_and_evolve(
        update_dir,
        skill_dir,
        optimizer_model,
        analysis_workers,
        seed,
        official_prompts=official_prompts,
        optimizer_generation_config=optimizer_generation_config,
    )
    snapshot = result_root / f"skill_generation_{generation + 1:03d}.md"
    shutil.copy2(skill_file, snapshot)
    return {"generation": generation, "trace_count": len(records), "updated": True, "skill": str(snapshot)}


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
    skill_file: Path | None,
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
    parser.add_argument("--split", default=str(DEFAULT_TRAIN_SPLIT))
    parser.add_argument("--eval-split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--train-config-dir", default=str(DEFAULT_TRAIN_CONFIG_DIR))
    parser.add_argument("--sites", default="shopping,shopping_admin,reddit,gitlab,wikipedia,map")
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--case-batch-size", type=int, default=16)
    parser.add_argument("--case-workers-per-sample", type=int, default=4)
    parser.add_argument("--eval-workers-per-endpoint", type=int, default=4)
    parser.add_argument("--sigma-start", type=float, default=float(os.environ.get("WEBRL_ES_SIGMA_START", "1e-3")))
    parser.add_argument("--sigma-end", type=float, default=float(os.environ.get("WEBRL_ES_SIGMA_END", os.environ.get("WEBRL_ES_SIGMA_START", "1e-3"))))
    parser.add_argument(
        "--sigma-schedule",
        default=os.environ.get("WEBRL_ES_SIGMA_SCHEDULE", "constant"),
        choices=["constant", "linear", "cosine"],
    )
    parser.add_argument(
        "--sigma-warmup-steps",
        type=int,
        default=int(os.environ.get("WEBRL_ES_SIGMA_WARMUP_STEPS", "0")),
        help="Number of initial generations to keep sigma fixed.",
    )
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--reward-normalization", default="zscore")
    parser.add_argument(
        "--skill-file",
        default="",
    )
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
        "--trace2skill-every-generation",
        action="store_true",
        help="After each ES generation, summarize that generation's traces into the run skill file.",
    )
    parser.add_argument("--init-empty-skill", action="store_true", help="Create an empty SKILL.md when --skill-file is missing.")
    parser.add_argument("--trace2skill-optimizer-model", default="gpt-4.1-mini")
    parser.add_argument("--trace2skill-analysis-workers", type=int, default=16)
    parser.add_argument("--trace2skill-html-limit", type=int, default=12000)
    parser.add_argument(
        "--trace2skill-official-prompts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--trace2skill-optimizer-generation-config", default="")
    parser.add_argument(
        "--tensorboard-dir",
        default="",
        help="TensorBoard log directory. Defaults to <result_root>/tensorboard.",
    )
    parser.add_argument("--history-file", default=os.environ.get("WEBRL_ES_HISTORY_FILE", ""))
    parser.add_argument("--resume-history", default=os.environ.get("WEBRL_ES_RESUME_HISTORY", ""))
    parser.add_argument("--resume-generations", type=int, default=int(os.environ.get("WEBRL_ES_RESUME_GENERATIONS", "-1")))
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
        raise RuntimeError("No endpoints provided.")
    result_root = ROOT / "runs/webrl_lite_full_es" / args.run_id
    result_root.mkdir(parents=True, exist_ok=True)
    history_path = Path(args.history_file).expanduser().resolve() if args.history_file else result_root / "history.json"
    tensorboard_dir = Path(args.tensorboard_dir) if args.tensorboard_dir else result_root / "tensorboard"
    writer = SummaryWriter(str(tensorboard_dir)) if SummaryWriter is not None else None
    allowed_sites = {site.strip() for site in args.sites.split(",") if site.strip()}
    eval_task_ids = load_tasks(Path(args.eval_split), allowed_sites, args.eval_limit)
    config_dir = Path(args.config_dir)
    if not config_dir.exists():
        raise FileNotFoundError(config_dir)
    if not args.eval_only and not args.split:
        raise ValueError(
            "--split is required for environment ES training. "
            "Use data/webarena/vab_nonlite_split/train/items.json with --train-config-dir."
        )
    train_task_ids = (
        load_tasks(Path(args.split), allowed_sites, args.episodes)
        if not args.eval_only and args.split
        else []
    )
    train_config_dir = Path(args.train_config_dir) if train_task_ids and args.train_config_dir else None
    if train_task_ids and train_config_dir is None:
        raise ValueError("--train-config-dir is required when --split is set.")
    if train_config_dir is not None and not train_config_dir.exists():
        raise FileNotFoundError(train_config_dir)
    if train_task_ids and train_config_dir is not None:
        validate_config_alignment(Path(args.split), train_task_ids, train_config_dir)
    skill_file = Path(args.skill_file) if args.skill_file else None
    if skill_file is not None and not skill_file.exists():
        if args.init_empty_skill:
            skill_file.parent.mkdir(parents=True, exist_ok=True)
            skill_file.write_text(EMPTY_WEB_ARENA_SKILL, encoding="utf-8")
        else:
            raise FileNotFoundError(skill_file)
    if args.trace2skill_every_generation and skill_file is None:
        raise ValueError("--trace2skill-every-generation requires --skill-file.")

    print(
        f"[setting] endpoints={endpoints} population={args.population} "
        f"case_batch_size={args.case_batch_size} case_workers_per_sample={args.case_workers_per_sample} "
        f"eval_workers_per_endpoint={args.eval_workers_per_endpoint} "
        f"sigma_start={args.sigma_start} sigma_end={args.sigma_end} sigma_schedule={args.sigma_schedule} "
        f"sigma_warmup_steps={sigma_warmup_steps} "
        f"alpha={args.alpha} "
        f"parameter_scope={args.parameter_scope} skill_file={skill_file or ''} "
        f"train_config_dir={train_config_dir or ''} eval_config_dir={config_dir} "
        f"tensorboard_dir={tensorboard_dir if writer is not None else ''}",
        flush=True,
    )

    history = [{"config": {"sigma_start": args.sigma_start, "sigma_end": args.sigma_end, "sigma_schedule": args.sigma_schedule, "sigma_warmup_steps": sigma_warmup_steps, "alpha": args.alpha, "population": args.population, "seed": args.seed, "history_file": str(history_path), "trace2skill_every_generation": args.trace2skill_every_generation}}]
    start_generation = 0
    if not args.eval_only or args.resume_history:
        init_records = []
        for endpoint in endpoints:
            init = post_json(f"{endpoint}/es/init", {"parameter_scope": args.parameter_scope, "verbose": True})
            init_records.append({"endpoint": endpoint, "init": init})
        print(f"[es_init] {init_records}", flush=True)
    if args.resume_history:
        source_history = read_history(args.resume_history)
        replay_limit = None if args.resume_generations < 0 else args.resume_generations
        resume_records = completed_update_records(source_history, limit=replay_limit)
        start_generation = validate_seed_sequence(resume_records, population=args.population, seed=args.seed)
        replay_log = replay_http_updates(
            endpoints=endpoints,
            records=resume_records,
            post_json=post_json,
            timeout=600,
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

    if args.eval_only:
        eval_rec = eval_tasks_distributed(
            endpoints=endpoints,
            task_ids=eval_task_ids,
            config_dir=config_dir,
            result_root=result_root,
            run_name="eval_only",
            skill_file=skill_file,
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
        atomic_write_history(history_path, history)
        write_eval_scalars(writer, "eval_only", eval_rec, start_generation)
        if writer is not None:
            writer.close()
        print(f"[eval_only] {eval_rec}", flush=True)
        return

    if not args.skip_initial_eval:
        initial_eval = eval_tasks_distributed(
            endpoints=endpoints,
            task_ids=eval_task_ids,
            config_dir=config_dir,
            result_root=result_root,
            run_name="initial_base_eval",
            skill_file=skill_file,
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
        atomic_write_history(history_path, history)
        write_eval_scalars(writer, "initial_eval", initial_eval, 0)
        print(f"[initial_eval] {initial_eval}", flush=True)

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
        batch_start = (generation * args.case_batch_size) % len(train_task_ids)
        selected = [train_task_ids[(batch_start + i) % len(train_task_ids)] for i in range(args.case_batch_size)]
        seeds = [rng.randrange(1, 2**31 - 1) for _ in range(args.population)]
        print(f"[generation {generation}] sigma={sigma_t:.12g} case_batch={selected}", flush=True)

        def eval_direction(i: int, endpoint: str):
            return i, eval_population_sample(
                    endpoint=endpoint,
                    seed=seeds[i],
                    sigma=sigma_t,
                    task_ids=selected,
                    config_dir=train_config_dir,
                    result_root=result_root,
                    run_name=f"gen_{generation:03d}_sample_{i:02d}_seed_{seeds[i]}",
                    skill_file=skill_file,
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

        sample_records = map_endpoint_serial(
            endpoints=endpoints,
            count=args.population,
            worker=eval_direction,
        )
        for i, rec in enumerate(sample_records):
            print(
                f"[sample] gen={generation} sample={i} endpoint={rec['endpoint']} "
                f"seed={rec['seed']} case_scores={rec['case_scores']} reward={rec['reward']}",
                flush=True,
            )
        rewards = [rec["reward"] for rec in sample_records]
        update_records = []
        for endpoint in endpoints:
            update = post_json(
                f"{endpoint}/es/update",
                {
                    "seeds": seeds,
                    "rewards": rewards,
                    "alpha": args.alpha,
                    "reward_normalization": args.reward_normalization,
                },
            )
            update_records.append({"endpoint": endpoint, "update": update})
        print(f"[update] gen={generation} {update_records}", flush=True)

        skill_update_rec = None
        if args.trace2skill_every_generation and skill_file is not None:
            skill_update_rec = evolve_skill_from_es_generation(
                result_root=result_root,
                generation=generation,
                skill_file=skill_file,
                optimizer_model=args.trace2skill_optimizer_model,
                analysis_workers=args.trace2skill_analysis_workers,
                seed=args.seed + generation + 1,
                html_limit=args.trace2skill_html_limit,
                official_prompts=args.trace2skill_official_prompts,
                optimizer_generation_config=args.trace2skill_optimizer_generation_config,
            )
            print(f"[trace2skill] gen={generation} {skill_update_rec}", flush=True)

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
                skill_file=skill_file,
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
            "sigma_start": args.sigma_start,
            "sigma_end": args.sigma_end,
            "sigma_schedule": args.sigma_schedule,
            "sigma_warmup_steps": sigma_warmup_steps,
            "seeds": seeds,
            "rewards": rewards,
            "alpha": args.alpha,
            "reward_normalization": args.reward_normalization,
            "samples": sample_records,
            "updates": update_records,
        }
        if skill_update_rec is not None:
            record["trace2skill"] = skill_update_rec
        if eval_rec is not None:
            record["eval"] = eval_rec
        history.append(record)
        atomic_write_history(history_path, history)
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
