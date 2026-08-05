#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "webarena-train-time/scripts"))

from run_webrl_lite_distributed_es_train import eval_tasks_distributed, write_eval_scalars  # noqa: E402
from run_webrl_lite_full_es_train import DEFAULT_CONFIG_DIR, DEFAULT_SPLIT, load_tasks, post_json  # noqa: E402

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None


def update_endpoint(endpoint: str, seeds: list[int], rewards: list[float], alpha: float, normalization: str) -> dict:
    update = post_json(
        f"{endpoint}/es/update",
        {
            "seeds": seeds,
            "rewards": rewards,
            "alpha": alpha,
            "reward_normalization": normalization,
        },
    )
    return {"endpoint": endpoint, "update": update}


def post_all(endpoints: list[str], path: str, payload: dict) -> list[dict]:
    records = []
    with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
        futures = {pool.submit(post_json, f"{endpoint}{path}", payload): endpoint for endpoint in endpoints}
        for future in as_completed(futures):
            endpoint = futures[future]
            records.append({"endpoint": endpoint, "response": future.result()})
    return sorted(records, key=lambda row: row["endpoint"])


def replay_history(
    *,
    endpoints: list[str],
    history: list[dict],
    generations: int,
    alpha: float,
    normalization: str,
) -> list[dict]:
    replay_records = []
    for record in history[:generations]:
        generation = int(record["generation"])
        seeds = [int(seed) for seed in record["seeds"]]
        rewards = [float(reward) for reward in record["rewards"]]
        update_records = []
        with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
            futures = {
                pool.submit(update_endpoint, endpoint, seeds, rewards, alpha, normalization): endpoint
                for endpoint in endpoints
            }
            for future in as_completed(futures):
                update_records.append(future.result())
        update_records.sort(key=lambda row: row["endpoint"])
        replay_record = {
            "generation": generation,
            "case_batch": record.get("case_batch"),
            "seeds": seeds,
            "rewards": rewards,
            "updates": update_records,
        }
        replay_records.append(replay_record)
        valid = [reward for reward in rewards if reward >= 0.0]
        print(
            f"[replay] generation={generation} rewards={len(rewards)} "
            f"valid={len(valid)} mean={sum(valid) / len(valid) if valid else -1.0}",
            flush=True,
        )
    return replay_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-history", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--endpoints", required=True)
    parser.add_argument("--generations", type=int, default=21)
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--reward-normalization", default="zscore")
    parser.add_argument("--parameter-scope", default="full", choices=["full", "all_linear", "lora"])
    parser.add_argument("--skip-reset", action="store_true")
    parser.add_argument("--skip-init", action="store_true")
    parser.add_argument("--eval-split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--sites", default="shopping,shopping_admin,reddit,gitlab,wikipedia,map")
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--eval-workers-per-endpoint", type=int, default=4)
    parser.add_argument("--eval-repeats", type=int, default=1)
    parser.add_argument("--skill-file", default="")
    parser.add_argument("--instruction-path", default="agent/prompts/jsons/p_webrl_chat_qwen_action.json")
    parser.add_argument("--model-name", default="Qwen3.5-27B")
    parser.add_argument("--mode", default="chat", choices=["completion", "chat"])
    parser.add_argument("--stop-token", default="")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    args = parser.parse_args()

    endpoints = [endpoint.strip().rstrip("/") for endpoint in args.endpoints.split(",") if endpoint.strip()]
    if not endpoints:
        raise RuntimeError("No endpoints provided.")
    source_history = json.loads(Path(args.source_history).read_text())
    if args.generations > len(source_history):
        raise ValueError(f"Requested {args.generations} generations, but history has {len(source_history)}.")

    result_root = ROOT / "runs/webrl_lite_full_es" / args.run_id
    result_root.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(result_root / "tensorboard")) if SummaryWriter is not None else None

    if not args.skip_reset:
        reset_records = post_all(endpoints, "/es/reset", {})
        (result_root / "reset_records.json").write_text(json.dumps(reset_records, indent=2) + "\n")
        print(f"[reset] {reset_records}", flush=True)
    if not args.skip_init:
        init_records = post_all(
            endpoints,
            "/es/init",
            {"parameter_scope": args.parameter_scope, "verbose": True},
        )
        (result_root / "init_records.json").write_text(json.dumps(init_records, indent=2) + "\n")
        print(f"[init] {init_records}", flush=True)

    replay_records = replay_history(
        endpoints=endpoints,
        history=source_history,
        generations=args.generations,
        alpha=args.alpha,
        normalization=args.reward_normalization,
    )
    (result_root / "replay_history.json").write_text(json.dumps(replay_records, indent=2) + "\n")

    allowed_sites = {site.strip() for site in args.sites.split(",") if site.strip()}
    eval_task_ids = load_tasks(Path(args.eval_split), allowed_sites, args.eval_limit)
    skill_file = Path(args.skill_file) if args.skill_file else None
    eval_records = []
    for repeat in range(1, args.eval_repeats + 1):
        run_name = f"eval_after_replay_{args.generations:03d}_repeat_{repeat:02d}"
        eval_rec = eval_tasks_distributed(
            endpoints=endpoints,
            task_ids=eval_task_ids,
            config_dir=Path(args.config_dir),
            result_root=result_root,
            run_name=run_name,
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
        eval_records.append(eval_rec)
        (result_root / f"eval_summary_repeat_{repeat:02d}.json").write_text(
            json.dumps(eval_rec, indent=2) + "\n"
        )
        write_eval_scalars(writer, f"eval/repeat_{repeat:02d}", eval_rec, args.generations)
        print(f"[eval_summary] {json.dumps(eval_rec, ensure_ascii=False)}", flush=True)

    averages = [float(record["average"]) for record in eval_records]
    mean = sum(averages) / len(averages) if averages else -1.0
    variance = sum((average - mean) ** 2 for average in averages) / len(averages) if averages else 0.0
    summary = {
        "generations": args.generations,
        "eval_repeats": args.eval_repeats,
        "run_averages": averages,
        "mean": mean,
        "std": math.sqrt(variance),
        "runs": eval_records,
    }
    (result_root / "eval_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if writer is not None:
        writer.add_scalar("eval/mean", mean, args.generations)
        writer.add_scalar("eval/std", summary["std"], args.generations)
        writer.flush()
    if writer is not None:
        writer.close()
    print(f"[eval_aggregate] {json.dumps(summary, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
