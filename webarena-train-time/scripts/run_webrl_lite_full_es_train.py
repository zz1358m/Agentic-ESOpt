#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib import request


ROOT = Path(os.environ.get("ROOT", Path(__file__).resolve().parents[2])).resolve()
VAB = Path(os.environ.get("VAB_ROOT", ROOT / "data/webarena/vab-lite")).resolve()
PY = Path(os.environ.get("PY", sys.executable))
DEFAULT_CONFIG_DIR = ROOT / "data/webarena/vab-lite/config_files/wa/test_webarena_lite"
DEFAULT_SPLIT = ROOT / "data/webarena/vab_lite_split/items.json"
DEFAULT_WEBRL_TRAJECTORIES = ROOT / "data/webarena/skillopt_splits/train/trajectories.jsonl"


def web_urls_from_env() -> dict[str, str]:
    host = os.environ.get("WEBARENA_HOST", "127.0.0.1").removeprefix("http://").removeprefix("https://").rstrip("/")
    defaults = {
        "SHOPPING": f"http://{host}:7770",
        "SHOPPING_ADMIN": f"http://{host}:7780/admin",
        "REDDIT": f"http://{host}:9999",
        "GITLAB": f"http://{host}:8023",
        "MAP": f"http://{host}:3000",
        "WIKIPEDIA": f"http://{host}:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing",
        "HOMEPAGE": f"http://{host}:4399",
    }
    return {key: os.environ.get(key) or os.environ.get(f"WA_{key}") or value for key, value in defaults.items()}


WEB_URLS = web_urls_from_env()


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode())


def load_tasks(split_path: Path, allowed_sites: set[str], limit: int) -> list[int]:
    resolved = split_path.resolve()
    blocked_parts = {("data", "webarena", "lite"), ("data", "webarena", "jitrl")}
    parts = resolved.parts
    for blocked in blocked_parts:
        for idx in range(0, len(parts) - len(blocked) + 1):
            if tuple(parts[idx : idx + len(blocked)]) == blocked:
                raise ValueError(
                    f"Refusing legacy/JitRL WebArena split: {resolved}. "
                    "Use data/webarena/vab_lite_split/items.json for WebRL/VAB setting."
                )
    items = json.loads(split_path.read_text())
    if len(items) == 165:
        old_task_ids = sum(1 for item in items if item.get("old_task_id") is not None)
        if old_task_ids < 100 and split_path.name == "items.json":
            raise ValueError(
                f"Split does not look like VAB/WebRL WebArena-Lite: {resolved}. "
                "Expected old_task_id metadata from data/webarena/vab_lite_split/items.json."
            )
    task_ids = []
    for item in items:
        sites = set(item.get("sites") or [])
        if sites and sites.issubset(allowed_sites):
            task_ids.append(int(item["task_id"]))
        if limit and len(task_ids) >= limit:
            break
    if not task_ids:
        raise RuntimeError(f"No tasks selected from {split_path}")
    return task_ids


def eval_tasks(
    *,
    endpoint: str,
    task_ids: list[int],
    config_dir: Path,
    result_root: Path,
    run_name: str,
    skill_file: Path | None,
) -> dict:
    scores = []
    for task_id in task_ids:
        score = run_episode(
            endpoint=endpoint,
            task_id=task_id,
            config_dir=config_dir,
            result_root=result_root,
            run_name=run_name,
            skill_file=skill_file,
        )
        scores.append({"task_id": task_id, "score": score})
        print(f"[eval] {run_name} task={task_id} score={score}", flush=True)
    valid = [row["score"] for row in scores if row["score"] >= 0.0]
    return {
        "run_name": run_name,
        "count": len(scores),
        "valid_count": len(valid),
        "average": sum(valid) / len(valid) if valid else -1.0,
        "max": max(valid) if valid else -1.0,
        "scores": scores,
    }


def make_single_config(task_id: int, source_config: Path, work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    source_file = source_config / f"{task_id}.json"
    if not source_file.exists():
        raise FileNotFoundError(source_file)
    raw = source_file.read_text()
    for key, value in WEB_URLS.items():
        raw = raw.replace(f"__{key}__", value)
    (work_dir / "0.json").write_text(raw)
    return work_dir


def run_episode(
    *,
    endpoint: str,
    task_id: int,
    config_dir: Path = DEFAULT_CONFIG_DIR,
    result_root: Path,
    run_name: str,
    skill_file: Path | None = None,
    instruction_path: str = "agent/prompts/jsons/p_webrl.json",
    model_name: str = "Llama-3.1-8B-Instruct",
    mode: str = "completion",
    stop_token: str = "<|eot_id|>",
    max_steps: int = 30,
    temperature: float = 0.0,
    top_p: float = 0.9,
    top_k: int | None = None,
    min_p: float | None = None,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
) -> float:
    with tempfile.TemporaryDirectory(prefix=f"webrl_es_task_{task_id}_") as tmp:
        cfg_dir = make_single_config(task_id, config_dir, Path(tmp) / "configs")
        result_dir = (result_root / run_name / f"task_{task_id}").resolve()
        result_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        for key in ("DISPLAY", "XAUTHORITY", "WAYLAND_DISPLAY"):
            env.pop(key, None)
        api_key_path = ROOT / "apikey"
        env.update(
            {
                "DATASET": "webarena",
                **WEB_URLS,
                "OPENAI_API_KEY": api_key_path.read_text().strip() if api_key_path.exists() else "dummy",
                "WEBRL_EVAL_MODEL": os.environ.get("WEBRL_EVAL_MODEL", "gpt-4.1-mini"),
                "PYTHONPATH": str(VAB),
            }
        )
        if skill_file is not None:
            env["WEBRL_SKILL_FILE"] = str(skill_file)
            env["WEBRL_FIXED_SKILL_POLICY"] = "1"
        cmd = [
            str(PY),
            "run.py",
            "--instruction_path",
            instruction_path,
            "--test_start_idx",
            "0",
            "--test_end_idx",
            "1",
            "--result_dir",
            str(result_dir),
            "--test_config_base_dir",
            str(cfg_dir),
            "--provider",
            "local_completion",
            "--model",
            model_name,
            "--mode",
            mode,
            "--planner_ip",
            f"{endpoint}/completions",
            "--stop_token",
            stop_token,
            "--temperature",
            str(temperature),
            "--top_p",
            str(top_p),
            "--max_obs_length",
            "0",
            "--max_tokens",
            "2048",
            "--viewport_width",
            "1280",
            "--viewport_height",
            "720",
            "--parsing_failure_th",
            "5",
            "--repeating_action_failure_th",
            "5",
            "--action_set_tag",
            "webrl_id",
            "--observation_type",
            "webrl",
            "--max_steps",
            str(max_steps),
        ]
        if top_k is not None:
            cmd.extend(["--top_k", str(top_k)])
        if min_p is not None:
            cmd.extend(["--min_p", str(min_p)])
        cmd.extend(
            [
                "--presence_penalty",
                str(presence_penalty),
                "--repetition_penalty",
                str(repetition_penalty),
            ]
        )
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(VAB),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=1200,
            )
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ""
            if isinstance(output, bytes):
                output = output.decode(errors="replace")
            (result_dir / "timeout.log").write_text(output, encoding="utf-8")
            return 0.0
        (result_dir / "run.log").write_text(proc.stdout)
        action_paths = sorted((result_dir / "actions").glob("*.json"))
        if not action_paths:
            return -1.0
        try:
            return float(json.loads(action_paths[-1].read_text()).get("score", 0.0))
        except Exception:
            return -1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", default="")
    parser.add_argument("--eval-split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--train-config-dir", default="")
    parser.add_argument(
        "--train-source",
        default=os.environ.get("WEBRL_ES_TRAIN_SOURCE", "environment"),
        choices=["environment", "webrl_sft"],
        help=(
            "environment runs perturbed policies in browser tasks. "
            "webrl_sft is reserved for offline WebRL trajectory objectives and is not implemented here."
        ),
    )
    parser.add_argument("--webrl-trajectories", default=str(DEFAULT_WEBRL_TRAJECTORIES))
    parser.add_argument("--sites", default="shopping,shopping_admin,reddit,gitlab,wikipedia,map")
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--case-batch-size", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=5e-4)
    parser.add_argument("--alpha", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--reward-normalization", default="zscore")
    parser.add_argument("--skill-file", default="")
    parser.add_argument("--parameter-scope", default="full", choices=["full", "all_linear", "lora"])
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--skip-initial-eval", action="store_true")
    parser.add_argument("--instruction-path", default="agent/prompts/jsons/p_webrl_chat_qwen_action.json")
    parser.add_argument("--model-name", default="Qwen3.5-27B")
    parser.add_argument("--mode", default="chat", choices=["completion", "chat"])
    parser.add_argument("--stop-token", default="")
    args = parser.parse_args()

    if not VAB.exists():
        raise FileNotFoundError(
            f"VAB/WebArena-Lite source not found: {VAB}. "
            "Set VAB_ROOT or place it at data/webarena/vab-lite."
        )

    result_root = ROOT / "runs/webrl_lite_full_es" / args.run_id
    result_root.mkdir(parents=True, exist_ok=True)
    allowed_sites = {site.strip() for site in args.sites.split(",") if site.strip()}
    eval_task_ids = load_tasks(Path(args.eval_split), allowed_sites, args.eval_limit)
    config_dir = Path(args.config_dir)
    if not config_dir.exists():
        raise FileNotFoundError(config_dir)
    if args.train_source == "webrl_sft":
        trajectory_path = Path(args.webrl_trajectories)
        if not trajectory_path.exists():
            raise FileNotFoundError(
                f"WebRL SFT trajectories not found: {trajectory_path}. "
                "Prepare data/webarena/vab_lite_split/items.json or pass --split/--eval-split explicitly."
            )
        raise NotImplementedError(
            "Offline WebRL-SFT ES training is not implemented in this runner. "
            "Use --train-source environment with --split and --train-config-dir for browser-interaction ES, "
            "or add an explicit trajectory-scoring objective before enabling WebRL-SFT ES."
        )
    if not args.split:
        raise ValueError(
            "--split is required for environment ES training. "
            "Use data/webarena/vab_lite_split/items.json only for VAB/WebRL WebArena-Lite evaluation; "
            "training needs a config-backed split plus --train-config-dir."
        )
    task_ids = load_tasks(Path(args.split), allowed_sites, args.episodes) if args.split else []
    train_config_dir = Path(args.train_config_dir) if args.train_config_dir else None
    if task_ids and train_config_dir is None:
        raise ValueError("--train-config-dir is required when --split is set.")
    if train_config_dir is not None and not train_config_dir.exists():
        raise FileNotFoundError(train_config_dir)
    skill_file = Path(args.skill_file) if args.skill_file else None
    if skill_file is not None and not skill_file.exists():
        raise FileNotFoundError(skill_file)

    init = post_json(f"{args.endpoint}/es/init", {"parameter_scope": args.parameter_scope, "verbose": True})
    print(f"[es_init] {init}", flush=True)
    print(
        f"[setting] population={args.population} case_batch_size={args.case_batch_size} "
        f"sigma={args.sigma} alpha={args.alpha} parameter_scope={args.parameter_scope} "
        f"skill_file={skill_file or ''} train_config_dir={train_config_dir or ''} eval_config_dir={config_dir}",
        flush=True,
    )

    rng = random.Random(args.seed)
    history = []
    if not args.skip_initial_eval:
        initial_eval = eval_tasks(
            endpoint=args.endpoint,
            task_ids=eval_task_ids,
            config_dir=config_dir,
            result_root=result_root,
            run_name="initial_base_eval",
            skill_file=skill_file,
            instruction_path=args.instruction_path,
            model_name=args.model_name,
            mode=args.mode,
            stop_token=args.stop_token,
        )
        history.append(
            {
                "generation": -1,
                "kind": "initial_base_eval",
                "eval": initial_eval,
            }
        )
        (result_root / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"[initial_eval] {initial_eval}", flush=True)

    for generation in range(args.generations):
        seeds = []
        rewards = []
        batch_start = (generation * args.case_batch_size) % len(task_ids)
        selected = [task_ids[(batch_start + i) % len(task_ids)] for i in range(args.case_batch_size)]
        print(f"[generation {generation}] case_batch={selected}", flush=True)
        sample_records = []
        for i in range(args.population):
            seed = rng.randrange(1, 2**31 - 1)
            seeds.append(seed)
            post_json(f"{args.endpoint}/es/apply", {"seed": seed, "sigma": args.sigma})
            case_scores = []
            try:
                for task_id in selected:
                    score = run_episode(
                        endpoint=args.endpoint,
                        task_id=task_id,
                        config_dir=train_config_dir,
                        result_root=result_root,
                        run_name=f"gen_{generation:03d}_sample_{i:02d}_seed_{seed}",
                        skill_file=skill_file,
                        instruction_path=args.instruction_path,
                        model_name=args.model_name,
                        mode=args.mode,
                        stop_token=args.stop_token,
                    )
                    case_scores.append(score)
            finally:
                post_json(f"{args.endpoint}/es/revert", {"seed": seed, "sigma": args.sigma})
            valid_scores = [score for score in case_scores if score != -1.0]
            reward = sum(valid_scores) / len(valid_scores) if valid_scores else -1.0
            rewards.append(reward)
            sample_records.append({"seed": seed, "case_scores": case_scores, "reward": reward})
            print(
                f"[sample] gen={generation} sample={i} seed={seed} "
                f"case_scores={case_scores} reward={reward}",
                flush=True,
            )

        update = post_json(
            f"{args.endpoint}/es/update",
            {
                "seeds": seeds,
                "rewards": rewards,
                "alpha": args.alpha,
                "reward_normalization": args.reward_normalization,
            },
        )
        eval_rec = eval_tasks(
            endpoint=args.endpoint,
            task_ids=eval_task_ids,
            config_dir=config_dir,
            result_root=result_root,
            run_name=f"eval_after_epoch_{generation + 1:03d}",
            skill_file=skill_file,
            instruction_path=args.instruction_path,
            model_name=args.model_name,
            mode=args.mode,
            stop_token=args.stop_token,
        )
        rec = {
            "generation": generation,
            "case_batch": selected,
            "seeds": seeds,
            "rewards": rewards,
            "samples": sample_records,
            "update": update,
            "eval": eval_rec,
        }
        history.append(rec)
        (result_root / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"[update] {update}", flush=True)


if __name__ == "__main__":
    main()
