#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib import error, request


ROOT = Path(os.environ.get("ROOT", Path(__file__).resolve().parents[2])).resolve()
VAB = Path(os.environ.get("VAB_ROOT", ROOT / "data/webarena/vab-lite")).resolve()
PY = Path(os.environ.get("PY", sys.executable))
DEFAULT_CONFIG_DIR = ROOT / "data/webarena/vab-lite/config_files/wa/test_webarena_lite"
DEFAULT_SPLIT = ROOT / "data/webarena/vab_lite_split/items.json"


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


def resolve_sigma_warmup_steps(generations: int, warmup_steps: int) -> int:
    if warmup_steps >= 0:
        return min(max(0, warmup_steps), max(0, generations))
    return max(0, generations // 4)


def sigma_for_generation(
    *,
    sigma_start: float,
    sigma_end: float,
    generation: int,
    generations: int,
    schedule: str,
    warmup_steps: int,
) -> float:
    if generations <= 0:
        return sigma_start
    if schedule != "cosine":
        raise ValueError(f"Unsupported sigma schedule: {schedule}")
    warmup_steps = resolve_sigma_warmup_steps(generations, warmup_steps)
    if generation < warmup_steps or warmup_steps >= generations:
        return sigma_start
    denominator = max(1, generations - 1 - warmup_steps)
    progress = min(1.0, max(0.0, (generation - warmup_steps) / denominator))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return sigma_end + (sigma_start - sigma_end) * cosine


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


def post_json_retry(
    url: str,
    payload: dict,
    *,
    attempts: int = 3,
    delay_seconds: float = 5.0,
) -> dict:
    last_error: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return post_json(url, payload)
        except (TimeoutError, error.URLError) as exc:
            last_error = exc
            if attempt >= attempts:
                break
            print(
                f"[http_retry] url={url} attempt={attempt}/{attempts} error={exc!r}",
                flush=True,
            )
            time.sleep(delay_seconds)
    assert last_error is not None
    raise last_error


def load_tasks(split_path: Path, allowed_sites: set[str], limit: int) -> list[int]:
    resolved = split_path.resolve()
    blocked_parts = {("data", "webarena", "lite")}
    parts = resolved.parts
    for blocked in blocked_parts:
        for idx in range(0, len(parts) - len(blocked) + 1):
            if tuple(parts[idx : idx + len(blocked)]) == blocked:
                raise ValueError(
                    f"Refusing legacy WebArena split: {resolved}. "
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


def validate_config_alignment(split_path: Path, task_ids: list[int], config_dir: Path) -> None:
    """Fail fast if split metadata points at different configs than config_dir."""
    items = json.loads(split_path.read_text())
    selected = set(task_ids)
    mismatches = []
    missing = []
    for item in items:
        task_id = int(item["task_id"])
        if task_id not in selected or not item.get("config_path"):
            continue
        expected = Path(item["config_path"])
        actual = config_dir / f"{task_id}.json"
        if not expected.exists() or not actual.exists():
            missing.append((task_id, str(expected), str(actual)))
            continue
        expected_json = json.dumps(json.loads(expected.read_text()), sort_keys=True)
        actual_json = json.dumps(json.loads(actual.read_text()), sort_keys=True)
        if expected_json != actual_json:
            mismatches.append((task_id, str(expected), str(actual)))
    if missing or mismatches:
        detail = {
            "missing": missing[:10],
            "mismatches": mismatches[:20],
            "missing_count": len(missing),
            "mismatch_count": len(mismatches),
        }
        raise ValueError(
            f"Config directory is not aligned with split config_path metadata: {json.dumps(detail, indent=2)}"
        )


def eval_tasks(
    *,
    endpoint: str,
    task_ids: list[int],
    config_dir: Path,
    result_root: Path,
    run_name: str,
    instruction_path: str,
    model_name: str,
    mode: str,
    stop_token: str,
) -> dict:
    scores = []
    for task_id in task_ids:
        score = run_episode(
            endpoint=endpoint,
            task_id=task_id,
            config_dir=config_dir,
            result_root=result_root,
            run_name=run_name,
            instruction_path=instruction_path,
            model_name=model_name,
            mode=mode,
            stop_token=stop_token,
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
                # This is the VAB benchmark grader, not a Trace2Skill analysis/evolution model.
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
            return 0.0
        try:
            return max(0.0, float(json.loads(action_paths[-1].read_text()).get("score", 0.0)))
        except Exception:
            return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
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
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--reward-normalization", default="zscore")
    parser.add_argument("--parameter-scope", default="full", choices=["full", "all_linear", "lora"])
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--skip-initial-eval", action="store_true")
    parser.add_argument("--instruction-path", default="agent/prompts/jsons/p_webrl.json")
    parser.add_argument("--model-name", default="Llama-3.1-8B-Instruct")
    parser.add_argument("--mode", default="completion", choices=["completion", "chat"])
    parser.add_argument("--stop-token", default="<|eot_id|>")
    args = parser.parse_args()
    sigma_end = args.sigma_start if args.sigma_end is None else args.sigma_end
    if args.sigma_start < 0 or sigma_end < 0:
        raise ValueError("Sigma values must be non-negative.")
    sigma_warmup_steps = resolve_sigma_warmup_steps(args.generations, args.sigma_warmup_steps)

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
    if not args.split:
        raise ValueError(
            "--split is required for ES training. "
            "Use data/webarena/vab_lite_split/items.json only for VAB/WebRL WebArena-Lite evaluation; "
            "training needs a config-backed split plus --train-config-dir."
        )
    task_ids = load_tasks(Path(args.split), allowed_sites, args.episodes) if args.split else []
    train_config_dir = Path(args.train_config_dir) if args.train_config_dir else None
    if task_ids and train_config_dir is None:
        raise ValueError("--train-config-dir is required when --split is set.")
    if train_config_dir is not None and not train_config_dir.exists():
        raise FileNotFoundError(train_config_dir)

    init = post_json(f"{args.endpoint}/es/init", {"parameter_scope": args.parameter_scope, "verbose": True})
    print(f"[es_init] {init}", flush=True)
    print(
        f"[setting] population={args.population} case_batch_size={args.case_batch_size} "
        f"sigma_start={args.sigma_start} sigma_end={sigma_end} sigma_schedule={args.sigma_schedule} "
        f"sigma_warmup_steps={sigma_warmup_steps} "
        f"alpha={args.alpha} parameter_scope={args.parameter_scope} "
        f"train_config_dir={train_config_dir or ''} eval_config_dir={config_dir}",
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
        sigma_t = sigma_for_generation(
            sigma_start=args.sigma_start,
            sigma_end=sigma_end,
            generation=generation,
            generations=args.generations,
            schedule=args.sigma_schedule,
            warmup_steps=sigma_warmup_steps,
        )
        seeds = []
        rewards = []
        batch_start = (generation * args.case_batch_size) % len(task_ids)
        selected = [task_ids[(batch_start + i) % len(task_ids)] for i in range(args.case_batch_size)]
        print(f"[generation {generation}] sigma={sigma_t:.12g} case_batch={selected}", flush=True)
        sample_records = []
        for i in range(args.population):
            seed = rng.randrange(1, 2**31 - 1)
            seeds.append(seed)
            post_json(f"{args.endpoint}/es/apply", {"seed": seed, "sigma": sigma_t})
            case_scores = []
            try:
                for task_id in selected:
                    score = run_episode(
                        endpoint=args.endpoint,
                        task_id=task_id,
                        config_dir=train_config_dir,
                        result_root=result_root,
                        run_name=f"gen_{generation:03d}_sample_{i:02d}_seed_{seed}",
                        instruction_path=args.instruction_path,
                        model_name=args.model_name,
                        mode=args.mode,
                        stop_token=args.stop_token,
                    )
                    case_scores.append(score)
            finally:
                post_json_retry(f"{args.endpoint}/es/revert", {"seed": seed, "sigma": sigma_t})
            reward = sum(case_scores) / len(case_scores) if case_scores else 0.0
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
            instruction_path=args.instruction_path,
            model_name=args.model_name,
            mode=args.mode,
            stop_token=args.stop_token,
        )
        rec = {
            "generation": generation,
            "case_batch": selected,
            "sigma": sigma_t,
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
