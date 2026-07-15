#!/usr/bin/env python
"""Run Trace2Skill on WebArena with WebRL-SFT target rollouts.

The Trace2Skill core is the official analysis/skill_evolver code. This wrapper
only adapts WebArena/VAB trajectories to the official Markdown-log interface.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKILLOPT_SRC = ROOT / "webarena-train-time" / "third_party" / "skillopt"
TRACE_SRC = ROOT / "webarena-train-time" / "methods" / "trace2skill" / "source"
EMPTY_WEB_ARENA_SKILL = """---
name: webarena-sft-trace-skill
description: Skill instructions for WebArena agents using WebRL id actions.
---

# WebArena Skill

"""
_RUN_BATCH = None


def load_run_batch():
    """Load the ignored SkillOpt rollout runtime only when a rollout starts."""

    global _RUN_BATCH
    if _RUN_BATCH is not None:
        return _RUN_BATCH
    rollout_file = SKILLOPT_SRC / "skillopt" / "envs" / "webarena_sft" / "rollout.py"
    if not rollout_file.is_file():
        raise FileNotFoundError(
            "Missing the WebArena rollout runtime at "
            f"{SKILLOPT_SRC}. Install it with the command in data/README.md."
        )
    if str(SKILLOPT_SRC) not in sys.path:
        sys.path.insert(0, str(SKILLOPT_SRC))
    from skillopt.envs.webarena_sft.rollout import run_batch

    _RUN_BATCH = run_batch
    return _RUN_BATCH


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def read_openai_key() -> str:
    key_file = ROOT / "apikey"
    if key_file.exists() and key_file.stat().st_size:
        return key_file.read_text(encoding="utf-8").strip()
    return os.environ.get("OPENAI_API_KEY", "dummy")


def ensure_split(split_dir: Path) -> None:
    if (split_dir / "train" / "items.json").exists() and (split_dir / "val" / "items.json").exists():
        return
    cmd = [
        sys.executable,
        str(ROOT / "webarena-train-time" / "scripts" / "prepare_webarena_nonlite_split.py"),
        "--output-dir",
        str(split_dir),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def load_lite_test_items() -> list[dict]:
    config_dir = ROOT / "data" / "webarena" / "vab-lite" / "config_files" / "wa" / "test_webarena_lite"
    items = []
    for path in sorted(config_dir.glob("*.json"), key=lambda p: int(p.stem)):
        task = load_json(path)
        task_id = int(path.stem)
        sites = [str(site) for site in task.get("sites", [])]
        items.append(
            {
                "id": str(task_id),
                "task_id": task_id,
                "intent": task.get("intent", ""),
                "sites": sites,
                "task_type": sites[0] if sites else "unknown",
                "eval_types": task.get("eval", {}).get("eval_types", []),
                "config_path": str(path.resolve()),
            }
        )
    if len(items) != 165:
        raise RuntimeError(f"Expected 165 WebArena-Lite test items, got {len(items)} from {config_dir}")
    return items


def select_train_items(items: list[dict], epoch: int, per_epoch: int) -> list[dict]:
    if per_epoch <= 0 or per_epoch >= len(items):
        return list(items)
    start = ((epoch - 1) * per_epoch) % len(items)
    doubled = items + items
    return doubled[start: start + per_epoch]


def repeated_items(items: list[dict], samples_per_instance: int) -> list[dict]:
    expanded = []
    for item in items:
        for sample_idx in range(samples_per_instance):
            copy = dict(item)
            copy["id"] = f"{item['id']}_s{sample_idx:02d}"
            copy["sample_idx"] = sample_idx
            expanded.append(copy)
    return expanded


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _truncate_text(text: str, limit: int = 12000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def _format_webarena_rounds(result: dict, pred_dir: Path) -> list[str]:
    trace_paths = sorted((pred_dir / "vab_result" / "traces").glob("*.jsonl"))
    trace_rows = _read_jsonl(trace_paths[0]) if trace_paths else []
    if not trace_rows:
        return []

    lines = []
    for row in trace_rows:
        idx = row.get("index", len(lines))
        prompt = str(row.get("prompt", ""))
        html = _truncate_text(str(row.get("html", "")))
        response = str(row.get("response", ""))
        if idx == 0:
            lines.extend(
                [
                    f"## Round {idx}",
                    "",
                    "### User",
                    "",
                    f"Task Instruction: {row.get('target') or result.get('task_description')}",
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
        else:
            lines.extend(
                [
                    f"## Round {idx}",
                    "",
                    "### User",
                    "",
                    prompt or "** Simplified html **",
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
    return lines


def write_trace_logs(results: list[dict], logs_dir: Path) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        outcome = "SUCCEED" if int(result.get("hard", 0)) else "FAILED"
        iid = str(result.get("id"))
        filename_iid = iid.replace("_", "-")
        actions = []
        pred_dir = Path(result.get("prediction_dir", ""))
        conv_path = pred_dir / "conversation.json"
        if conv_path.exists():
            for msg in load_json(conv_path):
                if "action" in msg:
                    actions.append(str(msg["action"]))
        if not actions and result.get("predicted_answer"):
            actions.append(str(result["predicted_answer"]))

        lines = [
            f"# Chat History {iid}",
            "",
            f"Task ID: {result.get('task_id')}",
            f"Site: {result.get('task_type')}",
            f"Task: {result.get('task_description')}",
            f"Score: {result.get('soft')}",
            f"Outcome: {outcome}",
            f"Failure reason: {result.get('fail_reason', '')}",
            "",
            "## WebArena Execution Trace",
            "",
        ]
        round_lines = _format_webarena_rounds(result, pred_dir)
        if round_lines:
            lines.extend(round_lines)
        else:
            lines.extend(["## Actions", ""])
            for idx, action in enumerate(actions):
                lines.append(f"{idx}. {action}")
        lines.extend(["", "---", "", "## RESULT", outcome, ""])
        (logs_dir / f"webarena_agent_{filename_iid}_{outcome}.md").write_text("\n".join(lines), encoding="utf-8")


def attach_prediction_dirs(results: list[dict], rollout_dir: Path) -> list[dict]:
    for result in results:
        result["prediction_dir"] = str(rollout_dir / "predictions" / str(result["id"]))
    return results


def run_webarena_rollout(
    *,
    items: list[dict],
    out_dir: Path,
    skill_file: Path,
    port: int,
    model_endpoints: str,
    model_name: str,
    instruction_path: str,
    mode: str,
    stop_token: str,
    local_enable_thinking: str,
    temperature: float,
    workers: int,
    timeout: int,
    max_steps: int,
) -> list[dict]:
    old_thinking = os.environ.get("WEBRL_LOCAL_ENABLE_THINKING")
    old_openai_key = os.environ.get("OPENAI_API_KEY")
    old_openai_base = os.environ.get("OPENAI_BASE_URL")
    os.environ["WEBRL_LOCAL_ENABLE_THINKING"] = str(local_enable_thinking)
    os.environ["OPENAI_API_KEY"] = read_openai_key()
    os.environ["OPENAI_BASE_URL"] = old_openai_base or "https://api.openai.com/v1"
    skill = skill_file.read_text(encoding="utf-8")
    try:
        results = load_run_batch()(
            items=items,
            out_root=str(out_dir),
            skill_content=skill,
            webarena_root=str(ROOT / "data" / "webarena" / "vab-lite"),
            python=sys.executable,
            model_endpoint=f"http://127.0.0.1:{port}/completions",
            model_endpoints=model_endpoints,
            model_name=model_name,
            instruction_path=instruction_path,
            max_steps=max_steps,
            max_tokens=2048,
            temperature=temperature,
            top_p=0.9,
            workers=workers,
            task_timeout=timeout,
            mode=mode,
            stop_token=stop_token,
            local_enable_thinking=local_enable_thinking,
        )
    finally:
        if old_thinking is None:
            os.environ.pop("WEBRL_LOCAL_ENABLE_THINKING", None)
        else:
            os.environ["WEBRL_LOCAL_ENABLE_THINKING"] = old_thinking
        if old_openai_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = old_openai_key
        if old_openai_base is None:
            os.environ.pop("OPENAI_BASE_URL", None)
        else:
            os.environ["OPENAI_BASE_URL"] = old_openai_base
    return attach_prediction_dirs(results, out_dir)


def run_cmd(cmd: list[str], cwd: Path, env: dict, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def run_analysis_and_evolve(
    epoch_dir: Path,
    skill_dir: Path,
    model: str,
    workers: int,
    seed: int,
    *,
    official_prompts: bool,
    optimizer_generation_config: str,
) -> None:
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = read_openai_key()
    env["OPENAI_BASE_URL"] = env.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    max_skill_lines = env.get("TRACE2SKILL_MAX_SKILL_LINES", "100")
    max_references = env.get("TRACE2SKILL_MAX_REFERENCES", "5")
    if not official_prompts:
        prompts = ROOT / "webarena-train-time" / "methods" / "trace2skill" / "prompts"
        env["TRACE2SKILL_ERROR_SYSTEM_PROMPT"] = str(prompts / "webarena_error_system.txt")
        env["TRACE2SKILL_ERROR_USER_PROMPT"] = str(prompts / "webarena_error_user.txt")
        env["TRACE2SKILL_SUCCESS_SYSTEM_PROMPT"] = str(prompts / "webarena_success_system.txt")
        env["TRACE2SKILL_SUCCESS_USER_PROMPT"] = str(prompts / "webarena_success_user.txt")

    logs_dir = epoch_dir / "trace_logs"
    err_dir = epoch_dir / "error_analysis"
    succ_dir = epoch_dir / "success_analysis"
    shutil.rmtree(err_dir, ignore_errors=True)
    shutil.rmtree(succ_dir, ignore_errors=True)
    failed_logs = list(logs_dir.glob("*_FAILED.md"))
    success_logs = list(logs_dir.glob("*_SUCCEED.md"))
    if failed_logs:
        cmd = [
                sys.executable,
                "analysis/run_error_analysis_llm.py",
                "--logs_dir",
                str(logs_dir),
                "--output_dir",
                str(err_dir),
                "--model",
                model,
                "--base_url",
                env["OPENAI_BASE_URL"],
                "--max_workers",
                str(workers),
            ]
        if optimizer_generation_config:
            cmd.extend(["--generation_config", optimizer_generation_config])
        run_cmd(
            cmd,
            TRACE_SRC,
            env,
            epoch_dir / "logs" / "error_analysis.log",
        )
    else:
        write_json(err_dir / "parsed_error_records.json", [])
    if success_logs:
        cmd = [
                sys.executable,
                "analysis/run_success_analysis_llm.py",
                "--logs_dir",
                str(logs_dir),
                "--output_dir",
                str(succ_dir),
                "--model",
                model,
                "--base_url",
                env["OPENAI_BASE_URL"],
                "--max_workers",
                str(workers),
            ]
        if optimizer_generation_config:
            cmd.extend(["--generation_config", optimizer_generation_config])
        run_cmd(
            cmd,
            TRACE_SRC,
            env,
            epoch_dir / "logs" / "success_analysis.log",
        )
    else:
        write_json(succ_dir / "parsed_success_records.json", [])

    cmd = [
            sys.executable,
            "-m",
            "skill_evolver.run_parallel_combined_skill_evolution",
            "--error-json",
            str(err_dir / "parsed_error_records.json"),
            "--success-json",
            str(succ_dir / "parsed_success_records.json"),
            "--skill-dir",
            str(skill_dir.resolve()),
            "--model",
            model,
            "--base-url",
            env["OPENAI_BASE_URL"],
            "--max-workers",
            str(workers),
            "--batch-size",
            "1",
            "--merge-batch-size",
            "5",
            "--patch-pipeline",
            "json",
            "--save-intermediates",
            "--intermediates-dir",
            str(epoch_dir / "evolution_intermediates"),
            "--changelog",
            str(epoch_dir / "change.log"),
            "--seed",
            str(seed),
            "--max-skill-lines",
            str(max_skill_lines),
            "--max-references",
            str(max_references),
        ]
    if optimizer_generation_config:
        cmd.extend(["--generation-config", optimizer_generation_config])
    run_cmd(
        cmd,
        TRACE_SRC,
        env,
        epoch_dir / "logs" / "evolve.log",
    )


def score_summary(results: list[dict]) -> dict:
    total = len(results)
    hard = sum(int(r.get("hard", 0)) for r in results)
    soft = sum(float(r.get("soft", 0.0)) for r in results)
    return {
        "total": total,
        "hard": hard,
        "hard_acc": hard / max(total, 1),
        "soft_avg": soft / max(total, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3, help="Deprecated name; interpreted as eval steps.")
    parser.add_argument("--steps", type=int, default=None, help="Number of Trace2Skill eval steps.")
    parser.add_argument("--run-id", default="trace2skill_webarena_sft")
    parser.add_argument("--split-dir", default=str(ROOT / "data" / "webarena" / "vab_nonlite_split"))
    parser.add_argument("--train-instances-per-epoch", type=int, default=8)
    parser.add_argument("--instances-per-eval-step", type=int, default=None)
    parser.add_argument(
        "--instances-per-update",
        type=int,
        default=None,
        help="Distinct train instances used for one Trace2Skill skill update. Defaults to instances-per-eval-step.",
    )
    parser.add_argument(
        "--updates-per-eval-step",
        type=int,
        default=1,
        help="Number of skill updates to run before val/Lite eval.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1,
        help="Run val/Lite eval every N steps and on the final step. Default preserves the old every-step behavior.",
    )
    parser.add_argument("--samples-per-instance", type=int, default=8)
    parser.add_argument("--train-temperature", type=float, default=1.0)
    parser.add_argument("--test-temperature", type=float, default=0.0)
    parser.add_argument("--train-workers", type=int, default=4)
    parser.add_argument("--test-workers", type=int, default=4)
    parser.add_argument("--analysis-workers", type=int, default=8)
    parser.add_argument(
        "--official-prompts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Trace2Skill official analysis prompts. Use --no-official-prompts for WebArena-specific prompts.",
    )
    parser.add_argument(
        "--optimizer-generation-config",
        default="",
        help="Optional generation config JSON/path passed through to official analysis/evolution entrypoints.",
    )
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--empty-skill", action="store_true", help="Initialize the run with an empty WebArena SKILL.md.")
    parser.add_argument("--port", type=int, default=11013)
    parser.add_argument(
        "--model-endpoints",
        default=(
            "http://127.0.0.1:11013/completions "
            "http://127.0.0.1:11014/completions "
            "http://127.0.0.1:11015/completions "
            "http://127.0.0.1:11016/completions"
        ),
    )
    parser.add_argument("--optimizer-model", default="gpt-4.1-mini")
    parser.add_argument("--target-model-name", default="Qwen3-14B")
    parser.add_argument("--instruction-path", default="agent/prompts/jsons/p_webrl_chat.json")
    parser.add_argument("--mode", default="chat")
    parser.add_argument("--stop-token", default="")
    parser.add_argument("--local-enable-thinking", default="false")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260605)
    args = parser.parse_args()
    if not (TRACE_SRC / "skill_evolver").is_dir():
        raise FileNotFoundError(
            f"Missing the Trace2Skill checkout at {TRACE_SRC}. "
            "Install it with the command in data/README.md."
        )
    if args.steps is not None:
        args.epochs = args.steps
    if args.instances_per_eval_step is not None:
        args.train_instances_per_epoch = args.instances_per_eval_step
    if args.instances_per_update is None:
        args.instances_per_update = args.train_instances_per_epoch

    split_dir = Path(args.split_dir)
    ensure_split(split_dir)
    train_items = load_json(split_dir / "train" / "items.json")
    val_items = load_json(split_dir / "val" / "items.json")
    test_items = load_lite_test_items()
    if args.val_limit:
        val_items = val_items[: args.val_limit]
    if args.test_limit:
        test_items = test_items[: args.test_limit]

    out_root = ROOT / "runs" / "trace2skill_webarena_sft" / args.run_id
    skill_dir = out_root / "skill"
    if not skill_dir.exists():
        if args.empty_skill:
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(EMPTY_WEB_ARENA_SKILL, encoding="utf-8")
        else:
            shutil.copytree(ROOT / "webarena-train-time" / "methods" / "trace2skill" / "skills" / "webagent", skill_dir)

    manifest = {
        "target_model": args.target_model_name,
        "target_model_name": args.target_model_name,
        "instruction_path": args.instruction_path,
        "mode": args.mode,
        "local_enable_thinking": args.local_enable_thinking,
        "samples_per_instance": args.samples_per_instance,
        "train_temperature": args.train_temperature,
        "test_temperature": args.test_temperature,
        "train_instances_per_epoch": args.train_instances_per_epoch,
        "instances_per_eval_step": args.train_instances_per_epoch,
        "instances_per_update": args.instances_per_update,
        "updates_per_eval_step": args.updates_per_eval_step,
        "eval_interval": args.eval_interval,
        "max_steps": args.max_steps,
        "val_items": len(val_items),
        "test_items": len(test_items),
        "val_split": "vab_nonlite_val",
        "test_split": "official_vab_webarena_lite_165",
        "model_endpoints": args.model_endpoints.split(),
    }
    write_json(out_root / "manifest.json", manifest)

    step_summaries = []
    global_update = 0
    for epoch in range(1, args.epochs + 1):
        epoch_dir = out_root / f"step_{epoch:03d}"
        eval_train_results = []
        eval_selected = []
        for update_idx in range(1, int(args.updates_per_eval_step) + 1):
            global_update += 1
            update_dir = epoch_dir / f"update_{update_idx:03d}"
            selected = select_train_items(train_items, global_update, args.instances_per_update)
            sampled = repeated_items(selected, args.samples_per_instance)
            eval_selected.extend(selected)
            write_json(update_dir / "train_items.json", selected)
            write_json(update_dir / "sampled_train_items.json", sampled)

            train_results = run_webarena_rollout(
                items=sampled,
                out_dir=update_dir / "train_rollout",
                skill_file=skill_dir / "SKILL.md",
                port=args.port,
                model_endpoints=args.model_endpoints,
                model_name=args.target_model_name,
                instruction_path=args.instruction_path,
                mode=args.mode,
                stop_token=args.stop_token,
                local_enable_thinking=args.local_enable_thinking,
                temperature=args.train_temperature,
                workers=args.train_workers,
                timeout=args.timeout,
                max_steps=args.max_steps,
            )
            eval_train_results.extend(train_results)
            write_json(update_dir / "train_results.json", train_results)
            write_trace_logs(train_results, update_dir / "trace_logs")
            run_analysis_and_evolve(
                update_dir,
                skill_dir,
                args.optimizer_model,
                args.analysis_workers,
                args.seed + global_update,
                official_prompts=args.official_prompts,
                optimizer_generation_config=args.optimizer_generation_config,
            )
            shutil.copy2(skill_dir / "SKILL.md", out_root / f"skill_update_{global_update:04d}.md")

        write_json(epoch_dir / "train_items.json", eval_selected)
        write_json(epoch_dir / "train_results.json", eval_train_results)

        should_eval = args.eval_interval > 0 and (epoch % args.eval_interval == 0 or epoch == args.epochs)
        val_results = []
        test_results = []
        if should_eval:
            val_results = run_webarena_rollout(
                items=val_items,
                out_dir=epoch_dir / "val_rollout",
                skill_file=skill_dir / "SKILL.md",
                port=args.port,
                model_endpoints=args.model_endpoints,
                model_name=args.target_model_name,
                instruction_path=args.instruction_path,
                mode=args.mode,
                stop_token=args.stop_token,
                local_enable_thinking=args.local_enable_thinking,
                temperature=args.test_temperature,
                workers=args.test_workers,
                timeout=args.timeout,
                max_steps=args.max_steps,
            )
            test_results = run_webarena_rollout(
                items=test_items,
                out_dir=epoch_dir / "test_rollout",
                skill_file=skill_dir / "SKILL.md",
                port=args.port,
                model_endpoints=args.model_endpoints,
                model_name=args.target_model_name,
                instruction_path=args.instruction_path,
                mode=args.mode,
                stop_token=args.stop_token,
                local_enable_thinking=args.local_enable_thinking,
                temperature=args.test_temperature,
                workers=args.test_workers,
                timeout=args.timeout,
                max_steps=args.max_steps,
            )
        summary = {
            "step": epoch,
            "updates_completed": global_update,
            "updates_this_step": int(args.updates_per_eval_step),
            "train": score_summary(eval_train_results),
            "eval_ran": should_eval,
            "val": score_summary(val_results) if should_eval else None,
            "test": score_summary(test_results) if should_eval else None,
        }
        step_summaries.append(summary)
        if should_eval:
            write_json(epoch_dir / "val_results.json", val_results)
            write_json(epoch_dir / "test_results.json", test_results)
        write_json(epoch_dir / "summary.json", summary)
        write_json(out_root / "step_summaries.json", step_summaries)
        write_json(out_root / "epoch_summaries.json", step_summaries)
        shutil.copy2(skill_dir / "SKILL.md", out_root / f"skill_step_{epoch:03d}.md")
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
