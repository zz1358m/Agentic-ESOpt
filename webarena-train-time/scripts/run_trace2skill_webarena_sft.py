#!/usr/bin/env python
"""Run Trace2Skill on WebArena with WebRL-SFT target rollouts.

The Trace2Skill core is the official analysis/skill_evolver code. This wrapper
only adapts WebArena/VAB trajectories to the official Markdown-log interface.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
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


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def generation_config_with_effort(config: str, effort: str | None) -> str:
    """Return a generation-config JSON string with a phase effort override."""
    if not effort:
        return config
    value: dict = {}
    if config:
        config_path = Path(config)
        raw = config_path.read_text(encoding="utf-8") if config_path.is_file() else config
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Optimizer generation config must be a JSON object.")
        value.update(parsed)
    value["reasoning_effort"] = effort
    return json.dumps(value, separators=(",", ":"))


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
        str(ROOT / "webarena-train-time" / "scripts" / "prepare_webarena_nonlite_skillopt_split.py"),
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


_INFRASTRUCTURE_FAILURE_PATTERN = re.compile(
    r"timeout(?:expired)?|timed out|connection(?: reset| refused)|"
    r"server disconnected|broken pipe|process exited|calledprocesserror",
    re.IGNORECASE,
)


def trace_result_class(result: dict, max_steps: int) -> str:
    """Classify a rollout before it is allowed to influence skill evolution."""
    fail_reason = str(result.get("fail_reason") or "")
    if result.get("agent_ok") is False or _INFRASTRUCTURE_FAILURE_PATTERN.search(fail_reason):
        return "infrastructure"

    hard = int(result.get("hard", 0))
    if hard:
        return "positive"

    turns = int(result.get("n_turns") or 0)
    if max_steps > 0 and turns >= max_steps:
        return "turn_limit"

    if not result_has_usable_trace(result):
        return "empty_trace"
    return "negative"


def result_has_usable_trace(result: dict) -> bool:
    """Return whether a rollout contains agent behavior worth analyzing."""
    if str(result.get("predicted_answer") or "").strip():
        return True

    prediction_dir = Path(str(result.get("prediction_dir") or ""))
    if not prediction_dir.is_dir():
        return False
    conversation = prediction_dir / "conversation.json"
    if conversation.is_file():
        try:
            if any(str(message.get("action") or "").strip() for message in load_json(conversation)):
                return True
        except (json.JSONDecodeError, OSError, TypeError):
            pass
    return any((prediction_dir / "vab_result" / "traces").glob("*.jsonl"))


def select_representative_results(
    results: list[dict], max_steps: int
) -> tuple[list[dict], dict]:
    """Select at most one positive and one valid negative per base task."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for result in results:
        grouped[str(result.get("task_id"))].append(result)

    selected: list[dict] = []
    selected_roles: dict[str, str] = {}
    excluded = Counter()
    task_outcomes = Counter()

    for task_id in sorted(grouped, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value)):
        classified = [(result, trace_result_class(result, max_steps)) for result in grouped[task_id]]
        positives = [result for result, label in classified if label == "positive"]
        negatives = [result for result, label in classified if label == "negative"]
        for _result, label in classified:
            if label not in {"positive", "negative"}:
                excluded[label] += 1

        if positives and negatives:
            task_outcomes["mixed"] += 1
        elif positives:
            task_outcomes["all_positive"] += 1
        elif negatives:
            task_outcomes["all_negative"] += 1
        else:
            task_outcomes["no_usable_trace"] += 1

        if positives:
            positive = min(
                positives,
                key=lambda result: (
                    int(result.get("n_turns") or 0),
                    float(result.get("wall_time_s") or 0.0),
                    str(result.get("id")),
                ),
            )
            selected.append(positive)
            selected_roles[str(positive.get("id"))] = "positive"

        if negatives:
            negative = max(
                negatives,
                key=lambda result: (
                    int(result.get("n_turns") or 0),
                    len(str(result.get("predicted_answer") or "")),
                    str(result.get("id")),
                ),
            )
            selected.append(negative)
            selected_roles[str(negative.get("id"))] = "negative"

    selected.sort(key=lambda result: (str(result.get("task_id")), selected_roles[str(result.get("id"))]))
    report = {
        "raw_rollouts": len(results),
        "task_count": len(grouped),
        "selected_rollouts": len(selected),
        "selected_positive": sum(role == "positive" for role in selected_roles.values()),
        "selected_negative": sum(role == "negative" for role in selected_roles.values()),
        "task_outcomes": dict(sorted(task_outcomes.items())),
        "excluded_rollouts": dict(sorted(excluded.items())),
        "selected": [
            {
                "id": str(result.get("id")),
                "task_id": result.get("task_id"),
                "role": selected_roles[str(result.get("id"))],
                "n_turns": result.get("n_turns"),
            }
            for result in selected
        ],
    }
    return selected, report


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


def _compact_html_for_response(text: str, response: str, limit: int = 12000) -> str:
    if len(text) <= limit:
        return text

    element_match = re.search(r'element\s*=\s*["\']([^"\']+)["\']', response)
    anchor = -1
    if element_match:
        element_id = re.escape(element_match.group(1))
        id_match = re.search(rf'\bid\s*=\s*["\']{element_id}["\']', text)
        if id_match:
            anchor = id_match.start()

    marker = "\n...[HTML omitted]...\n"
    content_budget = max(1, limit - 2 * len(marker))
    if anchor < 0:
        head_size = content_budget // 2
        return text[:head_size] + marker + text[-(content_budget - head_size) :]

    head_size = content_budget // 4
    tail_size = content_budget // 4
    target_size = content_budget - head_size - tail_size
    target_start = max(0, anchor - target_size // 2)
    target_end = min(len(text), target_start + target_size)
    target_start = max(0, target_end - target_size)
    sections = [(0, head_size), (target_start, target_end), (len(text) - tail_size, len(text))]
    merged: list[tuple[int, int]] = []
    for start, end in sorted(sections):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    compacted = marker.join(text[start:end] for start, end in merged)
    return compacted[:limit]


def _format_webarena_rounds(result: dict, pred_dir: Path) -> list[str]:
    trace_paths = sorted((pred_dir / "vab_result" / "traces").glob("*.jsonl"))
    trace_rows = _read_jsonl(trace_paths[0]) if trace_paths else []
    if not trace_rows:
        return []

    lines = []
    for row in trace_rows:
        idx = row.get("index", len(lines))
        prompt = str(row.get("prompt", ""))
        response = str(row.get("response", ""))
        html = _compact_html_for_response(str(row.get("html", "")), response)
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
    top_p: float,
    top_k: int | None,
    min_p: float | None,
    presence_penalty: float,
    repetition_penalty: float,
    workers: int,
    timeout: int,
    max_steps: int,
) -> list[dict]:
    if str(SKILLOPT_SRC) not in sys.path:
        sys.path.insert(0, str(SKILLOPT_SRC))
    try:
        from skillopt.envs.webarena_sft.rollout import run_batch
    except ModuleNotFoundError as exc:
        if exc.name == "skillopt":
            raise RuntimeError(
                "Missing SkillOpt WebArena runtime. Clone it into "
                f"{SKILLOPT_SRC} as described in data/README.md."
            ) from exc
        raise

    old_thinking = os.environ.get("WEBRL_LOCAL_ENABLE_THINKING")
    old_openai_key = os.environ.get("OPENAI_API_KEY")
    old_openai_base = os.environ.get("OPENAI_BASE_URL")
    os.environ["WEBRL_LOCAL_ENABLE_THINKING"] = str(local_enable_thinking)
    os.environ["OPENAI_API_KEY"] = read_openai_key()
    os.environ["OPENAI_BASE_URL"] = old_openai_base or "https://api.openai.com/v1"
    skill = skill_file.read_text(encoding="utf-8")
    try:
        results = run_batch(
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
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
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
    analysis_reasoning_effort: str | None = None,
    skill_reasoning_effort: str | None = None,
    consolidation_reasoning_effort: str | None = None,
    group_records_by_task: bool = True,
) -> None:
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = read_openai_key()
    env["OPENAI_BASE_URL"] = env.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    env["TRACE2SKILL_SKILL_DOMAIN"] = "WebArena web-agent"
    env["TRACE2SKILL_RUNTIME_POLICY"] = (
        "Use only the available WebRL actions and visible page evidence. "
        "Before exiting, satisfy the task's full completion contract, including requested cardinality, "
        "ties, multiple entities, and persisted state when applicable. When that complete answer is "
        "visible, exit immediately with it. When the requested state-changing task is fully and "
        "visibly confirmed complete, exit immediately. "
        "Prefer the shortest valid action sequence; learned guidance must not require extra "
        "exploration or verification after these completion conditions are satisfied."
    )
    max_skill_lines = env.get("TRACE2SKILL_MAX_SKILL_LINES", "50")
    max_skill_tokens = env.get("TRACE2SKILL_MAX_SKILL_TOKENS", "0")
    max_references = env.get("TRACE2SKILL_MAX_REFERENCES", "5")
    analysis_generation_config = generation_config_with_effort(
        optimizer_generation_config, analysis_reasoning_effort
    )
    skill_generation_config = generation_config_with_effort(
        optimizer_generation_config, skill_reasoning_effort
    )
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
        if analysis_generation_config:
            cmd.extend(["--generation_config", analysis_generation_config])
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
        if analysis_generation_config:
            cmd.extend(["--generation_config", analysis_generation_config])
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
            "--max-skill-tokens",
            str(max_skill_tokens),
            "--max-references",
            str(max_references),
            "--max-verification-rounds",
            "0",
        ]
    if group_records_by_task:
        cmd.append("--group-records-by-task")
    if skill_generation_config:
        cmd.extend(["--generation-config", skill_generation_config])
    if skill_reasoning_effort:
        cmd.extend(["--reasoning-effort", skill_reasoning_effort])
    if consolidation_reasoning_effort:
        cmd.extend(
            ["--consolidation-reasoning-effort", consolidation_reasoning_effort]
        )
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
    parser.add_argument(
        "--start-step",
        type=int,
        default=1,
        help="First global step to execute when continuing from a replayed skill.",
    )
    parser.add_argument(
        "--initial-skill",
        type=Path,
        default=None,
        help="SKILL.md file or skill directory used to initialize a new run.",
    )
    parser.add_argument(
        "--eval-initial-skill",
        action="store_true",
        help="Evaluate a replayed boundary skill before executing --start-step.",
    )
    parser.add_argument("--run-id", default="trace2skill_webarena_sft")
    parser.add_argument("--split-dir", default=str(ROOT / "data" / "webarena" / "skillopt_nonlite_sft"))
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
        "--skill-update-interval",
        type=int,
        default=1,
        help="Aggregate rollout traces and evolve the skill every N steps.",
    )
    parser.add_argument(
        "--evolve-at-end-only",
        action="store_true",
        help=(
            "Keep the initial skill fixed while collecting all rollout traces, "
            "then run skill evolution once after the final training step."
        ),
    )
    parser.add_argument(
        "--preload-trace-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory of previously selected trace markdown files to "
            "include in the next skill evolution pool."
        ),
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1,
        help="Run val/Lite eval every N steps and on the final step. Default preserves the old every-step behavior.",
    )
    parser.add_argument("--samples-per-instance", type=int, default=8)
    parser.add_argument(
        "--trace-selection",
        choices=["all", "representative"],
        default="representative",
        help=(
            "Trace evidence admitted to skill evolution. Representative keeps at "
            "most one positive and one valid negative per base task."
        ),
    )
    parser.add_argument("--train-temperature", type=float, default=1.0)
    parser.add_argument("--test-temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
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
    parser.add_argument(
        "--analysis-reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh"],
        default=None,
    )
    parser.add_argument(
        "--skill-reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh"],
        default=None,
    )
    parser.add_argument(
        "--consolidation-reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh"],
        default=None,
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
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument(
        "--eval-timeout",
        type=int,
        default=1200,
        help="Per-task timeout in seconds for validation and test rollouts.",
    )
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260605)
    args = parser.parse_args()
    if args.steps is not None:
        args.epochs = args.steps
    if args.instances_per_eval_step is not None:
        args.train_instances_per_epoch = args.instances_per_eval_step
    if args.instances_per_update is None:
        args.instances_per_update = args.train_instances_per_epoch
    if args.skill_update_interval <= 0:
        raise ValueError("--skill-update-interval must be positive.")
    if args.start_step < 1 or args.start_step > args.epochs:
        raise ValueError("--start-step must be between 1 and --steps/--epochs.")
    if (
        not args.evolve_at_end_only
        and (args.start_step - 1) % args.skill_update_interval != 0
    ):
        raise ValueError(
            "--start-step must follow a skill-update boundary so update seeds stay aligned."
        )
    if args.eval_initial_skill and args.start_step == 1:
        raise ValueError("--eval-initial-skill requires --start-step greater than 1.")

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
    preload_trace_paths: list[Path] = []
    if args.preload_trace_dir is not None:
        preload_trace_dir = args.preload_trace_dir.resolve()
        if not preload_trace_dir.is_dir():
            raise FileNotFoundError(
                f"Preloaded trace directory not found: {preload_trace_dir}"
            )
        preload_trace_paths = sorted(preload_trace_dir.glob("*.md"))
        if not preload_trace_paths:
            raise FileNotFoundError(
                f"No trace markdown files found in: {preload_trace_dir}"
            )
    if not skill_dir.exists():
        if args.initial_skill is not None:
            initial_skill = args.initial_skill.resolve()
            if initial_skill.is_dir():
                shutil.copytree(initial_skill, skill_dir)
            elif initial_skill.is_file():
                skill_dir.mkdir(parents=True)
                shutil.copy2(initial_skill, skill_dir / "SKILL.md")
            else:
                raise FileNotFoundError(f"Initial skill not found: {initial_skill}")
        elif args.empty_skill:
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(EMPTY_WEB_ARENA_SKILL, encoding="utf-8")
        else:
            shutil.copytree(ROOT / "webarena-train-time" / "methods" / "trace2skill" / "skills" / "webagent", skill_dir)

    manifest = {
        "target_model": args.target_model_name,
        "target_model_name": args.target_model_name,
        "optimizer_model": args.optimizer_model,
        "analysis_reasoning_effort": args.analysis_reasoning_effort,
        "skill_reasoning_effort": args.skill_reasoning_effort,
        "consolidation_reasoning_effort": args.consolidation_reasoning_effort,
        "instruction_path": args.instruction_path,
        "mode": args.mode,
        "local_enable_thinking": args.local_enable_thinking,
        "samples_per_instance": args.samples_per_instance,
        "trace_selection": args.trace_selection,
        "steps": args.epochs,
        "start_step": args.start_step,
        "initial_skill": str(args.initial_skill.resolve()) if args.initial_skill else None,
        "eval_initial_skill": args.eval_initial_skill,
        "train_temperature": args.train_temperature,
        "test_temperature": args.test_temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "train_instances_per_epoch": args.train_instances_per_epoch,
        "instances_per_eval_step": args.train_instances_per_epoch,
        "instances_per_update": args.instances_per_update,
        "updates_per_eval_step": args.updates_per_eval_step,
        "skill_update_interval": args.skill_update_interval,
        "evolve_at_end_only": args.evolve_at_end_only,
        "preload_trace_dir": (
            str(args.preload_trace_dir.resolve()) if args.preload_trace_dir else None
        ),
        "preloaded_trace_count": len(preload_trace_paths),
        "eval_interval": args.eval_interval,
        "train_timeout": args.timeout,
        "eval_timeout": args.eval_timeout,
        "max_steps": args.max_steps,
        "max_tokens": 2048,
        "max_skill_lines": int(os.environ.get("TRACE2SKILL_MAX_SKILL_LINES", "50")),
        "max_skill_tokens": int(os.environ.get("TRACE2SKILL_MAX_SKILL_TOKENS", "0")),
        "max_references": int(os.environ.get("TRACE2SKILL_MAX_REFERENCES", "5")),
        "max_verification_rounds": 0,
        "val_items": len(val_items),
        "test_items": len(test_items),
        "val_split": "vab_nonlite_val",
        "test_split": "official_vab_webarena_lite_165",
        "model_endpoints": args.model_endpoints.split(),
    }
    write_json(out_root / "manifest.json", manifest)

    step_summaries = []
    global_update = args.start_step - 1
    skill_update_count = (
        0
        if args.evolve_at_end_only
        else (args.start_step - 1) // args.skill_update_interval
    )
    pending_trace_logs = out_root / "pending_trace_logs"
    shutil.rmtree(pending_trace_logs, ignore_errors=True)
    pending_trace_logs.mkdir(parents=True)
    for trace_index, trace_path in enumerate(preload_trace_paths, start=1):
        preload_name = f"preload_{trace_index:05d}_{trace_path.name}"
        shutil.copy2(trace_path, pending_trace_logs / preload_name)
    if args.eval_initial_skill:
        boundary_step = args.start_step - 1
        boundary_dir = out_root / f"replay_boundary_eval_step_{boundary_step:03d}"
        val_results = run_webarena_rollout(
            items=val_items,
            out_dir=boundary_dir / "val_rollout",
            skill_file=skill_dir / "SKILL.md",
            port=args.port,
            model_endpoints=args.model_endpoints,
            model_name=args.target_model_name,
            instruction_path=args.instruction_path,
            mode=args.mode,
            stop_token=args.stop_token,
            local_enable_thinking=args.local_enable_thinking,
            temperature=args.test_temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            presence_penalty=args.presence_penalty,
            repetition_penalty=args.repetition_penalty,
            workers=args.test_workers,
            timeout=args.eval_timeout,
            max_steps=args.max_steps,
        )
        test_results = run_webarena_rollout(
            items=test_items,
            out_dir=boundary_dir / "test_rollout",
            skill_file=skill_dir / "SKILL.md",
            port=args.port,
            model_endpoints=args.model_endpoints,
            model_name=args.target_model_name,
            instruction_path=args.instruction_path,
            mode=args.mode,
            stop_token=args.stop_token,
            local_enable_thinking=args.local_enable_thinking,
            temperature=args.test_temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            presence_penalty=args.presence_penalty,
            repetition_penalty=args.repetition_penalty,
            workers=args.test_workers,
            timeout=args.eval_timeout,
            max_steps=args.max_steps,
        )
        boundary_summary = {
            "step": boundary_step,
            "source": "replayed_boundary_skill",
            "val": score_summary(val_results),
            "test": score_summary(test_results),
        }
        write_json(boundary_dir / "val_results.json", val_results)
        write_json(boundary_dir / "test_results.json", test_results)
        write_json(boundary_dir / "summary.json", boundary_summary)
        print(json.dumps(boundary_summary, indent=2), flush=True)
    for epoch in range(args.start_step, args.epochs + 1):
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
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                presence_penalty=args.presence_penalty,
                repetition_penalty=args.repetition_penalty,
                workers=args.train_workers,
                timeout=args.timeout,
                max_steps=args.max_steps,
            )
            eval_train_results.extend(train_results)
            write_json(update_dir / "train_results.json", train_results)
            write_trace_logs(train_results, update_dir / "trace_logs")
            if args.trace_selection == "representative":
                evolution_results, selection_report = select_representative_results(
                    train_results, args.max_steps
                )
            else:
                evolution_results = train_results
                selection_report = {
                    "raw_rollouts": len(train_results),
                    "task_count": len({str(result.get("task_id")) for result in train_results}),
                    "selected_rollouts": len(train_results),
                    "mode": "all",
                }
            write_json(update_dir / "selected_train_results.json", evolution_results)
            write_json(update_dir / "trace_selection.json", selection_report)
            selected_logs_dir = update_dir / "selected_trace_logs"
            shutil.rmtree(selected_logs_dir, ignore_errors=True)
            selected_logs_dir.mkdir(parents=True)
            for result in evolution_results:
                outcome = "SUCCEED" if int(result.get("hard", 0)) else "FAILED"
                filename_iid = str(result.get("id")).replace("_", "-")
                source_trace = update_dir / "trace_logs" / f"webarena_agent_{filename_iid}_{outcome}.md"
                if not source_trace.is_file():
                    raise FileNotFoundError(f"Selected trace log not found: {source_trace}")
                shutil.copy2(source_trace, selected_logs_dir / source_trace.name)
            for trace_path in selected_logs_dir.glob("*.md"):
                aggregate_name = (
                    f"step_{epoch:03d}_update_{update_idx:03d}_{trace_path.name}"
                )
                shutil.copy2(trace_path, pending_trace_logs / aggregate_name)

        write_json(epoch_dir / "train_items.json", eval_selected)
        write_json(epoch_dir / "train_results.json", eval_train_results)

        should_update_skill = (
            epoch == args.epochs
            if args.evolve_at_end_only
            else epoch % args.skill_update_interval == 0 or epoch == args.epochs
        )
        if should_update_skill:
            skill_update_count += 1
            evolution_dir = epoch_dir / "skill_update"
            shutil.rmtree(evolution_dir / "trace_logs", ignore_errors=True)
            shutil.copytree(pending_trace_logs, evolution_dir / "trace_logs")
            run_analysis_and_evolve(
                evolution_dir,
                skill_dir,
                args.optimizer_model,
                args.analysis_workers,
                args.seed + skill_update_count,
                official_prompts=args.official_prompts,
                optimizer_generation_config=args.optimizer_generation_config,
                analysis_reasoning_effort=args.analysis_reasoning_effort,
                skill_reasoning_effort=args.skill_reasoning_effort,
                consolidation_reasoning_effort=args.consolidation_reasoning_effort,
                group_records_by_task=True,
            )
            shutil.copy2(
                skill_dir / "SKILL.md",
                out_root / f"skill_update_{skill_update_count:04d}.md",
            )
            shutil.rmtree(pending_trace_logs)
            pending_trace_logs.mkdir(parents=True)

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
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                presence_penalty=args.presence_penalty,
                repetition_penalty=args.repetition_penalty,
                workers=args.test_workers,
                timeout=args.eval_timeout,
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
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                presence_penalty=args.presence_penalty,
                repetition_penalty=args.repetition_penalty,
                workers=args.test_workers,
                timeout=args.eval_timeout,
                max_steps=args.max_steps,
            )
        summary = {
            "step": epoch,
            "updates_completed": global_update,
            "updates_this_step": int(args.updates_per_eval_step),
            "skill_updated": should_update_skill,
            "skill_updates_completed": skill_update_count,
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
