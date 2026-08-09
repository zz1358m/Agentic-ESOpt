#!/usr/bin/env python
"""Evolve a WebArena skill from existing ES rollout traces."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRACE_WRAPPER_PATH = ROOT / "webarena-train-time" / "scripts" / "run_trace2skill_webarena_sft.py"

spec = importlib.util.spec_from_file_location("trace2skill_webarena_sft", TRACE_WRAPPER_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {TRACE_WRAPPER_PATH}")
trace2skill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace2skill)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def parse_score(task_dir: Path) -> float:
    actions_files = sorted((task_dir / "actions").glob("*.json"))
    for path in actions_files:
        try:
            return float(json.loads(path.read_text(encoding="utf-8")).get("score", 0.0))
        except Exception:
            pass
    log_path = task_dir / "run.log"
    if log_path.exists():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        matches = re.findall(r"Average score:\s*(-?[0-9.]+)", text)
        if matches:
            return float(matches[-1])
    return 0.0


def truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def write_one_trace_log(task_dir: Path, output_path: Path, html_limit: int) -> dict | None:
    trace_files = sorted((task_dir / "traces").glob("*.jsonl"))
    if not trace_files:
        return None
    rows = read_jsonl(trace_files[0])
    if not rows:
        return None
    score = parse_score(task_dir)
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


def generation_number(path: Path) -> int:
    match = re.match(r"gen_(\d+)_sample_", path.name)
    return int(match.group(1)) if match else -1


def sample_number(path: Path) -> int:
    match = re.match(r"gen_\d+_sample_(\d+)_", path.name)
    if match is None:
        raise ValueError(f"Cannot parse ES sample number from {path.name}")
    return int(match.group(1))


def last_completed_generation(es_run_dir: Path) -> int | None:
    history_path = es_run_dir / "history.json"
    if not history_path.exists():
        return None
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    generations = [int(row["generation"]) for row in history if "generation" in row and int(row["generation"]) >= 0]
    return max(generations) if generations else None


def collect_task_dirs(es_run_dir: Path, generations: int, max_traces: int) -> list[Path]:
    sample_dirs = [p for p in es_run_dir.glob("gen_*_sample_*") if p.is_dir()]
    last_done = last_completed_generation(es_run_dir)
    if last_done is not None:
        sample_dirs = [p for p in sample_dirs if generation_number(p) <= last_done]
    sample_dirs.sort(key=lambda p: (generation_number(p), p.name), reverse=True)
    if generations > 0:
        keep_gens = sorted({generation_number(p) for p in sample_dirs}, reverse=True)[:generations]
        sample_dirs = [p for p in sample_dirs if generation_number(p) in keep_gens]
    task_dirs = []
    for sample_dir in sample_dirs:
        task_dirs.extend(sorted([p for p in sample_dir.glob("task_*") if p.is_dir()]))
    task_dirs.sort(key=lambda p: (generation_number(p.parent), p.parent.name, p.name), reverse=True)
    if max_traces > 0:
        task_dirs = task_dirs[:max_traces]
    return task_dirs


def prepare_skill_dir(skill_dir: Path, initial_skill: str, empty_skill: bool) -> None:
    if skill_dir.exists():
        return
    skill_dir.mkdir(parents=True, exist_ok=True)
    if empty_skill:
        (skill_dir / "SKILL.md").write_text(trace2skill.EMPTY_WEB_ARENA_SKILL, encoding="utf-8")
    elif initial_skill:
        src = Path(initial_skill)
        if src.is_dir():
            shutil.copytree(src, skill_dir, dirs_exist_ok=True)
        else:
            shutil.copy2(src, skill_dir / "SKILL.md")
    else:
        shutil.copytree(
            ROOT / "webarena-train-time" / "methods" / "trace2skill" / "skills" / "webagent",
            skill_dir,
            dirs_exist_ok=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--es-run-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--initial-skill", default="")
    parser.add_argument(
        "--empty-skill",
        action="store_true",
        help="Start from the empty WebArena skill instead of the bundled seed skill.",
    )
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--max-traces", type=int, default=256)
    parser.add_argument("--html-limit", type=int, default=12000)
    parser.add_argument("--optimizer-model", default="gpt-4.1-mini")
    parser.add_argument("--analysis-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument(
        "--official-prompts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use upstream Trace2Skill prompts instead of the committed WebArena prompts.",
    )
    parser.add_argument("--optimizer-generation-config", default="")
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
    args = parser.parse_args()
    if args.empty_skill and args.initial_skill:
        raise ValueError("Use either --empty-skill or --initial-skill, not both.")

    es_run_dir = Path(args.es_run_dir)
    if not es_run_dir.exists():
        raise FileNotFoundError(es_run_dir)

    out_root = ROOT / "runs" / "trace2skill_webarena_sft" / args.run_id
    update_dir = out_root / "step_001" / "update_001"
    logs_dir = update_dir / "trace_logs"
    if logs_dir.exists():
        shutil.rmtree(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    skill_dir = out_root / "skill"
    prepare_skill_dir(skill_dir, args.initial_skill, args.empty_skill)

    task_dirs = collect_task_dirs(es_run_dir, args.generations, args.max_traces)
    records = []
    for task_dir in task_dirs:
        score = parse_score(task_dir)
        outcome = "SUCCEED" if score >= 1.0 else "FAILED"
        task_id = task_dir.name.removeprefix("task_")
        generation = generation_number(task_dir.parent)
        sample_id = f"g{generation:03d}-{task_id}-s{sample_number(task_dir.parent):02d}"
        record = write_one_trace_log(
            task_dir,
            logs_dir / f"webarena_agent_{sample_id}_{outcome}.md",
            args.html_limit,
        )
        if record:
            records.append(record)
    if not records:
        raise RuntimeError(f"No usable WebArena trajectories found under {es_run_dir}.")
    trace2skill.write_json(update_dir / "source_traces.json", records)
    trace2skill.write_json(
        out_root / "manifest.json",
        {
            "source": "es_traces",
            "es_run_dir": str(es_run_dir),
            "initial_skill": args.initial_skill,
            "empty_skill": args.empty_skill,
            "generations": args.generations,
            "max_traces": args.max_traces,
            "html_limit": args.html_limit,
            "trace_count": len(records),
            "optimizer_model": args.optimizer_model,
            "analysis_workers": args.analysis_workers,
            "official_prompts": args.official_prompts,
            "analysis_reasoning_effort": args.analysis_reasoning_effort,
            "skill_reasoning_effort": args.skill_reasoning_effort,
            "consolidation_reasoning_effort": args.consolidation_reasoning_effort,
            "seed": args.seed,
            "max_skill_lines": int(os.environ.get("TRACE2SKILL_MAX_SKILL_LINES", "50")),
            "max_skill_tokens": int(os.environ.get("TRACE2SKILL_MAX_SKILL_TOKENS", "0")),
            "max_references": int(os.environ.get("TRACE2SKILL_MAX_REFERENCES", "5")),
        },
    )
    print(json.dumps({"trace_logs": len(records), "out_root": str(out_root)}, indent=2), flush=True)
    trace2skill.run_analysis_and_evolve(
        update_dir,
        skill_dir,
        args.optimizer_model,
        args.analysis_workers,
        args.seed,
        official_prompts=args.official_prompts,
        optimizer_generation_config=args.optimizer_generation_config,
        analysis_reasoning_effort=args.analysis_reasoning_effort,
        skill_reasoning_effort=args.skill_reasoning_effort,
        consolidation_reasoning_effort=args.consolidation_reasoning_effort,
    )
    shutil.copy2(skill_dir / "SKILL.md", out_root / "skill_step_001.md")
    print(json.dumps({"done": True, "skill": str(out_root / "skill_step_001.md")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
