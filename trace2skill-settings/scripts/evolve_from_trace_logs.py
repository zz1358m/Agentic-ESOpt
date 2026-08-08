#!/usr/bin/env python
"""Run Trace2Skill evolution from prepared Markdown trace logs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACE_SRC = ROOT / "webarena-train-time" / "methods" / "trace2skill" / "source"
INITIAL_SKILLS = {
    "math_reasoning": ROOT / "trace2skill-settings" / "skills" / "math_reasoning",
    "docvqa": ROOT / "trace2skill-settings" / "skills" / "docvqa",
}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def read_openai_key() -> str:
    key_file = ROOT / "apikey"
    if key_file.exists() and key_file.stat().st_size:
        return key_file.read_text(encoding="utf-8").strip()
    return os.environ.get("OPENAI_API_KEY", "dummy")


def run_cmd(cmd: list[str], cwd: Path, env: dict, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", choices=sorted(INITIAL_SKILLS), required=True)
    parser.add_argument("--trace-logs", required=True, help="Directory containing *_FAILED.md and *_SUCCEED.md logs.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--initial-skill",
        default="",
        help="Optional SKILL.md file or skill directory to evolve instead of the setting default.",
    )
    parser.add_argument("--trace2skill-root", default=os.environ.get("TRACE2SKILL_ROOT", str(DEFAULT_TRACE_SRC)))
    parser.add_argument("--optimizer-model", default=os.environ.get("TRACE2SKILL_OPTIMIZER_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--analysis-model", default=os.environ.get("TRACE2SKILL_ANALYSIS_MODEL", ""))
    parser.add_argument("--evolution-model", default=os.environ.get("TRACE2SKILL_EVOLUTION_MODEL", ""))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("TRACE2SKILL_WORKERS", "8")))
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--max-skill-lines", type=int, default=int(os.environ.get("TRACE2SKILL_MAX_SKILL_LINES", "20")))
    parser.add_argument("--generation-config", default=os.environ.get("TRACE2SKILL_GENERATION_CONFIG", ""))
    parser.add_argument(
        "--analysis-generation-config",
        default=os.environ.get("TRACE2SKILL_ANALYSIS_GENERATION_CONFIG", ""),
    )
    parser.add_argument(
        "--evolution-generation-config",
        default=os.environ.get("TRACE2SKILL_EVOLUTION_GENERATION_CONFIG", ""),
    )
    parser.add_argument(
        "--trajectory-manifest",
        default="",
        help="Optional prepare_es_trajectory_logs manifest; aggregates analysis into one MAP unit per task.",
    )
    parser.add_argument(
        "--max-references",
        type=int,
        default=int(os.environ.get("TRACE2SKILL_MAX_REFERENCES", "5")),
    )
    parser.add_argument(
        "--evolution-temperature",
        type=float,
        default=float(os.environ.get("TRACE2SKILL_EVOLUTION_TEMPERATURE", "1.0")),
    )
    args = parser.parse_args()

    analysis_model = args.analysis_model or args.optimizer_model
    evolution_model = args.evolution_model or args.optimizer_model
    analysis_generation_config = args.analysis_generation_config or args.generation_config
    evolution_generation_config = args.evolution_generation_config or args.generation_config

    trace_src = Path(args.trace2skill_root)
    if not trace_src.exists():
        raise FileNotFoundError(f"Trace2Skill source not found: {trace_src}")
    source_logs = Path(args.trace_logs)
    if not source_logs.exists():
        raise FileNotFoundError(source_logs)

    out_root = ROOT / "runs" / "trace2skill_extra" / args.run_id
    update_dir = out_root / "step_001" / "update_001"
    logs_dir = update_dir / "trace_logs"
    shutil.rmtree(logs_dir, ignore_errors=True)
    shutil.copytree(source_logs, logs_dir)

    skill_dir = out_root / "skill"
    if skill_dir.exists():
        shutil.rmtree(skill_dir)
    if args.initial_skill:
        initial_skill = Path(args.initial_skill).expanduser().resolve()
        if initial_skill.is_dir():
            if not (initial_skill / "SKILL.md").is_file():
                raise FileNotFoundError(initial_skill / "SKILL.md")
            shutil.copytree(initial_skill, skill_dir)
        elif initial_skill.is_file():
            skill_dir.mkdir(parents=True)
            shutil.copy2(initial_skill, skill_dir / "SKILL.md")
        else:
            raise FileNotFoundError(initial_skill)
    else:
        initial_skill = INITIAL_SKILLS[args.setting]
        shutil.copytree(initial_skill, skill_dir)

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = read_openai_key()
    env["OPENAI_BASE_URL"] = env.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    env["TRACE2SKILL_MAX_SKILL_LINES"] = str(args.max_skill_lines)

    failed_logs = list(logs_dir.glob("*_FAILED.md"))
    success_logs = list(logs_dir.glob("*_SUCCEED.md"))
    err_dir = update_dir / "error_analysis"
    succ_dir = update_dir / "success_analysis"
    if failed_logs:
        cmd = [
            sys.executable,
            "analysis/run_error_analysis_llm.py",
            "--logs_dir",
            str(logs_dir),
            "--output_dir",
            str(err_dir),
            "--model",
            analysis_model,
            "--base_url",
            env["OPENAI_BASE_URL"],
            "--max_workers",
            str(args.workers),
        ]
        if analysis_generation_config:
            cmd.extend(["--generation_config", analysis_generation_config])
        run_cmd(cmd, trace_src, env, update_dir / "logs" / "error_analysis.log")
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
            analysis_model,
            "--base_url",
            env["OPENAI_BASE_URL"],
            "--max_workers",
            str(args.workers),
        ]
        if analysis_generation_config:
            cmd.extend(["--generation_config", analysis_generation_config])
        run_cmd(cmd, trace_src, env, update_dir / "logs" / "success_analysis.log")
    else:
        write_json(succ_dir / "parsed_success_records.json", [])

    error_json = err_dir / "parsed_error_records.json"
    success_json = succ_dir / "parsed_success_records.json"
    if args.trajectory_manifest:
        evidence_dir = update_dir / "task_evidence"
        error_json = evidence_dir / "task_evidence_records.json"
        success_json = evidence_dir / "empty_success_records.json"
        cmd = [
            sys.executable,
            str(ROOT / "trace2skill-settings" / "scripts" / "aggregate_analysis_records.py"),
            "--manifest",
            str(Path(args.trajectory_manifest).resolve()),
            "--error-json",
            str(err_dir / "parsed_error_records.json"),
            "--success-json",
            str(succ_dir / "parsed_success_records.json"),
            "--output-json",
            str(error_json),
            "--empty-success-json",
            str(success_json),
            "--summary-json",
            str(evidence_dir / "summary.json"),
        ]
        run_cmd(cmd, ROOT, env, update_dir / "logs" / "aggregate.log")

    cmd = [
        sys.executable,
        "-m",
        "skill_evolver.run_parallel_combined_skill_evolution",
        "--error-json",
        str(error_json),
        "--success-json",
        str(success_json),
        "--skill-dir",
        str(skill_dir.resolve()),
        "--model",
        evolution_model,
        "--base-url",
        env["OPENAI_BASE_URL"],
        "--max-workers",
        str(args.workers),
        "--batch-size",
        "1",
        "--merge-batch-size",
        "5",
        "--patch-pipeline",
        "json",
        "--save-intermediates",
        "--intermediates-dir",
        str(update_dir / "evolution_intermediates"),
        "--changelog",
        str(update_dir / "change.log"),
        "--seed",
        str(args.seed),
        "--max-skill-lines",
        str(args.max_skill_lines),
        "--max-references",
        str(args.max_references),
        "--temperature",
        str(args.evolution_temperature),
    ]
    if evolution_generation_config:
        cmd.extend(["--generation-config", evolution_generation_config])
    run_cmd(cmd, trace_src, env, update_dir / "logs" / "evolve.log")
    generated_skill = skill_dir / "SKILL.md"
    if len(generated_skill.read_text(encoding="utf-8").splitlines()) > args.max_skill_lines:
        pre_compression = update_dir / "skill_before_hard_compression.md"
        shutil.copy2(generated_skill, pre_compression)
        cmd = [
            sys.executable,
            str(ROOT / "trace2skill-settings" / "scripts" / "compress_skill_markdown.py"),
            "--input",
            str(pre_compression),
            "--output",
            str(generated_skill),
            "--model",
            evolution_model,
            "--base-url",
            env["OPENAI_BASE_URL"],
            "--max-lines",
            str(args.max_skill_lines),
            "--generation-config",
            evolution_generation_config or "{}",
            "--attempts",
            "5",
        ]
        run_cmd(cmd, ROOT, env, update_dir / "logs" / "hard_compression.log")
    shutil.copy2(skill_dir / "SKILL.md", out_root / "skill_step_001.md")
    write_json(
        out_root / "manifest.json",
        {
            "setting": args.setting,
            "trace_logs": str(source_logs),
            "initial_skill": str(initial_skill),
            "trace2skill_root": str(trace_src),
            "optimizer_model": args.optimizer_model,
            "analysis_model": analysis_model,
            "evolution_model": evolution_model,
            "analysis_generation_config": analysis_generation_config,
            "evolution_generation_config": evolution_generation_config,
            "workers": args.workers,
            "max_skill_lines": args.max_skill_lines,
            "max_references": args.max_references,
            "evolution_temperature": args.evolution_temperature,
            "trajectory_manifest": args.trajectory_manifest,
            "output_skill": str(out_root / "skill_step_001.md"),
        },
    )
    print(json.dumps({"done": True, "skill": str(out_root / "skill_step_001.md")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
