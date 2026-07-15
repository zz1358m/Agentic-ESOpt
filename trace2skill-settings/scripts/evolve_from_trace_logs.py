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
    parser.add_argument("--trace2skill-root", default=os.environ.get("TRACE2SKILL_ROOT", str(DEFAULT_TRACE_SRC)))
    parser.add_argument("--optimizer-model", default=os.environ.get("TRACE2SKILL_OPTIMIZER_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("TRACE2SKILL_WORKERS", "8")))
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--max-skill-lines", type=int, default=int(os.environ.get("TRACE2SKILL_MAX_SKILL_LINES", "20")))
    parser.add_argument("--generation-config", default=os.environ.get("TRACE2SKILL_GENERATION_CONFIG", ""))
    args = parser.parse_args()

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
    shutil.copytree(INITIAL_SKILLS[args.setting], skill_dir)

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
            args.optimizer_model,
            "--base_url",
            env["OPENAI_BASE_URL"],
            "--max_workers",
            str(args.workers),
        ]
        if args.generation_config:
            cmd.extend(["--generation_config", args.generation_config])
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
            args.optimizer_model,
            "--base_url",
            env["OPENAI_BASE_URL"],
            "--max_workers",
            str(args.workers),
        ]
        if args.generation_config:
            cmd.extend(["--generation_config", args.generation_config])
        run_cmd(cmd, trace_src, env, update_dir / "logs" / "success_analysis.log")
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
        args.optimizer_model,
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
    ]
    if args.generation_config:
        cmd.extend(["--generation-config", args.generation_config])
    run_cmd(cmd, trace_src, env, update_dir / "logs" / "evolve.log")
    shutil.copy2(skill_dir / "SKILL.md", out_root / "skill_step_001.md")
    write_json(
        out_root / "manifest.json",
        {
            "setting": args.setting,
            "trace_logs": str(source_logs),
            "trace2skill_root": str(trace_src),
            "optimizer_model": args.optimizer_model,
            "workers": args.workers,
            "max_skill_lines": args.max_skill_lines,
            "output_skill": str(out_root / "skill_step_001.md"),
        },
    )
    print(json.dumps({"done": True, "skill": str(out_root / "skill_step_001.md")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
