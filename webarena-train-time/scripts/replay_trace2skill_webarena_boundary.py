#!/usr/bin/env python
"""Re-evolve a WebArena boundary skill from curated historical rollouts."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import run_trace2skill_webarena_sft as trace2skill


def trace_log_path(step_dir: Path, result: dict) -> Path:
    outcome = "SUCCEED" if int(result.get("hard", 0)) else "FAILED"
    filename_iid = str(result.get("id")).replace("_", "-")
    return (
        step_dir
        / "update_001"
        / "trace_logs"
        / f"webarena_agent_{filename_iid}_{outcome}.md"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-skill", type=Path, required=True)
    parser.add_argument("--boundary-step", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--optimizer-model", default="gpt-5.4-nano")
    parser.add_argument("--analysis-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260606)
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
    parser.add_argument("--official-prompts", action="store_true")
    args = parser.parse_args()

    source_run = args.source_run.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    skill_dir = output_dir / "skill"
    trace_logs_dir = output_dir / "trace_logs"
    skill_dir.mkdir(parents=True, exist_ok=True)
    trace_logs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.initial_skill.resolve(), skill_dir / "SKILL.md")

    selected_results: list[dict] = []
    reports = []
    for step in range(1, args.boundary_step + 1):
        step_dir = source_run / f"step_{step:03d}"
        results_path = step_dir / "train_results.json"
        if not results_path.is_file():
            raise FileNotFoundError(f"Missing historical results: {results_path}")
        results = trace2skill.load_json(results_path)
        selected, report = trace2skill.select_representative_results(
            results, args.max_steps
        )
        report["step"] = step
        reports.append(report)
        selected_results.extend(selected)
        for result in selected:
            source_log = trace_log_path(step_dir, result)
            if not source_log.is_file():
                raise FileNotFoundError(f"Missing selected trace log: {source_log}")
            destination = trace_logs_dir / f"step_{step:03d}_update_001_{source_log.name}"
            shutil.copy2(source_log, destination)

    summary = {
        "source_run": str(source_run),
        "boundary_step": args.boundary_step,
        "selection_mode": "representative",
        "raw_rollouts": sum(report["raw_rollouts"] for report in reports),
        "selected_rollouts": len(selected_results),
        "selected_positive": sum(report["selected_positive"] for report in reports),
        "selected_negative": sum(report["selected_negative"] for report in reports),
        "excluded_rollouts": {
            key: sum(report["excluded_rollouts"].get(key, 0) for report in reports)
            for key in sorted(
                {
                    key
                    for report in reports
                    for key in report["excluded_rollouts"]
                }
            )
        },
        "steps": reports,
    }
    trace2skill.write_json(output_dir / "trace_selection.json", summary)
    trace2skill.write_json(output_dir / "selected_train_results.json", selected_results)
    trace2skill.write_json(
        output_dir / "manifest.json",
        {
            **summary,
            "optimizer_model": args.optimizer_model,
            "analysis_reasoning_effort": args.analysis_reasoning_effort,
            "skill_reasoning_effort": args.skill_reasoning_effort,
            "consolidation_reasoning_effort": args.consolidation_reasoning_effort,
            "max_steps": args.max_steps,
        },
    )
    print(json.dumps(summary, indent=2), flush=True)

    trace2skill.run_analysis_and_evolve(
        output_dir,
        skill_dir,
        args.optimizer_model,
        args.analysis_workers,
        args.seed,
        official_prompts=args.official_prompts,
        optimizer_generation_config=args.optimizer_generation_config,
        analysis_reasoning_effort=args.analysis_reasoning_effort,
        skill_reasoning_effort=args.skill_reasoning_effort,
        consolidation_reasoning_effort=args.consolidation_reasoning_effort,
        group_records_by_task=False,
    )
    shutil.copy2(skill_dir / "SKILL.md", output_dir / f"skill_step_{args.boundary_step:03d}.md")
    print(json.dumps({"done": True, "skill": str(skill_dir / "SKILL.md")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
