#!/usr/bin/env python3
"""Build the auditable final report for the fixed DAPO-400 Math GRPO run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
CURVE_METRICS = {
    "training/epoch",
    "actor/lr",
    "actor/entropy",
    "actor/kl_loss",
    "actor/pg_loss",
    "actor/grad_norm",
    "critic/score/mean",
    "critic/rewards/mean",
    "response_length/mean",
    "response_length/max",
    "num_turns/mean",
    "num_turns/max",
    "timing_s/gen",
    "timing_s/update_actor",
    "timing_s/step",
    "perf/max_memory_allocated_gb",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: str) -> int | float | None:
    try:
        number = float(value)
    except ValueError:
        return None
    return int(number) if number.is_integer() else number


def parse_training_curve(path: Path) -> list[dict[str, int | float]]:
    """Extract one compact, deterministic metric row per completed step."""
    by_step: dict[int, dict[str, int | float]] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = ANSI.sub("", raw_line).replace("\r", "")
            match = re.search(r"(?:^|\s)step:(\d+)\s+-\s+", line)
            if match is None:
                continue
            step = int(match.group(1))
            row: dict[str, int | float] = {"step": step}
            for item in line[match.end() :].split(" - "):
                if ":" not in item:
                    continue
                key, value = item.rsplit(":", 1)
                key = key.strip()
                parsed = _number(value.strip())
                if key in CURVE_METRICS and parsed is not None:
                    row[key] = parsed
            by_step[step] = row
    return [by_step[step] for step in sorted(by_step)]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def inspect_eval_dir(path: Path, expected: dict[str, int]) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    all_keys: list[str] = []
    request_errors = 0
    for dataset, expected_records in expected.items():
        rows = _jsonl(path / "outputs" / f"{dataset}.jsonl")
        keys = [str(row.get("key", "")) for row in rows]
        errors = sum(bool(row.get("error")) for row in rows)
        scores = [float(row["score"]) for row in rows if float(row.get("score", -1)) >= 0]
        by_task: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            score = float(row.get("score", -1))
            if score >= 0:
                by_task[str(row.get("task_id", ""))].append(score)
        datasets[dataset] = {
            "records": len(rows),
            "expected_records": expected_records,
            "unique_keys": len(set(keys)),
            "request_errors": errors,
            "mean_score": sum(scores) / len(scores) if scores else -1.0,
            "max_at_n": sum(max(values) for values in by_task.values()) / len(by_task) if by_task else -1.0,
            "complete": len(rows) == expected_records and len(set(keys)) == expected_records and errors == 0,
        }
        all_keys.extend(keys)
        request_errors += errors
    total_expected = sum(expected.values())
    return {
        "path": str(path.resolve()),
        "records": len(all_keys),
        "expected_records": total_expected,
        "unique_keys": len(set(all_keys)),
        "request_errors": request_errors,
        "complete": len(all_keys) == total_expected and len(set(all_keys)) == total_expected and request_errors == 0,
        "datasets": datasets,
    }


def inspect_checkpoints(path: Path, *, total_steps: int, save_freq: int) -> dict[str, Any]:
    expected = list(range(save_freq, total_steps + 1, save_freq))
    actual = sorted(
        int(candidate.name.removeprefix("global_step_"))
        for candidate in path.glob("global_step_*")
        if candidate.name.removeprefix("global_step_").isdigit()
    )
    hf_dirs = [path / f"global_step_{step}" / "actor" / "huggingface" for step in expected]
    hf_complete = {
        str(step): directory.is_dir() and (directory / "config.json").is_file()
        for step, directory in zip(expected, hf_dirs, strict=True)
    }
    return {
        "path": str(path.resolve()),
        "expected_steps": expected,
        "actual_steps": actual,
        "hf_complete": hf_complete,
        "complete": actual == expected and all(hf_complete.values()),
    }


def dependency_versions() -> dict[str, str]:
    versions = {}
    for name in ("torch", "transformers", "datasets", "sglang", "ray", "flash-attn", "hydra-core"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _working_tree_change_digest(root: Path, diff: str, untracked: str) -> str:
    """Hash the tracked diff plus names and bytes of every untracked file."""
    digest = hashlib.sha256()
    digest.update(diff.encode())
    separator = "\0" if "\0" in untracked else "\n"
    paths = sorted(path for path in untracked.split(separator) if path)
    for relative in paths:
        content = (root / relative).read_bytes()
        digest.update(b"\0untracked\0")
        digest.update(relative.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(str(len(content)).encode())
        digest.update(b"\0")
        digest.update(content)
    return digest.hexdigest()


def code_version(root: Path = ROOT) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, text=True, capture_output=True
        ).stdout

    diff = git("diff", "--binary", "HEAD")
    untracked = git("ls-files", "--others", "--exclude-standard", "-z")
    digest = _working_tree_change_digest(root, diff, untracked)
    return {
        "head": git("rev-parse", "HEAD").strip(),
        "branch": git("branch", "--show-current").strip(),
        "dirty": bool(git("status", "--short").strip()),
        "working_tree_change_sha256": digest,
        "status": git("status", "--short").splitlines(),
    }


def render_markdown(report: dict[str, Any]) -> str:
    before = report["evaluation"]["before"]["datasets"]
    after = report["evaluation"]["after"]["datasets"]
    checks = report["acceptance"]
    lines = [
        "# DAPO-400 Multi-turn GRPO Math Report",
        "",
        f"Overall status: **{checks['status']}**",
        "",
        "| Dataset | Before mean | After mean | Delta | Before max@16 | After max@16 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("dapo100", "aime2026"):
        b, a = before[dataset], after[dataset]
        lines.append(
            f"| {dataset} | {b['mean_score']:.6f} | {a['mean_score']:.6f} | "
            f"{a['mean_score'] - b['mean_score']:+.6f} | {b['max_at_n']:.6f} | {a['max_at_n']:.6f} |"
        )
    lines.extend(["", "## Acceptance", ""])
    for key, value in checks["checks"].items():
        lines.append(f"- {key}: {'PASS' if value else 'FAIL'}")
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- Code HEAD: `{report['code']['head']}`",
            f"- Working-tree change SHA-256: `{report['code']['working_tree_change_sha256']}`",
            f"- Dependencies: `{json.dumps(report['dependencies'], sort_keys=True)}`",
            f"- GPU resources: `{json.dumps(report['gpu_resources'].get('gpus', []), sort_keys=True)}`",
            f"- Limit fallback history: `{json.dumps(report.get('run_history', []), sort_keys=True)}`",
            "",
            "## Full experiment config",
            "",
            "```json",
            json.dumps(report["experiment"], ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
            "## Data manifest and hashes",
            "",
            "```json",
            json.dumps(report["data_manifest"], ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
            "## GPU and dependency manifest",
            "",
            "```json",
            json.dumps(
                {
                    "gpu_resources": report["gpu_resources"],
                    "dependencies": report["dependencies"],
                    "code": report["code"],
                    "limit_fallback_history": report.get("run_history", []),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
            "## Full training curve",
            "",
            f"All {len(report['training_curve'])} completed training steps:",
            "",
            "```json",
            json.dumps(report["training_curve"], ensure_ascii=False, indent=2, sort_keys=True),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--gpu-resources", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--trajectory-summary", type=Path, required=True)
    parser.add_argument("--before-eval-dir", type=Path, required=True)
    parser.add_argument("--after-eval-dir", type=Path, required=True)
    parser.add_argument("--run-history", type=Path)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = read_json(args.experiment_config)
    before = inspect_eval_dir(args.before_eval_dir, {"dapo100": 1600, "aime2026": 480})
    after = inspect_eval_dir(args.after_eval_dir, {"dapo100": 1600, "aime2026": 480})
    checkpoints = inspect_checkpoints(args.checkpoint_dir, total_steps=300, save_freq=20)
    trajectories = read_json(args.trajectory_summary)
    expected_phases = {"train": 48000, "validation": 6100, "baseline": 2080, "post": 2080}
    by_phase = trajectories.get("by_phase", {})
    checks = {
        "300 training steps": len(parse_training_curve(args.train_log)) == 300,
        "15 recoverable checkpoints with HF models": checkpoints["complete"],
        "exact trajectory counts": all(int(by_phase.get(key, -1)) == value for key, value in expected_phases.items()),
        "before evaluation complete without request errors": before["complete"],
        "after evaluation complete without request errors": after["complete"],
    }
    report = {
        "experiment": experiment,
        "code": code_version(),
        "dependencies": dependency_versions(),
        "gpu_resources": read_json(args.gpu_resources),
        "data_manifest": read_json(args.data_manifest),
        "run_history": read_json(args.run_history) if args.run_history else [],
        "training_curve": parse_training_curve(args.train_log),
        "checkpoints": checkpoints,
        "trajectories": trajectories,
        "evaluation": {"before": before, "after": after},
        "acceptance": {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks},
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown_out.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["acceptance"], indent=2))


if __name__ == "__main__":
    main()
