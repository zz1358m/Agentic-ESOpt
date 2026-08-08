#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import archive_and_eval_sample_prefixes as common


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ahd-test-time" / "results" / "AHD-SampleES-Current1000"
PROGRESS = (
    ROOT
    / "runs"
    / "ahd_sample_es_invalid_reward_tsp_kp_3rep_cosine_maskedzero_gpu0_3_20260719_150222"
    / "progress.jsonl"
)
TASK_DIRS = {"construct_tsp": "TSP_construct", "construct_kp": "KP_construct"}
SETTINGS = {task: common.SETTINGS[task] for task in TASK_DIRS}


def source_rows() -> list[dict[str, Any]]:
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    for line in PROGRESS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (
            row.get("invalid_reward_strategy") == "current"
            and row.get("task") in TASK_DIRS
        ):
            latest[(str(row["task"]), int(row["rep"]))] = row
    expected = {(task, rep) for task in TASK_DIRS for rep in (1, 2, 3)}
    if set(latest) != expected:
        raise RuntimeError(f"expected {expected}, found {set(latest)}")
    if any(row.get("status") != "completed" for row in latest.values()):
        raise RuntimeError("not all current sample_es runs completed")
    return [latest[key] for key in sorted(latest)]


def payload(path: Path) -> dict[str, Any]:
    item = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(item, list):
        item = item[0]
    if not isinstance(item, dict) or not str(item.get("code", "")).strip():
        raise ValueError(f"invalid result payload: {path}")
    return item


def archive() -> list[dict[str, Any]]:
    OUT.mkdir(parents=True, exist_ok=True)
    for dirname in TASK_DIRS.values():
        (OUT / dirname).mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for row in source_rows():
        task = str(row["task"])
        rep = int(row["rep"])
        source = Path(row["result_path"])
        item = payload(source)
        code_file = OUT / TASK_DIRS[task] / f"{task}_rep{rep}_final_best_code.py"
        header = (
            f"# source: {source}\n"
            "# method: sample_es, invalid_reward=current, sigma_schedule=cosine\n"
            "# population=20, generations=50, samples=1000, sigma=0.001->0, alpha=0.0005\n"
            f"# task: {task}, rep: {rep}\n"
            f"# train_objective: {item.get('objective')}\n\n"
        )
        code_file.write_text(header + str(item["code"]).strip() + "\n", encoding="utf-8")
        manifest.append(
            {
                "prefix": 1000,
                "task": task,
                "rep": rep,
                "train_objective": item.get("objective"),
                "code_file": str(code_file),
                "source_json": str(source),
                "method": "sample_es",
                "invalid_reward_strategy": "current",
                "sigma_schedule": "cosine",
            }
        )
    manifest.sort(key=lambda row: (str(row["task"]), int(row["rep"])))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    common.write_csv(OUT / "manifest.csv", manifest)
    write_train_report(manifest)
    return manifest


def write_train_report(manifest: list[dict[str, Any]]) -> None:
    lines = [
        "# AHD SampleES Current 1000: training best",
        "",
        "sample_es, current invalid reward, cosine sigma 0.001 -> 0, three independent runs.",
        "",
        "| Task | Rep1 | Rep2 | Rep3 |",
        "|---|---:|---:|---:|",
    ]
    for task in TASK_DIRS:
        values = {int(row["rep"]): row["train_objective"] for row in manifest if row["task"] == task}
        lines.append(f"| {task} | {values[1]} | {values[2]} | {values[3]} |")
    (OUT / "TRAIN_BEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def jobs(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for record in manifest:
        for setting, params in SETTINGS[str(record["task"])]:
            result.append({**record, "setting": setting, "params": list(params)})
    return result


def key(row: dict[str, Any]) -> tuple[str, int, str]:
    return str(row["task"]), int(row["rep"]), str(row["setting"])


def save_detail(rows: list[dict[str, Any]]) -> None:
    rows.sort(key=key)
    (OUT / "test_detail.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    common.write_csv(OUT / "test_detail.csv", rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["task"]), str(row["setting"])), []).append(row)
    result: list[dict[str, Any]] = []
    for (task, setting), items in sorted(groups.items()):
        items.sort(key=lambda row: int(row["rep"]))
        values = [float(row["metric"]) for row in items if isinstance(row.get("metric"), (int, float))]
        result.append(
            {
                "task": task,
                "setting": setting,
                "objective": items[0].get("objective"),
                "rep1": next((row.get("metric") for row in items if int(row["rep"]) == 1), None),
                "rep2": next((row.get("metric") for row in items if int(row["rep"]) == 2), None),
                "rep3": next((row.get("metric") for row in items if int(row["rep"]) == 3), None),
                "mean_over_reps": statistics.mean(values) if values else None,
                "std_over_reps": statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None,
                "valid_reps": len(values),
                "total_failures": sum(int(row.get("failure_count") or 0) for row in items),
                "errors": "; ".join(str(row["error"]) for row in items if row.get("error")),
            }
        )
    return result


def finalize(rows: list[dict[str, Any]]) -> None:
    save_detail(rows)
    summary = summarize(rows)
    (OUT / "test_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    common.write_csv(OUT / "test_summary.csv", summary)

    def fmt(value: Any) -> str:
        return "NA" if value is None else f"{float(value):.8g}"

    lines = [
        "# AHD SampleES Current 1000: test results",
        "",
        "Full test split; sample_es current branch with cosine sigma decay.",
        "",
        "| Task | Setting | Rep1 | Rep2 | Rep3 | Mean | Std | Failures |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['task']} | {row['setting']} | {fmt(row['rep1'])} | {fmt(row['rep2'])} | "
            f"{fmt(row['rep3'])} | {fmt(row['mean_over_reps'])} | {fmt(row['std_over_reps'])} | "
            f"{row['total_failures']} |"
        )
    lines.extend(["", "KP: larger is better. TSP: smaller is better.", ""])
    (OUT / "TEST_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-only", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    manifest = archive()
    print(f"archived {len(manifest)} current sample_es codes into {OUT}", flush=True)
    if args.archive_only:
        return
    existing = []
    detail = OUT / "test_detail.json"
    if detail.exists():
        existing = json.loads(detail.read_text(encoding="utf-8"))
    completed = {key(row) for row in existing if not row.get("error")}
    pending = [job for job in jobs(manifest) if key(job) not in completed]
    by_key = {key(row): row for row in existing}
    print(f"evaluation jobs: completed={len(completed)} pending={len(pending)}", flush=True)
    if pending:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max(1, args.workers), mp_context=context) as executor:
            futures = {executor.submit(common.evaluate_job, job): job for job in pending}
            for index, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                by_key[key(row)] = row
                save_detail(list(by_key.values()))
                print(
                    f"[{index}/{len(pending)}] {row['task']} rep{row['rep']} {row['setting']} "
                    f"metric={row.get('metric')} error={row.get('error', '')}",
                    flush=True,
                )
    rows = list(by_key.values())
    finalize(rows)
    print(f"all done: rows={len(rows)} errors={sum(bool(row.get('error')) for row in rows)}", flush=True)


if __name__ == "__main__":
    main()
