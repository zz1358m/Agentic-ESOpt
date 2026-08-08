#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import archive_and_eval_sample_prefixes as common


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "ahd-test-time" / "results"
OUTPUTS = {
    1000: RESULTS / "AHD-SampleES-Current1000",
    2000: RESULTS / "AHD-SampleES-Current2000",
}
PROGRESS_FILES = (
    ROOT
    / "runs"
    / "ahd_sample_es_current_cosine_mixed_queue_a_gpu0_3_20260720_030717"
    / "progress.jsonl",
    ROOT
    / "runs"
    / "ahd_sample_es_current_cosine_mixed_queue_b_gpu4_7_20260720_030717"
    / "progress.jsonl",
)
TASK_DIRS = common.TASK_DIRS
NEW_1000_TASKS = {"construct_asp", "aco_tsp", "aco_cvrp", "aco_bpp"}
ALL_TASKS = set(TASK_DIRS)


def progress_rows() -> dict[tuple[int, str, int], dict[str, Any]]:
    latest: dict[tuple[int, str, int], dict[str, Any]] = {}
    for path in PROGRESS_FILES:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row["sample_total"]), str(row["task"]), int(row["rep"]))
            latest[key] = row
    expected = {
        *((1000, task, rep) for task in NEW_1000_TASKS for rep in (1, 2, 3)),
        *((2000, task, rep) for task in ALL_TASKS for rep in (1, 2, 3)),
    }
    found = set(latest)
    if found != expected:
        raise RuntimeError(f"progress keys mismatch: missing={expected-found}, extra={found-expected}")
    incomplete = [key for key, row in latest.items() if row.get("status") != "completed"]
    if incomplete:
        raise RuntimeError(f"incomplete runs: {incomplete}")
    return latest


def payload(path: Path) -> dict[str, Any]:
    item = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(item, list):
        if not item:
            raise ValueError(f"empty result: {path}")
        item = item[0]
    if not isinstance(item, dict) or not str(item.get("code", "")).strip():
        raise ValueError(f"invalid result: {path}")
    return item


def valid_samples(result_path: Path) -> tuple[int | None, int | None]:
    sample_file = result_path.parents[1] / "samples.jsonl"
    if not sample_file.exists():
        return None, None
    count = 0
    valid = 0
    for line in sample_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        count += 1
        objective = json.loads(line).get("objective")
        if isinstance(objective, (int, float)) and math.isfinite(objective):
            valid += 1
    return valid, count


def archive_record(prefix: int, row: dict[str, Any]) -> dict[str, Any]:
    task = str(row["task"])
    rep = int(row["rep"])
    source = Path(row["result_path"])
    item = payload(source)
    out = OUTPUTS[prefix]
    code_file = out / TASK_DIRS[task] / f"{task}_rep{rep}_final_best_code.py"
    code = str(item["code"]).strip()
    repairs: list[str] = []
    if task.startswith("aco_") and "np." in code and not re.search(
        r"^\s*(?:import\s+numpy\s+as\s+np|from\s+numpy\s+import)", code, re.MULTILINE
    ):
        code = "import numpy as np\n\n" + code
        repairs.append("add_missing_numpy_import")
    cvrp_exp = "np.exp(-calculate_heuristics(i, j) / get_distance(i, j))"
    if task == "aco_cvrp" and cvrp_exp in code:
        code = code.replace(
            cvrp_exp,
            "np.exp(np.minimum(-calculate_heuristics(i, j) / get_distance(i, j), 60.0))",
        )
        repairs.append("exp_log_upper_clip_60")
    header = (
        f"# source: {source}\n"
        "# method: sample_es, invalid_reward=current, sigma_schedule=cosine\n"
        f"# population=20, generations={prefix // 20}, samples={prefix}, "
        "sigma=0.001->0, alpha=0.0005\n"
        f"# task: {task}, rep: {rep}\n"
        f"# train_objective: {item.get('objective')}\n\n"
    )
    if repairs:
        header += f"# evaluation_repairs: {','.join(repairs)}\n\n"
    code_file.write_text(header + code + "\n", encoding="utf-8")
    valid, count = valid_samples(source)
    return {
        "prefix": prefix,
        "task": task,
        "rep": rep,
        "train_objective": item.get("objective"),
        "valid_samples": valid,
        "sample_count": count,
        "code_file": str(code_file),
        "source_json": str(source),
        "method": "sample_es",
        "invalid_reward_strategy": "current",
        "sigma_schedule": "cosine",
        "evaluation_repairs": ",".join(repairs),
    }


def archive() -> dict[int, list[dict[str, Any]]]:
    rows = progress_rows()
    for out in OUTPUTS.values():
        out.mkdir(parents=True, exist_ok=True)
        for dirname in TASK_DIRS.values():
            (out / dirname).mkdir(parents=True, exist_ok=True)

    old_manifest_path = OUTPUTS[1000] / "manifest.json"
    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    manifests: dict[int, list[dict[str, Any]]] = {
        1000: [row for row in old_manifest if row["task"] in {"construct_tsp", "construct_kp"}],
        2000: [],
    }
    for key, row in sorted(rows.items()):
        prefix = key[0]
        manifests[prefix].append(archive_record(prefix, row))

    for prefix, manifest in manifests.items():
        manifest.sort(key=lambda row: (str(row["task"]), int(row["rep"])))
        expected = {(task, rep) for task in ALL_TASKS for rep in (1, 2, 3)}
        found = {(str(row["task"]), int(row["rep"])) for row in manifest}
        if found != expected:
            raise RuntimeError(f"manifest {prefix} mismatch: missing={expected-found}, extra={found-expected}")
        out = OUTPUTS[prefix]
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        common.write_csv(out / "manifest.csv", manifest)
        common.write_train_report(out, prefix, manifest)
    return manifests


def jobs(manifests: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for prefix, manifest in manifests.items():
        for record in manifest:
            for setting, params in common.SETTINGS[str(record["task"])] :
                result.append({**record, "prefix": prefix, "setting": setting, "params": list(params)})
    return result


def save_rows(rows: list[dict[str, Any]]) -> None:
    for prefix, out in OUTPUTS.items():
        selected = sorted(
            [row for row in rows if int(row["prefix"]) == prefix],
            key=common.result_key,
        )
        (out / "test_detail.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
        common.write_csv(out / "test_detail.csv", selected)


def finalize(rows: list[dict[str, Any]]) -> None:
    save_rows(rows)
    old_roots = common.PREFIX_ROOTS
    common.PREFIX_ROOTS = OUTPUTS
    try:
        common.finalize(rows)
    finally:
        common.PREFIX_ROOTS = old_roots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-only", action="store_true")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    manifests = archive()
    print("archived full 18-code manifests for Current1000 and Current2000", flush=True)
    if args.archive_only:
        return

    existing: list[dict[str, Any]] = []
    for out in OUTPUTS.values():
        detail = out / "test_detail.json"
        if detail.exists():
            existing.extend(json.loads(detail.read_text(encoding="utf-8")))
    completed = {common.result_key(row) for row in existing if not row.get("error")}
    pending = [job for job in jobs(manifests) if common.result_key(job) not in completed]
    by_key = {common.result_key(row): row for row in existing}
    print(f"evaluation jobs: completed={len(completed)} pending={len(pending)}", flush=True)

    if pending:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max(1, args.workers), mp_context=context) as executor:
            futures = {executor.submit(common.evaluate_job, job): job for job in pending}
            for index, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                by_key[common.result_key(row)] = row
                save_rows(list(by_key.values()))
                print(
                    f"[{index}/{len(pending)}] sample_es{row['prefix']} {row['task']} "
                    f"rep{row['rep']} {row['setting']} metric={row.get('metric')} "
                    f"error={row.get('error', '')} wall={row.get('seconds_wall')}s",
                    flush=True,
                )
    rows = list(by_key.values())
    finalize(rows)
    errors = [row for row in rows if row.get("error")]
    print(f"all done: detail_rows={len(rows)} errors={len(errors)}", flush=True)


if __name__ == "__main__":
    main()
