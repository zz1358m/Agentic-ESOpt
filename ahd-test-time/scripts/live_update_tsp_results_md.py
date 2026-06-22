#!/usr/bin/env python3
import argparse
import glob
import json
import os
import statistics
import tempfile
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP


PARAMS = [
    ("3e-4", "5e-4"),
    ("3e-4", "1e-3"),
    ("1e-3", "5e-4"),
    ("1e-3", "1e-3"),
    ("3e-3", "1e-3"),
    ("1e-3", "3e-3"),
    ("3e-3", "3e-3"),
]

BASELINE = [6.59586, 6.66389, 7.00437]
START = "<!-- live-status:start -->"
END = "<!-- live-status:end -->"


def latest_for(sigma, alpha, rep):
    matches = glob.glob(
        f"cache/active_runs/*grid_sigma{sigma}_alpha{alpha}_rep{rep}/results/pops_best"
    )
    if not matches:
        return {"status": "missing", "gen": None, "obj": None, "file": None}
    files = glob.glob(os.path.join(matches[0], "population_generation_*.json"))
    if not files:
        return {"status": "running", "gen": None, "obj": None, "file": matches[0]}
    files.sort(key=lambda p: int(os.path.basename(p).rsplit("_", 1)[1].split(".")[0]))
    path = files[-1]
    gen = int(os.path.basename(path).rsplit("_", 1)[1].split(".")[0])
    with open(path, "r", encoding="utf-8") as fh:
        obj = json.load(fh).get("objective")
    return {
        "status": "complete" if gen >= 25 else "running",
        "gen": gen,
        "obj": obj,
        "file": path,
    }


def fmt(value):
    if value is None:
        return "-"
    return str(Decimal(str(value)).quantize(Decimal("0.00001"), rounding=ROUND_HALF_UP))


def row_for(sigma, alpha):
    reps = [latest_for(sigma, alpha, rep) for rep in (1, 2, 3)]
    finals = [r["obj"] for r in reps if r["status"] == "complete" and r["obj"] is not None]
    latest = []
    for idx, r in enumerate(reps, start=1):
        if r["status"] == "complete":
            latest.append(f"r{idx}=done:{fmt(r['obj'])}")
        elif r["status"] == "running" and r["gen"] is not None:
            latest.append(f"r{idx}=g{r['gen']}:{fmt(r['obj'])}")
        elif r["status"] == "running":
            latest.append(f"r{idx}=running:no-json")
        else:
            latest.append(f"r{idx}=missing")
    mean = statistics.mean(finals) if finals else None
    return (
        f"| Dynamic-Agent | {sigma} | {alpha} | {len(finals)}/3 | "
        f"{', '.join(latest)} | {fmt(mean)} |"
    )


def build_block():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    baseline_mean = statistics.mean(BASELINE)
    completed_rows = []
    for sigma, alpha in PARAMS:
        reps = [latest_for(sigma, alpha, rep) for rep in (1, 2, 3)]
        finals = [r["obj"] for r in reps if r["status"] == "complete" and r["obj"] is not None]
        if len(finals) == 3:
            completed_rows.append((statistics.mean(finals), sigma, alpha))
    best_line = "-"
    if completed_rows:
        mean, sigma, alpha = min(completed_rows)
        best_line = f"`sigma={sigma}, alpha={alpha}`, mean `{fmt(mean)}`, delta vs baseline `{fmt(mean - baseline_mean)}`"
    lines = [
        START,
        "## Live Status",
        "",
        f"Updated: `{now}`",
        "",
        f"Baseline mean: `{fmt(baseline_mean)}`",
        "",
        f"Best completed 3-rep ES setting: {best_line}",
        "",
        "| Method | Sigma | Alpha | Completed reps | Latest per rep | Completed-rep mean |",
        "|---|---:|---:|---:|---|---:|",
        f"| EoH baseline | - | - | 3/3 | {', '.join(fmt(v) for v in BASELINE)} | {fmt(baseline_mean)} |",
    ]
    lines.extend(row_for(sigma, alpha) for sigma, alpha in PARAMS)
    lines.extend(["", END])
    return "\n".join(lines)


def update(path):
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    block = build_block()
    if START in text and END in text:
        before = text.split(START, 1)[0]
        after = text.split(END, 1)[1]
        new_text = before + block + after
    else:
        marker = "Task: `tsp_construct`"
        if marker in text:
            new_text = text.replace(marker, block + "\n\n" + marker, 1)
        else:
            new_text = block + "\n\n" + text
    if new_text == text:
        return
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".live-md-", dir=directory, text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(new_text)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="cache/tsp_construct_results_summary.md")
    parser.add_argument("--interval", type=int, default=120)
    args = parser.parse_args()
    while True:
        update(args.path)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
