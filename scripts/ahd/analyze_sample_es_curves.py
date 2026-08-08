#!/usr/bin/env python3
"""Compare plain sampling and sample-ES learning curves from saved generations."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ACTIVE = ROOT / "cache" / "active_runs"
OUT = ROOT / "results" / "sample_es_vs_sample_curves_20260719"
TASKS = [
    "construct_tsp",
    "construct_kp",
    "construct_asp",
    "aco_tsp",
    "aco_cvrp",
    "aco_bpp",
]
TITLES = {
    "construct_tsp": "Constructive TSP",
    "construct_kp": "Constructive KP",
    "construct_asp": "Constructive ASP",
    "aco_tsp": "ACO TSP",
    "aco_cvrp": "ACO CVRP",
    "aco_bpp": "ACO BPP",
}


def sample_dir(task: str, rep: int) -> Path:
    stamp = "20260718_060041" if task.startswith("construct_") else "20260718_091608"
    if rep > 1:
        stamp = "20260718_144918"
    name = f"{task}_sample_t1000_rep{rep}_{stamp}"
    return ACTIVE / f"{task}_train_sample_t1000_{name}"


def es_dir(task: str, rep: int) -> Path:
    run = f"{task}_sample_es_reload_constant_pop20_gen50_rep{rep}_20260718_145842"
    prefix = f"{task}_train_sample_es_pop20_gen50_sigma0.001_alpha0.0005_"
    return ACTIVE / f"{prefix}{run}"


def completed(path: Path) -> bool:
    return (path / "results" / "history" / "sample_generation_50.json").is_file()


def load_curve(path: Path) -> dict[str, np.ndarray]:
    means, bests, valid_rates = [], [], []
    cumulative = math.inf
    for generation in range(1, 51):
        history = path / "results" / "history" / f"sample_generation_{generation}.json"
        records = json.loads(history.read_text())
        values = [
            float(record["objective"])
            for record in records
            if isinstance(record.get("objective"), (int, float))
            and math.isfinite(record["objective"])
        ]
        means.append(float(np.mean(values)) if values else np.nan)
        if values:
            cumulative = min(cumulative, min(values))
        bests.append(cumulative if math.isfinite(cumulative) else np.nan)
        valid_rates.append(len(values) / len(records) if records else 0.0)
    return {
        "mean": np.asarray(means),
        "best": np.asarray(bests),
        "valid_rate": np.asarray(valid_rates),
    }


def smooth(values: np.ndarray, width: int = 5) -> np.ndarray:
    result = np.empty_like(values)
    for index in range(len(values)):
        start = max(0, index - width + 1)
        window = values[start : index + 1]
        result[index] = np.nanmean(window) if np.isfinite(window).any() else np.nan
    return result


def stack(curves: list[dict[str, np.ndarray]], metric: str) -> np.ndarray:
    return np.stack([curve[metric] for curve in curves])


def plot_metric(
    data: dict, metric: str, filename: str, ylabel: str, do_smooth: bool, title: str
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True)
    generations = np.arange(1, 51)
    styles = {"sample": ("#4C78A8", "Sample"), "es": ("#E45756", "Sample + ES")}
    for axis, task in zip(axes.flat, TASKS):
        for method in ("sample", "es"):
            values = stack(data[task][method], metric)
            if do_smooth:
                values = np.stack([smooth(row) for row in values])
            center = np.asarray([
                np.nanmean(column) if np.isfinite(column).any() else np.nan
                for column in values.T
            ])
            low = np.asarray([
                np.nanmin(column) if np.isfinite(column).any() else np.nan
                for column in values.T
            ])
            high = np.asarray([
                np.nanmax(column) if np.isfinite(column).any() else np.nan
                for column in values.T
            ])
            color, label = styles[method]
            count = values.shape[0]
            axis.plot(generations, center, color=color, linewidth=2, label=f"{label} (n={count})")
            if count > 1:
                axis.fill_between(generations, low, high, color=color, alpha=0.15)
        axis.set_title(TITLES[task])
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        axis.set_xlabel("Generation (20 samples each)")
        axis.set_ylabel(ylabel)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {}
    rows = []
    for task in TASKS:
        es_reps = [rep for rep in (1, 2, 3) if completed(es_dir(task, rep))]
        data[task] = {
            "sample": [load_curve(sample_dir(task, rep)) for rep in es_reps],
            "es": [load_curve(es_dir(task, rep)) for rep in es_reps],
        }
        for method in ("sample", "es"):
            curves = data[task][method]
            batch_mean = stack(curves, "mean")
            best = stack(curves, "best")
            valid = stack(curves, "valid_rate")
            rows.append(
                {
                    "task": task,
                    "method": method,
                    "runs": len(curves),
                    "best_at_1000_mean": np.nanmean(best[:, -1]),
                    "batch_mean_gen_1_10": np.nanmean(batch_mean[:, :10]),
                    "batch_mean_gen_41_50": np.nanmean(batch_mean[:, 40:]),
                    "late_minus_early": np.nanmean(batch_mean[:, 40:])
                    - np.nanmean(batch_mean[:, :10]),
                    "valid_rate_gen_1_10": np.nanmean(valid[:, :10]),
                    "valid_rate_gen_41_50": np.nanmean(valid[:, 40:]),
                }
            )

    plot_metric(
        data,
        "best",
        "cumulative_best.png",
        "Cumulative best objective",
        False,
        "Plain sampling vs constant-sigma ES: cumulative best (lower is better)",
    )
    plot_metric(
        data,
        "mean",
        "batch_valid_mean_rolling5.png",
        "Valid objective mean (rolling 5 gen)",
        True,
        "Plain sampling vs constant-sigma ES: batch quality (lower is better)",
    )
    plot_metric(
        data,
        "valid_rate",
        "valid_rate_rolling5.png",
        "Valid fraction (rolling 5 gen)",
        True,
        "Plain sampling vs constant-sigma ES: valid-sample rate (higher is better)",
    )
    with (OUT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "run_sources.json").write_text(
        json.dumps(
            {
                task: {
                    "es": [str(es_dir(task, rep)) for rep in (1, 2, 3) if completed(es_dir(task, rep))],
                    "sample": [
                        str(sample_dir(task, rep))
                        for rep in (1, 2, 3)
                        if completed(es_dir(task, rep))
                    ],
                }
                for task in TASKS
            },
            indent=2,
        )
    )
    print(OUT)


if __name__ == "__main__":
    main()
