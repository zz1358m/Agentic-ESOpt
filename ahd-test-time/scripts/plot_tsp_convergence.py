#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


GROUPS = [
    {
        "label": "EoH baseline",
        "color": "#1f77b4",
        "runs": [
            ROOT / "cache/results_tsp_eoh_8b_pop10_gen25_temp10_mcts_train50_seed1234",
            ROOT / "cache/active_runs/results_tsp_eoh_8b_pop10_gen25_temp10_mcts_train50_seed1234_run2",
            ROOT / "cache/active_runs/results_tsp_eoh_8b_pop10_gen25_temp10_mcts_train50_seed1234_run3",
        ],
    },
    {
        "label": "ES sigma=3e-4 alpha=5e-4",
        "color": "#2ca02c",
        "runs": [
            ROOT / "cache/active_runs/results_tsp_eoh_evolve_8b_cuda4567_sigma0.0003_alpha0.0005_pop10_gen25_esbatch10_temp10_mcts_train50_seed1234_grid_sigma3e-4_alpha5e-4_rep1",
            ROOT / "cache/active_runs/results_tsp_eoh_evolve_8b_cuda4567_sigma0.0003_alpha0.0005_pop10_gen25_esbatch10_temp10_mcts_train50_seed1234_grid_sigma3e-4_alpha5e-4_rep2",
            ROOT / "cache/active_runs/results_tsp_eoh_evolve_8b_cuda4567_sigma0.0003_alpha0.0005_pop10_gen25_esbatch10_temp10_mcts_train50_seed1234_grid_sigma3e-4_alpha5e-4_rep3",
        ],
    },
    {
        "label": "ES sigma=3e-4 alpha=1e-3",
        "color": "#ff7f0e",
        "runs": [
            ROOT / "cache/active_runs/results_tsp_eoh_evolve_8b_cuda4567_sigma0.0003_alpha0.001_pop10_gen25_esbatch10_temp10_mcts_train50_seed1234_grid_sigma3e-4_alpha1e-3_rep1",
            ROOT / "cache/active_runs/results_tsp_eoh_evolve_8b_cuda4567_sigma0.0003_alpha0.001_pop10_gen25_esbatch10_temp10_mcts_train50_seed1234_grid_sigma3e-4_alpha1e-3_rep2",
            ROOT / "cache/active_runs/results_tsp_eoh_evolve_8b_cuda4567_sigma0.0003_alpha0.001_pop10_gen25_esbatch10_temp10_mcts_train50_seed1234_grid_sigma3e-4_alpha1e-3_rep3",
        ],
    },
    {
        "label": "ES sigma=1e-3 alpha=5e-4",
        "color": "#d62728",
        "runs": [
            ROOT / "cache/active_runs/results_tsp_eoh_evolve_8b_cuda4567_sigma0.001_alpha0.0005_pop10_gen25_esbatch10_temp10_mcts_train50_seed1234_grid_sigma1e-3_alpha5e-4_rep1",
            ROOT / "cache/active_runs/results_tsp_eoh_evolve_8b_cuda4567_sigma0.001_alpha0.0005_pop10_gen25_esbatch10_temp10_mcts_train50_seed1234_grid_sigma1e-3_alpha5e-4_rep2",
            ROOT / "cache/active_runs/results_tsp_eoh_evolve_8b_cuda4567_sigma0.001_alpha0.0005_pop10_gen25_esbatch10_temp10_mcts_train50_seed1234_grid_sigma1e-3_alpha5e-4_rep3",
        ],
    },
    {
        "label": "ES sigma=1e-3 alpha=1e-3",
        "color": "#9467bd",
        "runs": [
            ROOT / "cache/active_runs/results_tsp_eoh_evolve_8b_cuda4567_sigma0.001_alpha0.001_pop10_gen25_esbatch10_temp10_mcts_train50_seed1234_grid_sigma1e-3_alpha1e-3_rep1",
            ROOT / "cache/active_runs/results_tsp_eoh_evolve_8b_cuda4567_sigma0.001_alpha0.001_pop10_gen25_esbatch10_temp10_mcts_train50_seed1234_grid_sigma1e-3_alpha1e-3_rep2",
            ROOT / "cache/active_runs/results_tsp_eoh_evolve_8b_cuda4567_sigma0.001_alpha0.001_pop10_gen25_esbatch10_temp10_mcts_train50_seed1234_grid_sigma1e-3_alpha1e-3_rep3",
        ],
    },
    {
        "label": "ES sigma=3e-3 alpha=1e-3",
        "color": "#8c564b",
        "runs": [
            ROOT / "cache/active_runs/results_tsp_eoh_evolve_8b_cuda4567_sigma0.003_alpha0.001_pop10_gen25_esbatch10_temp10_mcts_train50_seed1234_grid_sigma3e-3_alpha1e-3_rep1",
            ROOT / "cache/active_runs/results_tsp_eoh_evolve_8b_cuda4567_sigma0.003_alpha0.001_pop10_gen25_esbatch10_temp10_mcts_train50_seed1234_grid_sigma3e-3_alpha1e-3_rep2",
            ROOT / "cache/active_runs/results_tsp_eoh_evolve_8b_cuda4567_sigma0.003_alpha0.001_pop10_gen25_esbatch10_temp10_mcts_train50_seed1234_grid_sigma3e-3_alpha1e-3_rep3",
        ],
    },
    {
        "label": "ES sigma=1e-3 alpha=3e-3",
        "color": "#17becf",
        "runs": [
            ROOT / "cache/active_runs/results_tsp_eoh_evolve_8b_cuda4567_sigma0.001_alpha0.003_pop10_gen25_esbatch10_temp10_mcts_train50_seed1234_grid_sigma1e-3_alpha3e-3_rep1",
            ROOT / "cache/active_runs/results_tsp_eoh_evolve_8b_cuda4567_sigma0.001_alpha0.003_pop10_gen25_esbatch10_temp10_mcts_train50_seed1234_grid_sigma1e-3_alpha3e-3_rep2",
            ROOT / "cache/active_runs/results_tsp_eoh_evolve_8b_cuda4567_sigma0.001_alpha0.003_pop10_gen25_esbatch10_temp10_mcts_train50_seed1234_grid_sigma1e-3_alpha3e-3_rep3",
        ],
    },
]


def load_curve(run_dir):
    pops = run_dir / "results/pops_best"
    if not pops.exists():
        return []

    curve = []
    for path in sorted(pops.glob("population_generation_*.json"), key=generation_of):
        gen = generation_of(path)
        try:
            with path.open() as f:
                obj = json.load(f).get("objective")
            obj = float(obj)
        except Exception:
            continue
        if math.isfinite(obj):
            curve.append((gen, obj))
    return curve


def generation_of(path):
    stem = Path(path).stem
    return int(stem.rsplit("_", 1)[1])


def mean_curve(curves):
    points = []
    for gen in range(1, 26):
        vals = [dict(curve)[gen] for curve in curves if gen in dict(curve)]
        if vals:
            points.append((gen, sum(vals) / len(vals), len(vals)))
    return points


def final_value(curve):
    return curve[-1][1] if curve else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "cache/plots/tsp_construct_convergence.png"))
    parser.add_argument("--csv", default=str(ROOT / "cache/plots/tsp_construct_convergence_summary.csv"))
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    csv = Path(args.csv)
    csv.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)

    rows = ["group,run,generation,objective,status"]
    plotted = False
    for group in GROUPS:
        curves = []
        for idx, run in enumerate(group["runs"], start=1):
            curve = load_curve(run)
            if not curve:
                rows.append(f"{group['label']},rep{idx},,,missing")
                continue
            curves.append(curve)
            xs = [x for x, _ in curve]
            ys = [y for _, y in curve]
            status = "complete" if xs and xs[-1] >= 25 else "partial"
            rows.extend(
                f"{group['label']},rep{idx},{gen},{obj:.5f},{status}"
                for gen, obj in curve
            )
            ax.plot(
                xs,
                ys,
                color=group["color"],
                linewidth=1.15,
                linestyle=(0, (3, 3)),
                alpha=0.28,
            )
            plotted = True

        mean = mean_curve(curves)
        if mean:
            xs = [x for x, _, _ in mean]
            ys = [y for _, y, _ in mean]
            ns = [n for _, _, n in mean]
            label = f"{group['label']} mean (n={ns[-1]})"
            ax.plot(xs, ys, color=group["color"], linewidth=2.6, label=label)
            plotted = True

    ax.set_title("TSP Construct Convergence: final-population best objective")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best objective so far (lower is better)")
    ax.set_xlim(1, 25)
    ax.set_xticks(range(1, 26, 2))
    ax.set_ylim(bottom=6.2, top=7.0)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    if plotted:
        fig.savefig(out)
    else:
        ax.text(0.5, 0.5, "No data found", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(out)
    csv.write_text("\n".join(rows) + "\n")
    print(out)
    print(csv)


if __name__ == "__main__":
    main()
