#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HORIZONS = (5, 10, 15)
COLORS = {5: "#3B5B92", 10: "#C96A3A", 15: "#2F8F83"}
MARKERS = {5: "o", 10: "s", 15: "D"}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9.0,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.0,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": True,
        }
    )


def plot(output: Path) -> None:
    configure_style()
    fig, ax = plt.subplots(figsize=(4.05, 2.8))
    step_accuracy = np.linspace(0.70, 1.0, 301)

    for horizon in HORIZONS:
        trajectory_correctness = step_accuracy**horizon
        markevery = [0, 50, 100, 150, 200, 250, 300]
        ax.plot(
            step_accuracy,
            trajectory_correctness,
            color=COLORS[horizon],
            linewidth=2.0,
            marker=MARKERS[horizon],
            markevery=markevery,
            markersize=3.8,
            markerfacecolor="white",
            markeredgewidth=0.9,
            label=f"$H={horizon}$",
        )

    ax.axvline(0.95, color="#6B7280", linestyle=":", linewidth=1.0)
    for horizon, offset in zip(HORIZONS, (0.035, 0.0, -0.035)):
        value = 0.95**horizon
        ax.annotate(
            f"{value:.3f}",
            xy=(0.95, value),
            xytext=(0.956, value + offset),
            fontsize=7.5,
            color=COLORS[horizon],
        )

    ax.set_xlim(0.70, 1.00)
    ax.set_ylim(0, 1.00)
    ax.set_xticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel(r"Per-turn correctness, $p$")
    ax.set_ylabel(r"Full-trajectory success, $S_H$")
    ax.set_title("Turn errors compound exponentially with horizon", pad=4)
    ax.grid(color="#D8DEE9", linewidth=0.65, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncols=1)

    fig.tight_layout(pad=0.35)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="figures/Sudoku-GRPO-Step-Trajectory.pdf")
    args = parser.parse_args()
    plot(Path(args.output))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
