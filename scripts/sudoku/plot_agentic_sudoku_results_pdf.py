#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_agentic_sudoku_results import as_float, collect_curves, read_sheet


HORIZONS = ["5", "10", "15"]
HORIZON_LABELS = {"5": "5-horizon", "10": "10-horizon", "15": "15-horizon"}
CURVE_METHODS = ["GRPO", "ES pop16", "ES pop32"]
COLORS = {
    "GRPO": "#3B5B92",
    "ES pop16": "#C96A3A",
    "ES pop32": "#2F8F83",
}


def present(values: list[float | None]) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for idx, value in enumerate(values):
        if value is not None:
            xs.append(idx)
            ys.append(value)
    return xs, ys


def collect_summary(rows: dict[str, dict[str, str]]) -> dict[str, dict[str, list[float]]]:
    layout = {
        "5": ("C", "D"),
        "10": ("E", "F"),
        "15": ("G", "H"),
    }
    summary: dict[str, dict[str, list[float]]] = {
        label: {"mean": [], "std": []} for label in ["GRPO", "ES pop32"]
    }
    for row in rows.values():
        raw_label = row.get("A", "").strip()
        # Use the recommended sampling configuration for the main comparison.
        if raw_label == "+ GRPO-temp=0.7 top-p=0.8 top-k=20":
            label = "GRPO"
        elif raw_label == "+ Dynamic-Agent (G=32)":
            label = "ES pop32"
        else:
            label = None
        if label is None:
            continue
        for horizon in HORIZONS:
            mean_col, std_col = layout[horizon]
            summary[label]["mean"].append(as_float(row.get(mean_col, "")) or 0.0)
            summary[label]["std"].append(as_float(row.get(std_col, "")) or 0.0)
    return summary


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9.0,
            "axes.titlesize": 9.0,
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


def polish_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D8DEE9", linewidth=0.7, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6B7280")
    ax.spines["bottom"].set_color("#6B7280")
    ax.tick_params(colors="#374151", width=0.8, length=3)
    ax.set_axisbelow(True)


def save_test_panel(summary: dict[str, dict[str, list[float]]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(2.62, 2.75))
    x = np.arange(len(HORIZONS)) * 0.62
    width = 0.21
    offsets = {"GRPO": -width / 2, "ES pop32": width / 2}
    names = {"GRPO": "GRPO", "ES pop32": "Dynamic-Agent"}

    for method in ["GRPO", "ES pop32"]:
        means = np.array(summary[method]["mean"])
        stds = np.array(summary[method]["std"])
        bars = ax.bar(
            x + offsets[method],
            means,
            width,
            yerr=stds,
            capsize=3.5,
            label=names[method],
            color=COLORS[method],
            edgecolor="#1F2937",
            linewidth=0.45,
            error_kw={"elinewidth": 0.9, "ecolor": "#1F2937", "capthick": 0.9},
            alpha=0.96,
        )
        ax.bar_label(bars, labels=[f"{value:.1f}" for value in means], padding=1.5, fontsize=7.0, color="#111827")

    ax.set_xticks(x, [HORIZON_LABELS[horizon] for horizon in HORIZONS])
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(x[0] - 0.25, x[-1] + 0.25)
    ax.set_title("Long-horizon Sudoku test success", pad=3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.21), frameon=False, handlelength=0.9, ncols=2, columnspacing=0.65)
    polish_axis(ax)
    fig.tight_layout(pad=0.18)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def save_curve_panel(steps: list[int], curves: dict[str, dict[str, list[float | None]]], horizon: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(2.62, 2.75))
    labels = {"GRPO": "GRPO", "ES pop16": "Dynamic-Agent, G=16", "ES pop32": "Dynamic-Agent, G=32"}
    markers = {"GRPO": "o", "ES pop16": "s", "ES pop32": "D"}
    linestyles = {"GRPO": "-", "ES pop16": "--", "ES pop32": "-"}

    for method in CURVE_METHODS:
        values = curves[horizon].get(method, [])
        idxs, ys = present(values)
        xs = [steps[idx] for idx in idxs]
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            label=labels[method],
            color=COLORS[method],
            linestyle=linestyles[method],
            linewidth=1.85,
            marker=markers[method],
            markersize=4.0,
            markerfacecolor="white",
            markeredgewidth=1.0,
        )

    ax.set_title(f"{HORIZON_LABELS[horizon]} training curve", pad=3)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Success rate (%)")
    ax.set_xlim(-2, 102)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.legend(
        loc="best",
        frameon=False,
        fontsize=6.3,
        handlelength=0.9,
        labelspacing=0.25,
        borderaxespad=0.35,
    )
    polish_axis(ax)
    fig.tight_layout(pad=0.18)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", default="results.xlsx")
    parser.add_argument("--sheet", default="sukodu")
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()

    rows = read_sheet(Path(args.workbook), args.sheet)
    steps, curves = collect_curves(rows)
    summary = collect_summary(rows)
    output_dir = Path(args.output_dir)

    configure_style()
    save_test_panel(summary, output_dir / "Sudoku-Agentic-Test.pdf")
    save_curve_panel(steps, curves, "5", output_dir / "Sudoku-Agentic-Curve-5.pdf")
    save_curve_panel(steps, curves, "15", output_dir / "Sudoku-Agentic-Curve-15.pdf")
    print(f"Wrote Sudoku agentic PDF panels to {output_dir}")


if __name__ == "__main__":
    main()
