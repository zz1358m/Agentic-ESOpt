#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def read_sheet(path: Path, sheet_name: str) -> dict[str, dict[str, str]]:
    with zipfile.ZipFile(path) as zf:
        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", NS):
                shared.append("".join(t.text or "" for t in si.findall(".//a:t", NS)))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        sheets = {
            sheet.attrib["name"]: sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            for sheet in workbook.findall(".//a:sheet", NS)
        }
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        relmap = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        target = relmap[sheets[sheet_name]]
        target = f"xl/{target}" if not target.startswith("xl/") else target
        root = ET.fromstring(zf.read(target))

        rows: dict[str, dict[str, str]] = {}
        for row in root.findall(".//a:row", NS):
            row_idx = row.attrib["r"]
            values = {}
            for cell in row.findall("a:c", NS):
                ref = cell.attrib.get("r", "")
                col = "".join(ch for ch in ref if ch.isalpha())
                value_node = cell.find("a:v", NS)
                value = ""
                if value_node is not None:
                    value = value_node.text or ""
                    if cell.attrib.get("t") == "s":
                        value = shared[int(value)]
                values[col] = value
            rows[row_idx] = values
        return rows


def as_float(value: str) -> float | None:
    value = str(value).strip()
    if not value or value == "-":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def collect_curves(rows: dict[str, dict[str, str]]) -> tuple[list[int], dict[str, dict[str, list[float | None]]]]:
    step_cols = ["B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
    steps = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    curves: dict[str, dict[str, list[float | None]]] = {}
    recommended_grpo_block = False
    for row in rows.values():
        label = row.get("A", "").strip()
        if label.startswith("GRPO-temp="):
            recommended_grpo_block = label == "GRPO-temp=0.7 top-p=0.8 top-k=20"
            continue
        if not label.startswith("Test-"):
            continue
        parts = label.split("-")
        if len(parts) == 2:
            if not recommended_grpo_block:
                continue
            horizon = parts[1]
            method = "GRPO"
        elif len(parts) == 3:
            horizon = parts[1]
            method = f"ES pop{parts[2]}"
        else:
            continue
        if horizon not in {"5", "10", "15"} or method == "ES pop8":
            continue
        curves.setdefault(horizon, {})[method] = [as_float(row.get(col, "")) for col in step_cols]
    return steps, curves


def best_value(values: list[float | None]) -> float:
    present = [v for v in values if v is not None]
    return max(present) if present else 0.0


def polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def draw_svg(steps: list[int], curves: dict[str, dict[str, list[float | None]]]) -> str:
    width, height = 1280, 720
    colors = {"GRPO": "#2563eb", "ES pop16": "#f97316", "ES pop32": "#16a34a"}
    horizons = ["5", "10", "15"]
    methods = ["GRPO", "ES pop16", "ES pop32"]
    labels = {"5": "5-horizon", "10": "10-horizon", "15": "15-horizon"}

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="1280" height="720" fill="#ffffff"/>',
        '<text x="64" y="56" font-family="Arial, sans-serif" font-size="30" font-weight="700" fill="#111827">Agentic Sudoku: ES learns faster as horizons get longer</text>',
        '<text x="64" y="88" font-family="Arial, sans-serif" font-size="16" fill="#4b5563">Binary success reward; each run trains/evaluates one fixed mask horizon.</text>',
    ]

    # Summary bars.
    chart_x, chart_y, chart_w, chart_h = 74, 126, 1130, 230
    out.append(f'<text x="{chart_x}" y="{chart_y - 24}" font-family="Arial, sans-serif" font-size="18" font-weight="700" fill="#111827">Best test success across training (%)</text>')
    for tick in range(0, 101, 25):
        y = chart_y + chart_h - tick / 100 * chart_h
        out.append(f'<line x1="{chart_x}" x2="{chart_x + chart_w}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{chart_x - 12}" y="{y + 5:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#6b7280">{tick}</text>')
    group_w = chart_w / len(horizons)
    bar_w = 38
    for hi, horizon in enumerate(horizons):
        cx = chart_x + group_w * hi + group_w / 2
        out.append(f'<text x="{cx:.1f}" y="{chart_y + chart_h + 28}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" font-weight="700" fill="#374151">{labels[horizon]}</text>')
        for mi, method in enumerate(methods):
            value = best_value(curves[horizon].get(method, []))
            x = cx - 72 + mi * 52
            h = value / 100 * chart_h
            y = chart_y + chart_h - h
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" rx="4" fill="{colors[method]}"/>')
            out.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 7:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#111827">{value:.1f}</text>')

    legend_x = 840
    for idx, method in enumerate(methods):
        x = legend_x + idx * 120
        out.append(f'<rect x="{x}" y="105" width="16" height="16" rx="3" fill="{colors[method]}"/>')
        out.append(f'<text x="{x + 22}" y="118" font-family="Arial, sans-serif" font-size="13" fill="#374151">{method}</text>')

    # Small-multiple convergence curves.
    panel_y, panel_h, panel_w, gap = 430, 190, 340, 50
    for hi, horizon in enumerate(horizons):
        x0 = 74 + hi * (panel_w + gap)
        y0 = panel_y
        out.append(f'<text x="{x0}" y="{y0 - 22}" font-family="Arial, sans-serif" font-size="17" font-weight="700" fill="#111827">{labels[horizon]} test curve</text>')
        out.append(f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" fill="#f9fafb" stroke="#e5e7eb" rx="8"/>')
        for tick in [0, 50, 100]:
            y = y0 + panel_h - tick / 100 * panel_h
            out.append(f'<line x1="{x0}" x2="{x0 + panel_w}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e5e7eb"/>')
            out.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#6b7280">{tick}</text>')
        for tick in [0, 50, 100]:
            x = x0 + tick / 100 * panel_w
            out.append(f'<text x="{x:.1f}" y="{y0 + panel_h + 18}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#6b7280">{tick}</text>')
        for method in methods:
            values = curves[horizon].get(method, [])
            pts = []
            for step, value in zip(steps, values):
                if value is None or math.isnan(value):
                    continue
                x = x0 + step / 100 * panel_w
                y = y0 + panel_h - value / 100 * panel_h
                pts.append((x, y))
            if len(pts) >= 2:
                out.append(f'<polyline points="{polyline(pts)}" fill="none" stroke="{colors[method]}" stroke-width="3"/>')
            for x, y in pts:
                out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.8" fill="{colors[method]}"/>')

    out.append('<text x="64" y="684" font-family="Arial, sans-serif" font-size="14" fill="#4b5563">Takeaway: ES G=32 reaches or exceeds GRPO final performance earlier on 10- and 15-horizon tasks, while retaining higher best test success.</text>')
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", default="results.xlsx")
    parser.add_argument("--sheet", default="sukodu")
    parser.add_argument("--output", default="figures/sudoku_agentic_es_vs_grpo.svg")
    args = parser.parse_args()
    rows = read_sheet(Path(args.workbook), args.sheet)
    steps, curves = collect_curves(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(draw_svg(steps, curves), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
