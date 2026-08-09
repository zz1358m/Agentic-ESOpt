#!/usr/bin/env python3
"""Export Math smoke, GRPO, validation, and evaluation trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verl_trace2skill.trajectory import (  # noqa: E402
    export_trajectory_records,
    load_raw_trajectory_records,
    normalize_evaluation_record,
)


def _evaluation_files(path: Path) -> Iterable[Path]:
    outputs = path / "outputs" if (path / "outputs").is_dir() else path
    return sorted(outputs.glob("*.jsonl")) if outputs.is_dir() else [outputs]


def _load_evaluation(path: Path, *, phase: str, epoch: int, step: int) -> list[dict]:
    records = []
    for source in _evaluation_files(path):
        with source.open(encoding="utf-8") as handle:
            records.extend(
                normalize_evaluation_record(
                    json.loads(line),
                    phase=phase,
                    epoch=epoch,
                    step=step,
                )
                for line in handle
                if line.strip()
            )
    return records


def _load_raw(path: Path, *, phase: str) -> list[dict]:
    # Raw attempts remain immutable on disk.  The normalized acceptance export
    # uses exactly one complete (latest) attempt per logical training step.
    records = load_raw_trajectory_records(path, latest_only=True)
    for record in records:
        old_phase = str(record.get("phase", "unknown"))
        if old_phase != phase:
            trajectory_id = str(record["trajectory_id"])
            prefix = old_phase + "-"
            record["trajectory_id"] = phase + "-" + (
                trajectory_id[len(prefix) :] if trajectory_id.startswith(prefix) else trajectory_id
            )
            record["phase"] = phase
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-dir", type=Path)
    parser.add_argument("--train-dir", type=Path)
    parser.add_argument("--validation-dir", type=Path)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--post-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--final-epoch", type=int, default=15)
    parser.add_argument("--final-step", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    for path, phase in (
        (args.smoke_dir, "smoke"),
        (args.train_dir, "train"),
        (args.validation_dir, "validation"),
    ):
        if path is not None:
            records.extend(_load_raw(path.expanduser().resolve(), phase=phase))
    if args.baseline_dir is not None:
        records.extend(
            _load_evaluation(
                args.baseline_dir.expanduser().resolve(),
                phase="baseline",
                epoch=0,
                step=0,
            )
        )
    if args.post_dir is not None:
        records.extend(
            _load_evaluation(
                args.post_dir.expanduser().resolve(),
                phase="post",
                epoch=args.final_epoch,
                step=args.final_step,
            )
        )

    out_dir = args.out_dir.expanduser().resolve()
    summary = export_trajectory_records(records, out_dir=out_dir)
    summary["by_phase"] = dict(Counter(str(record.get("phase", "unknown")) for record in records))
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
