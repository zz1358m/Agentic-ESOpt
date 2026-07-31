#!/usr/bin/env python3
"""Download pinned Qwen3.5/DocVQA assets, convert the text model, and validate identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


MODEL_REPO = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DOCVQA_REPO = "lmms-lab/DocVQA"
DOCVQA_REVISION = "539088ef8a8ada01ac8e2e6d4e372586748a265e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], *, cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True, env=os.environ.copy())


def download_model(destination: Path) -> str:
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=MODEL_REPO,
            revision=MODEL_REVISION,
            local_dir=destination,
        )
        return "huggingface"
    except Exception as hf_error:  # noqa: BLE001 - only the official ModelScope mirror is allowed as fallback
        print(f"Hugging Face model download failed: {hf_error}", file=sys.stderr, flush=True)
        try:
            from modelscope import snapshot_download as modelscope_snapshot_download
        except ImportError as exc:
            raise RuntimeError("Hugging Face failed and ModelScope is not installed") from exc
        # ModelScope and Hugging Face do not share commit hashes. The fallback
        # deliberately resolves the official ModelScope repository's default
        # revision after the pinned Hugging Face revision has failed.
        modelscope_snapshot_download(MODEL_REPO, local_dir=str(destination))
        return "modelscope"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run-root", type=Path, default=Path("runs/docvqa_grpo"))
    parser.add_argument("--historical-jsonl", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    run_root = args.run_root.expanduser()
    if not run_root.is_absolute():
        run_root = (root / run_root).resolve()
    historical = args.historical_jsonl.expanduser().resolve()
    assets = run_root / "assets"
    source_model = assets / "Qwen3.5-4B-source"
    text_model = assets / "Qwen3.5-4B-text"
    data_dir = root / "data/trace2skill/docvqa"
    assets.mkdir(parents=True, exist_ok=True)

    provider = download_model(source_model)
    run(
        [
            args.python,
            str(root / "scripts/trace2skill/convert_qwen35_to_text_qwen3next.py"),
            "--src",
            str(source_model),
            "--dst",
            str(text_model),
        ],
        cwd=root,
    )
    run(
        [
            args.python,
            str(root / "trace2skill-settings/scripts/prepare_data.py"),
            "--setting",
            "docvqa",
            "--output-dir",
            str(data_dir),
            "--seed",
            "42",
            "--docvqa-evolve-count",
            "50",
            "--docvqa-revision",
            DOCVQA_REVISION,
            "--docvqa-order-reference",
            str(historical),
        ],
        cwd=root,
    )
    alignment_path = run_root / "data_alignment.json"
    run(
        [
            args.python,
            str(root / "scripts/docvqa/validate_experiment_data.py"),
            "--train",
            str(data_dir / "evolve.jsonl"),
            "--test",
            str(data_dir / "test.jsonl"),
            "--historical",
            str(historical),
            "--limit",
            "100",
            "--samples",
            "4",
            "--out",
            str(alignment_path),
        ],
        cwd=root,
    )

    index_path = text_model / "model.safetensors.index.json"
    manifest = {
        "model": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "provider": provider,
            "source_path": str(source_model),
            "text_path": str(text_model),
            "weight_index_sha256": sha256(index_path),
        },
        "dataset": {
            "repo": DOCVQA_REPO,
            "revision": DOCVQA_REVISION,
            "split": "validation",
            "path": str(data_dir),
            "train_records": 50,
            "test_records": 5299,
            "fixed_eval_records": 100,
        },
        "historical_jsonl": str(historical),
        "alignment_report": str(alignment_path),
    }
    (run_root / "assets_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
