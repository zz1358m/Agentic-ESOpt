#!/usr/bin/env python3
"""Verify a converted Qwen3.5 text checkpoint and run a fixed forward pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate_weight_index(model_path: Path) -> dict[str, int]:
    index = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index.get("weight_map") or {}
    if not weight_map:
        raise RuntimeError("weight index is empty")
    shards = set(weight_map.values())
    missing = sorted(shard for shard in shards if not (model_path / shard).is_file())
    if missing:
        raise RuntimeError(f"missing checkpoint shards: {missing}")
    empty = sorted(shard for shard in shards if (model_path / shard).stat().st_size == 0)
    if empty:
        raise RuntimeError(f"empty checkpoint shards: {empty}")
    return {"weights": len(weight_map), "shards": len(shards)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    model_path = args.model_path.expanduser().resolve()
    index_info = validate_weight_index(model_path)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    loaded = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": args.device},
        trust_remote_code=True,
        output_loading_info=True,
    )
    model, loading_info = loaded
    unexpected = loading_info.get("unexpected_keys") or []
    mismatched = loading_info.get("mismatched_keys") or []
    missing = [key for key in (loading_info.get("missing_keys") or []) if key != "lm_head.weight"]
    if missing or unexpected or mismatched:
        raise RuntimeError(
            f"checkpoint load incomplete: missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
        )
    inputs = tokenizer("Fixed checkpoint verification prompt.", return_tensors="pt").to(args.device)
    with torch.inference_mode():
        logits = model(**inputs).logits
    if tuple(logits.shape[:2]) != tuple(inputs["input_ids"].shape) or not torch.isfinite(logits).all():
        raise RuntimeError(f"invalid forward output: shape={tuple(logits.shape)}")
    report = {
        **index_info,
        "model_type": model.config.model_type,
        "vocab_size": int(logits.shape[-1]),
        "prompt_tokens": int(inputs["input_ids"].shape[-1]),
        "device": args.device,
        "forward_finite": True,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "mismatched_keys": mismatched,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
