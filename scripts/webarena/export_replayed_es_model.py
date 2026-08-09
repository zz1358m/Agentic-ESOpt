#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from algorithms.es import SeedReplayModelES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source-history", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--generations", type=int, default=21)
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--reward-normalization", default="zscore")
    parser.add_argument("--parameter-scope", default="full", choices=["full", "all_linear", "lora"])
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16"])
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = json.loads(Path(args.source_history).read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    es = SeedReplayModelES()
    init = es.init(model, parameter_scope=args.parameter_scope, verbose=True)
    records = [{"kind": "init", "response": init}]
    for record in history[: args.generations]:
        generation = int(record["generation"])
        seeds = [int(seed) for seed in record["seeds"]]
        rewards = [float(reward) for reward in record["rewards"]]
        update = es.update(
            seeds=seeds,
            rewards=rewards,
            alpha=args.alpha,
            reward_normalization=args.reward_normalization,
        )
        records.append({"generation": generation, "update": update})
        valid = [reward for reward in rewards if reward >= 0.0]
        print(
            f"[export_replay] generation={generation} rewards={len(rewards)} "
            f"valid={len(valid)} update={update}",
            flush=True,
        )

    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size=args.max_shard_size)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "es_replay_export_manifest.json").write_text(
        json.dumps(
            {
                "model_path": args.model_path,
                "source_history": args.source_history,
                "generations": args.generations,
                "alpha": args.alpha,
                "reward_normalization": args.reward_normalization,
                "parameter_scope": args.parameter_scope,
                "dtype": args.dtype,
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[saved] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
