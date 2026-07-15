#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "webarena-train-time" / "scripts"))

import run_trace2skill_webarena_sft as trace2skill  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--skill-file", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--model-name", default="Qwen3.5-27B")
    parser.add_argument("--instruction-path", default="agent/prompts/jsons/p_webrl_chat_qwen_action.json")
    parser.add_argument(
        "--model-endpoints",
        default=(
            "http://127.0.0.1:12013/completions "
            "http://127.0.0.1:12014/completions "
            "http://127.0.0.1:12015/completions "
            "http://127.0.0.1:12016/completions"
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    skill_file = Path(args.skill_file)
    items = trace2skill.load_lite_test_items()
    results = trace2skill.run_webarena_rollout(
        items=items,
        out_dir=out_dir / "test_rollout",
        skill_file=skill_file,
        port=12013,
        model_endpoints=args.model_endpoints,
        model_name=args.model_name,
        instruction_path=args.instruction_path,
        mode="chat",
        stop_token="",
        local_enable_thinking="false",
        temperature=args.temperature,
        workers=args.workers,
        timeout=args.timeout,
        max_steps=args.max_steps,
    )
    trace2skill.write_json(out_dir / "test_results.json", results)
    hard = sum(int(r.get("hard", 0)) for r in results)
    soft = sum(float(r.get("soft", 0.0)) for r in results) / len(results)
    summary = {
        "count": len(results),
        "hard": hard,
        "hard_acc": hard / len(results),
        "soft_avg": soft,
        "skill": str(skill_file),
    }
    trace2skill.write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
