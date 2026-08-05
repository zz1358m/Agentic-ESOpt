#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "webarena-train-time" / "scripts"))

import run_trace2skill_webarena_sft as trace2skill  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--skill-file",
        default="",
        help="Skill markdown file. Omit for a true empty/no-skill evaluation.",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=1200)
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
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.skill_file:
        skill_file = Path(args.skill_file).resolve()
        if not skill_file.is_file():
            raise FileNotFoundError(f"Skill file not found: {skill_file}")
        skill_label = str(skill_file)
    else:
        skill_file = out_dir / "EMPTY_SKILL.md"
        skill_file.write_text("", encoding="utf-8")
        skill_label = None
    items = trace2skill.load_lite_test_items()
    repeat_summaries = []
    for repeat in range(1, args.repeats + 1):
        repeat_dir = out_dir / f"repeat_{repeat:02d}"
        results = trace2skill.run_webarena_rollout(
            items=items,
            out_dir=repeat_dir / "test_rollout",
            skill_file=skill_file,
            port=12013,
            model_endpoints=args.model_endpoints,
            model_name=args.model_name,
            instruction_path=args.instruction_path,
            mode="chat",
            stop_token="",
            local_enable_thinking="false",
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            presence_penalty=args.presence_penalty,
            repetition_penalty=args.repetition_penalty,
            workers=args.workers,
            timeout=args.timeout,
            max_steps=args.max_steps,
        )
        trace2skill.write_json(repeat_dir / "test_results.json", results)
        hard = sum(int(result.get("hard", 0)) for result in results)
        failures = sum(result.get("agent_ok") is False for result in results)
        repeat_summary = {
            "repeat": repeat,
            "count": len(results),
            "hard": hard,
            "hard_acc": hard / len(results),
            "soft_avg": sum(float(result.get("soft", 0.0)) for result in results)
            / len(results),
            "runner_failures": failures,
            "conditional_acc": hard / (len(results) - failures)
            if len(results) > failures
            else None,
        }
        repeat_summaries.append(repeat_summary)
        trace2skill.write_json(repeat_dir / "summary.json", repeat_summary)
        print(json.dumps(repeat_summary, indent=2), flush=True)

    hard_counts = [summary["hard"] for summary in repeat_summaries]
    summary = {
        "repeats": args.repeats,
        "count_per_repeat": len(items),
        "hard_counts": hard_counts,
        "mean_hard": statistics.mean(hard_counts),
        "mean_hard_acc": statistics.mean(
            summary["hard_acc"] for summary in repeat_summaries
        ),
        "skill": skill_label,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "presence_penalty": args.presence_penalty,
            "repetition_penalty": args.repetition_penalty,
            "max_steps": args.max_steps,
            "timeout": args.timeout,
        },
        "runs": repeat_summaries,
    }
    trace2skill.write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
