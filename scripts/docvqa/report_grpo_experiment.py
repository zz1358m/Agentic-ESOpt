#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from verl_trace2skill.docvqa_results import compare_results, summarize_results  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def render_markdown(report: dict[str, Any]) -> str:
    before = report["before"]
    after = report["after"]
    comparison = report["comparison"]
    ci = comparison["bootstrap_95_ci"]
    return (
        "# Qwen3.5-4B DocVQA GRPO Report\n\n"
        "| Metric | Before | After |\n"
        "|---|---:|---:|\n"
        f"| Mean ANLS | {before['mean_anls']:.6f} | {after['mean_anls']:.6f} |\n"
        f"| Accuracy (ANLS > 0.5) | {before['mean_accuracy']:.6f} | {after['mean_accuracy']:.6f} |\n"
        f"| Tool call rate | {before['tool_call_rate']:.6f} | {after['tool_call_rate']:.6f} |\n"
        f"| Tool success rate | {before['tool_success_rate']:.6f} | {after['tool_success_rate']:.6f} |\n"
        f"| Mean ReAct turns | {before['mean_turns']:.3f} | {after['mean_turns']:.3f} |\n"
        f"| Mean latency (s) | {before['mean_latency_s']:.3f} | {after['mean_latency_s']:.3f} |\n"
        f"| Request errors | {before['request_errors']} | {after['request_errors']} |\n"
        f"| Format retries | {before['format_retries']} | {after['format_retries']} |\n"
        f"| Bash timeouts | {before['bash_timeouts']} | {after['bash_timeouts']} |\n\n"
        f"Token usage before: `{json.dumps(before['usage'], sort_keys=True)}`  \n"
        f"Token usage after: `{json.dumps(after['usage'], sort_keys=True)}`  \n"
        f"Termination diagnostics before: `{json.dumps(before['react_errors'], sort_keys=True)}`  \n"
        f"Termination diagnostics after: `{json.dumps(after['react_errors'], sort_keys=True)}`  \n\n"
        f"Paired tasks: {comparison['paired_tasks']}  \n"
        f"Mean ANLS delta: {comparison['mean_anls_delta']:.6f}  \n"
        f"Question-cluster bootstrap 95% CI: [{ci[0]:.6f}, {ci[1]:.6f}]\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pre/post DocVQA GRPO JSONL results.")
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    before_rows = read_jsonl(args.before)
    after_rows = read_jsonl(args.after)
    report = {
        "before_path": str(args.before.resolve()),
        "after_path": str(args.after.resolve()),
        "before": summarize_results(before_rows),
        "after": summarize_results(after_rows),
        "comparison": compare_results(
            before_rows,
            after_rows,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.markdown_out.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
