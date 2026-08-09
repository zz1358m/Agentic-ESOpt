#!/usr/bin/env python3
"""Compress a generated SKILL.md to a hard line limit with an OpenAI model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI


def clean_markdown(text: str) -> str:
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    return text + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--max-lines", type=int, required=True)
    parser.add_argument("--generation-config", default="{}")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--profile", choices=["default", "docvqa_minimal"], default="default")
    args = parser.parse_args()

    source = args.input.read_text(encoding="utf-8")
    if len(source.splitlines()) <= args.max_lines:
        args.output.write_text(source, encoding="utf-8")
        return

    config = json.loads(args.generation_config or "{}")
    reasoning_effort = config.get("reasoning_effort", "medium")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        key_file = Path(__file__).resolve().parents[3] / "apikey"
        api_key = key_file.read_text(encoding="utf-8").strip()
    client = OpenAI(api_key=api_key, base_url=args.base_url)
    profile_rules = ""
    if args.profile == "docvqa_minimal":
        profile_rules = """
- Produce a minimal DocVQA execution policy, not a catalog of task-specific templates.
- Include the exact one-line action form: Action: {"name":"bash","arguments":{"command":"<command>"}}
- A bash command must execute successfully before the final answer. Never advise stopping tool use after a parse failure.
- Never use dummy actions such as `echo ready` or an existence-only check; the first successful action must OCR `/workspace/document.png`.
- Use at most three successful bash/OCR actions: one full-image OCR, then only targeted crop/OCR if needed.
- Once answer evidence appears, immediately emit `Final answer: <short answer>` and stop.
- Do not use `UNKNOWN`; return the best OCR-grounded span after bounded attempts.
- Remove all named examples, benchmark-specific cases, calculations, field-type catalogs, and special-case sections.
"""
    prompt = f"""Compress the following SKILL.md to at most {args.max_lines} Markdown lines, counting blank lines.

Requirements:
- Output only the complete replacement SKILL.md, with no code fence or commentary.
- Preserve only generalizable, actionable guidance supported by the source.
- Deduplicate overlapping rules and examples aggressively.
- Keep mandatory tool-use and exact final-answer protocol constraints.
- Do not mention training tasks, trajectory IDs, evidence records, or this compression request.
- Do not create or refer to any reference files.
- The hard limit of {args.max_lines} lines must be obeyed.
{profile_rules}

SOURCE SKILL.md:
{source}
"""
    last = ""
    for attempt in range(1, args.attempts + 1):
        response = client.responses.create(
            model=args.model,
            reasoning={"effort": reasoning_effort},
            input=prompt,
        )
        last = clean_markdown(response.output_text)
        line_count = len(last.splitlines())
        if line_count <= args.max_lines and last.lstrip().startswith("#"):
            args.output.write_text(last, encoding="utf-8")
            print(json.dumps({"attempt": attempt, "lines": line_count}))
            return
        prompt += (
            f"\n\nYour previous output had {line_count} lines or invalid Markdown. "
            f"Rewrite it more compactly and stay at or below {args.max_lines} lines."
        )
    raise RuntimeError(
        f"Could not compress skill to <= {args.max_lines} lines after {args.attempts} attempts; "
        f"last output had {len(last.splitlines())} lines"
    )


if __name__ == "__main__":
    main()
