#!/usr/bin/env python3
"""Consolidate one MAP patch per math task into a compact inference skill."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


SYSTEM_PROMPT = """You are consolidating failure-derived guidance for a mathematical ReAct agent.

Each supplied Evidence Unit represents exactly one distinct math problem. Count every unit at most once, regardless of how many rollout samples, failures, lessons, or edits it contains.

Produce a complete SKILL.md, not a patch. Follow these rules:
1. Retain a learned rule only when it recurs across independent Evidence Units or is a universal runtime obligation: valid bash Action syntax, rigorous verification, full constraint checking, or exact final-answer termination.
2. Weight support by distinct Evidence Units, never by repetitions or item count inside one unit.
3. Merge overlapping rules into short executable instructions.
4. Remove task-family modules, named formula templates, benchmark-specific facts, sample answers, constants, and rare one-problem techniques.
5. Favor a small general workflow: interpret precisely; derive; use bash/Python deliberately; validate against the original constraints; prove completeness when required; finalize in the exact format.
6. Keep the entire file at most 80 lines and at most 1200 o200k_base tokens, including YAML frontmatter.
7. Use only the YAML fields name and description. Use imperative instructions. Do not create references or other files.
8. Do not mention this consolidation process, evidence counts, trajectories, benchmarks, or sampling in the generated skill.

Return only the complete Markdown file beginning with --- and ending with the final instruction. Do not use a code fence."""


REFINE_PROMPT = """Rewrite the supplied mathematical-reasoning SKILL.md so it satisfies every constraint below:
- at most {max_lines} lines and {max_tokens} o200k_base tokens;
- retain only general cross-problem reasoning, tool verification, constraint, completeness, and final-format guidance;
- remove named topic modules and specialized formula templates;
- preserve valid YAML frontmatter with only name and description;
- output only the complete Markdown file without a code fence.

Current SKILL.md:
<skill>
{skill}
</skill>"""


SPECIALIZED_PATTERN = re.compile(
    r"\b(determinant|cofactor|markov|newton identit|2-adic|v2\(|minimax|"
    r"geometry-to-probability|digit concatenation|cyclic indexing|game-theory p/n)\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-patches", type=Path, required=True)
    parser.add_argument("--initial-skill", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.4-nano")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--max-lines", type=int, default=80)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--expected-units", type=int, default=358)
    return parser.parse_args()


def clean_markdown(text: str) -> str:
    text = text.strip()
    fenced = re.fullmatch(r"```(?:markdown)?\s*\n(.*?)\n```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("---")
    if start > 0:
        text = text[start:]
    return text.rstrip() + "\n"


def skill_stats(text: str, encoding: Any) -> dict[str, int | bool]:
    return {
        "lines": len(text.splitlines()),
        "tokens_o200k": len(encoding.encode(text)),
        "contains_specialized_module": bool(SPECIALIZED_PATTERN.search(text)),
    }


def main() -> None:
    args = parse_args()
    try:
        import tiktoken
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional Trace2Skill dependencies: openai tiktoken"
        ) from exc

    patches = sorted(args.map_patches.glob("patch_*.json"))
    if len(patches) != args.expected_units:
        raise ValueError(
            f"Expected {args.expected_units} task-level evidence units, found {len(patches)}"
        )
    evidence = []
    for index, path in enumerate(patches, start=1):
        value = json.loads(path.read_text(encoding="utf-8"))
        # Keep one compact unit per task. Changelog text, JSON operation syntax,
        # empty locator fields, and batch metadata add tokens but no evidence.
        proposed_rules = "\n".join(
            str(edit.get("content", "")).strip()
            for edit in value.get("edits", [])
            if str(edit.get("content", "")).strip()
        )
        evidence.append(
            f"## Evidence Unit {index}\n"
            f"Reason: {value.get('reasoning', '')}\n"
            f"Proposed rules:\n{proposed_rules}"
        )

    initial_skill = args.initial_skill.read_text(encoding="utf-8")
    user_prompt = (
        "Create the compact mathematical-reasoning skill from the initial draft and the "
        "task-level evidence units below.\n\n"
        f"# Initial draft\n{initial_skill}\n\n"
        "# Task-level evidence\n"
        + "\n\n".join(evidence)
    )

    key_file = REPO_ROOT / "apikey"
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and key_file.is_file():
        api_key = key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or create <repo>/apikey.")
    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    response = client.chat.completions.create(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    skill = clean_markdown(response.choices[0].message.content or "")
    encoding = tiktoken.get_encoding("o200k_base")

    refinement_responses = []
    for _ in range(2):
        stats = skill_stats(skill, encoding)
        if (
            stats["lines"] <= args.max_lines
            and stats["tokens_o200k"] <= args.max_tokens
            and not stats["contains_specialized_module"]
        ):
            break
        refine = client.chat.completions.create(
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": REFINE_PROMPT.format(
                        max_lines=args.max_lines,
                        max_tokens=args.max_tokens,
                        skill=skill,
                    ),
                },
            ],
        )
        skill = clean_markdown(refine.choices[0].message.content or "")
        refinement_responses.append(skill)

    stats = skill_stats(skill, encoding)
    if stats["lines"] > args.max_lines or stats["tokens_o200k"] > args.max_tokens:
        raise ValueError(f"Generated skill exceeds size limits: {stats}")
    if stats["contains_specialized_module"]:
        raise ValueError(f"Generated skill retained a specialized module: {stats}")
    if not skill.startswith("---\n"):
        raise ValueError("Generated skill is missing YAML frontmatter")

    skill_dir = args.output_dir / "skill"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(skill, encoding="utf-8")
    (args.output_dir / "skill_step_001.md").write_text(skill, encoding="utf-8")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "evidence_unit": "one distinct math task",
                "evidence_units": len(patches),
                "source_map_patches": str(args.map_patches),
                "max_lines": args.max_lines,
                "max_tokens_o200k": args.max_tokens,
                "stats": stats,
                "refinement_calls": len(refinement_responses),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"skill": str(skill_dir / "SKILL.md"), **stats}, indent=2))


if __name__ == "__main__":
    main()
