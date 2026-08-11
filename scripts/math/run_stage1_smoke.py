#!/usr/bin/env python3
"""Run one Math DAPO ReAct trajectory against an HTTP model endpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "math-train-time"))

from scripts.stage1_model_client import (  # noqa: E402
    build_sampling_payload,
    call_completion,
    render_prompt,
)

from envs.math_reasoning import (  # noqa: E402
    extract_math_answer,
    final_answer_line,
    load_tasks,
    math_react_messages,
    math_score,
    parse_react_action,
    react_observation_text,
    run_bash,
    strip_think,
    trace_markdown,
)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--model", default="Qwen3.5-4B")
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--python-timeout", type=float, default=20.0)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tool_dir = output_dir / "tool_workdir"
    tool_dir.mkdir(exist_ok=True)
    tasks = load_tasks(args.data.expanduser().resolve())
    if args.row_index < 0 or args.row_index >= len(tasks):
        raise IndexError(f"row index {args.row_index} is outside a dataset with {len(tasks)} rows")
    task = tasks[args.row_index]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path.expanduser().resolve())
    messages = math_react_messages(task)
    steps = []
    used_bash = False
    completion = ""
    termination_reason = "max_turns"

    for turn in range(1, max(1, args.max_turns) + 1):
        completion = call_completion(
            args.endpoint,
            build_sampling_payload(
                model=args.model,
                prompt=render_prompt(tokenizer, messages),
                max_tokens=args.max_tokens,
            ),
            timeout=args.timeout,
        )
        messages.append({"role": "assistant", "content": completion})
        cleaned = strip_think(completion)
        action = parse_react_action(cleaned)
        final = final_answer_line(cleaned)
        if final is not None and action is None:
            if used_bash:
                steps.append({"turn": turn, "assistant": completion})
                termination_reason = "final_answer"
                break
            observation = "You must call the bash Action before answering."
            messages.append({"role": "user", "content": react_observation_text("format_check", observation)})
            steps.append({"turn": turn, "assistant": completion, "observation": observation})
            continue
        if action is None:
            observation = (
                'No valid action was parsed. Use Action: {"name":"bash",'
                '"arguments":{"command":"<shell command>"}} or a final answer after bash use.'
            )
            messages.append({"role": "user", "content": react_observation_text("format_check", observation)})
            steps.append({"turn": turn, "assistant": completion, "observation": observation})
            continue
        if action["name"] != "bash":
            observation = f"Unknown action {action['name']!r}; available action is bash."
        else:
            command = str(action["arguments"].get("command", ""))
            if command.strip():
                used_bash = True
                observation = run_bash(command, tool_dir, timeout=args.python_timeout, limit=6000)
            else:
                observation = "No shell command was provided."
        messages.append({"role": "user", "content": react_observation_text(action["name"], observation)})
        steps.append(
            {
                "turn": turn,
                "assistant": completion,
                "action": action,
                "observation": observation,
            }
        )

    score, score_method = math_score(completion, task.answer)
    if not used_bash:
        score = 0.0
        score_method = "no_bash_tool_use"
    result = {
        "task_id": task.id,
        "task": {
            "id": task.id,
            "question": task.question,
            "answer": task.answer,
            "source": task.source,
        },
        "score": score,
        "score_method": f"math_paper_react_cli_{score_method}",
        "prediction": extract_math_answer(completion),
        "completion": completion,
        "used_bash": used_bash,
        "termination_reason": termination_reason,
        "answer_status": "answered" if final_answer_line(strip_think(completion)) is not None else "missing_final_answer",
        "steps": steps,
        "endpoint": args.endpoint,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output_dir / "trace.md").write_text(
        trace_markdown(task=task, row=result, transcript=steps),
        encoding="utf-8",
    )
    print(json.dumps({key: result[key] for key in ("task_id", "score", "score_method", "prediction", "used_bash", "termination_reason")}))


if __name__ == "__main__":
    main()
