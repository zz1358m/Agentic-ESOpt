#!/usr/bin/env python3
"""Run one DocVQA bash/OCR ReAct trajectory against an HTTP model endpoint."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docvqa-train-time" / "envs"))

from docvqa import anls, load_tasks  # noqa: E402
from scripts.stage1_model_client import (  # noqa: E402
    build_sampling_payload,
    call_completion,
    render_prompt,
)
from algorithms.verl_trace2skill.docvqa_protocol import (  # noqa: E402
    bash_action_command,
    build_docvqa_messages,
    extract_final_answer,
    observation_message,
    react_step,
)
from algorithms.verl_trace2skill.docvqa_sandbox import run_sandboxed_bash  # noqa: E402

def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def trace_markdown(result: dict) -> str:
    task = result["task"]
    lines = [
        f"# DocVQA Stage 1 trace: {result['task_id']}",
        "",
        f"Question: {task['question']}",
        f"Expected answers: {json.dumps(task['answers'], ensure_ascii=False)}",
        f"Prediction: {result['prediction']}",
        f"ANLS: {result['anls']}",
        f"Tool used: {result['used_bash']}",
        f"Termination: {result['termination_reason']}",
        "",
    ]
    for index, step in enumerate(result["steps"], 1):
        lines.extend([f"## Round {index}", "", "### Assistant", "", step["assistant"], ""])
        if step.get("action"):
            lines.extend(["### Action", "", "```json", json.dumps(step["action"], ensure_ascii=False), "```", ""])
        if step.get("observation"):
            lines.extend(["### Observation", "", "```text", step["observation"], "```", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--model", default="Qwen3.5-4B")
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--bash-timeout", type=float, default=30.0)
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
    source_image = Path(task.image).resolve()
    copied_image = tool_dir / "source_document.png"
    shutil.copy2(source_image, copied_image)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path.expanduser().resolve())
    messages = build_docvqa_messages(task.question)
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
        decision = react_step(strip_think(completion), used_tool=used_bash)
        if decision.kind == "final":
            steps.append({"turn": turn, "assistant": completion})
            termination_reason = "final_answer"
            break
        if decision.kind == "retry":
            observation = str(decision.message or "Invalid response format.")
            messages.append({"role": "user", "content": observation_message("format_check", observation)})
            steps.append({"turn": turn, "assistant": completion, "observation": observation})
            continue

        assert decision.action is not None
        command, error = bash_action_command(decision.action)
        if error:
            observation = error
        else:
            used_bash = True
            sandbox = run_sandboxed_bash(
                str(command),
                image_path=copied_image,
                timeout=args.bash_timeout,
                max_output_chars=6000,
            )
            observation = sandbox.text
        messages.append({"role": "user", "content": observation_message("bash", observation)})
        steps.append(
            {
                "turn": turn,
                "assistant": completion,
                "action": decision.action,
                "observation": observation,
            }
        )

    prediction = extract_final_answer(strip_think(completion)) or ""
    anls_score = anls(prediction, task.answers)
    if not used_bash:
        anls_score = 0.0
    result = {
        "task_id": task.id,
        "task": {
            "id": task.id,
            "question": task.question,
            "answers": list(task.answers),
            "image": str(source_image),
            "source": task.source,
        },
        "score": float(anls_score),
        "anls": float(anls_score),
        "acc": 1.0 if anls_score > 0.5 else 0.0,
        "score_method": "docvqa_paper_react_cli_anls" if used_bash else "docvqa_paper_react_cli_no_tool_use",
        "prediction": prediction,
        "completion": completion,
        "used_bash": used_bash,
        "termination_reason": termination_reason,
        "answer_status": "answered" if prediction else "missing_final_answer",
        "source_image": str(source_image),
        "copied_image": str(copied_image),
        "tool_image": "/workspace/document.png",
        "steps": steps,
        "endpoint": args.endpoint,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output_dir / "trace.md").write_text(trace_markdown(result), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("task_id", "anls", "prediction", "used_bash", "termination_reason")}))


if __name__ == "__main__":
    main()
