#!/usr/bin/env python3
"""DocVQA ReAct ES training on the in-process vLLM Math-ES backend.

The ES scheduler and optimizer are intentionally shared with the Math runner.
Only the task payload, paper-aligned CLI rollout, and DocVQA reward are specialized.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DOCVQA_PATH = ROOT / "docvqa-train-time"
MATH_RUNNER = ROOT / "math-train-time" / "scripts" / "run_math_es_vllm_train.py"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "math-train-time"))
sys.path.insert(0, str(MATH_RUNNER.parent))

import run_math_es_vllm_train as core  # noqa: E402
sys.path.insert(0, str(DOCVQA_PATH / "envs"))
from docvqa import DocVQAEnv, DocVQATask, anls  # noqa: E402
from algorithms.verl_trace2skill.docvqa_protocol import (  # noqa: E402
    DOCVQA_IMAGE_PATH,
    build_docvqa_messages,
    incremental_message_token_count,
)
from algorithms.verl_trace2skill.docvqa_sandbox import run_sandboxed_bash  # noqa: E402


DEFAULT_TRAIN = ROOT / "data" / "trace2skill" / "docvqa" / "evolve.jsonl"
DEFAULT_EVAL = ROOT / "data" / "trace2skill" / "docvqa" / "test.jsonl"


def task_payload(task: DocVQATask) -> dict[str, Any]:
    return {
        "id": task.id,
        "question": task.question,
        "answers": list(task.answers),
        "image": task.image,
        "source": task.source,
    }


def job_to_payload(job: Any) -> dict[str, Any]:
    return {
        "task": task_payload(job.task),
        "row_index": int(job.row_index),
        "sample_index": int(job.sample_index),
    }


def react_messages(task: DocVQATask, skill: str) -> list[dict[str, Any]]:
    return build_docvqa_messages(task.question, skill)


def extract_trace2skill_answer(text: str) -> str:
    cleaned = core.strip_think(text)
    final = core.final_answer_line(cleaned)
    if final is not None:
        return final
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return lines[-1] if lines else cleaned.strip()


def parse_trace2skill_action(text: str) -> dict[str, Any] | None:
    """Keep DocVQA action parsing byte-for-byte equivalent to Trace2Skill eval."""
    match = re.search(r"Action:\s*(\{.*?\})\s*$", text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        match = re.search(r"Action:\s*(\{.*\})", text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    raw = match.group(1).strip()
    decoder = json.JSONDecoder()
    try:
        action, _ = decoder.raw_decode(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(action, dict):
        return None
    name = action.get("name")
    arguments = action.get("arguments", {})
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    return {"name": name, "arguments": arguments}


class DocVqaVllmActor(core.MathVllmActor):
    def _sampling_params(self, **kwargs: Any):
        """Trace2Skill max_tokens=0 omits the cap instead of falling back to 4096."""
        from vllm import SamplingParams

        max_tokens = int(kwargs["max_tokens"])
        return SamplingParams(
            temperature=float(kwargs["temperature"]),
            top_p=float(kwargs["top_p"]),
            top_k=int(kwargs["top_k"]),
            min_p=float(kwargs["min_p"]),
            presence_penalty=float(kwargs["presence_penalty"]),
            repetition_penalty=float(kwargs["repetition_penalty"]),
            max_tokens=max_tokens if max_tokens > 0 else None,
            seed=int(kwargs["seed"]),
            stop=["Observation:"],
        )

    def rollout_batch(
        self,
        *,
        jobs: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        presence_penalty: float,
        repetition_penalty: float,
        max_turns: int,
        python_timeout: float,
        tool_observation_limit: int,
        trim_context: bool,
        rollout_token_budget: int,
        max_total_tokens: int,
        seed: int,
    ) -> list[dict[str, Any]]:
        states: list[dict[str, Any]] = []
        for job in jobs:
            payload = job["task"]
            row_index = int(job["row_index"])
            sample_index = int(job.get("sample_index", 0))
            seed_base = int(seed) + sample_index * 1_000_003 + row_index
            source_image_path = Path(str(payload["image"])).resolve()
            if not source_image_path.exists():
                raise FileNotFoundError(source_image_path)

            # Bubblewrap mounts the source image read-only at the canonical
            # /workspace/document.png path, so trajectories cannot overwrite
            # the dataset or leak artifacts into one another.
            tool_workdir = (
                self.tool_work_root
                / core.safe_workdir_name(str(payload["id"]), fallback=f"row{row_index:05d}")
                / f"row{row_index:05d}_sample{sample_index:02d}_seed{seed_base}"
            )
            tool_workdir.mkdir(parents=True, exist_ok=True)
            task = DocVQATask(
                id=str(payload["id"]),
                question=str(payload["question"]),
                answers=[str(answer) for answer in payload.get("answers", [])],
                image=DOCVQA_IMAGE_PATH,
                source=str(payload.get("source", "")),
            )
            messages = react_messages(task, self.skill)
            states.append(
                {
                    "task": task,
                    "task_payload": payload,
                    "row_index": row_index,
                    "sample_index": sample_index,
                    "image_path": source_image_path,
                    "source_image_path": source_image_path,
                    "tool_workdir": tool_workdir,
                    "messages": messages,
                    "steps": [],
                    "used_bash": False,
                    "completion": "",
                    "react_error": None,
                    "termination_reason": None,
                    "context_trims": 0,
                    "max_context_tokens": 0,
                    "generated_tokens": 0,
                    "trajectory_tokens": 0,
                    "attempted_trajectory_tokens": 0,
                    "done": False,
                    "started_at": time.time(),
                    "seed_base": seed_base,
                }
            )

        turn = 0
        while int(max_turns) <= 0 or turn < int(max_turns):
            active = [state for state in states if not state["done"]]
            if not active:
                break
            generation_states: list[dict[str, Any]] = []
            inputs: list[str] = []
            params: list[Any] = []
            for state in active:
                output_tokens = int(max_tokens) if int(max_tokens) > 0 else 1
                prompt, trims, prompt_tokens = self._fit_prompt_to_context(
                    state["messages"], reserve_tokens=output_tokens, trim_context=trim_context
                )
                state["context_trims"] += trims
                if prompt is None:
                    state["done"] = True
                    state["termination_reason"] = "context_length_exceeded"
                    state["react_error"] = "context_length_exceeded"
                    continue
                generation_states.append(state)
                state["current_prompt_tokens"] = int(prompt_tokens)
                state["max_context_tokens"] = max(int(state["max_context_tokens"]), int(prompt_tokens))
                inputs.append(prompt)
                params.append(
                    self._sampling_params(
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        min_p=min_p,
                        presence_penalty=presence_penalty,
                        repetition_penalty=repetition_penalty,
                        seed=state["seed_base"] + turn * 97,
                    )
                )
            if not generation_states:
                continue

            current_turn = turn + 1
            turn += 1
            output_tokens = int(max_tokens) if int(max_tokens) > 0 else self.default_max_tokens
            request_sizes = [int(s["current_prompt_tokens"]) + output_tokens for s in generation_states]
            request_batches = core.token_budget_batches(request_sizes, rollout_token_budget)
            generated: list[tuple[dict[str, Any], Any]] = []
            for indexes in request_batches:
                batch_states = [generation_states[index] for index in indexes]
                batch_inputs = [inputs[index] for index in indexes]
                batch_params = [params[index] for index in indexes]
                try:
                    outputs = self.llm.generate(batch_inputs, batch_params, use_tqdm=False)
                    generated.extend(zip(batch_states, outputs))
                    continue
                except Exception:
                    pass
                for state, model_input, sampling in zip(batch_states, batch_inputs, batch_params):
                    try:
                        output = self.llm.generate([model_input], [sampling], use_tqdm=False)[0]
                        generated.append((state, output))
                    except Exception as exc:
                        state["done"] = True
                        state["termination_reason"] = "request_error"
                        state["react_error"] = f"{type(exc).__name__}: {exc}"

            for state, output in generated:
                try:
                    completion = self._output_text(output)
                    candidates = getattr(output, "outputs", None) or []
                    token_ids = getattr(candidates[0], "token_ids", None) if candidates else None
                    completion_tokens = len(token_ids) if token_ids is not None else self._prompt_token_count(completion)
                    state["generated_tokens"] += int(completion_tokens)
                    state["trajectory_tokens"] += int(completion_tokens)
                    state["attempted_trajectory_tokens"] = int(state["trajectory_tokens"])
                    state["max_context_tokens"] = max(
                        int(state["max_context_tokens"]),
                        int(state["current_prompt_tokens"]) + int(completion_tokens),
                    )
                    state["completion"] = completion
                    state["messages"].append({"role": "assistant", "content": completion})
                    if max_total_tokens > 0 and int(state["trajectory_tokens"]) >= int(max_total_tokens):
                        state["done"] = True
                        state["termination_reason"] = "max_response_tokens_exceeded"
                        state["react_error"] = "max_response_tokens_exceeded"
                        continue
                    cleaned = core.strip_think(completion)
                    final_match = core.final_answer_line(cleaned) is not None
                    action = parse_trace2skill_action(cleaned)
                    if final_match and not action:
                        if state["used_bash"]:
                            state["done"] = True
                            state["termination_reason"] = "final_answer"
                            continue
                        warning = (
                            "You must use a bash Action to inspect/process the image file before answering. "
                            "Tool observations are text only; then provide Final answer."
                        )
                        warning_message = {
                            "role": "user",
                            "content": core.react_observation_text("format_check", warning),
                        }
                        state["messages"].append(warning_message)
                        state["steps"].append({"turn": current_turn, "assistant": completion, "observation": warning})
                        observation_tokens = incremental_message_token_count(
                            self.tokenizer,
                            warning_message,
                            apply_chat_template_kwargs={"enable_thinking": False},
                        )
                        attempted = int(state["trajectory_tokens"]) + int(observation_tokens)
                        state["attempted_trajectory_tokens"] = attempted
                        if max_total_tokens > 0 and attempted >= int(max_total_tokens):
                            state["done"] = True
                            state["termination_reason"] = "max_response_tokens_exceeded"
                            state["react_error"] = "max_response_tokens_exceeded"
                        else:
                            state["trajectory_tokens"] = attempted
                        continue
                    if not action:
                        warning = (
                            'No valid action was parsed. Use exactly:\n'
                            'Action:\n{"name": "bash", "arguments": {"command": "<shell command>"}}\n'
                            "or, after tool use, Final answer: <short answer>"
                        )
                        warning_message = {
                            "role": "user",
                            "content": core.react_observation_text("format_check", warning),
                        }
                        state["messages"].append(warning_message)
                        state["steps"].append({"turn": current_turn, "assistant": completion, "observation": warning})
                        observation_tokens = incremental_message_token_count(
                            self.tokenizer,
                            warning_message,
                            apply_chat_template_kwargs={"enable_thinking": False},
                        )
                        attempted = int(state["trajectory_tokens"]) + int(observation_tokens)
                        state["attempted_trajectory_tokens"] = attempted
                        if max_total_tokens > 0 and attempted >= int(max_total_tokens):
                            state["done"] = True
                            state["termination_reason"] = "max_response_tokens_exceeded"
                            state["react_error"] = "max_response_tokens_exceeded"
                        else:
                            state["trajectory_tokens"] = attempted
                        continue
                    name = action["name"]
                    arguments = action["arguments"]
                    if name == "bash":
                        command = str(arguments.get("command", ""))
                        if command.strip():
                            state["used_bash"] = True
                            sandbox_result = run_sandboxed_bash(
                                command,
                                image_path=state["image_path"],
                                timeout=float(python_timeout),
                                max_output_chars=int(tool_observation_limit),
                            )
                            observation = sandbox_result.text
                        else:
                            observation = "No shell command was provided."
                    else:
                        observation = f"Unknown action '{name}'. Available action is bash."
                    observation_message = {
                        "role": "user",
                        "content": core.react_observation_text(name, observation),
                    }
                    state["messages"].append(observation_message)
                    state["steps"].append(
                        {
                            "turn": current_turn,
                            "assistant": completion,
                            "action": action,
                            "observation": observation,
                        }
                    )
                    observation_tokens = incremental_message_token_count(
                        self.tokenizer,
                        observation_message,
                        apply_chat_template_kwargs={"enable_thinking": False},
                    )
                    attempted = int(state["trajectory_tokens"]) + int(observation_tokens)
                    state["attempted_trajectory_tokens"] = attempted
                    if max_total_tokens > 0 and attempted >= int(max_total_tokens):
                        state["done"] = True
                        state["termination_reason"] = "max_response_tokens_exceeded"
                        state["react_error"] = "max_response_tokens_exceeded"
                    else:
                        state["trajectory_tokens"] = attempted
                except Exception as exc:
                    state["done"] = True
                    state["termination_reason"] = "request_error"
                    state["react_error"] = f"{type(exc).__name__}: {exc}"

        for state in states:
            if not state["done"]:
                state["termination_reason"] = "max_turns"
                state["react_error"] = "max_react_turns_exceeded"

        rows: list[dict[str, Any]] = []
        for state in states:
            task = state["task"]
            completion = state["completion"]
            if not state["used_bash"]:
                completion = ""
            prediction = extract_trace2skill_answer(completion)
            termination = str(state["termination_reason"] or "request_error")
            anls_score = anls(prediction, task.answers)
            # Train on the paper-aligned continuous ANLS signal.  Thresholded
            # accuracy remains available below as a diagnostic metric only.
            score = float(anls_score)
            method = "anls"
            if termination in {"context_length_exceeded", "request_error"}:
                score = 0.0
                method = termination
            elif not state["used_bash"]:
                score = 0.0
                method = "no_tool_use"
            rows.append(
                {
                    "key": f"{task.id}:sample{int(state['sample_index']):02d}",
                    "task_id": task.id,
                    "task": state["task_payload"],
                    "row_index": int(state["row_index"]),
                    "sample_index": int(state["sample_index"]),
                    "score": float(score),
                    "answers": list(task.answers),
                    "anls": float(anls_score),
                    "vlns": float(anls_score),
                    "acc": 1.0 if anls_score > 0.5 else 0.0,
                    "prediction": prediction,
                    "response": completion,
                    "completion": completion,
                    "image": str(state["source_image_path"]),
                    "tool_image": task.image,
                    "tool_workdir": str(state["tool_workdir"]),
                    "latency_s": time.time() - float(state["started_at"]),
                    "mode": "paper_react_cli_vllm",
                    "used_bash": bool(state["used_bash"]),
                    "termination_reason": termination,
                    "answer_status": "answered" if core.final_answer_line(core.strip_think(completion)) is not None else "missing_final_answer",
                    "context_trims": int(state["context_trims"]),
                    "max_context_tokens": int(state["max_context_tokens"]),
                    "generated_tokens": int(state["generated_tokens"]),
                    "trajectory_tokens": int(state["trajectory_tokens"]),
                    "attempted_trajectory_tokens": int(state["attempted_trajectory_tokens"]),
                    "react_error": state["react_error"],
                    "react_steps": state["steps"],
                    "trace_rounds": len(state["steps"]),
                    "score_method": f"docvqa_paper_react_cli_{method}",
                }
            )
        rows.sort(key=lambda row: (int(row["row_index"]), int(row["sample_index"])))
        return rows


def docvqa_trace_markdown(row: dict[str, Any]) -> str:
    task = row.get("task") or {}
    score = float(row.get("score", 0.0))
    outcome = "SUCCEED" if score > 0.5 else "FAILED"
    failure_reason = row.get("react_error") or ("ANLS did not exceed 0.5." if outcome == "FAILED" else "")
    lines = [
        f"# Chat History docvqa_{row.get('task_id', '')}",
        "",
        f"Task ID: {row.get('task_id', '')}",
        "Setting: docvqa",
        f"Sample index: {row.get('sample_index', 0)}",
        f"Image: {row.get('image', task.get('image', ''))}",
        f"Question: {task.get('question', '')}",
        f"Expected answers: {json.dumps(row.get('answers', task.get('answers', [])), ensure_ascii=False)}",
        f"Prediction: {row.get('prediction', '')}",
        f"ANLS: {row.get('anls', score)}",
        f"Threshold accuracy: {row.get('acc', 0.0)}",
        f"Score: {score}",
        f"Score method: {row.get('score_method', '')}",
        f"Outcome: {outcome}",
        f"Termination reason: {row.get('termination_reason', '')}",
        f"Failure reason: {failure_reason}",
        "",
        "## Problem",
        "",
        str(task.get("question", "")),
        "",
        "## Trace",
        "",
    ]
    for index, turn in enumerate(row.get("react_steps") or [], 1):
        lines.extend(
            [
                f"## Round {index}",
                "",
                "### Assistant",
                "",
                str(turn.get("assistant", "")).strip(),
                "",
            ]
        )
        if turn.get("action"):
            lines.extend(
                [
                    "### Action",
                    "",
                    "```json",
                    json.dumps(turn["action"], ensure_ascii=False),
                    "```",
                    "",
                ]
            )
        observation = str(turn.get("observation", "")).strip()
        if observation:
            lines.extend(["### Observation", "", "```text", observation, "```", ""])
    lines.extend(
        [
            "## Final prediction",
            "",
            str(row.get("prediction", "")),
            "",
            "---",
            "",
            "## RESULT",
            outcome,
            "",
        ]
    )
    return "\n".join(lines)


def write_trace_logs(
    trace_dir: Path | None,
    rows: list[dict[str, Any]],
    filename_prefix: str = "",
) -> None:
    if trace_dir is None:
        return
    trace_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        outcome = "SUCCEED" if float(row.get("score", 0.0)) > 0.5 else "FAILED"
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row.get("task_id", "")))
        path = trace_dir / (
            f"docvqa_agent_{filename_prefix}{safe_id}_"
            f"sample{int(row.get('sample_index', 0)):02d}_{outcome}.md"
        )
        # Model output can occasionally contain an unpaired UTF-16 surrogate.
        # Keep trace persistence best-effort so malformed text cannot abort an
        # otherwise valid population evaluation and ES update.
        path.write_text(docvqa_trace_markdown(row), encoding="utf-8", errors="replace")
        row["trace_log"] = str(path)


def eval_tasks_vllm(**kwargs: Any) -> dict[str, Any]:
    label = str(kwargs.get("label", ""))
    if label.startswith("aime"):
        return {"skipped": True, "reason": "DocVQA has one held-out dataset"}
    return ORIGINAL_EVAL(**kwargs)


ORIGINAL_EVAL = core.eval_tasks_vllm
core.MathReasoningEnv = DocVQAEnv
core.MathVllmActor = DocVqaVllmActor
core.job_to_payload = job_to_payload
core.write_trace_logs = write_trace_logs
core.eval_tasks_vllm = eval_tasks_vllm
core.DEFAULT_TRAIN = DEFAULT_TRAIN
core.DEFAULT_EVAL = DEFAULT_EVAL
core.DEFAULT_AIME = DEFAULT_EVAL


def main() -> None:
    os.environ.setdefault("MATH_ES_RESULT_SUBDIR", "runs/docvqa_es_vllm")
    core.main()


if __name__ == "__main__":
    main()
