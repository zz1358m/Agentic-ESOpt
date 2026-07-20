#!/usr/bin/env python3
"""Run Trace2Skill math and DocVQA test sets through a vLLM OpenAI server.

Defaults cover DAPO held-out 100, AIME 2026, and DocVQA validation with 16
stochastic samples per item. The ReAct runners disable Qwen's built-in thinking
mode so the explicit tool loop controls every intermediate turn.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import httpx

try:
    from math_verify import parse as math_verify_parse
    from math_verify import verify as math_verify_compare
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:  # pragma: no cover - cluster env dependency check catches this for PBS runs
    math_verify_parse = None
    math_verify_compare = None
    ExprExtractionConfig = None
    LatexExtractionConfig = None


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATH_ROOT = ROOT / "data/trace2skill/math_reasoning"
DEFAULT_DOCVQA_ROOT = Path(os.environ.get("DOCVQA_ROOT", ROOT))
DEFAULT_OUT = Path(
    os.environ.get(
        "TRACE2SKILL_VLLM_OUT",
        ROOT / "runs/trace2skill_vllm/qwen35-27b-eval16",
    )
)


class ContextLengthExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    kind: str
    path: Path
    enable_thinking: bool
    max_tokens: int
    limit: int | None = None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=True) + "\n")


def normalize_math_answer(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", value)
    while len(value) >= 2 and value.startswith("{") and value.endswith("}"):
        value = value[1:-1].strip()
    value = value.replace(",", "")
    value = re.sub(r"\s+", "", value)
    value = value.strip(".")
    return value.lower()


def decimal_value(value: str) -> Decimal | None:
    value = normalize_math_answer(value)
    try:
        if "/" in value:
            left, right = value.split("/", 1)
            return Decimal(left) / Decimal(right)
        return Decimal(value)
    except (InvalidOperation, ZeroDivisionError, ValueError):
        return None


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def last_boxed_content(text: str) -> str | None:
    marker = r"\boxed{"
    start_marker = text.rfind(marker)
    if start_marker < 0:
        start_marker = text.rfind("\x08oxed{")
        if start_marker < 0:
            return None
        start = start_marker + len("\x08oxed{")
    else:
        start = start_marker + len(marker)
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1
    if depth == 0:
        return text[start : pos - 1].strip()
    return None


def final_answer_line(text: str) -> str | None:
    matches = re.findall(
        r"^\s*(?:final answer|answer)\s*[:\uFF1A]\s*(.+?)\s*$",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not matches:
        return None
    value = matches[-1].strip()
    boxed = last_boxed_content(value)
    return boxed if boxed is not None else value


def extract_math_answer(text: str) -> str:
    text = strip_think(text)
    final = final_answer_line(text)
    if final is not None:
        return final
    boxed = last_boxed_content(text)
    if boxed is not None:
        return boxed
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", text.replace(",", ""))
    if numbers:
        return numbers[-1].strip()
    return text.strip().splitlines()[-1].strip() if text.strip() else ""


def explicit_math_answer(text: str) -> str | None:
    text = strip_think(text)
    final = final_answer_line(text)
    if final is not None:
        return final
    return last_boxed_content(text)


def math_exact_match(prediction_text: str, answer: str) -> float:
    prediction = explicit_math_answer(prediction_text)
    if prediction is None:
        return 0.0
    pred_norm = normalize_math_answer(prediction)
    ans_norm = normalize_math_answer(answer)
    if pred_norm == ans_norm:
        return 1.0
    pred_num = decimal_value(pred_norm)
    ans_num = decimal_value(ans_norm)
    if pred_num is not None and ans_num is not None and abs(pred_num - ans_num) <= Decimal("1e-8"):
        return 1.0
    return 0.0


def math_verify_match(prediction_text: str, answer: str) -> float | None:
    if (
        math_verify_parse is None
        or math_verify_compare is None
        or ExprExtractionConfig is None
        or LatexExtractionConfig is None
    ):
        return None
    extraction_config = (LatexExtractionConfig(), ExprExtractionConfig())
    prediction = explicit_math_answer(prediction_text)
    if prediction is None:
        return None
    try:
        gold = math_verify_parse(str(answer), extraction_config=extraction_config)
        pred = math_verify_parse(prediction, extraction_config=extraction_config)
        if not gold or not pred:
            return None
        return 1.0 if math_verify_compare(gold, pred) else 0.0
    except Exception:  # noqa: BLE001 - fall back to exact match for malformed generations
        return None


def math_score(prediction_text: str, answer: str) -> tuple[float, str]:
    if explicit_math_answer(prediction_text) is None:
        return 0.0, "missing_final_answer"
    verified = math_verify_match(prediction_text, answer)
    if verified is not None:
        return verified, "math_verify"
    return math_exact_match(prediction_text, answer), "exact_fallback"


def normalize_doc_text(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if ca == cb else 1),
                )
            )
        previous = current
    return previous[-1]


def doc_anls(prediction: str, answers: list[str]) -> float:
    pred = normalize_doc_text(prediction)
    if not pred or not answers:
        return 0.0
    scores = []
    for answer in answers:
        ans = normalize_doc_text(answer)
        if not ans:
            continue
        distance = levenshtein(pred, ans)
        norm = distance / max(len(pred), len(ans), 1)
        scores.append(1.0 - norm if norm < 0.5 else 0.0)
    return max(scores) if scores else 0.0


def doc_acc_from_anls(anls_score: float) -> float:
    return 1.0 if anls_score > 0.5 else 0.0


def extract_doc_answer(text: str) -> str:
    text = strip_think(text)
    final = final_answer_line(text)
    if final is not None:
        return final
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def math_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Solve the math problem carefully. You may reason step by step. "
                "Put the final result in \\boxed{} at the end."
            ),
        },
        {"role": "user", "content": str(row.get("question", ""))},
    ]


def math_react_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    system = """You are a math reasoning agent. Solve the problem using a command-line Python ReAct loop.

You are not allowed to answer from the problem alone. First use the bash tool to run command-line Python for calculation, checking, symbolic manipulation, or search over cases. Then finish with the final answer.

Available action:

Action:
{"name": "bash", "arguments": {"command": "<shell command>"}}

Use command-line Python deliberately, for example python -c "...", for arithmetic, algebraic verification, brute force checks, or symbolic computation. When finished, output exactly:

Final answer: \\boxed{<answer>}

Do not include tool outputs in the final answer."""
    user = (
        "Task: Solve the following math problem.\n\n"
        f"{row.get('question', '')}\n\n"
        "You must call the bash action at least once before giving the final answer."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def resolve_docvqa_image(row: dict[str, Any], docvqa_root: Path) -> Path:
    image_path = Path(str(row.get("image", "")))
    if not image_path.is_absolute():
        image_path = docvqa_root / image_path
    return image_path


def docvqa_react_messages(row: dict[str, Any], docvqa_root: Path) -> list[dict[str, Any]]:
    image_path = resolve_docvqa_image(row, docvqa_root)
    system = """You are a DocVQA agent. You answer questions about document images using a command-line and Python ReAct loop.

You are not allowed to answer from the question alone. You must inspect or process the local image file using command-line tools and Python commands, then answer from the textual observations you produced.

Available action:

Action:
{"name": "bash", "arguments": {"command": "<shell command>"}}

The bash action runs in the image directory. Use shell commands and command-line Python, for example python -c "...", to inspect or process the provided image path.
Tool observations are text only. Do not expect the image to be displayed back to you.
When finished, output exactly:

Final answer: <short answer>

Return only the requested short answer after the Final answer prefix. Do not include reasoning in the final answer."""
    user = (
        "Task: Answer the document visual question.\n"
        f"Image path: {image_path}\n"
        f"Question: {row.get('question', '')}\n"
        "You must call at least one bash action before giving the final answer."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def response_text(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message")
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(choices[0].get("text", ""))


def is_context_length_error(response: httpx.Response) -> bool:
    try:
        body = response.json()
    except ValueError:
        body = {}
    text = response.text.lower()
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            param = str(error.get("param", "")).lower()
            message = str(error.get("message", "")).lower()
            text += "\n" + message
            if param == "input_tokens" and "context length" in message:
                return True
    return "context length" in text and "maximum input length" in text


def usage_add(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any] | None:
    if not left:
        return right
    if not right:
        return left
    merged = dict(left)
    for key, value in right.items():
        if isinstance(value, (int, float)) and isinstance(merged.get(key), (int, float)):
            merged[key] += value
        elif key not in merged:
            merged[key] = value
    return merged


def parse_react_action(text: str) -> dict[str, Any] | None:
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


def truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = max(limit // 2, 1)
    return text[:half] + "\n...[truncated]...\n" + text[-half:]


def run_bash(command: str, cwd: Path, timeout: float, limit: int) -> str:
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"Bash timed out after {timeout:.1f}s."
    except Exception as exc:  # Match the official bash tool: tool failures become observations.
        return f"[ERROR] Failed to execute command: {exc}"
    output = ""
    if proc.stdout:
        output += proc.stdout
    if proc.stderr:
        output += ("\n[stderr]\n" if output else "[stderr]\n") + proc.stderr
    if not output:
        output = f"Bash exited with code {proc.returncode} and no output."
    else:
        output += f"\n[exit_code] {proc.returncode}"
    return truncate_text(output, limit)


def react_observation_text(name: str, text: str) -> str:
    return f"Observation from {name}:\n{text}"


async def post_chat(
    *,
    client: httpx.AsyncClient,
    chat_url: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    args: argparse.Namespace,
    seed: int,
    enable_thinking: bool,
) -> tuple[str, dict[str, Any] | None]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "seed": seed,
        "stop": ["Observation:"],
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if max_tokens > 0:
        payload["max_tokens"] = max_tokens
    last_error: Exception | None = None
    for attempt in range(args.request_retries + 1):
        try:
            response = await client.post(chat_url, json=payload)
            if response.status_code == 400 and is_context_length_error(response):
                raise ContextLengthExceeded(response.text)
            response.raise_for_status()
            response_json = response.json()
            return response_text(response_json), response_json.get("usage") if isinstance(response_json, dict) else None
        except httpx.HTTPStatusError as exc:
            last_error = exc
            status = exc.response.status_code
            if status < 500 or attempt >= args.request_retries:
                raise
        except httpx.TransportError as exc:
            last_error = exc
            if attempt >= args.request_retries:
                raise
        await asyncio.sleep(min(2.0 * (attempt + 1), 10.0))
    assert last_error is not None
    raise last_error


async def run_math_react(
    *,
    client: httpx.AsyncClient,
    chat_url: str,
    model: str,
    row: dict[str, Any],
    row_index: int,
    sample_index: int,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any] | None, list[dict[str, Any]], str | None]:
    messages = math_react_messages(row)
    steps: list[dict[str, Any]] = []
    total_usage: dict[str, Any] | None = None
    used_bash = False
    seed_base = args.seed + sample_index * 1_000_003 + row_index

    for turn in range(args.math_max_turns):
        try:
            completion, usage = await post_chat(
                client=client,
                chat_url=chat_url,
                model=model,
                messages=messages,
                max_tokens=args.math_max_tokens,
                args=args,
                seed=seed_base + turn * 97,
                enable_thinking=False,
            )
        except ContextLengthExceeded:
            if not used_bash:
                return "", total_usage, steps, "no_bash_tool_use"
            return "", total_usage, steps, "max_react_turns_exceeded"
        total_usage = usage_add(total_usage, usage)
        messages.append({"role": "assistant", "content": completion})

        cleaned = strip_think(completion)
        final_match = final_answer_line(cleaned) is not None
        action = parse_react_action(cleaned)
        if final_match and not action:
            if used_bash:
                return completion, total_usage, steps, None
            warning = (
                "You must call the bash Action before answering. "
                "Use command-line Python to compute or verify the solution, then provide Final answer."
            )
            messages.append({"role": "user", "content": react_observation_text("format_check", warning)})
            steps.append({"turn": turn + 1, "assistant": completion, "observation": warning})
            continue

        if not action:
            warning = (
                'No valid action was parsed. Use exactly:\n'
                'Action:\n{"name": "bash", "arguments": {"command": "<shell command>"}}\n'
                'or finish after bash use with: Final answer: \\boxed{<answer>}'
            )
            messages.append({"role": "user", "content": react_observation_text("format_check", warning)})
            steps.append({"turn": turn + 1, "assistant": completion, "observation": warning})
            continue

        name = action["name"]
        arguments = action["arguments"]
        if name == "bash":
            command = str(arguments.get("command", ""))
            if not command.strip():
                observation = "No shell command was provided."
            else:
                used_bash = True
                observation = run_bash(
                    command,
                    ROOT,
                    timeout=args.math_python_timeout,
                    limit=args.tool_observation_limit,
                )
        else:
            observation = f"Unknown action '{name}'. Available action is bash."

        messages.append({"role": "user", "content": react_observation_text(name, observation)})
        steps.append(
            {
                "turn": turn + 1,
                "assistant": completion,
                "action": action,
                "observation": observation,
            }
        )

    if not used_bash:
        return "", total_usage, steps, "no_bash_tool_use"
    return "", total_usage, steps, "max_react_turns_exceeded"


async def run_docvqa_react(
    *,
    client: httpx.AsyncClient,
    chat_url: str,
    model: str,
    row: dict[str, Any],
    row_index: int,
    sample_index: int,
    docvqa_root: Path,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any] | None, list[dict[str, Any]], str | None]:
    image_path = resolve_docvqa_image(row, docvqa_root)
    messages = docvqa_react_messages(row, docvqa_root)
    steps: list[dict[str, Any]] = []
    total_usage: dict[str, Any] | None = None
    used_tool = False
    last_completion = ""
    seed_base = args.seed + sample_index * 1_000_003 + row_index

    for turn in range(args.docvqa_max_turns):
        try:
            completion, usage = await post_chat(
                client=client,
                chat_url=chat_url,
                model=model,
                messages=messages,
                max_tokens=args.docvqa_max_tokens,
                args=args,
                seed=seed_base + turn * 97,
                enable_thinking=False,
            )
        except ContextLengthExceeded:
            if not used_tool:
                return "", total_usage, steps, "no_tool_use"
            return last_completion, total_usage, steps, "max_react_turns_exceeded"
        last_completion = completion
        total_usage = usage_add(total_usage, usage)
        messages.append({"role": "assistant", "content": completion})

        cleaned = strip_think(completion)
        final_match = final_answer_line(cleaned) is not None
        action = parse_react_action(cleaned)
        if final_match and not action:
            if used_tool:
                return completion, total_usage, steps, None
            warning = (
                "You must use a bash Action to inspect/process the image file before answering. "
                "Tool observations are text only; then provide Final answer."
            )
            messages.append({"role": "user", "content": react_observation_text("format_check", warning)})
            steps.append({"turn": turn + 1, "assistant": completion, "observation": warning})
            continue

        if not action:
            warning = (
                'No valid action was parsed. Use exactly:\n'
                'Action:\n{"name": "bash", "arguments": {"command": "<shell command>"}}\n'
                'or finish after tool use with: Final answer: <short answer>'
            )
            messages.append({"role": "user", "content": react_observation_text("format_check", warning)})
            steps.append({"turn": turn + 1, "assistant": completion, "observation": warning})
            continue

        name = action["name"]
        arguments = action["arguments"]
        if name == "bash":
            command = str(arguments.get("command", ""))
            if not command.strip():
                observation = "No shell command was provided."
            else:
                used_tool = True
                observation = run_bash(
                    command,
                    image_path.parent,
                    timeout=args.docvqa_python_timeout,
                    limit=args.tool_observation_limit,
                )
            messages.append({"role": "user", "content": react_observation_text(name, observation)})
        else:
            observation = f"Unknown action '{name}'. Available action is bash."
            messages.append({"role": "user", "content": react_observation_text(name, observation)})

        steps.append(
            {
                "turn": turn + 1,
                "assistant": completion,
                "action": action,
                "observation": observation,
            }
        )

    if not used_tool:
        return "", total_usage, steps, "no_tool_use"
    return last_completion, total_usage, steps, "max_react_turns_exceeded"


def completed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys = set()
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("key"):
                keys.add(str(row["key"]))
    return keys


async def request_one(
    *,
    client: httpx.AsyncClient,
    chat_url: str,
    model: str,
    dataset: DatasetSpec,
    row: dict[str, Any],
    row_index: int,
    sample_index: int,
    docvqa_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    task_id = str(row.get("id", row_index))
    key = f"{dataset.name}:{task_id}:sample{sample_index:02d}"
    max_tokens = args.math_max_tokens if dataset.kind == "math" else args.docvqa_max_tokens
    started_at = time.time()
    error = None
    react_error = None
    usage: dict[str, Any] | None = None
    react_steps: list[dict[str, Any]] | None = None
    try:
        if dataset.kind == "math":
            completion, usage, react_steps, react_error = await run_math_react(
                client=client,
                chat_url=chat_url,
                model=model,
                row=row,
                row_index=row_index,
                sample_index=sample_index,
                args=args,
            )
        else:
            completion, usage, react_steps, react_error = await run_docvqa_react(
                client=client,
                chat_url=chat_url,
                model=model,
                row=row,
                row_index=row_index,
                sample_index=sample_index,
                docvqa_root=docvqa_root,
                args=args,
            )
    except Exception as exc:  # noqa: BLE001 - keep per-case errors in output
        completion = ""
        error = f"{type(exc).__name__}: {exc}"
    latency = time.time() - started_at

    if dataset.kind == "math":
        target = str(row.get("answer", ""))
        prediction = extract_math_answer(completion)
        if error:
            score = -1.0
            score_method = "request_error"
        else:
            score, score_method = math_score(completion, target)
        extra = {
            "target": target,
            "mode": "paper_react_cli",
            "react_error": react_error,
            "react_steps": react_steps or [],
            "score_method": f"math_paper_react_cli_{score_method}" if not error else score_method,
        }
    else:
        answers = [str(answer) for answer in row.get("answers", [])]
        prediction = extract_doc_answer(completion)
        anls = doc_anls(prediction, answers)
        acc = doc_acc_from_anls(anls)
        score = acc
        extra = {
            "answers": answers,
            "image": row.get("image", ""),
            "mode": "paper_react_cli",
            "react_error": react_error,
            "react_steps": react_steps or [],
            "anls": anls,
            "vlns": anls,
            "acc": acc,
            "score_method": "docvqa_paper_react_cli_anls_gt_0.5_acc" if not error else "request_error",
        }

    return {
        "key": key,
        "dataset": dataset.name,
        "task_id": task_id,
        "row_index": row_index,
        "sample_index": sample_index,
        "enable_thinking": dataset.enable_thinking,
        "score": score,
        "prediction": prediction,
        "completion": completion,
        "latency_s": latency,
        "error": error,
        "usage": usage,
        **extra,
    }


def summarize_output(path: Path) -> dict[str, Any]:
    rows = []
    if path.exists():
        with path.open(encoding="utf-8", errors="replace") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    def primary_score(row: dict[str, Any]) -> float:
        if "anls" in row:
            return doc_acc_from_anls(float(row.get("anls", -1.0)))
        return float(row.get("score", -1.0))

    valid = [primary_score(row) for row in rows if primary_score(row) >= 0.0]
    by_sample: dict[int, list[float]] = {}
    by_task: dict[str, list[float]] = {}
    score_methods: dict[str, int] = {}
    metric_means: dict[str, float] = {}
    for metric in ("anls", "vlns", "acc"):
        values = [float(row[metric]) for row in rows if metric in row and float(row[metric]) >= 0.0]
        if values:
            metric_means[f"mean_{metric}"] = sum(values) / len(values)
            metric_by_task: dict[str, list[float]] = {}
            for row in rows:
                if metric not in row or float(row[metric]) < 0.0:
                    continue
                metric_by_task.setdefault(str(row.get("task_id", row.get("row_index", ""))), []).append(float(row[metric]))
            if metric_by_task:
                metric_means[f"max@16_{metric}"] = sum(max(scores) for scores in metric_by_task.values()) / len(
                    metric_by_task
                )
    for row in rows:
        score = primary_score(row)
        if score >= 0.0:
            by_sample.setdefault(int(row.get("sample_index", -1)), []).append(score)
            by_task.setdefault(str(row.get("task_id", row.get("row_index", ""))), []).append(score)
        if row.get("score_method"):
            method = str(row["score_method"])
            score_methods[method] = score_methods.get(method, 0) + 1
    max_at_16 = sum(max(scores) for scores in by_task.values()) / len(by_task) if by_task else -1.0
    return {
        "output": str(path),
        "records": len(rows),
        "valid_records": len(valid),
        "mean_score": sum(valid) / len(valid) if valid else -1.0,
        "max@16": max_at_16,
        "score_methods": score_methods,
        **metric_means,
        "by_sample": {
            str(idx): {
                "count": len(scores),
                "mean_score": sum(scores) / len(scores) if scores else -1.0,
            }
            for idx, scores in sorted(by_sample.items())
        },
    }


async def run_dataset(dataset: DatasetSpec, args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(dataset.path)
    if dataset.limit is not None:
        rows = rows[: dataset.limit]
    out_path = args.out_dir / "outputs" / f"{dataset.name}.jsonl"
    summary_path = args.out_dir / "summaries" / f"{dataset.name}.summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    done = completed_keys(out_path) if args.resume else set()
    print(
        f"[{dataset.name}] start rows={len(rows)} samples={args.samples} "
        f"resume_done={len(done)} thinking={dataset.enable_thinking} "
        f"max_tokens={dataset.max_tokens} "
        f"mode={'paper_react_cli' if dataset.kind in ('math', 'docvqa') else 'n/a'}",
        flush=True,
    )
    jobs = []
    for row_index, row in enumerate(rows):
        task_id = str(row.get("id", row_index))
        for sample_index in range(args.samples):
            key = f"{dataset.name}:{task_id}:sample{sample_index:02d}"
            if key not in done:
                jobs.append((row_index, row, sample_index))
    print(f"[{dataset.name}] pending_jobs={len(jobs)}", flush=True)

    chat_url = args.base_url.rstrip("/") + "/chat/completions"
    timeout = None if args.timeout <= 0 else args.timeout
    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=0)
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    abort_event = asyncio.Event()
    started_at = time.time()
    completed = 0
    error_count = 0

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        async def guarded(job: tuple[int, dict[str, Any], int]) -> None:
            nonlocal completed, error_count
            if abort_event.is_set():
                return
            row_index, row, sample_index = job
            async with semaphore:
                if abort_event.is_set():
                    return
                rec = await request_one(
                    client=client,
                    chat_url=chat_url,
                    model=args.model,
                    dataset=dataset,
                    row=row,
                    row_index=row_index,
                    sample_index=sample_index,
                    docvqa_root=args.docvqa_root,
                    args=args,
            )
            async with write_lock:
                write_jsonl_record(out_path, rec)
                completed += 1
                if rec.get("error"):
                    error_count += 1
                    if error_count >= args.max_errors:
                        abort_event.set()
                if completed == 1 or completed % args.log_every == 0 or completed == len(jobs):
                    elapsed = time.time() - started_at
                    rate = completed / elapsed if elapsed > 0 else math.nan
                    print(
                        f"[{dataset.name}] completed {completed}/{len(jobs)} "
                        f"errors={error_count} score={rec.get('score')} rate={rate:.2f}/s",
                        flush=True,
                    )

        await asyncio.gather(*(guarded(job) for job in jobs))

    summary = summarize_output(out_path)
    summary.update(
        {
            "dataset": dataset.name,
            "kind": dataset.kind,
            "data": str(dataset.path),
            "items": len(rows),
            "samples": args.samples,
            "expected_records": len(rows) * args.samples,
            "enable_thinking": dataset.enable_thinking,
            "mode": "paper_react_cli" if dataset.kind in ("math", "docvqa") else None,
            "math_max_turns": args.math_max_turns if dataset.kind == "math" else None,
            "docvqa_max_turns": args.docvqa_max_turns if dataset.kind == "docvqa" else None,
            "sampling": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "min_p": args.min_p,
                "presence_penalty": args.presence_penalty,
                "repetition_penalty": args.repetition_penalty,
            },
        }
    )
    write_json(summary_path, summary)
    if error_count >= args.max_errors:
        raise RuntimeError(
            f"{dataset.name} aborted after {error_count} request errors; "
            f"see {out_path} and server logs"
        )
    return summary


def build_datasets(args: argparse.Namespace) -> list[DatasetSpec]:
    dataset_arg = args.datasets.replace("\\,", ",")
    wanted = {item.strip() for item in dataset_arg.split(",") if item.strip()}
    all_specs = {
        "dapo100": DatasetSpec(
            name="dapo100",
            kind="math",
            path=args.math_root / "dapo_test.jsonl",
            enable_thinking=False,
            max_tokens=args.math_max_tokens,
            limit=100,
        ),
        "aime2026": DatasetSpec(
            name="aime2026",
            kind="math",
            path=args.math_root / "aime_2026.jsonl",
            enable_thinking=False,
            max_tokens=args.math_max_tokens,
        ),
        "docvqa": DatasetSpec(
            name="docvqa",
            kind="docvqa",
            path=args.docvqa_root / "data/trace2skill/docvqa/test.jsonl",
            enable_thinking=False,
            max_tokens=args.docvqa_max_tokens,
            limit=args.docvqa_limit if args.docvqa_limit > 0 else None,
        ),
    }
    unknown = wanted - set(all_specs)
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    return [all_specs[name] for name in ("dapo100", "aime2026", "docvqa") if name in wanted]


async def main_async(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    specs = build_datasets(args)
    manifest = {
        "model": args.model,
        "base_url": args.base_url,
        "datasets": [spec.name for spec in specs],
        "samples": args.samples,
        "limits": {spec.name: spec.limit for spec in specs if spec.limit is not None},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(args.out_dir / "manifest.json", manifest)
    summaries = {}
    for spec in specs:
        if not spec.path.exists():
            raise FileNotFoundError(spec.path)
        summaries[spec.name] = await run_dataset(spec, args)
    write_json(args.out_dir / "summary.json", summaries)
    print(json.dumps(summaries, indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--model", default="Qwen3.5-27B")
    parser.add_argument("--math-root", type=Path, default=DEFAULT_MATH_ROOT)
    parser.add_argument("--docvqa-root", type=Path, default=DEFAULT_DOCVQA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--datasets", default="dapo100,aime2026,docvqa")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--request-retries", type=int, default=3)
    parser.add_argument("--max-errors", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=2.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--math-max-tokens", type=int, default=512)
    parser.add_argument("--math-max-turns", type=int, default=30)
    parser.add_argument("--math-python-timeout", type=float, default=20.0)
    parser.add_argument("--docvqa-max-tokens", type=int, default=512)
    parser.add_argument("--docvqa-max-turns", type=int, default=30)
    parser.add_argument("--docvqa-python-timeout", type=float, default=20.0)
    parser.add_argument("--docvqa-limit", type=int, default=0)
    parser.add_argument("--tool-observation-limit", type=int, default=6000)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
