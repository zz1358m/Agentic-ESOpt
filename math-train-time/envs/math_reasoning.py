from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib import request

try:
    from math_verify import parse as math_verify_parse
    from math_verify import verify as math_verify_compare
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:  # pragma: no cover - cluster env dependency check catches this
    math_verify_parse = None
    math_verify_compare = None
    ExprExtractionConfig = None
    LatexExtractionConfig = None


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class MathTask:
    id: str
    question: str
    answer: str
    source: str


@dataclass(frozen=True)
class MathRolloutJob:
    task: MathTask
    row_index: int
    sample_index: int = 0


class ContextLengthExceeded(RuntimeError):
    pass


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_tasks(path: Path, limit: int = 0) -> list[MathTask]:
    tasks = [
        MathTask(
            id=str(row.get("id", idx)),
            question=str(row.get("question", "")),
            answer=str(row.get("answer", "")),
            source=str(row.get("source", "")),
        )
        for idx, row in enumerate(read_jsonl(path))
    ]
    if limit > 0:
        tasks = tasks[:limit]
    if not tasks:
        raise RuntimeError(f"No math tasks loaded from {path}")
    return tasks


def post_json(url: str, payload: dict[str, Any], timeout: int | float = 900) -> dict[str, Any]:
    data = json.dumps(payload).encode()
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def chat_completion_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/v1/chat/completions"):
        return endpoint
    if endpoint.endswith("/chat/completions"):
        return endpoint
    if endpoint.endswith("/v1"):
        return f"{endpoint}/chat/completions"
    return f"{endpoint}/v1/chat/completions"


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
        r"^\s*(?:final answer|answer)\s*[:：]\s*(.+?)\s*$",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not matches:
        return None
    value = matches[-1].strip()
    boxed = last_boxed_content(value)
    return boxed if boxed is not None else value


def explicit_math_answer(text: str) -> str | None:
    text = strip_think(text)
    final = final_answer_line(text)
    if final is not None:
        return final
    return last_boxed_content(text)


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
            result = Decimal(left) / Decimal(right)
        else:
            result = Decimal(value)
    except (InvalidOperation, ZeroDivisionError, ValueError):
        return None
    return result if result.is_finite() else None


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
    if pred_num is not None and ans_num is not None:
        try:
            if abs(pred_num - ans_num) <= Decimal("1e-8"):
                return 1.0
        except (InvalidOperation, ValueError):
            pass
    return 0.0


def math_verify_match(prediction_text: str, answer: str) -> float | None:
    if (
        math_verify_parse is None
        or math_verify_compare is None
        or ExprExtractionConfig is None
        or LatexExtractionConfig is None
    ):
        return None
    prediction = explicit_math_answer(prediction_text)
    if prediction is None:
        return None
    extraction_config = (LatexExtractionConfig(), ExprExtractionConfig())
    try:
        gold = math_verify_parse(str(answer), extraction_config=extraction_config)
        pred = math_verify_parse(prediction, extraction_config=extraction_config)
        if not gold or not pred:
            return None
        return 1.0 if math_verify_compare(gold, pred) else 0.0
    except Exception:
        return None


def math_score(prediction_text: str, answer: str) -> tuple[float, str]:
    if explicit_math_answer(prediction_text) is None:
        return 0.0, "missing_final_answer"
    verified = math_verify_match(prediction_text, answer)
    if verified is not None:
        return verified, "math_verify"
    return math_exact_match(prediction_text, answer), "exact_fallback"


def math_react_messages(task: MathTask, skill: str = "") -> list[dict[str, Any]]:
    skill_text = skill.strip()
    skill_block = f"\n\nAdditional skill instructions:\n{skill_text}" if skill_text else ""
    system = f"""You are a math reasoning agent. Solve the problem using a command-line Python ReAct loop.

You are not allowed to answer from the problem alone. First use the bash tool to run command-line Python for calculation, checking, symbolic manipulation, or search over cases. Then finish with the final answer.

Available action:

Action:
{{"name": "bash", "arguments": {{"command": "<shell command>"}}}}

Use command-line Python deliberately, for example python -c "...", for arithmetic, algebraic verification, brute force checks, or symbolic computation. When finished, output exactly:

Final answer: \\boxed{{<answer>}}

Do not include tool outputs in the final answer.{skill_block}"""
    user = (
        "Task: Solve the following math problem.\n\n"
        f"{task.question}\n\n"
        "You must call the bash action at least once before giving the final answer."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def trim_oldest_react_exchange(messages: list[dict[str, Any]]) -> bool:
    """Drop the oldest generated turn while preserving the system prompt and task."""
    if len(messages) <= 2:
        return False
    remove_count = 2 if len(messages) >= 4 else 1
    del messages[2 : 2 + remove_count]
    return True


def response_text(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message")
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(choices[0].get("text", ""))


def is_context_length_error(response_json: dict[str, Any], response_text_value: str) -> bool:
    text = response_text_value.lower()
    error = response_json.get("error")
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
        match = re.search(r"Action:\s*(\{.*)", text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    raw = match.group(1).strip()
    decoder = json.JSONDecoder()
    try:
        action, _ = decoder.raw_decode(raw)
    except json.JSONDecodeError:
        action = parse_relaxed_bash_action(raw)
    if not isinstance(action, dict):
        return None
    name = action.get("name")
    arguments = action.get("arguments", {})
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    return {"name": name, "arguments": arguments}


def parse_relaxed_bash_action(raw: str) -> dict[str, Any] | None:
    if not re.search(r"""["']name["']\s*:\s*["']bash["']""", raw, flags=re.IGNORECASE):
        return None
    match = re.search(r"""["']command["']\s*:\s*(["'])""", raw, flags=re.IGNORECASE)
    if not match:
        return None
    quote = match.group(1)
    start = match.end()
    end = start
    escaped = False
    while end < len(raw):
        ch = raw[end]
        if escaped:
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == quote:
            break
        end += 1
    command_raw = raw[start:end] if end < len(raw) else raw[start:]
    command = decode_relaxed_json_string(command_raw).strip()
    if not command:
        return None
    return {"name": "bash", "arguments": {"command": command}}


def decode_relaxed_json_string(raw: str) -> str:
    try:
        return json.loads('"' + raw.replace('"', '\\"') + '"')
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(f'"{raw}"')
    except json.JSONDecodeError:
        return (
            raw.replace(r"\\", "\\")
            .replace(r"\"", '"')
            .replace(r"\'", "'")
            .replace(r"\n", "\n")
            .replace(r"\t", "\t")
        )


def truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = max(limit // 2, 1)
    return text[:half] + "\n...[truncated]...\n" + text[-half:]


def safe_workdir_name(value: str, *, fallback: str = "item", limit: int = 96) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._-")
    return (name[:limit] or fallback)


def limit_bash_command(command: str) -> str:
    memory_limit_gb = float(os.environ.get("MATH_TOOL_MEMORY_LIMIT_GB", "8"))
    if memory_limit_gb <= 0:
        return command
    memory_limit_kb = max(1024, int(memory_limit_gb * 1024 * 1024))
    return f"ulimit -v {memory_limit_kb}; {command}"


def is_dangerous_bash_command(command: str) -> bool:
    pattern = r"(^|[;&|()`$<>\s])(?:sudo|su|kill|pkill|killall|reboot|shutdown|halt|poweroff|screen|tmux|ray)(\s|$)"
    return re.search(pattern, command, flags=re.IGNORECASE) is not None


def run_bash(command: str, cwd: Path, timeout: float, limit: int) -> str:
    log_path = cwd / ".math_tool_commands.log"
    try:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] timeout={timeout}\n{command}\n")
    except Exception:
        pass
    if is_dangerous_bash_command(command):
        return "Blocked unsafe shell command in math tool sandbox."
    proc: subprocess.Popen[str] | None = None
    try:
        proc = subprocess.Popen(
            limit_bash_command(command),
            shell=True,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            start_new_session=True,
        )
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = proc.communicate()
        return f"Bash timed out after {timeout:.1f}s."
    except Exception as exc:
        return f"[ERROR] Failed to execute command: {exc}"
    output = ""
    if stdout:
        output += stdout
    if stderr:
        output += ("\n[stderr]\n" if output else "[stderr]\n") + stderr
    if not output:
        output = f"Bash exited with code {proc.returncode} and no output."
    else:
        output += f"\n[exit_code] {proc.returncode}"
    return truncate_text(output, limit)


def react_observation_text(name: str, text: str) -> str:
    return f"Observation from {name}:\n{text}"


def post_chat(
    *,
    chat_url: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    presence_penalty: float,
    repetition_penalty: float,
    timeout: int | float,
    seed: int,
    request_retries: int,
    enable_thinking: bool = False,
) -> tuple[str, dict[str, Any] | None]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "presence_penalty": presence_penalty,
        "repetition_penalty": repetition_penalty,
        "seed": seed,
        "stop": ["Observation:"],
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        "enable_thinking": enable_thinking,
    }
    if max_tokens > 0:
        payload["max_tokens"] = max_tokens

    last_error: Exception | None = None
    for attempt in range(request_retries + 1):
        try:
            data = json.dumps(payload).encode()
            req = request.Request(chat_url, data=data, headers={"Content-Type": "application/json"}, method="POST")
            with request.urlopen(req, timeout=timeout) as resp:
                text = resp.read().decode()
                response_json = json.loads(text)
            if is_context_length_error(response_json, text):
                raise ContextLengthExceeded(text)
            return response_text(response_json), response_json.get("usage") if isinstance(response_json, dict) else None
        except ContextLengthExceeded:
            raise
        except Exception as exc:
            last_error = exc
            if attempt >= request_retries:
                raise
            time.sleep(min(2.0 * (attempt + 1), 10.0))
    assert last_error is not None
    raise last_error


def trace_markdown(*, task: MathTask, row: dict[str, Any], transcript: list[dict[str, Any]]) -> str:
    score = float(row.get("score", -1.0))
    outcome = "SUCCEED" if score >= 1.0 else "FAILED"
    lines = [
        f"# Chat History math_reasoning_{task.id}",
        "",
        f"Task ID: {task.id}",
        "Setting: math_reasoning",
        f"Sample index: {row.get('sample_index', 0)}",
        f"Score: {score}",
        f"Score method: {row.get('score_method', '')}",
        f"Outcome: {outcome}",
        f"Failure reason: {row.get('react_error') or ('Score was 0.' if score < 1.0 else '')}",
        "",
        "## Problem",
        "",
        task.question,
        "",
        f"Expected answer: {task.answer}",
        "",
        "## Trace",
        "",
    ]
    for idx, turn in enumerate(transcript, 1):
        lines.extend(
            [
                f"## Round {idx}",
                "",
                "### Assistant",
                "",
                str(turn.get("assistant", "")).strip(),
                "",
            ]
        )
        if turn.get("action"):
            lines.extend(["### Action", "", "```json", json.dumps(turn["action"], ensure_ascii=False), "```", ""])
        observation = str(turn.get("observation", "")).strip()
        if observation:
            lines.extend(["### Observation", "", "```text", observation, "```", ""])
    lines.extend(
        [
            "## Prediction",
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


class MathReasoningEnv:
    def __init__(
        self,
        data_path: str | Path,
        *,
        limit: int = 0,
        skill_file: str | Path | None = None,
        tool_work_root: str | Path | None = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.tasks = load_tasks(self.data_path, limit)
        self.tool_work_root = Path(tool_work_root) if tool_work_root else (
            ROOT / "runs" / "math_tool_workdirs" / safe_workdir_name(self.data_path.stem)
        )
        self.tool_work_root.mkdir(parents=True, exist_ok=True)
        self.skill = ""
        if skill_file:
            path = Path(skill_file)
            if path.exists():
                self.skill = path.read_text(encoding="utf-8")

    def rollout_one(
        self,
        *,
        endpoint: str,
        job: MathRolloutJob,
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        presence_penalty: float,
        repetition_penalty: float,
        timeout: int,
        request_retries: int,
        max_turns: int,
        python_timeout: float,
        tool_observation_limit: int,
        seed: int,
    ) -> dict[str, Any]:
        task = job.task
        key = f"{task.id}:sample{job.sample_index:02d}"
        messages = math_react_messages(task, self.skill)
        steps: list[dict[str, Any]] = []
        total_usage: dict[str, Any] | None = None
        used_bash = False
        completion = ""
        react_error = None
        termination_reason = None
        context_trims = 0
        started_at = time.time()
        seed_base = seed + job.sample_index * 1_000_003 + job.row_index
        workdir = (
            self.tool_work_root
            / safe_workdir_name(task.id, fallback=f"row{job.row_index:05d}")
            / f"row{job.row_index:05d}_sample{job.sample_index:02d}_seed{seed_base}"
        )
        workdir.mkdir(parents=True, exist_ok=True)

        try:
            turn = 0
            while max_turns <= 0 or turn < max_turns:
                while True:
                    try:
                        completion, usage = post_chat(
                            chat_url=chat_completion_url(endpoint),
                            model=model,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            min_p=min_p,
                            presence_penalty=presence_penalty,
                            repetition_penalty=repetition_penalty,
                            timeout=timeout,
                            seed=seed_base + turn * 97,
                            request_retries=request_retries,
                            enable_thinking=False,
                        )
                        break
                    except ContextLengthExceeded:
                        if not trim_oldest_react_exchange(messages):
                            react_error = "context_length_exceeded"
                            termination_reason = "context_length_exceeded"
                            break
                        context_trims += 1
                if termination_reason == "context_length_exceeded":
                    break
                turn += 1
                total_usage = usage_add(total_usage, usage)
                messages.append({"role": "assistant", "content": completion})

                cleaned = strip_think(completion)
                final_match = final_answer_line(cleaned) is not None
                action = parse_react_action(cleaned)
                if final_match and not action:
                    if used_bash:
                        termination_reason = "final_answer"
                        break
                    warning = (
                        "You must call the bash Action before answering. "
                        "Use command-line Python to compute or verify the solution, then provide Final answer."
                    )
                    messages.append({"role": "user", "content": react_observation_text("format_check", warning)})
                    steps.append({"turn": turn, "assistant": completion, "observation": warning})
                    continue

                if not action:
                    warning = (
                        'No valid action was parsed. Use exactly:\n'
                        'Action:\n{"name": "bash", "arguments": {"command": "<shell command>"}}\n'
                        'or finish after bash use with: Final answer: \\boxed{<answer>}'
                    )
                    messages.append({"role": "user", "content": react_observation_text("format_check", warning)})
                    steps.append({"turn": turn, "assistant": completion, "observation": warning})
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
                            workdir,
                            timeout=python_timeout,
                            limit=tool_observation_limit,
                        )
                else:
                    observation = f"Unknown action '{name}'. Available action is bash."

                messages.append({"role": "user", "content": react_observation_text(name, observation)})
                steps.append(
                    {
                        "turn": turn,
                        "assistant": completion,
                        "action": action,
                        "observation": observation,
                    }
                )
            if termination_reason is None:
                react_error = "max_react_turns_exceeded"
                termination_reason = "max_turns"

            answer_status = "answered" if final_answer_line(strip_think(completion)) is not None else "missing_final_answer"
            if termination_reason == "context_length_exceeded":
                score = 0.0
                score_method = "context_length_exceeded"
            else:
                score, score_method = math_score(completion, task.answer)
            if not used_bash:
                score = 0.0
                score_method = "no_bash_tool_use"
            return {
                "key": key,
                "task_id": task.id,
                "row_index": job.row_index,
                "sample_index": job.sample_index,
                "score": score,
                "answer": task.answer,
                "prediction": extract_math_answer(completion),
                "response": completion,
                "completion": completion,
                "latency_s": time.time() - started_at,
                "usage": total_usage,
                "mode": "paper_react_cli",
                "tool_workdir": str(workdir),
                "used_bash": used_bash,
                "termination_reason": termination_reason or "final_answer",
                "answer_status": answer_status,
                "context_trims": context_trims,
                "react_error": react_error,
                "react_steps": steps,
                "trace_rounds": len(steps),
                "score_method": f"math_paper_react_cli_{score_method}",
            }
        except Exception as exc:
            return {
                "key": key,
                "task_id": task.id,
                "row_index": job.row_index,
                "sample_index": job.sample_index,
                "score": 0.0,
                "answer": task.answer,
                "prediction": "",
                "response": "",
                "completion": "",
                "latency_s": time.time() - started_at,
                "mode": "paper_react_cli",
                "tool_workdir": str(workdir),
                "used_bash": used_bash,
                "termination_reason": "request_error",
                "answer_status": "missing_final_answer",
                "context_trims": context_trims,
                "react_error": f"{type(exc).__name__}: {exc}",
                "error": f"{type(exc).__name__}: {exc}",
                "score_method": "request_error",
                "react_steps": steps,
            }

    def rollout_batch(
        self,
        *,
        endpoint: str,
        jobs: list[MathRolloutJob],
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        presence_penalty: float,
        repetition_penalty: float,
        timeout: int,
        request_retries: int,
        max_turns: int,
        python_timeout: float,
        tool_observation_limit: int,
        seed: int,
        concurrency: int,
        trace_dir: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        if not jobs:
            return []
        rows: list[dict[str, Any]] = []
        max_workers = max(1, min(int(concurrency), len(jobs)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(
                    self.rollout_one,
                    endpoint=endpoint,
                    job=job,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    presence_penalty=presence_penalty,
                    repetition_penalty=repetition_penalty,
                    timeout=timeout,
                    request_retries=request_retries,
                    max_turns=max_turns,
                    python_timeout=python_timeout,
                    tool_observation_limit=tool_observation_limit,
                    seed=seed,
                )
                for job in jobs
            ]
            for future in as_completed(futures):
                rows.append(future.result())

        trace_root = Path(trace_dir) if trace_dir else None
        if trace_root is not None:
            trace_root.mkdir(parents=True, exist_ok=True)
            task_by_key = {(job.task.id, job.sample_index): job.task for job in jobs}
            for row in rows:
                task = task_by_key[(str(row["task_id"]), int(row.get("sample_index", 0)))]
                outcome = "SUCCEED" if float(row.get("score", -1.0)) >= 1.0 else "FAILED"
                safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", task.id)
                sample_index = int(row.get("sample_index", 0))
                path = trace_root / f"math_agent_{safe_id}_sample{sample_index:02d}_{outcome}.md"
                path.write_text(trace_markdown(task=task, row=row, transcript=row.get("react_steps", [])), encoding="utf-8")
                row["trace_log"] = str(path)

        rows.sort(key=lambda row: (int(row.get("row_index", 0)), int(row.get("sample_index", 0))))
        return rows
