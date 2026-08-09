from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

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
    # Tool rollouts run unattended.  In addition to process/system controls,
    # reject desktop launchers so a model-generated command cannot repeatedly
    # open browser windows or interactive GUI prompts on the host session.
    pattern = (
        r"(^|[;&|()`$<>\s])(?:"
        r"sudo|su|pkexec|gksu|gksudo|kdesu|"
        r"kill|pkill|killall|reboot|shutdown|halt|poweroff|screen|tmux|ray|"
        r"xdg-open|gio|sensible-browser|x-www-browser|"
        r"firefox|google-chrome(?:-stable)?|chromium(?:-browser)?|brave(?:-browser)?|"
        r"opera|epiphany|zenity|kdialog|xmessage|notify-send"
        r")(\s|$)"
    )
    return re.search(pattern, command, flags=re.IGNORECASE) is not None


def run_bash(command: str, cwd: Path, timeout: float, limit: int) -> str:
    log_path = cwd / ".math_tool_commands.log"
    try:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] timeout={timeout}\n{command}\n")
    except Exception:
        pass
    if is_dangerous_bash_command(command):
        return "Blocked unsafe or interactive desktop command in math tool sandbox."
    child_env = os.environ.copy()
    for name in ("DISPLAY", "WAYLAND_DISPLAY", "DBUS_SESSION_BUS_ADDRESS"):
        child_env.pop(name, None)
    child_env["BROWSER"] = "/bin/false"
    proc: subprocess.Popen[str] | None = None
    try:
        proc = subprocess.Popen(
            limit_bash_command(command),
            shell=True,
            executable="/bin/bash",
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env=child_env,
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
