"""Normalize raw rollout generations into replayable trajectory records."""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


_ACTION_MARKER = re.compile(r"Action:\s*", re.IGNORECASE)
_OBSERVATION_MARKER = re.compile(r"Observation from bash:\s*\n?", re.IGNORECASE)
_STEP_BOUNDARY = re.compile(
    r"\n\s*(?:Action:|Final answer:|Answer:|assistant\s*\n|user\s*\n)",
    re.IGNORECASE,
)
_RAW_DUMP_NAME = re.compile(r"^(\d+)(?:\.attempt(\d+))?\.jsonl$")


def parse_react_steps(output: str) -> list[dict[str, Any]]:
    """Extract valid JSON bash actions and their following observations."""
    parsed_actions: list[tuple[int, int, dict[str, Any]]] = []
    for marker in _ACTION_MARKER.finditer(output):
        payload_start = marker.end()
        candidate = output[payload_start:].lstrip()
        leading_space = len(output[payload_start:]) - len(candidate)
        try:
            action, consumed = json.JSONDecoder().raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(action, dict):
            continue
        action_end = payload_start + leading_space + consumed

        # Format-check observations quote the protocol back to the model.  They
        # are user feedback, not actions emitted by the assistant.
        feedback_start = output.rfind("Observation from format_check:", 0, marker.start())
        assistant_start = output.rfind("\nassistant\n", 0, marker.start())
        if feedback_start > assistant_start:
            continue
        arguments = action.get("arguments")
        command = arguments.get("command") if isinstance(arguments, dict) else None
        if action.get("name") != "bash" or not isinstance(command, str) or not command.strip():
            continue
        parsed_actions.append((marker.start(), action_end, action))

    steps = []
    for index, (_, action_end, action) in enumerate(parsed_actions):
        observation_match = _OBSERVATION_MARKER.search(output, action_end)
        next_action_start = (
            parsed_actions[index + 1][0] if index + 1 < len(parsed_actions) else None
        )
        if observation_match is None or (
            next_action_start is not None and next_action_start < observation_match.start()
        ):
            observation = None
        else:
            observation_start = observation_match.end()
            boundary = _STEP_BOUNDARY.search(output, observation_start)
            observation_end = boundary.start() if boundary else len(output)
            observation = output[observation_start:observation_end].strip()
        # Only an action followed by the tool's observation was actually
        # executed.  Valid-looking calls rejected by the first-turn policy or
        # cut off at the response limit remain visible in the full transcript,
        # but must not become phantom replay steps.
        if observation is not None:
            steps.append({"action": action, "observation": observation})
    return steps


def normalize_evaluation_record(
    evaluation: dict[str, Any],
    *,
    phase: str,
    epoch: int,
    step: int,
) -> dict[str, Any]:
    """Convert standalone evaluator output to the common trajectory shape."""
    dataset = str(evaluation.get("dataset", "math"))
    source_id = evaluation.get("task_id", evaluation.get("row_index", "unknown"))
    sample_index = int(evaluation.get("sample_index", 0))
    react_steps = evaluation.get("react_steps") or []
    transcript = []
    tool_used = False
    for react_step in react_steps:
        assistant = str(react_step.get("assistant", ""))
        observation = str(react_step.get("observation", ""))
        if assistant:
            transcript.append(assistant)
        if observation:
            transcript.append(f"Observation from bash:\n{observation}")
        tool_used = tool_used or any(
            parsed.get("action", {}).get("name") == "bash"
            for parsed in parse_react_steps(
                assistant + (f"\nObservation from bash:\n{observation}" if observation else "")
            )
        )
    completion = str(evaluation.get("completion", ""))
    if completion:
        transcript.append(completion)
    usage = evaluation.get("usage") or {}
    return {
        **evaluation,
        "trajectory_id": (
            f"{_identity_fragment(phase)}-{_identity_fragment(dataset)}-"
            f"{_identity_fragment(source_id)}-sample{sample_index:02d}"
        ),
        "phase": phase,
        "epoch": epoch,
        "global_step": step,
        "source_id": source_id,
        "row_index": evaluation.get("row_index"),
        "split": dataset,
        "rollout_index": sample_index,
        "input": str(evaluation.get("question", "")),
        "output": "\n".join(transcript),
        "gts": evaluation.get("target", evaluation.get("answers")),
        "steps": react_steps,
        "tool_used": float(tool_used),
        "num_turns": len(react_steps) * 2 + 3,
        "prompt_tokens": usage.get("prompt_tokens"),
        "response_tokens": usage.get("completion_tokens"),
    }


def load_raw_trajectory_records(
    dump_dir: Path,
    *,
    latest_only: bool = False,
) -> list[dict[str, Any]]:
    """Load immutable dumps, optionally selecting the latest attempt per step."""
    dump_files = []
    for path in dump_dir.glob("*.jsonl"):
        match = _RAW_DUMP_NAME.match(path.name)
        if match:
            dump_files.append((int(match.group(1)), int(match.group(2) or 1), path))
    dump_files.sort()
    if latest_only:
        latest_by_step: dict[int, tuple[int, int, Path]] = {}
        for item in dump_files:
            latest_by_step[item[0]] = item
        dump_files = [latest_by_step[step] for step in sorted(latest_by_step)]

    records = []
    for _, attempt, path in sorted(dump_files):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                record["raw_attempt"] = attempt
                if attempt > 1 and not latest_only:
                    record["trajectory_id"] = f"{record['trajectory_id']}-attempt{attempt:02d}"
                records.append(record)
    return records


def _identity_fragment(value: Any) -> str:
    fragment = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")
    return fragment or "unknown"


def build_trajectory_records(
    generations: Iterable[dict[str, Any]],
    *,
    phase: str,
    step: int,
    steps_per_epoch: int,
) -> list[dict[str, Any]]:
    """Add stable experiment identity to decoded rollout generations."""
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive")

    epoch = 0 if step <= 0 else (step - 1) // steps_per_epoch + 1
    rollout_counts: dict[str, int] = defaultdict(int)
    records = []
    for generation in generations:
        record = dict(generation)
        parsed_steps = parse_react_steps(str(record.get("output", "")))
        record.setdefault("steps", parsed_steps)
        record.setdefault(
            "tool_used",
            float(any(step.get("action", {}).get("name") == "bash" for step in parsed_steps)),
        )
        extra_info = record.get("extra_info") or {}
        source_id = extra_info.get("id", extra_info.get("index", "unknown"))
        group_id = str(record.get("uid") or source_id)
        rollout_index = rollout_counts[group_id]
        rollout_counts[group_id] += 1
        record.update(
            {
                "trajectory_id": (
                    f"{_identity_fragment(phase)}-step{step:06d}-"
                    f"{_identity_fragment(source_id)}-rollout{rollout_index:02d}"
                ),
                "phase": phase,
                "epoch": epoch,
                "global_step": step,
                "source_id": source_id,
                "row_index": extra_info.get("index"),
                "split": extra_info.get("split"),
                "rollout_index": rollout_index,
            }
        )
        records.append(record)
    return records


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_trajectory_records(
    records: Iterable[dict[str, Any]],
    *,
    dump_dir: Path,
    step: int,
) -> Path:
    """Write one immutable JSONL dump, preserving conflicting retries."""
    dump_dir.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(_json_safe(record), ensure_ascii=False) + "\n"
        for record in records
    )
    path = dump_dir / f"{step}.jsonl"
    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            return path
        attempt = 2
        while (dump_dir / f"{step}.attempt{attempt:02d}.jsonl").exists():
            attempt += 1
        path = dump_dir / f"{step}.attempt{attempt:02d}.jsonl"

    temporary = dump_dir / f".{path.name}.{os.getpid()}.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


def _markdown_record(record: dict[str, Any]) -> str:
    status = "SUCCEED" if float(record.get("score", 0.0)) > 0.0 else "FAILED"
    metadata = {
        key: record.get(key)
        for key in (
            "trajectory_id",
            "phase",
            "epoch",
            "global_step",
            "source_id",
            "row_index",
            "split",
            "rollout_index",
            "score",
            "acc",
            "tool_used",
            "num_turns",
            "prompt_tokens",
            "response_tokens",
        )
        if key in record
    }
    return (
        f"# {record['trajectory_id']} {status}\n\n"
        "## Metadata\n\n"
        f"```json\n{json.dumps(_json_safe(metadata), ensure_ascii=False, indent=2)}\n```\n\n"
        "## Prompt\n\n"
        f"{record.get('input', '')}\n\n"
        "## Response\n\n"
        f"{record.get('output', '')}\n\n"
        "## Ground truth\n\n"
        f"{record.get('gts', '')}\n\n"
        "## Parsed steps\n\n"
        f"```json\n{json.dumps(_json_safe(record.get('steps', [])), ensure_ascii=False, indent=2)}\n```\n"
    )


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def export_trajectory_records(
    records: Iterable[dict[str, Any]],
    *,
    out_dir: Path,
) -> dict[str, int]:
    """Export normalized JSONL and sharded Trace2Skill-compatible Markdown."""
    normalized = []
    seen = set()
    succeed = 0
    for original in records:
        record = dict(original)
        trajectory_id = str(record["trajectory_id"])
        if trajectory_id in seen:
            raise ValueError(f"duplicate trajectory_id: {trajectory_id}")
        seen.add(trajectory_id)
        record.setdefault("steps", parse_react_steps(str(record.get("output", ""))))
        normalized.append(record)

        successful = float(record.get("score", 0.0)) > 0.0
        succeed += int(successful)
        status = "SUCCEED" if successful else "FAILED"
        markdown_path = (
            out_dir
            / "markdown"
            / _identity_fragment(record.get("phase", "unknown"))
            / f"epoch_{int(record.get('epoch', 0)):02d}"
            / f"step_{int(record.get('global_step', 0)):06d}"
            / f"{_identity_fragment(trajectory_id)}_{status}.md"
        )
        _atomic_write_text(markdown_path, _markdown_record(record))

    consolidated = "".join(
        json.dumps(_json_safe(record), ensure_ascii=False) + "\n"
        for record in normalized
    )
    _atomic_write_text(out_dir / "trajectories.jsonl", consolidated)
    return {
        "records": len(normalized),
        "succeed": succeed,
        "failed": len(normalized) - succeed,
    }
