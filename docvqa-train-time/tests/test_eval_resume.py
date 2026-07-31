from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/trace2skill/run_trace2skill_vllm_eval16.py"
SPEC = importlib.util.spec_from_file_location("run_trace2skill_vllm_eval16", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_prepare_resume_output_archives_errors_and_keeps_unique_successes(tmp_path: Path) -> None:
    output = tmp_path / "docvqa.jsonl"
    rows = [
        {"key": "docvqa:a:sample00", "error": None, "score": 1.0},
        {"key": "docvqa:b:sample00", "error": "ValueError: embedded null byte", "score": 0.0},
        {"key": "docvqa:a:sample00", "error": None, "score": 0.5},
    ]
    output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    completed = MODULE.prepare_resume_output(output)

    assert completed == {"docvqa:a:sample00"}
    kept = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert kept == [{"key": "docvqa:a:sample00", "error": None, "score": 0.5}]
    archive = output.with_name("docvqa.request_errors.jsonl")
    archived = [json.loads(line) for line in archive.read_text(encoding="utf-8").splitlines()]
    assert archived == [rows[1]]
