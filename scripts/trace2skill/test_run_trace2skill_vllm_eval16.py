from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


RUNNER = Path(__file__).with_name("run_trace2skill_vllm_eval16.py")
SPEC = importlib.util.spec_from_file_location("trace2skill_vllm_eval16", RUNNER)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class TraceLogOutcomeTests(unittest.TestCase):
    def write_trace(self, kind: str, score: float) -> Path:
        dataset = runner.DatasetSpec(
            name=f"test_{kind}",
            kind=kind,
            path=Path("unused.jsonl"),
            enable_thinking=False,
            max_tokens=512,
        )
        return runner.write_trace_markdown(
            Path(self.tmpdir.name),
            dataset,
            {"question": "question", "answer": "answer", "answers": ["answer"]},
            {"task_id": "task", "sample_index": 0, "score": score, "react_steps": []},
        )

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_math_requires_exact_correctness(self) -> None:
        self.assertTrue(self.write_trace("math", 1.0).name.endswith("_SUCCEED.md"))
        self.assertTrue(self.write_trace("math", 0.9).name.endswith("_FAILED.md"))

    def test_docvqa_uses_anls_success_threshold(self) -> None:
        self.assertTrue(self.write_trace("docvqa", 0.8).name.endswith("_SUCCEED.md"))
        self.assertTrue(self.write_trace("docvqa", 0.5).name.endswith("_FAILED.md"))


if __name__ == "__main__":
    unittest.main()
