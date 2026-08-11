from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


RUNNER = Path(__file__).with_name("run_trace2skill_vllm_eval16.py")
SPEC = importlib.util.spec_from_file_location("trace2skill_vllm_eval16", RUNNER)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class TraceLogOutcomeTests(unittest.TestCase):
    def test_math_datasets_keep_their_historical_final_eval_seeds(self) -> None:
        args = SimpleNamespace(
            datasets="dapo100,aime2026",
            math_root=Path("math"),
            math_max_tokens=4096,
            math_limit=0,
            dapo_seed=20270652,
            aime_seed=20280652,
            docvqa_data=None,
            docvqa_evolve_data=None,
            docvqa_root=Path("docvqa"),
            docvqa_max_tokens=512,
            docvqa_limit=0,
        )

        specs = runner.build_datasets(args)

        self.assertEqual(
            {spec.name: spec.seed for spec in specs},
            {"dapo100": 20270652, "aime2026": 20280652},
        )

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


class RequestAuditFieldsTests(unittest.IsolatedAsyncioTestCase):
    async def test_math_result_records_endpoint_and_termination_reason(self) -> None:
        dataset = runner.DatasetSpec(
            name="dapo100",
            kind="math",
            path=Path("unused.jsonl"),
            enable_thinking=False,
            max_tokens=4096,
        )
        args = SimpleNamespace(
            math_max_tokens=4096,
            docvqa_max_tokens=512,
            math_react_prompt="repo-react-v1",
            math_skill_text="",
        )
        with mock.patch.object(
            runner,
            "run_math_react",
            new=mock.AsyncMock(return_value=("Final answer: 1", {}, [], None)),
        ), mock.patch.object(runner, "math_score", return_value=(1.0, "exact")):
            result = await runner.request_one(
                client=mock.Mock(),
                chat_url="http://127.0.0.1:18180/v1/chat/completions",
                model="model",
                dataset=dataset,
                row={"id": "task", "question": "1?", "answer": "1"},
                row_index=0,
                sample_index=0,
                base_seed=20270652,
                docvqa_root=Path("unused"),
                args=args,
            )

        self.assertEqual(result["endpoint"], "http://127.0.0.1:18180/v1")
        self.assertEqual(result["termination_reason"], "final_answer")


if __name__ == "__main__":
    unittest.main()
