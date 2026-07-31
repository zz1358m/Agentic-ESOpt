from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/docvqa/run_four_gpu_eval.py"
SPEC = importlib.util.spec_from_file_location("run_four_gpu_eval", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FourGpuEvalTests(unittest.TestCase):
    def test_last_four_eval_topology_resolves_stable_uuids(self) -> None:
        expected = ",".join(f"GPU-target-{index}" for index in range(4, 8))
        query_output = "\n".join(
            f"{index - 1}, GPU-target-{index}, NVIDIA A100" for index in range(4, 8)
        )

        physical_gpus, identities = MODULE.resolve_eval_gpus(
            "4,5,6,7",
            expected,
            query_output=query_output,
        )

        self.assertEqual(physical_gpus, ("4", "5", "6", "7"))
        self.assertEqual([identity.uuid for identity in identities], expected.split(","))

    def test_each_server_is_single_gpu_with_128k_context(self) -> None:
        command = MODULE.server_command("python", Path("/model"), 18080)
        self.assertEqual(command[command.index("--tp-size") + 1], "1")
        self.assertEqual(command[command.index("--context-length") + 1], "131072")

    def test_result_validation_retries_a_transient_partial_read(self) -> None:
        valid = json.dumps({"key": "docvqa:a:sample00", "error": None}) + "\n"
        with mock.patch.object(Path, "read_text", side_effect=['{"key": "partial', valid]):
            MODULE.validate_results(Path("/tmp/docvqa.jsonl"), 1, attempts=2, retry_delay=0)

    def test_complete_resume_result_can_skip_model_startup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "docvqa.jsonl"
            output.write_text(
                json.dumps({"key": "docvqa:a:sample00", "error": None}) + "\n",
                encoding="utf-8",
            )

            self.assertTrue(MODULE.results_complete(output, 1))

    def test_validation_requires_the_exact_docvqa_task_and_sample_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data = root / "test.jsonl"
            data.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "task-a"}),
                        json.dumps({"id": "task-b"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            expected_keys = MODULE.expected_result_keys(data, limit=2, samples=2)
            output = root / "docvqa.jsonl"
            output.write_text(
                "\n".join(
                    json.dumps({"key": f"wrong:{index}", "error": None})
                    for index in range(4)
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "key_set_matches=False"):
                MODULE.validate_results(
                    output,
                    4,
                    expected_keys=expected_keys,
                    attempts=1,
                    retry_delay=0,
                )

    def test_resume_manifest_must_match_the_requested_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "four_gpu_manifest.json"
            manifest.write_text(
                json.dumps({"model_path": "/old-model", "seed": 42}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "manifest fingerprint mismatch"):
                MODULE.validate_manifest_fingerprint(
                    manifest,
                    {"model_path": "/new-model", "seed": 42},
                )

    def test_validation_does_not_treat_unicode_next_line_as_jsonl_separator(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "docvqa.jsonl"
            output.write_text(
                json.dumps(
                    {"key": "docvqa:a:sample00", "error": None, "completion": "before\u0085after"},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            MODULE.validate_results(output, 1, attempts=1, retry_delay=0)


if __name__ == "__main__":
    unittest.main()
