from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "trace2skill" / "prepare_verl_trace2skill_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_verl_trace2skill_data", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrepareVerlDocVQATests(unittest.TestCase):
    def test_zero_jsonl_limit_reads_the_full_trace2skill_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            path.write_text(
                "".join(json.dumps({"id": index}) + "\n" for index in range(3)),
                encoding="utf-8",
            )

            self.assertEqual(len(MODULE._read_jsonl(path, 0)), 3)
            self.assertEqual(len(MODULE._read_jsonl(path, 2)), 2)

    def test_script_is_directly_executable_from_repo_root(self) -> None:
        process = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=SCRIPT.parents[2],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_docvqa_record_routes_to_paper_react_agent_and_sandbox_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image = root / "page.png"
            image.write_bytes(b"image")
            records = [
                {
                    "id": "docvqa_1",
                    "question": "What is shown?",
                    "answers": ["invoice"],
                    "image": "page.png",
                }
            ]

            row = MODULE._docvqa_rows(records, "train", root)[0]

        prompt = row["prompt"][1]["content"]
        self.assertEqual(row["agent_name"], "paper_react_cli_agent")
        self.assertIn("Image path: /workspace/document.png", prompt)
        self.assertNotIn(str(image), prompt)
        self.assertEqual(
            row["extra_info"]["tools_kwargs"]["bash"]["create_kwargs"]["image_path"],
            str(image.resolve()),
        )

    def test_routing_validator_rejects_wrong_agent(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "expected agent_name"):
            MODULE._validate_agent_routing([{"agent_name": "single_turn_agent"}], "paper_react_cli_agent")


if __name__ == "__main__":
    unittest.main()
