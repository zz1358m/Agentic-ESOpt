#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/math/run_stage1_smoke.py"
TOKENIZER = Path(
    "/mnt/data7t/ES4LLM/data/trained_checkpoints/"
    "Qwen3.5-4B-MATH-ReAct-Agentic-ESOpt"
)


class _MathHandler(BaseHTTPRequestHandler):
    responses = [
        'Action:\n{"name":"bash","arguments":{"command":"python -c \'print(2+2)\'"}}',
        r"Final answer: \boxed{4}",
    ]
    calls = 0

    def do_POST(self) -> None:  # noqa: N802
        size = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(size)
        index = min(type(self).calls, len(type(self).responses) - 1)
        type(self).calls += 1
        body = json.dumps({"content": [type(self).responses[index]]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


class MathStage1SmokeContractTests(unittest.TestCase):
    def test_cli_runs_bash_then_formal_scorer_and_writes_result(self) -> None:
        _MathHandler.calls = 0
        server = ThreadingHTTPServer(("127.0.0.1", 0), _MathHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                data = root / "dapo.jsonl"
                data.write_text(
                    json.dumps(
                        {
                            "id": "smoke",
                            "question": "What is 2+2?",
                            "answer": "4",
                            "source": "test",
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                output = root / "out"
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(RUNNER),
                        "--endpoint",
                        f"http://127.0.0.1:{server.server_port}/completions",
                        "--tokenizer-path",
                        str(TOKENIZER),
                        "--data",
                        str(data),
                        "--output-dir",
                        str(output),
                        "--max-turns",
                        "3",
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                result = json.loads((output / "result.json").read_text(encoding="utf-8"))
                self.assertTrue(result["used_bash"])
                self.assertEqual(result["termination_reason"], "final_answer")
                self.assertEqual(result["prediction"], "4")
                self.assertEqual(result["score"], 1.0)
                self.assertIn("4", result["steps"][0]["observation"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
