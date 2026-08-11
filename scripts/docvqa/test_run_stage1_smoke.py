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

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/docvqa/run_stage1_smoke.py"
TOKENIZER = Path(
    "/mnt/data7t/ES4LLM/data/trained_checkpoints/"
    "Qwen3.5-4B-DocVQA-ReAct-Agentic-ESOpt"
)


class _DocVQAHandler(BaseHTTPRequestHandler):
    responses = [
        'Action:\n{"name":"bash","arguments":{"command":"tesseract document.png stdout 2>/dev/null"}}',
        "Final answer: smoke",
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


class DocVQAStage1SmokeContractTests(unittest.TestCase):
    def test_cli_copies_image_runs_tesseract_and_scores_anls(self) -> None:
        _DocVQAHandler.calls = 0
        server = ThreadingHTTPServer(("127.0.0.1", 0), _DocVQAHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                image = root / "source.png"
                canvas = Image.new("RGB", (600, 180), "white")
                ImageDraw.Draw(canvas).text((30, 60), "smoke", fill="black")
                canvas.save(image)
                data = root / "docvqa.jsonl"
                data.write_text(
                    json.dumps(
                        {
                            "id": "smoke",
                            "question": "What word is shown?",
                            "answers": ["smoke"],
                            "image": str(image),
                            "source": "test",
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                output = root / "out"
                completed = subprocess.run(
                    [
                        "aa-exec",
                        "-p",
                        "busybox",
                        "--",
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
                    timeout=60,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                result = json.loads((output / "result.json").read_text(encoding="utf-8"))
                self.assertTrue(result["used_bash"])
                self.assertTrue((output / "tool_workdir/source_document.png").is_file())
                self.assertEqual(result["prediction"], "smoke")
                self.assertEqual(result["anls"], 1.0)
                self.assertIn("tesseract", result["steps"][0]["action"]["arguments"]["command"])
                self.assertIn("[exit_code] 0", result["steps"][0]["observation"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
