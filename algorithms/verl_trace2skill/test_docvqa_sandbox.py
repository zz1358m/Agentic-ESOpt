from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image, ImageDraw, ImageFont

from algorithms.verl_trace2skill.docvqa_sandbox import run_sandboxed_bash


ROOT = Path(__file__).resolve().parents[2]


class DocVQASandboxTests(unittest.TestCase):
    def test_exposes_only_virtual_document_and_hides_host_secret(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image = root / "source.png"
            secret = root / "answers.jsonl"
            image.write_bytes(b"fake-png")
            secret.write_text("gold answer", encoding="utf-8")

            result = run_sandboxed_bash(
                (
                    "test -r /workspace/document.png && "
                    f"test ! -e {secret} && "
                    "test \"$(cat /workspace/document.png)\" = fake-png && printf sandbox-ok"
                ),
                image_path=image,
                timeout=5,
                max_output_chars=1000,
            )

        self.assertEqual(result.returncode, 0, result.text)
        self.assertIn("sandbox-ok", result.text)

    def test_tesseract_can_ocr_the_current_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image = Path(tmpdir) / "document.png"
            canvas = Image.new("RGB", (900, 220), "white")
            draw = ImageDraw.Draw(canvas)
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                96,
            )
            draw.text((30, 40), "INVOICE 42", fill="black", font=font)
            canvas.save(image)

            result = run_sandboxed_bash(
                "tesseract /workspace/document.png stdout 2>/dev/null",
                image_path=image,
            )

        self.assertEqual(result.returncode, 0, result.text)
        self.assertIn("INVOICE", result.text.upper())
        self.assertIn("42", result.text)

    def test_repository_and_network_are_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image = Path(tmpdir) / "document.png"
            image.write_bytes(b"not-an-image")
            command = (
                f"test ! -e {ROOT / 'README.md'} && "
                "python -c \"import socket,sys; s=socket.socket(); s.settimeout(1); "
                "r=s.connect_ex(('1.1.1.1',80)); print('network-blocked', r); sys.exit(r == 0)\""
            )

            result = run_sandboxed_bash(command, image_path=image)

        self.assertEqual(result.returncode, 0, result.text)
        self.assertIn("network-blocked", result.text)

    def test_truncates_large_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image = Path(tmpdir) / "source.png"
            image.write_bytes(b"fake-png")

            result = run_sandboxed_bash(
                "i=0; while [ $i -lt 200 ]; do printf x; i=$((i+1)); done",
                image_path=image,
                timeout=5,
                max_output_chars=80,
            )

        self.assertEqual(result.returncode, 0, result.text)
        self.assertLessEqual(len(result.text), 100)
        self.assertIn("...[truncated]...", result.text)

    def test_limits_virtual_memory_to_eight_gibibytes_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image = Path(tmpdir) / "document.png"
            image.write_bytes(b"not-an-image")

            result = run_sandboxed_bash(
                "ulimit -v",
                image_path=image,
            )

        self.assertEqual(result.returncode, 0, result.text)
        self.assertEqual(result.text.splitlines()[0], "8388608")

    def test_limits_native_math_libraries_to_one_thread(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image = Path(tmpdir) / "document.png"
            image.write_bytes(b"not-an-image")

            result = run_sandboxed_bash(
                "printf '%s %s %s %s' \"$OPENBLAS_NUM_THREADS\" \"$OMP_NUM_THREADS\" "
                "\"$MKL_NUM_THREADS\" \"$NUMEXPR_NUM_THREADS\"",
                image_path=image,
            )

        self.assertEqual(result.returncode, 0, result.text)
        self.assertEqual(result.text.splitlines()[0], "1 1 1 1")

    def test_pip_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image = Path(tmpdir) / "document.png"
            image.write_bytes(b"not-an-image")

            result = run_sandboxed_bash(
                "python -m pip --version; pip --version",
                image_path=image,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("pip ", result.text.lower())

    def test_which_resolves_sandbox_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image = Path(tmpdir) / "document.png"
            image.write_bytes(b"not-an-image")

            result = run_sandboxed_bash(
                "which python",
                image_path=image,
            )

        self.assertEqual(result.returncode, 0, result.text)
        self.assertIn(str(Path(sys.prefix).absolute()), result.text)

    def test_uses_running_python_prefix_when_conda_prefix_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image = root / "document.png"
            stale_prefix = root / "stale-conda"
            image.write_bytes(b"not-an-image")
            (stale_prefix / "bin").mkdir(parents=True)

            with mock.patch.dict("os.environ", {"CONDA_PREFIX": str(stale_prefix)}):
                result = run_sandboxed_bash(
                    "python -c 'import sys; print(sys.executable)'",
                    image_path=image,
                )

        self.assertEqual(result.returncode, 0, result.text)
        expected_prefix = str(Path(sys.prefix).absolute())
        self.assertIn(expected_prefix, result.text)

    def test_invalid_null_byte_command_becomes_tool_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image = Path(tmpdir) / "document.png"
            image.write_bytes(b"not-an-image")

            result = run_sandboxed_bash(
                "printf before\x00after",
                image_path=image,
            )

        self.assertEqual(result.returncode, 126)
        self.assertIn("embedded null byte", result.text)


if __name__ == "__main__":
    unittest.main()
