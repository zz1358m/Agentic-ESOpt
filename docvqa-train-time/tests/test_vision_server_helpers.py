from __future__ import annotations

import base64
import importlib.util
import io
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "hf_vision_es_server.py"
SPEC = importlib.util.spec_from_file_location("hf_vision_es_server", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
SERVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERVER)


class _FakeImage:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def convert(self, mode: str):
        self.mode = mode
        return self


class _FakeImageModule:
    @staticmethod
    def open(stream: io.BytesIO) -> _FakeImage:
        return _FakeImage(stream.read())


class VisionServerHelperTests(unittest.TestCase):
    def test_normalize_openai_image_content(self) -> None:
        payload = b"small-image-payload"
        data_url = "data:image/png;base64," + base64.b64encode(payload).decode("ascii")
        normalized, images = SERVER.normalize_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "read this"},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            image_module=_FakeImageModule,
            max_image_bytes=1024,
        )

        self.assertEqual(
            normalized,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "read this"},
                        {"type": "image"},
                    ],
                }
            ],
        )
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].payload, payload)
        self.assertTrue(images[0].loaded)
        self.assertEqual(images[0].mode, "RGB")

    def test_rejects_remote_image_urls(self) -> None:
        with self.assertRaisesRegex(ValueError, "data:image"):
            SERVER.normalize_messages(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/document.png"},
                            }
                        ],
                    }
                ],
                image_module=_FakeImageModule,
                max_image_bytes=1024,
            )

    def test_rejects_oversized_image(self) -> None:
        data_url = "data:image/png;base64," + base64.b64encode(b"12345").decode("ascii")
        with self.assertRaisesRegex(ValueError, "byte limit"):
            SERVER.decode_data_image(
                data_url,
                image_module=_FakeImageModule,
                max_bytes=4,
            )


if __name__ == "__main__":
    unittest.main()
