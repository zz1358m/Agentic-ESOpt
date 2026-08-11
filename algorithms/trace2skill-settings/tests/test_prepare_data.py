from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_data", SCRIPT)
assert SPEC and SPEC.loader
prepare_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_data)


class FakeDataset:
    column_names = ["question", "image"]

    def __init__(self) -> None:
        self.cast_calls: list[tuple[str, object]] = []

    def cast_column(self, name: str, feature: object) -> "FakeDataset":
        self.cast_calls.append((name, feature))
        return self

    def __iter__(self):
        yield {"question": "q", "image": {"bytes": b"png", "path": "x.png"}}


class PrepareDataTests(unittest.TestCase):
    def test_hf_loader_can_preserve_encoded_image_bytes(self) -> None:
        dataset = FakeDataset()

        class FakeImage:
            def __init__(self, *, decode: bool) -> None:
                self.decode = decode

        fake_datasets = types.SimpleNamespace(
            DatasetDict=dict,
            Image=FakeImage,
            load_dataset=lambda *args, **kwargs: dataset,
        )
        with mock.patch.dict(sys.modules, {"datasets": fake_datasets}):
            rows = prepare_data.load_hf_dataset_rows(
                "repo/name",
                split="validation",
                decode_images=False,
            )

        self.assertEqual(rows[0]["image"]["bytes"], b"png")
        self.assertEqual(len(dataset.cast_calls), 1)
        self.assertEqual(dataset.cast_calls[0][0], "image")
        self.assertFalse(dataset.cast_calls[0][1].decode)


if __name__ == "__main__":
    unittest.main()
