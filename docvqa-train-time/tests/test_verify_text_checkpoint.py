from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/docvqa/verify_text_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("verify_text_checkpoint", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class VerifyTextCheckpointTests(unittest.TestCase):
    def test_validates_index_and_all_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model-00001.safetensors").write_bytes(b"weights")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"model.embed_tokens.weight": "model-00001.safetensors"}}),
                encoding="utf-8",
            )
            self.assertEqual(MODULE.validate_weight_index(root), {"weights": 1, "shards": 1})

    def test_rejects_missing_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"model.embed_tokens.weight": "missing.safetensors"}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "missing checkpoint shards"):
                MODULE.validate_weight_index(root)

    def test_rejects_empty_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model-00001.safetensors").touch()
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"model.embed_tokens.weight": "model-00001.safetensors"}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "empty checkpoint shards"):
                MODULE.validate_weight_index(root)


if __name__ == "__main__":
    unittest.main()
