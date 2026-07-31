from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "trace2skill-settings/scripts/prepare_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_data", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrepareDocVQAOrderTests(unittest.TestCase):
    def test_reference_order_skips_training_ids_and_preserves_rest(self) -> None:
        remaining = [{"id": task_id} for task_id in ("docvqa_c", "docvqa_b", "docvqa_d")]
        reference = [
            {"task_id": "docvqa_a"},  # selected for training, so absent
            {"task_id": "docvqa_b"},
            {"task_id": "docvqa_b"},  # repeated sample
            {"task_id": "docvqa_c"},
        ]

        ordered = MODULE.order_docvqa_test_rows(remaining, reference)

        self.assertEqual([row["id"] for row in ordered], ["docvqa_b", "docvqa_c", "docvqa_d"])

    def test_reference_uses_row_index_not_concurrent_file_order(self) -> None:
        remaining = [{"id": task_id} for task_id in ("docvqa_a", "docvqa_b")]
        reference = [
            {"task_id": "docvqa_b", "row_index": 1, "sample_index": 0},
            {"task_id": "docvqa_a", "row_index": 0, "sample_index": 0},
        ]

        ordered = MODULE.order_docvqa_test_rows(remaining, reference)

        self.assertEqual([row["id"] for row in ordered], ["docvqa_a", "docvqa_b"])

    def test_image_canonicalization_preserves_the_existing_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir)
            old_image = output / "images/docvqa_7.png"
            old_image.parent.mkdir(parents=True)
            old_image.write_bytes(b"png")
            rows = [{"id": "docvqa_7", "image": str(old_image)}]

            MODULE.canonicalize_prepared_docvqa_images(rows, output_dir=output)

            self.assertTrue(old_image.is_file())
            self.assertTrue((output / "images/7.png").is_file())
            self.assertEqual(Path(rows[0]["image"]).name, "7.png")


if __name__ == "__main__":
    unittest.main()
