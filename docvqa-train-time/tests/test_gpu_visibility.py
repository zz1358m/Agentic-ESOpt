from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "docvqa" / "gpu_visibility.py"
SPEC = importlib.util.spec_from_file_location("gpu_visibility", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class GpuVisibilityTests(unittest.TestCase):
    def test_resolves_physical_indices_to_exact_uuids(self) -> None:
        output = "\n".join(
            f"{index}, GPU-uuid-{index}, NVIDIA A100-SXM4-80GB"
            for index in range(8)
        )

        identities = MODULE.resolve_physical_gpus("1,3,5,7", query_output=output)

        self.assertEqual([item.index for item in identities], [1, 3, 5, 7])
        self.assertEqual(
            MODULE.cuda_visible_devices(identities),
            "GPU-uuid-1,GPU-uuid-3,GPU-uuid-5,GPU-uuid-7",
        )

    def test_auto_selects_the_highest_four_reported_indices(self) -> None:
        output = "\n".join(
            f"{index}, GPU-{index}, NVIDIA A100" for index in (0, 2, 4, 7, 9, 11)
        )

        identities = MODULE.resolve_physical_gpus("auto", query_output=output)

        self.assertEqual([item.index for item in identities], [4, 7, 9, 11])
        self.assertEqual(MODULE.cuda_visible_devices(identities), "GPU-4,GPU-7,GPU-9,GPU-11")

    def test_rejects_missing_or_duplicate_physical_indices(self) -> None:
        output = "\n".join(
            [
                *[f"{index}, GPU-{index}, NVIDIA A100" for index in range(4)],
                "2, GPU-two-again, NVIDIA A100",
            ]
        )
        with self.assertRaisesRegex(ValueError, "duplicate physical GPU index"):
            MODULE.resolve_physical_gpus("0,1,2,3", query_output=output)

        output_without_three = "\n".join(
            f"{index}, GPU-{index}, NVIDIA A100" for index in range(3)
        )
        with self.assertRaisesRegex(ValueError, "physical GPU 3 was not found"):
            MODULE.resolve_physical_gpus("0,1,2,3", query_output=output_without_three)

    def test_rejects_non_four_or_duplicate_explicit_plans(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly four unique"):
            MODULE.validate_docvqa_physical_ids("0,1,2")
        with self.assertRaisesRegex(ValueError, "exactly four unique"):
            MODULE.validate_docvqa_physical_ids("0,1,1,2")

    def test_stable_uuids_survive_compacted_runtime_indices(self) -> None:
        output = "\n".join(
            ["0, GPU-unused, NVIDIA A100"]
            + [f"{index - 3}, GPU-target-{index}, NVIDIA A100" for index in range(4, 8)]
        )
        expected = ",".join(f"GPU-target-{index}" for index in range(4, 8))

        identities = MODULE.resolve_physical_gpus(
            "4,5,6,7",
            query_output=output,
            expected_uuids=expected,
        )

        self.assertEqual([identity.index for identity in identities], [4, 5, 6, 7])
        self.assertEqual(MODULE.cuda_visible_devices(identities), expected)

    def test_resolves_last_four_by_stable_uuid_after_restart(self) -> None:
        output = "\n".join(
            [
                "0, GPU-unused, NVIDIA A100",
                "3, GPU-target-4, NVIDIA A100",
                "4, GPU-target-5, NVIDIA A100",
                "5, GPU-target-6, NVIDIA A100",
                "6, GPU-target-7, NVIDIA A100",
            ]
        )
        expected = ",".join(f"GPU-target-{index}" for index in range(4, 8))

        identities = MODULE.resolve_physical_gpus(
            "auto",
            query_output=output,
            expected_uuids=expected,
        )

        self.assertEqual([identity.index for identity in identities], [3, 4, 5, 6])
        self.assertEqual(MODULE.cuda_visible_devices(identities), expected)

    def test_sglang_child_uses_numeric_physical_index(self) -> None:
        identity = MODULE.GpuIdentity(
            index=7,
            uuid="GPU-target-7",
            name="NVIDIA A100-SXM4-80GB",
        )

        self.assertEqual(MODULE.sglang_visible_device(identity), "7")


if __name__ == "__main__":
    unittest.main()
