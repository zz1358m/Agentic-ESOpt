from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("report_grpo_experiment.py")
SPEC = importlib.util.spec_from_file_location("math_report", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MathReportTests(unittest.TestCase):
    def test_training_curve_parses_ansi_console_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "train.log"
            log.write_text(
                "\x1b[36m(TaskRunner pid=1)\x1b[0m step:7 - actor/kl_loss:0.25 "
                "- critic/score/mean:0.5 - training/epoch:1\n",
                encoding="utf-8",
            )
            curve = MODULE.parse_training_curve(log)

        self.assertEqual(curve, [{"step": 7, "actor/kl_loss": 0.25, "critic/score/mean": 0.5, "training/epoch": 1}])

    def test_eval_inspection_checks_counts_keys_and_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = root / "outputs"
            outputs.mkdir()
            rows = [
                {
                    "key": f"dapo100:q{i}:sample{s:02d}",
                    "task_id": f"q{i}",
                    "sample_index": s,
                    "score": float(i == 0 and s == 0),
                    "error": None,
                }
                for i in range(2)
                for s in range(3)
            ]
            (outputs / "dapo100.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            result = MODULE.inspect_eval_dir(root, {"dapo100": 6})

        self.assertEqual(result["records"], 6)
        self.assertEqual(result["unique_keys"], 6)
        self.assertEqual(result["request_errors"], 0)
        self.assertAlmostEqual(result["datasets"]["dapo100"]["mean_score"], 1 / 6)
        self.assertAlmostEqual(result["datasets"]["dapo100"]["max_at_n"], 0.5)

    def test_working_tree_digest_includes_untracked_file_contents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "new.py").write_text("first\n", encoding="utf-8")
            first = MODULE._working_tree_change_digest(root, "diff", "new.py\n")
            (root / "new.py").write_text("second\n", encoding="utf-8")
            second = MODULE._working_tree_change_digest(root, "diff", "new.py\n")

        self.assertNotEqual(first, second)

    def test_markdown_contains_config_data_hashes_and_full_curve(self) -> None:
        dataset = {"mean_score": 0.25, "max_at_n": 0.5}
        report = {
            "experiment": {"total_steps": 300, "rollout_n": 8},
            "data_manifest": {
                "files": {
                    "train": {"sha256": "859079example", "records": 400},
                }
            },
            "evaluation": {
                "before": {"datasets": {"dapo100": dataset, "aime2026": dataset}},
                "after": {"datasets": {"dapo100": dataset, "aime2026": dataset}},
            },
            "acceptance": {"status": "PASS", "checks": {"exact counts": True}},
            "code": {"head": "abc", "working_tree_change_sha256": "def"},
            "dependencies": {"torch": "2.8"},
            "gpu_resources": {"gpus": [{"physical_index": 3, "uuid": "GPU-3"}]},
            "run_history": [],
            "training_curve": [
                {"step": 1, "critic/score/mean": 0.25, "actor/kl_loss": 0.001},
                {"step": 2, "critic/score/mean": 0.50, "actor/kl_loss": 0.002},
            ],
        }

        markdown = MODULE.render_markdown(report)

        self.assertIn("## Full experiment config", markdown)
        self.assertIn('"total_steps": 300', markdown)
        self.assertIn("## Data manifest and hashes", markdown)
        self.assertIn("859079example", markdown)
        self.assertIn("## Full training curve", markdown)
        self.assertIn('"step": 2', markdown)
        self.assertIn('"critic/score/mean": 0.5', markdown)


if __name__ == "__main__":
    unittest.main()
