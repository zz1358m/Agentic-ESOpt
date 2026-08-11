#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts/sudoku/run_checkpoint_eval.py"
TRAINER = ROOT / "sudoku-train-time/scripts/run_sudoku_es_train.py"


def load_launcher():
    spec = importlib.util.spec_from_file_location("sudoku_checkpoint_eval", LAUNCHER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {LAUNCHER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _CompletionHandler(BaseHTTPRequestHandler):
    paths: list[str] = []

    def do_POST(self) -> None:  # noqa: N802
        type(self).paths.append(self.path)
        size = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(size)
        body = json.dumps({"content": ["set 1 1 5"]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


class CheckpointLauncherContractTests(unittest.TestCase):
    def test_endpoint_plan_maps_two_endpoints_to_each_of_eight_gpu_uuids(self) -> None:
        launcher = load_launcher()
        gpus = [
            launcher.GpuIdentity(index=index, uuid=f"GPU-test-{index}", name="A100")
            for index in range(8)
        ]

        plan = launcher.build_endpoint_plan(gpus, first_port=12100)

        self.assertEqual([row.port for row in plan], list(range(12100, 12116)))
        self.assertEqual(len(plan), 16)
        for gpu in gpus:
            self.assertEqual(sum(row.gpu_uuid == gpu.uuid for row in plan), 2)

    def test_endpoint_plan_maps_four_endpoints_to_each_of_four_gpu_uuids(self) -> None:
        launcher = load_launcher()
        gpus = [
            launcher.GpuIdentity(index=index, uuid=f"GPU-test-{index}", name="A100")
            for index in range(4)
        ]

        plan = launcher.build_endpoint_plan(gpus, first_port=12100)

        self.assertEqual([row.port for row in plan], list(range(12100, 12116)))
        self.assertEqual(len(plan), 16)
        for gpu in gpus:
            self.assertEqual(sum(row.gpu_uuid == gpu.uuid for row in plan), 4)

        waves = launcher.build_startup_waves(plan)
        self.assertEqual([len(wave) for wave in waves], [4, 4, 4, 4])
        self.assertEqual(
            [row.port for wave in waves for row in wave],
            list(range(12100, 12116)),
        )
        for wave in waves:
            self.assertEqual(len({row.gpu_uuid for row in wave}), 4)

    def test_four_or_eight_unique_physical_gpus_are_required(self) -> None:
        launcher = load_launcher()
        with self.assertRaisesRegex(ValueError, "four or eight unique"):
            launcher.validate_physical_gpu_ids("0,1,2")
        with self.assertRaisesRegex(ValueError, "four or eight unique"):
            launcher.validate_physical_gpu_ids("0,1,2,3,4,5,6,6")
        self.assertEqual(
            launcher.validate_physical_gpu_ids("0,1,2,3"),
            tuple(range(4)),
        )
        self.assertEqual(
            launcher.validate_physical_gpu_ids("0,1,2,3,4,5,6,7"),
            tuple(range(8)),
        )

    def test_explicit_four_gpu_selection_resolves_four_physical_devices(self) -> None:
        launcher = load_launcher()
        smi_output = "\n".join(
            f"{index}, GPU-test-{index}, A100" for index in range(8)
        )
        with mock.patch.object(launcher.subprocess, "run") as run:
            run.return_value.stdout = smi_output

            gpus = launcher.resolve_gpus("0,1,2,3")

        self.assertEqual([gpu.index for gpu in gpus], [0, 1, 2, 3])
        self.assertEqual(len({gpu.uuid for gpu in gpus}), 4)

    def test_inference_seed_is_unique_per_endpoint_and_reaches_server_cli(self) -> None:
        launcher = load_launcher()
        gpus = [
            launcher.GpuIdentity(index=index, uuid=f"GPU-test-{index}", name="A100")
            for index in range(8)
        ]

        plan = launcher.build_endpoint_plan(
            gpus,
            first_port=12100,
            inference_seed=20260811,
        )

        self.assertEqual(
            [row.inference_seed for row in plan],
            list(range(20260811, 20260827)),
        )
        command = launcher.build_server_command(
            python="python",
            model_path=Path("checkpoint"),
            assignment=plan[0],
        )
        seed_index = command.index("--seed")
        self.assertEqual(command[seed_index + 1], "20260811")

    def test_history_validation_rejects_incomplete_or_invalid_runs(self) -> None:
        launcher = load_launcher()
        valid_score = {
            "task_id": "eval-1",
            "score": 1.0,
            "endpoint": "http://127.0.0.1:12100/completions",
            "turns": [{"response": "set r1c1 5", "valid": True}],
        }
        valid_run = {
            "count": 1,
            "valid_count": 1,
            "scores": [valid_score],
            "repeat": 0,
        }
        history = [
            {
                "config": {
                    "mode": "checkpoint_eval_only",
                    "batched_eval": True,
                    "endpoint_batch_size": 32,
                }
            },
            {
                "generation": -1,
                "eval": {
                    "repeat_count": 1,
                    "count": 1,
                    "average": 1.0,
                    "runs": [valid_run],
                },
            },
        ]

        summary = launcher.validate_eval_history(history, expected_count=1, expected_repeats=1)
        self.assertEqual(summary["completed_trajectories"], 1)

        invalid = json.loads(json.dumps(history))
        invalid[-1]["eval"]["runs"][0]["valid_count"] = 0
        invalid[-1]["eval"]["runs"][0]["scores"][0]["score"] = -1.0
        with self.assertRaisesRegex(RuntimeError, "valid_count"):
            launcher.validate_eval_history(invalid, expected_count=1, expected_repeats=1)

        no_legal_actions = json.loads(json.dumps(history))
        no_legal_actions[-1]["eval"]["runs"][0]["scores"][0]["score"] = 0.0
        no_legal_actions[-1]["eval"]["runs"][0]["scores"][0]["turns"][0]["valid"] = False
        with self.assertRaisesRegex(RuntimeError, "no legal actions"):
            launcher.validate_eval_history(no_legal_actions, expected_count=1, expected_repeats=1)

    def test_formal_trainer_command_uses_original_batched_eval_profile(self) -> None:
        launcher = load_launcher()

        command = launcher.build_trainer_command(
            python="python",
            endpoints=["http://127.0.0.1:12100/completions"],
            eval_data=Path("eval.jsonl"),
            eval_limit=32,
            eval_repeats=3,
            result_root=Path("run"),
            served_model_name="Qwen3.5-4B",
        )

        self.assertIn("--batched-eval", command)
        batch_size_index = command.index("--endpoint-batch-size")
        self.assertEqual(command[batch_size_index + 1], "32")


class EvalOnlyCliContractTests(unittest.TestCase):
    def test_eval_only_writes_generation_minus_one_without_es_requests(self) -> None:
        _CompletionHandler.paths = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), _CompletionHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                data_path = tmp_path / "eval.jsonl"
                puzzle = [
                    [0, 3, 4, 6, 7, 8, 9, 1, 2],
                    [6, 7, 2, 1, 9, 5, 3, 4, 8],
                    [1, 9, 8, 3, 4, 2, 5, 6, 7],
                    [8, 5, 9, 7, 6, 1, 4, 2, 3],
                    [4, 2, 6, 8, 5, 3, 7, 9, 1],
                    [7, 1, 3, 9, 2, 4, 8, 5, 6],
                    [9, 6, 1, 5, 3, 7, 2, 8, 4],
                    [2, 8, 7, 4, 1, 9, 6, 3, 5],
                    [3, 4, 5, 2, 8, 6, 1, 7, 9],
                ]
                solution = [row[:] for row in puzzle]
                solution[0][0] = 5
                data_path.write_text(
                    json.dumps(
                        {
                            "id": "one-missing",
                            "puzzle": puzzle,
                            "solution": solution,
                            "mask_count": 1,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                result_root = tmp_path / "result"
                endpoint = f"http://127.0.0.1:{server.server_port}"
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(TRAINER),
                        "--eval-only",
                        "--endpoints",
                        endpoint,
                        "--eval-data",
                        str(data_path),
                        "--mask-count",
                        "1",
                        "--eval-limit",
                        "1",
                        "--eval-repeats",
                        "1",
                        "--max-turns",
                        "3",
                        "--result-root",
                        str(result_root),
                    ],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    timeout=30,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                history = json.loads((result_root / "history.json").read_text(encoding="utf-8"))

                self.assertEqual(history[-1]["generation"], -1)
                self.assertEqual(history[-1]["eval"]["average"], 1.0)
                self.assertEqual(_CompletionHandler.paths, ["/completions"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
