from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from algorithms.es.model_es_client import ModelESClient


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts" / "ahd" / "construct_tsp_stage3.py"
ORCHESTRATOR = ROOT / "scripts" / "ahd" / "run_construct_tsp_stage3.sh"


def load_tool():
    spec = importlib.util.spec_from_file_location("construct_tsp_stage3", TOOL)
    if spec is None or spec.loader is None:
        raise ImportError(TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def update_record(index: int, generation: int, operator: str, endpoint_count: int = 8) -> dict:
    sigma = 0.001 * (1 + __import__("math").cos(__import__("math").pi * generation / 24)) / 2
    return {
        "update_index": index,
        "generation": generation,
        "operator": operator,
        "seeds": list(range(index * 10, index * 10 + 10)),
        "rewards": [float(value) for value in range(10)],
        "alpha": 0.0005,
        "sigma": sigma,
        "sigma_start": 0.001,
        "sigma_end": 0.0,
        "sigma_schedule": "cosine",
        "update_applied": True,
        "generation_concurrency": 8,
        "evaluation_concurrency": 4,
        "engine_count": endpoint_count,
        "update": [
            {"ok": True, "n": 10, "endpoint": f"http://127.0.0.1:{11013 + endpoint}"}
            for endpoint in range(endpoint_count)
        ],
    }


def write_runtime_audits(root: Path, generations: int) -> None:
    candidate_rows = []
    evaluator_rows = []
    for generation in range(generations):
        for operator_index, operator in enumerate(("e1", "e2", "m1", "m2")):
            candidate_rows.append(
                {
                    "generation": generation,
                    "operator": operator,
                    "candidate_count": 10,
                    "code_count": 10,
                    "finite_objective_count": 10,
                }
            )
            evaluator_rows.append(
                {
                    "generation": generation,
                    "operator": operator,
                    "configured_workers": 4,
                    "max_concurrent_processes": 4,
                    "process_pids": [10000 + generation * 16 + operator_index * 4 + i for i in range(4)],
                    "candidate_count": 10,
                }
            )
    history_dir = root / "results" / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    (history_dir / "operator_candidates.json").write_text(json.dumps(candidate_rows))
    (root / "results" / "es" / "evaluator_processes.json").write_text(json.dumps(evaluator_rows))


class ConstructTspStage3Test(unittest.TestCase):
    def test_plan_exposes_frozen_formal_and_smoke_contracts(self):
        result = subprocess.run(
            [sys.executable, str(TOOL), "plan"],
            check=True,
            text=True,
            capture_output=True,
        )
        plan = json.loads(result.stdout)
        self.assertEqual(plan["topology"], {"endpoint_count": 8, "evaluation_workers": 4})
        self.assertEqual(plan["formal"]["candidate_count"], 1000)
        self.assertEqual(plan["formal"]["logical_updates"], 50)
        self.assertEqual(plan["formal"]["endpoint_updates"], 400)
        self.assertEqual(plan["formal"]["repeats"], [1, 2, 3])
        self.assertEqual(plan["smoke"]["directions"], 10)
        self.assertEqual(plan["smoke"]["logical_updates"], 1)
        self.assertEqual(plan["parameters"]["seed"], 2024)

    def test_model_es_update_audit_identifies_its_endpoint(self):
        class FakeClient(ModelESClient):
            def _post(self, path, payload=None):
                self.asserted_path = path
                return {"ok": True, "n": 10}

        client = FakeClient("http://127.0.0.1:11020/completions")
        result = client.update(seeds=list(range(10)), rewards=[1.0] * 10, alpha=0.0005)
        self.assertEqual(result["endpoint"], "http://127.0.0.1:11020")

    def test_formal_launcher_keeps_8_generation_workers_separate_from_4_evaluators(self):
        text = ORCHESTRATOR.read_text(encoding="utf-8")
        self.assertIn('ES_MAX_WORKERS="8"', text)
        self.assertIn('EVALUATION_WORKERS="4"', text)

    def test_formal_validator_accepts_exact_budget_and_emits_final_code(self):
        tool = load_tool()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            (results / "es").mkdir(parents=True)
            (results / "pops_best").mkdir()
            history = [
                update_record(2 * generation + offset, generation, operator)
                for generation in range(25)
                for offset, operator in enumerate(("m1", "m2"))
            ]
            (results / "es" / "history.json").write_text(json.dumps(history))
            write_runtime_audits(root, 25)
            for generation in range(1, 26):
                (results / "pops_best" / f"population_generation_{generation}.json").write_text(
                    json.dumps({"objective": 7.0 - generation / 100, "code": "def select_next_node(*args):\n    return 0\n"})
                )
            runner_log = root / "runner.log"
            runner_log.write_text("- Model ES initialized: " + repr([{"ok": True}] * 8) + "\n")
            final_code = root / "final.py"
            report = tool.validate_run(
                root,
                repeat=2,
                smoke=False,
                runner_log=runner_log,
                final_code=final_code,
            )
            self.assertEqual(report["candidate_count"], 1000)
            self.assertEqual(report["logical_updates"], 50)
            self.assertEqual(report["endpoint_updates"], 400)
            self.assertEqual(report["first_best"], 6.99)
            self.assertEqual(report["last_best"], 6.75)
            self.assertEqual(len(report["generation_comparison"]), 25)
            self.assertEqual(report["generation_comparison"][0]["repository_best"], 6.66389)
            self.assertIn("def select_next_node", final_code.read_text())

    def test_formal_validator_requires_fresh_8_client_initialization_evidence(self):
        tool = load_tool()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "results" / "es").mkdir(parents=True)
            history = [
                update_record(2 * generation + offset, generation, operator)
                for generation in range(25)
                for offset, operator in enumerate(("m1", "m2"))
            ]
            (root / "results" / "es" / "history.json").write_text(json.dumps(history))
            with self.assertRaisesRegex(RuntimeError, "initialize exactly 8"):
                tool.validate_run(root, repeat=1, smoke=False)

    def test_smoke_validator_requires_8_initialized_clients_and_one_sync_update(self):
        tool = load_tool()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "results" / "es").mkdir(parents=True)
            (root / "results" / "es" / "history.json").write_text(
                json.dumps([update_record(0, 0, "m1")])
            )
            write_runtime_audits(root, 1)
            log = root / "runner.log"
            log.write_text("- Model ES initialized: " + repr([{"ok": True}] * 8) + "\n")
            report = tool.validate_run(root, repeat=0, smoke=True, runner_log=log)
            self.assertEqual(report["initialized_endpoints"], 8)
            self.assertEqual(report["logical_updates"], 1)
            self.assertEqual(report["endpoint_updates"], 8)

    def test_smoke_validator_rejects_unidentified_endpoint_updates(self):
        tool = load_tool()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "results" / "es").mkdir(parents=True)
            record = update_record(0, 0, "m1")
            for update in record["update"]:
                update.pop("endpoint")
            (root / "results" / "es" / "history.json").write_text(json.dumps([record]))
            write_runtime_audits(root, 1)
            log = root / "runner.log"
            log.write_text("- Model ES initialized: " + repr([{"ok": True}] * 8) + "\n")
            with self.assertRaisesRegex(RuntimeError, "endpoint set"):
                tool.validate_run(root, repeat=0, smoke=True, runner_log=log)

    def test_topology_validator_maps_each_owned_pid_to_its_physical_gpu_uuid(self):
        tool = load_tool()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root / "logs"
            logs.mkdir()
            inventory = root / "gpu.csv"
            compute = root / "compute.csv"
            inventory.write_text("".join(f"{gpu}, GPU-{gpu}\n" for gpu in range(8)))
            compute.write_text("".join(f"{9000 + gpu}, GPU-{gpu}, 16000\n" for gpu in range(8)))
            for gpu, port in enumerate(range(11013, 11021)):
                (logs / f"server_gpu{gpu}_port{port}.pid").write_text(str(9000 + gpu))
            report = tool.validate_topology(inventory, compute, logs)
            self.assertEqual(report["endpoint_count"], 8)
            self.assertEqual(len(report["servers"]), 8)
            self.assertEqual(report["servers"][7]["gpu_uuid"], "GPU-7")

    def test_cpu_final_gate_rejects_any_invalid_test_instance(self):
        tool = load_tool()
        rows = [
            {"setting": "N=20", "count": 64, "valid_count": 64, "mean": 4.2},
            {"setting": "N=50", "count": 64, "valid_count": 63, "mean": 6.5},
            {"setting": "N=100", "count": 64, "valid_count": 64, "mean": 9.0},
        ]
        with self.assertRaisesRegex(RuntimeError, "invalid instances"):
            tool.validate_cpu_results(rows)

    def test_cpu_comparison_reports_repo_and_paper_differences(self):
        tool = load_tool()
        current = [
            {"setting": "N=20", "count": 64, "valid_count": 64, "mean": 4.4, "std": 0.2},
            {"setting": "N=50", "count": 64, "valid_count": 64, "mean": 6.8, "std": 0.3},
            {"setting": "N=100", "count": 64, "valid_count": 64, "mean": 9.2, "std": 0.4},
        ]
        repository = [
            {"setting": "N=20", "count": 64, "valid_count": 64, "mean": 4.2, "std": 0.1},
            {"setting": "N=50", "count": 64, "valid_count": 64, "mean": 6.5, "std": 0.2},
            {"setting": "N=100", "count": 64, "valid_count": 64, "mean": 9.0, "std": 0.3},
        ]
        comparison = tool.compare_evaluations(current, repository)
        self.assertAlmostEqual(comparison[0]["repository_absolute_difference"], 0.2)
        self.assertAlmostEqual(comparison[0]["paper_absolute_difference"], 4.4 - 4.2107)
        self.assertIsNone(comparison[2]["paper_absolute_difference"])

    def test_three_repeat_aggregate_reports_mean_std_and_differences(self):
        tool = load_tool()
        reports = []
        for repeat, final, repo in ((1, 6.6, 6.5), (2, 6.8, 6.6), (3, 6.4, 6.7)):
            reports.append(
                {
                    "repeat": repeat,
                    "trend": {"last_best": final, "original_log": {"last_best": repo}},
                    "cpu_final_eval": {
                        "comparison": [
                            {
                                "setting": "N=20",
                                "current_mean": final - 2,
                                "repository_mean": repo - 2,
                                "paper_mean": 4.2107,
                            }
                        ]
                    },
                }
            )
        aggregate = tool.aggregate_repeat_reports(reports)
        self.assertAlmostEqual(aggregate["train_final"]["current_mean"], 6.6)
        self.assertAlmostEqual(aggregate["train_final"]["current_std"], (0.08 / 3) ** 0.5)
        self.assertAlmostEqual(aggregate["test"][0]["repository_absolute_difference"], 0.0)

    def test_orchestrator_uses_canonical_launcher_and_never_kills_unowned_processes(self):
        text = ORCHESTRATOR.read_text(encoding="utf-8")
        self.assertIn('LAUNCHER="$ROOT/scripts/ahd/run_ahd_1000.sh"', text)
        self.assertIn('"$LAUNCHER" agentic-esopt-eoh', text)
        self.assertIn('ES_OPERATORS="m1,m2"', text)
        self.assertIn('ES_OPERATORS="m1"', text)
        self.assertNotIn("pkill", text)
        self.assertNotIn("killall", text)

    def test_public_cleanup_cli_accepts_an_empty_owned_run_root(self):
        with TemporaryDirectory() as temporary:
            completed = subprocess.run(
                [str(ORCHESTRATOR), "cleanup"],
                env={
                    **__import__("os").environ,
                    "RUN_ROOT": temporary,
                    "EXPECTED_COMMIT": "static-test",
                    "AHD_STAGE1_GATE": temporary,
                    "AHD_STAGE2_GATE": temporary,
                },
                check=True,
                text=True,
                capture_output=True,
            )
        self.assertIn("cleanup checked only PID files owned", completed.stdout)


if __name__ == "__main__":
    unittest.main()
