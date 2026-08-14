import json
import unittest
from pathlib import Path

from scripts.validate_data import validate_public_payload


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "public" / "data"


class PublicDataTests(unittest.TestCase):
    def load(self, name: str):
        payload = json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))
        validate_public_payload(payload)
        return payload

    def test_every_public_payload_is_present_and_sanitized(self):
        for name in ("sudoku", "math", "docvqa", "webarena", "ahd", "scaling"):
            with self.subTest(name=name):
                self.load(name)

    def test_math_and_docvqa_keep_only_confirmed_replay_nodes(self):
        math = self.load("math")
        docvqa = self.load("docvqa")
        self.assertEqual({case["dataset"] for case in math["cases"]}, {"DAPO", "AIME 2026"})
        self.assertGreaterEqual(len(docvqa["cases"]), 2)
        for case in math["cases"]:
            self.assertEqual([item["generation"] for item in case["checkpoints"]], [9, 19, 24, 25])
        for case in docvqa["cases"]:
            self.assertEqual([item["generation"] for item in case["checkpoints"]], [-1, 9, 19, 29, 39])
        self.assertEqual(math["cases"][0]["answer"], "35")
        self.assertEqual(docvqa["cases"][0]["answers"], ["Round-Robin", "Round-Robin tennis match"])
        self.assertEqual(
            [item["score"] for item in docvqa["cases"][0]["checkpoints"]],
            [0.0, 0.0, 0.75, 1.0, 1.0],
        )

    def test_sudoku_masks_and_scaling_values_match_the_approved_result_set(self):
        sudoku = self.load("sudoku")
        scaling = self.load("scaling")
        self.assertEqual(sorted(case["maskCount"] for case in sudoku["cases"]), [5, 10, 15])
        self.assertEqual([row["best"] for row in scaling["finalResults"]], [5.10, 35.42, 30.21, 37.50])
        self.assertEqual(scaling["finalResults"][1]["finalRelative"], 677.0)
        self.assertEqual(scaling["metadata"]["title"], "Model-size and ES Population Scaling")
        self.assertEqual(
            scaling["configurations"],
            [
                {"axis": "modelSize", "values": ["4B", "9B"]},
                {"axis": "esPopulationSize", "symbol": "G", "values": [8, 16], "meaning": "perturbation directions per ES update"},
            ],
        )
        self.assertEqual(
            {(row["model"], row["population"]) for row in scaling["finalResults"]},
            {("4B", 8), ("4B", 16), ("9B", 8), ("9B", 16)},
        )
        mask15_eval = next(curve for curve in sudoku["curves"] if curve["id"] == "mask15-eval")
        self.assertEqual(mask15_eval["points"][-1], {"generation": 99, "value": 0.53125, "std": 0.025516})
        replay = next(case for case in sudoku["cases"] if case.get("capabilityCheckpoints"))
        self.assertEqual(replay["id"], "eval-000064")
        self.assertEqual(replay["maskCount"], 5)
        self.assertEqual(
            [item["optimizationStep"] for item in replay["capabilityCheckpoints"]],
            [-1, 9, 19, 29, 39],
        )
        predictions = [json.dumps(item["prediction"]) for item in replay["capabilityCheckpoints"]]
        self.assertEqual(len(set(predictions)), 2)
        self.assertEqual(sum(left != right for left, right in zip(predictions, predictions[1:])), 1)
        self.assertEqual([item["score"] for item in replay["capabilityCheckpoints"]], [0.0, 1.0, 1.0, 1.0, 1.0])
        self.assertIn("maximum three-repeat base-to-final gain", replay["evidenceScope"])
        self.assertEqual(sudoku["checkpoints"], [])
        self.assertFalse(any("turns" in item for item in replay["capabilityCheckpoints"]))

    def test_webarena_publishes_outcomes_but_no_invented_browser_trajectory(self):
        webarena = self.load("webarena")
        self.assertGreaterEqual(len({case["site"] for case in webarena["cases"]}), 5)
        for case in webarena["cases"]:
            outcomes = case["outcomes"]
            self.assertEqual(len(outcomes), 12)
            self.assertFalse(any("observations" in item or "actions" in item for item in outcomes))
        self.assertIn("not retained", webarena["metadata"]["note"])
        serialized = json.dumps(webarena)
        self.assertNotIn("/tmp/", serialized)
        self.assertNotIn("127.0.0.1", serialized)
        self.assertIn("[local endpoint]", serialized)
        capability = next(case for case in webarena["cases"] if case["id"] == "task-4")
        self.assertEqual(
            [item["optimizationStep"] for item in capability["capabilityCheckpoints"]],
            [10, 50, 70],
        )
        self.assertEqual(
            [item["score"] for item in capability["capabilityCheckpoints"]],
            [0.0, 1.0, 1.0],
        )
        no_skill = [item["hard"] for item in capability["outcomes"] if item["setting"] == "No Skill"]
        esopt = [item["hard"] for item in capability["outcomes"] if item["setting"] == "Agentic ESOpt"]
        self.assertEqual(no_skill, [0, 0, 0])
        self.assertEqual(esopt, [1, 1, 1])
        self.assertIn("maximum three-repeat No Skill-to-ESOpt gain", capability["evidenceScope"])
        self.assertNotIn("output", capability["capabilityCheckpoints"][0])
        self.assertTrue(capability["capabilityCheckpoints"][0]["outputUnavailable"])
        self.assertTrue(capability["capabilityCheckpoints"][-1]["output"])
        self.assertEqual(webarena["checkpoints"], [])
        self.assertIn("not a separate training-split score", webarena["metadata"]["note"])

    def test_ahd_has_real_objectives_and_execution_pass_code_evolution(self):
        ahd = self.load("ahd")
        configurations = ahd["configurations"]
        self.assertEqual({item["problem"] for item in configurations}, {"TSP", "KP", "ASP", "CVRP", "BPP"})
        self.assertEqual({item["mode"] for item in configurations}, {"Constructive", "ACO"})
        self.assertEqual({item["outerMethod"] for item in configurations}, {"Sample", "EoH"})
        self.assertEqual({item["agenticESOpt"] for item in configurations}, {False, True})
        self.assertEqual({item["budget"] for item in configurations}, {1000, 2000})
        self.assertEqual({item["repeat"] for item in configurations}, {1, 2, 3})
        selected = next(case for case in ahd["cases"] if case["id"] == "tsp-aco-sample-agentic-1000-r1")
        selected_config = next(item for item in configurations if item["id"] == selected["configId"])
        self.assertEqual(selected_config["mode"], "ACO")
        self.assertEqual(selected_config["outerMethod"], "Sample")
        self.assertTrue(selected_config["agenticESOpt"])
        self.assertIn("def heuristics", selected["finalHeuristic"])
        self.assertTrue(any("aco_tsp" in source for source in selected_config["sourceFiles"]))
        self.assertEqual(
            [item["optimizationStep"] for item in selected["capabilityCheckpoints"]],
            [1, 12, 50],
        )
        self.assertEqual(
            [item["objective"] for item in selected["capabilityCheckpoints"]],
            [6.48937, 5.9408, 5.90256],
        )
        self.assertLess(selected["capabilityCheckpoints"][-1]["objective"], 6.0)
        self.assertEqual(
            selected["capabilityCheckpoints"][-1]["testInstanceMinimum"]["value"],
            5.331992149353027,
        )
        self.assertEqual(
            selected["capabilityCheckpoints"][-1]["testInstanceMinimum"]["scope"],
            "TSP-50 · minimum across 64 frozen-test instances",
        )
        self.assertFalse(
            any("testInstanceMinimum" in item for item in selected["capabilityCheckpoints"][:-1])
        )
        self.assertEqual(len({item["heuristic"] for item in selected["capabilityCheckpoints"]}), 3)
        self.assertIn("Favorable eligible code-evolution case", selected["evidenceScope"])
        self.assertIn("execution-side PASS", ahd["metadata"]["note"])
        self.assertIn("final review remains pending", ahd["metadata"]["note"])
        self.assertFalse(any("optimizationStep" in item for item in ahd["checkpoints"]))

    def test_build_audit_records_source_and_manuscript_checks(self):
        audit = json.loads((ROOT / "data_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["status"], "passed")
        self.assertGreaterEqual(len(audit["sourceConsistency"]), 39)
        self.assertTrue(all(item["status"] == "passed" for item in audit["sourceConsistency"]))
        manuscript_checks = [item for item in audit["sourceConsistency"] if "cell by cell" in item["check"]]
        self.assertEqual(len(manuscript_checks), 28)
        self.assertTrue(all("cell by cell" in item["check"] for item in manuscript_checks))


if __name__ == "__main__":
    unittest.main()
