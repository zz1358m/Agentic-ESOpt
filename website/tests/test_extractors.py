import json
import tempfile
import unittest
from pathlib import Path

from scripts.extract_ahd import parse_search_log
from scripts.extract_capability import (
    extract_ahd_heuristic_checkpoints,
    extract_sudoku_case_checkpoints,
    parse_webarena_case_progression,
    parse_webarena_task_scores,
)
from scripts.extract_history import extract_history_case
from scripts.extract_sudoku import parse_rl_training_log, parse_training_log
from scripts.validate_data import validate_public_payload


class ExtractorTests(unittest.TestCase):
    def test_sudoku_capability_extractor_keeps_same_case_outputs(self):
        history = [
            {
                "generation": -1,
                "eval": {
                    "average": 0.25,
                    "runs": [{"scores": [{
                        "task_id": "eval-1",
                        "score": 0.0,
                        "prediction": [[1, 1], [2, 2]],
                        "feedback": "invalid row 1",
                        "turns": [{"turn": 0, "response": "set r1c2 1", "board": [[1, 0], [2, 2]]}],
                    }]}],
                },
            },
            {
                "generation": 9,
                "eval": {
                    "average": 0.75,
                    "runs": [{"scores": [{
                        "task_id": "eval-1",
                        "score": 1.0,
                        "prediction": [[1, 2], [2, 1]],
                        "feedback": "The grid is valid.",
                        "turns": [{"turn": 0, "response": "set r1c2 2", "board": [[1, 0], [2, 1]]}],
                    }]}],
                },
            },
        ]

        checkpoints = extract_sudoku_case_checkpoints(history, case_id="eval-1")

        self.assertEqual([item["optimizationStep"] for item in checkpoints], [-1, 9])
        self.assertEqual([item["aggregateMetric"] for item in checkpoints], [0.25, 0.75])
        self.assertEqual([item["score"] for item in checkpoints], [0.0, 1.0])
        self.assertNotEqual(checkpoints[0]["prediction"], checkpoints[1]["prediction"])
        self.assertFalse(any("turns" in item or "turnCount" in item for item in checkpoints))

    def test_webarena_progression_links_scores_without_inventing_outputs(self):
        text = "\n".join([
            "[eval] eval_after_epoch_010 endpoint=http://127.0.0.1:12013 task=75 score=0.0",
            "[eval] eval_after_epoch_040 endpoint=http://127.0.0.1:12013 task=75 score=0.0",
            "[eval] eval_after_epoch_070 endpoint=http://127.0.0.1:12013 task=75 score=1.0",
        ])

        parsed_scores = parse_webarena_task_scores(text)
        checkpoints = parse_webarena_case_progression(
            text,
            task_id=75,
            aggregate_by_epoch={10: 0.2, 40: 0.3, 70: 0.4},
            selected_epochs=[10, 40, 70],
            final_output="Lo, Chen, Chu",
            parsed_scores=parsed_scores,
        )

        self.assertEqual(parsed_scores, {75: {10: 0.0, 40: 0.0, 70: 1.0}})
        self.assertEqual([item["score"] for item in checkpoints], [0.0, 0.0, 1.0])
        self.assertTrue(checkpoints[0]["outputUnavailable"])
        self.assertNotIn("output", checkpoints[0])
        self.assertEqual(checkpoints[-1]["output"], "Lo, Chen, Chu")
        with self.assertRaisesRegex(ValueError, "lacks a linked score"):
            parse_webarena_case_progression(
                text,
                task_id=75,
                aggregate_by_epoch={10: 0.2},
                selected_epochs=[10],
                final_output="",
                parsed_scores={},
            )

    def test_ahd_capability_extractor_keeps_real_code_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for generation, objective, code in ((1, 8.0, "def h():\n    return 1\n"), (4, 7.0, "def h():\n    return 2\n"), (25, 6.0, "def h():\n    return 3\n")):
                (root / f"population_generation_{generation}.json").write_text(json.dumps({
                    "algorithm": f"version {generation}",
                    "objective": objective,
                    "code": code,
                }))

            checkpoints = extract_ahd_heuristic_checkpoints(root, generations=[1, 4, 25])

        self.assertEqual([item["optimizationStep"] for item in checkpoints], [1, 4, 25])
        self.assertEqual([item["objective"] for item in checkpoints], [8.0, 7.0, 6.0])
        self.assertEqual(len({item["heuristic"] for item in checkpoints}), 3)

    def test_sudoku_parser_separates_samples_and_periodic_evaluation(self):
        text = """\
[eval] generation=-1 split=train repeats=3 solved_avg=7.33/32 average=0.229167 std=0.029463
[eval] generation=-1 split=eval repeats=3 solved_avg=8.67/32 average=0.270833 std=0.038976
[sample] gen=0 idx=0 reward=0.03125
[sample] gen=0 idx=1 reward=0.09375
[sample] gen=1 idx=0 reward=0.125
"""
        parsed = parse_training_log(text)

        self.assertEqual(parsed["trainCurve"], [
            {"generation": 0, "value": 0.0625},
            {"generation": 1, "value": 0.125},
        ])
        self.assertEqual(parsed["periodicEval"][0]["generation"], -1)
        self.assertEqual(parsed["periodicEval"][0]["value"], 0.270833)
        self.assertEqual(parsed["periodicTrain"][0]["value"], 0.229167)

    def test_history_extractor_keeps_only_selected_case_and_safe_fields(self):
        history = [
            {"config": {"history_file": "/mnt/private/alice/run/history.json"}},
            {
                "generation": 9,
                "reward_mean": 0.5,
                "dapo_eval": {
                    "mean_score": 0.75,
                    "scores": [
                        {
                            "task_id": "keep",
                            "task": {"question": "Q?", "answer": "42", "source": "dataset"},
                            "score": 1.0,
                            "prediction": "42",
                            "react_steps": [
                                {
                                    "turn": 1,
                                    "assistant": "Action:\nrun tool",
                                    "observation": "from /home/alice/work result",
                                    "action": {"name": "bash", "arguments": {"command": "python /workspace/a.py"}},
                                }
                            ],
                            "tool_workdir": "/mnt/private/alice/work",
                        },
                        {"task_id": "drop", "score": 0.0},
                    ],
                },
            },
        ]

        payload = extract_history_case(
            history,
            case_id="keep",
            checkpoint_generations=[9],
            task_name="math",
            metric_name="Exact match",
        )

        self.assertEqual(payload["curves"][0]["points"], [{"generation": 9, "value": 0.5}])
        self.assertEqual(payload["cases"][0]["checkpoints"][0]["prediction"], "42")
        dumped = json.dumps(payload)
        self.assertNotIn("drop", dumped)
        self.assertNotIn("/mnt/private", dumped)
        self.assertNotIn("/home/alice", dumped)
        self.assertNotIn("tool_workdir", dumped)

    def test_rl_sudoku_parser_keeps_step_reward_turns_and_periodic_eval(self):
        text = """\
[eval] step=0 repeats=3 solved_avg=3.00/32 average=0.093750 std=0.044194
[train] step=1 reward=0.003906 solved=1/256 avg_turns=16.6953 examples=4274
"""
        parsed = parse_rl_training_log(text)
        self.assertEqual(parsed["trainCurve"], [{"generation": 1, "value": 0.003906, "averageTurns": 16.6953}])
        self.assertEqual(parsed["periodicEval"], [{"generation": 0, "value": 0.09375, "std": 0.044194}])

    def test_ahd_parser_tracks_population_best_and_invalid_candidates(self):
        text = """\
 OP: e1, [1 / 4] |
 Obj:  8.0| Obj:  inf| Obj:  7.5|
--- 1 of 2 populations finished. Time Cost: 1.0 m
Pop Objs:  7.5 8.0
 OP: m2, [4 / 4] |
 Obj:  7.2| Obj:  inf|
--- 2 of 2 populations finished. Time Cost: 2.0 m
Pop Objs:  7.2 7.5 8.0
"""
        parsed = parse_search_log(text)

        self.assertEqual(parsed["generations"][0]["best"], 7.5)
        self.assertEqual(parsed["generations"][1]["bestSoFar"], 7.2)
        self.assertEqual(parsed["generations"][0]["invalidCandidates"], 1)
        self.assertEqual(parsed["generations"][1]["operator"], "m2")

    def test_validator_rejects_private_paths_and_unknown_curve_kinds(self):
        valid = {
            "metadata": {"task": "math", "method": "Agentic ESOpt", "sourceFiles": ["history.json"]},
            "curves": [{"id": "train", "kind": "train", "points": [{"generation": 0, "value": 0.5}]}],
            "checkpoints": [],
            "cases": [],
            "finalResults": [],
        }
        validate_public_payload(valid)

        bad_kind = json.loads(json.dumps(valid))
        bad_kind["curves"][0]["kind"] = "mystery"
        with self.assertRaisesRegex(ValueError, "curve kind"):
            validate_public_payload(bad_kind)

        bad_path = json.loads(json.dumps(valid))
        bad_path["metadata"]["sourceFiles"] = ["/home/alice/private.log"]
        with self.assertRaisesRegex(ValueError, "private path"):
            validate_public_payload(bad_path)

        bad_contact = json.loads(json.dumps(valid))
        bad_contact["metadata"]["note"] = "Contact alice@example.org or 212-555-0199"
        with self.assertRaisesRegex(ValueError, "contact information"):
            validate_public_payload(bad_contact)


if __name__ == "__main__":
    unittest.main()
