from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from es.run_state import (
    atomic_write_history,
    completed_update_records,
    history_prefix_through_updates,
    map_endpoint_serial,
    read_history,
    sigma_at_step,
    validate_es_run_shape,
    validate_seed_sequence,
)


class SigmaScheduleTest(unittest.TestCase):
    def test_cosine_includes_requested_endpoints(self) -> None:
        values = [
            sigma_at_step(
                sigma_start=1.0,
                sigma_end=0.25,
                step=step,
                total_steps=5,
                schedule="cosine",
            )
            for step in range(5)
        ]
        self.assertEqual(values[0], 1.0)
        self.assertEqual(values[-1], 0.25)
        self.assertTrue(all(left >= right for left, right in zip(values, values[1:])))

    def test_warmup_does_not_change_final_value(self) -> None:
        values = [
            sigma_at_step(
                sigma_start=2.0,
                sigma_end=1.0,
                step=step,
                total_steps=6,
                schedule="linear",
                warmup_steps=2,
            )
            for step in range(6)
        ]
        self.assertEqual(values[:3], [2.0, 2.0, 2.0])
        self.assertEqual(values[-1], 1.0)

    def test_warmup_cannot_hide_requested_end_value(self) -> None:
        values = [
            sigma_at_step(
                sigma_start=2.0,
                sigma_end=0.5,
                step=step,
                total_steps=4,
                schedule="linear",
                warmup_steps=99,
            )
            for step in range(4)
        ]
        self.assertEqual(values, [2.0, 2.0, 2.0, 0.5])

    def test_non_finite_endpoints_are_rejected(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    sigma_at_step(
                        sigma_start=value,
                        sigma_end=0.5,
                        step=0,
                        total_steps=2,
                        schedule="linear",
                    )

class HistoryTest(unittest.TestCase):
    def test_atomic_round_trip_and_filter(self) -> None:
        history = [
            {"config": {}},
            {"generation": -1, "eval": {}},
            {"generation": 0, "seeds": [1, 2], "rewards": [0.0, 1.0]},
            {"generation": 1, "seeds": [3, 4], "rewards": [1.0, 1.0]},
            {
                "generation": 2,
                "seeds": [5, 6],
                "rewards": [1.0, 0.0],
                "update_applied": False,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.json"
            atomic_write_history(path, history)
            loaded = read_history(path)
        self.assertEqual(loaded, history)
        self.assertEqual(len(completed_update_records(loaded)), 2)
        prefix = history_prefix_through_updates(loaded, 1)
        self.assertEqual([row.get("generation") for row in prefix], [None, -1, 0])

    def test_seed_sequence_validation(self) -> None:
        import random

        rng = random.Random(7)
        records = [
            {
                "generation": generation,
                "seeds": [rng.randrange(1, 2**31 - 1) for _ in range(2)],
                "rewards": [0.0, 1.0],
            }
            for generation in range(3)
        ]
        self.assertEqual(validate_seed_sequence(records, population=2, seed=7), 3)


class RunShapeTest(unittest.TestCase):
    def test_training_dimensions_must_be_positive(self) -> None:
        validate_es_run_shape(generations=1, population=1, case_batch_size=1)
        with self.assertRaises(ValueError):
            validate_es_run_shape(generations=0, population=1, case_batch_size=1)
        with self.assertRaises(ValueError):
            validate_es_run_shape(generations=1, population=0, case_batch_size=1)


class EndpointSchedulingTest(unittest.TestCase):
    def test_same_endpoint_never_overlaps(self) -> None:
        lock = threading.Lock()
        active = {"a": 0, "b": 0}
        peaks = {"a": 0, "b": 0}

        def worker(index: int, endpoint: str):
            with lock:
                active[endpoint] += 1
                peaks[endpoint] = max(peaks[endpoint], active[endpoint])
            time.sleep(0.005)
            with lock:
                active[endpoint] -= 1
            return index, f"{endpoint}:{index}"

        results = map_endpoint_serial(endpoints=["a", "b"], count=8, worker=worker)
        self.assertEqual(results, [f"{'a' if index % 2 == 0 else 'b'}:{index}" for index in range(8)])
        self.assertEqual(peaks, {"a": 1, "b": 1})


if __name__ == "__main__":
    unittest.main()
