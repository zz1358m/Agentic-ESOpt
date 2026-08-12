from __future__ import annotations

import json
import random
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory

import numpy as np
import torch

from eoh.methods.eoh.eoh_interface_EC import InterfaceEC
from algorithms.es.seeded_model_es import SeedReplayModelES


class InProcessModelESClient:
    """Exercise the same state machine that the HTTP model server exposes."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.state = SeedReplayModelES()
        self.state.init(model, parameter_scope="full", verbose=False)

    def apply_perturbation(self, *, seed: int, sigma: float):
        return self.state.apply(seed=seed, sigma=sigma)

    def revert_perturbation(self, *, seed: int, sigma: float):
        return self.state.revert(seed=seed, sigma=sigma)

    def update(self, **kwargs):
        return self.state.update(**kwargs)

    def status(self):
        return self.state.status()


class PerturbationAwareEvolution:
    def __init__(self, client: InProcessModelESClient):
        self.client = client
        self.calls = 0

    def i1(self):
        active = self.client.status()["active_perturbation"]
        if active is None:
            raise AssertionError("generation ran without an active target-ES perturbation")
        self.calls += 1
        return f"def candidate_{self.calls}():\n    return 1\n", f"candidate-{self.calls}"


def parameter_vector(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1).cpu() for parameter in model.parameters()])


class FourMethodESAdapterTest(unittest.TestCase):
    def make_interface(self, clients, history_path: Path, resume_history=None):
        interface = object.__new__(InterfaceEC)
        interface.paras = SimpleNamespace(
            llm_es_sigma=0.1,
            llm_es_sigma_start=0.1,
            llm_es_sigma_end=0.0,
            llm_es_sigma_schedule="cosine",
            llm_es_sigma_warmup_steps=0,
            llm_es_alpha=0.05,
            llm_es_directions=4,
            llm_es_seed=2024,
            llm_es_reward_mode="negative_objective",
            llm_es_reward_floor=-1e30,
            llm_es_reward_normalization="zscore",
            llm_es_reward_normalization_ddof=0,
            llm_es_reward_normalization_eps=1e-8,
            llm_es_invalid_reward_strategy="current",
            llm_es_batch_relative_invalid_reward=True,
            llm_es_invalid_reward_margin=1.0,
            llm_es_invalid_reward_fallback_fraction=0.01,
            llm_es_invalid_reward_min_gap=1.0,
            llm_es_disable_update=False,
            llm_es_max_workers=len(clients),
            llm_es_operators=["i1"],
            llm_es_resume_history=str(resume_history) if resume_history else None,
            ec_m1m2_multiplier=1.0,
        )
        interface.pop_size = 4
        interface.m = 2
        interface.select = None
        interface.n_p = 2
        interface.debug = True
        interface.use_numba = False
        interface.invalid_objective = float("inf")
        interface.model_es_enabled = True
        interface.model_es_clients = clients
        interface.model_es_evolutions = [PerturbationAwareEvolution(client) for client in clients]
        interface.model_es_client_locks = [threading.Lock() for _ in clients]
        interface.model_es_rng = random.Random(2024)
        interface.current_generation_index = 0
        interface.total_generations = 2
        interface.model_es_history_path = history_path
        interface.model_es_history = []
        interface._evaluate_offspring_batch_with_timeout = lambda pairs: [
            float(index + 1) for index, _ in enumerate(pairs)
        ]
        return interface

    def test_target_state_machine_update_sync_schedule_and_replay(self):
        torch.manual_seed(7)
        initial_model = torch.nn.Linear(3, 2, bias=True)
        initial_state = initial_model.state_dict()

        def new_client():
            model = torch.nn.Linear(3, 2, bias=True)
            model.load_state_dict(initial_state)
            return InProcessModelESClient(model)

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            clients = [new_client(), new_client()]
            baseline = parameter_vector(clients[0].model).clone()
            interface = self.make_interface(clients, root / "history.json")

            _, first_batch = interface.get_algorithm([], "i1", offspring_count=4)
            self.assertEqual(len(first_batch), 4)
            self.assertTrue(all(np.isfinite(item["objective"]) for item in first_batch))
            self.assertTrue(all(client.status()["active_perturbation"] is None for client in clients))
            self.assertTrue(all(client.status()["update_history"] == 1 for client in clients))
            torch.testing.assert_close(parameter_vector(clients[0].model), parameter_vector(clients[1].model))
            self.assertFalse(torch.equal(parameter_vector(clients[0].model), baseline))

            interface.set_generation_context(1, total_generations=2)
            interface.get_algorithm([], "i1", offspring_count=4)
            self.assertTrue(all(client.status()["active_perturbation"] is None for client in clients))
            self.assertTrue(all(client.status()["update_history"] == 2 for client in clients))
            torch.testing.assert_close(parameter_vector(clients[0].model), parameter_vector(clients[1].model))

            history = json.loads((root / "history.json").read_text(encoding="utf-8"))
            self.assertEqual([record["sigma"] for record in history], [0.1, 0.0])
            self.assertEqual([record["sigma_schedule"] for record in history], ["cosine", "cosine"])
            self.assertEqual([record["engine_count"] for record in history], [2, 2])
            self.assertEqual([record["generation_concurrency"] for record in history], [2, 2])
            self.assertEqual([record["evaluation_concurrency"] for record in history], [2, 2])
            candidate_audit = json.loads((root / "operator_candidates.json").read_text(encoding="utf-8"))
            self.assertEqual([record["candidate_count"] for record in candidate_audit], [4, 4])

            replay_clients = [new_client(), new_client()]
            replay = self.make_interface(
                replay_clients,
                root / "replayed-history.json",
                resume_history=root / "history.json",
            )
            replay._restore_model_es_history()
            self.assertTrue(all(client.status()["update_history"] == 2 for client in replay_clients))
            torch.testing.assert_close(
                parameter_vector(replay_clients[0].model),
                parameter_vector(clients[0].model),
            )
            torch.testing.assert_close(
                parameter_vector(replay_clients[0].model),
                parameter_vector(replay_clients[1].model),
            )

    def test_cpu_evaluator_audit_records_four_real_processes(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            interface = self.make_interface([], root / "history.json")
            interface.n_p = 4
            interface.timeout = 5
            del interface._evaluate_offspring_batch_with_timeout
            interface.current_operator = "m1"
            def evaluate(code):
                import time

                time.sleep(0.1)
                return float(code)

            interface.interface_eval = SimpleNamespace(evaluate=evaluate)
            interface.evaluator_audit_path = root / "evaluator_processes.json"
            interface.evaluator_audit = []
            pairs = [
                (None, {"code": str(value), "objective": None})
                for value in (1, 2, 3, 4)
            ]

            self.assertEqual(interface._evaluate_offspring_batch_with_timeout(pairs), [1.0, 2.0, 3.0, 4.0])
            audit = json.loads((root / "evaluator_processes.json").read_text(encoding="utf-8"))
            self.assertEqual(audit[0]["configured_workers"], 4)
            self.assertEqual(audit[0]["max_concurrent_processes"], 4)
            self.assertEqual(len(set(audit[0]["process_pids"])), 4)
            self.assertEqual(len(audit[0]["process_intervals_monotonic"]), 4)
            self.assertEqual(audit[0]["timed_out_process_intervals_monotonic"], [])
            latest_start = max(interval[0] for interval in audit[0]["process_intervals_monotonic"])
            earliest_finish = min(interval[1] for interval in audit[0]["process_intervals_monotonic"])
            self.assertLess(latest_start, earliest_finish)


if __name__ == "__main__":
    unittest.main()
