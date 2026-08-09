from __future__ import annotations

import unittest

import torch

from algorithms.es.seeded_model_es import SeedReplayModelES


class SeedReplayModelESTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = torch.nn.Linear(3, 2, bias=False)
        self.initial = self.model.weight.detach().clone()
        self.es = SeedReplayModelES()
        self.es.init(self.model, parameter_scope="full", verbose=False)

    def test_perturbation_reverts(self) -> None:
        self.es.apply(seed=11, sigma=1e-2)
        self.assertFalse(torch.equal(self.model.weight, self.initial))
        self.es.revert(seed=11, sigma=1e-2)
        self.assertTrue(torch.allclose(self.model.weight, self.initial, atol=1e-7, rtol=0.0))

    def test_update_can_be_reset(self) -> None:
        self.es.update(seeds=[11, 12], rewards=[0.0, 1.0], alpha=1e-2)
        self.assertFalse(torch.equal(self.model.weight, self.initial))
        result = self.es.reset()
        self.assertEqual(result["reverted_updates"], 1)
        self.assertTrue(torch.allclose(self.model.weight, self.initial, atol=1e-7, rtol=0.0))

    def test_invalid_numeric_inputs_fail_before_mutation(self) -> None:
        with self.assertRaises(ValueError):
            self.es.apply(seed=1, sigma=float("nan"))
        with self.assertRaises(ValueError):
            self.es.update(seeds=[1], rewards=[float("nan")], alpha=1e-2)
        self.assertTrue(torch.equal(self.model.weight, self.initial))

    def test_update_requires_reverted_perturbation(self) -> None:
        self.es.apply(seed=3, sigma=1e-2)
        with self.assertRaises(RuntimeError):
            self.es.update(seeds=[3], rewards=[1.0], alpha=1e-2)
        self.es.revert(seed=3, sigma=1e-2)

    def test_reinitialization_restores_the_base_model(self) -> None:
        self.es.update(seeds=[11, 12], rewards=[0.0, 1.0], alpha=1e-2)
        result = self.es.init(self.model, parameter_scope="full", verbose=False)
        self.assertEqual(result["previous_state_reset"]["reverted_updates"], 1)
        self.assertTrue(torch.allclose(self.model.weight, self.initial, atol=1e-7, rtol=0.0))


if __name__ == "__main__":
    unittest.main()
