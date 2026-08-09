from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from algorithms.verl_trace2skill.vllm_dense_qwen3next_compat import patch_vllm_dense_qwen3next


class VllmDenseQwen3NextCompatTest(unittest.TestCase):
    @staticmethod
    def _config(*, num_experts: int):
        return SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(
                    num_experts=num_experts,
                    tie_word_embeddings=False,
                )
            )
        )

    def test_zero_experts_skips_upstream_moe_initialization(self):
        calls = []

        class Model:
            def __init__(self, *, vllm_config, prefix=""):
                self.vllm_config = vllm_config

            def set_moe_parameters(self):
                calls.append("upstream")

        fake_module = SimpleNamespace(Qwen3NextForCausalLM=Model)
        with patch.dict("sys.modules", {
            "vllm.model_executor.models.qwen3_next": fake_module,
        }):
            # Patch the package attribute used by ``from ... import``.
            import vllm.model_executor.models as models

            with patch.object(models, "qwen3_next", fake_module, create=True):
                self.assertTrue(patch_vllm_dense_qwen3next())

        instance = Model(vllm_config=self._config(num_experts=0))
        instance.set_moe_parameters()
        self.assertEqual(calls, [])
        self.assertEqual(instance.num_moe_layers, 0)

    def test_real_moe_keeps_upstream_behavior(self):
        calls = []

        class Model:
            def __init__(self, *, vllm_config, prefix=""):
                self.vllm_config = vllm_config

            def set_moe_parameters(self):
                calls.append("upstream")

        fake_module = SimpleNamespace(Qwen3NextForCausalLM=Model)
        import vllm.model_executor.models as models

        with patch.object(models, "qwen3_next", fake_module, create=True):
            self.assertTrue(patch_vllm_dense_qwen3next())

        instance = Model(vllm_config=self._config(num_experts=8))
        instance.set_moe_parameters()
        self.assertEqual(calls, ["upstream"])


if __name__ == "__main__":
    unittest.main()
