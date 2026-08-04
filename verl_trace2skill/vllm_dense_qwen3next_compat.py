"""Narrow vLLM 0.19 compatibility for converted dense Qwen3Next models."""

from __future__ import annotations

import logging


logger = logging.getLogger(__name__)


def patch_vllm_dense_qwen3next() -> bool:
    """Skip MoE metadata initialization when ``num_experts`` is zero.

    vLLM's Qwen3Next implementation supports dense MLP layers, but its causal
    LM constructor unconditionally initializes the ``MixtureOfExperts`` mixin.
    Converted Qwen3.5 dense checkpoints therefore fail after model creation
    with ``No Qwen3Next layer found``. Preserve the upstream behavior for real
    MoE checkpoints and bypass only the explicit zero-expert configuration.
    """
    try:
        from vllm.model_executor.models import qwen3_next as module
    except Exception as exc:  # noqa: BLE001 - optional startup compatibility
        logger.warning("dense Qwen3Next compatibility patch could not import vLLM: %r", exc)
        return False

    model_class = module.Qwen3NextForCausalLM
    if getattr(model_class, "_trace2skill_dense_compat", False):
        return True

    original = model_class.set_moe_parameters
    original_init = model_class.__init__

    def set_moe_parameters(self):
        config = self.vllm_config.model_config.hf_text_config
        if int(getattr(config, "num_experts", 0)) == 0:
            self.expert_weights = []
            self.moe_layers = []
            self.num_moe_layers = 0
            return None
        return original(self)

    def init_with_tied_head(self, *, vllm_config, prefix=""):
        original_init(self, vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_text_config
        if int(getattr(config, "num_experts", 0)) == 0 and bool(
            getattr(config, "tie_word_embeddings", False)
        ):
            del self.lm_head.weight
            self.lm_head.weight = self.model.embed_tokens.weight

    model_class.set_moe_parameters = set_moe_parameters
    model_class.__init__ = init_with_tied_head
    model_class._trace2skill_dense_compat = True
    logger.info("enabled dense Qwen3Next compatibility for vLLM")
    return True
