"""Narrow SGLang 0.5.2 compatibility for converted dense Qwen3Next models."""

from __future__ import annotations

import logging
import os
from typing import Any


logger = logging.getLogger(__name__)


def enable_eager_patch_for_spawn(env: Any = os.environ) -> None:
    """Ensure a spawned SGLang scheduler patches itself after GPU isolation."""
    env["TRACE2SKILL_EAGER_PATCH_DENSE_QWEN3NEXT"] = "1"


def remap_dense_tied_weight_name(name: str) -> str:
    if name == "lm_head.weight":
        return "model.embed_tokens.weight"
    return name


def patch_sglang_dense_qwen3next() -> bool:
    """Make the 0.5.2 Qwen3Next loader honor ``num_experts=0``.

    SGLang 0.5.2 only shipped the sparse Qwen3Next variant. The official
    Qwen3.5-4B text backbone has the same hybrid-attention layout but dense
    MLPs. The repository converter records this as ``num_experts=0`` and
    ``mlp_only_layers=all``. This patch changes only that zero-expert case.
    """
    try:
        from sglang.srt.models import qwen3_next as module
    except Exception as exc:  # noqa: BLE001 - startup logs optional compatibility failures
        logger.warning("dense Qwen3Next compatibility patch could not import SGLang: %r", exc)
        return False

    model_class = module.Qwen3NextForCausalLM
    if getattr(model_class, "_trace2skill_dense_compat", False):
        return True

    sparse_mlp_class = module.Qwen2MoeSparseMoeBlock
    dense_mlp_class = module.Qwen2MoeMLP

    class DenseMLPAdapter(dense_mlp_class):
        def forward(self, hidden_states, forward_batch=None, use_reduce_scatter: bool = False):
            del forward_batch
            return super().forward(
                hidden_states,
                should_allreduce_fusion=False,
                use_reduce_scatter=use_reduce_scatter,
            )

    def dense_or_sparse_mlp(
        layer_id: int,
        config: Any,
        quant_config=None,
        alt_stream=None,
        prefix: str = "",
    ):
        if int(getattr(config, "num_experts", 0)) == 0:
            return DenseMLPAdapter(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=prefix,
            )
        return sparse_mlp_class(
            layer_id=layer_id,
            config=config,
            quant_config=quant_config,
            alt_stream=alt_stream,
            prefix=prefix,
        )

    module.Qwen2MoeSparseMoeBlock = dense_or_sparse_mlp

    original_init = model_class.__init__
    original_load_weights = model_class.load_weights

    def init_with_tied_head(self, config, quant_config=None, prefix: str = ""):
        original_init(self, config, quant_config=quant_config, prefix=prefix)
        if int(getattr(config, "num_experts", 0)) == 0 and bool(config.tie_word_embeddings):
            del self.lm_head.weight
            self.lm_head.weight = self.model.embed_tokens.weight

    original_expert_config = model_class.get_model_config_for_expert_location.__func__

    def expert_config_for_dense(cls, config):
        if int(getattr(config, "num_experts", 0)) == 0:
            return None
        return original_expert_config(cls, config)

    model_class.__init__ = init_with_tied_head

    def load_weights_with_tied_head(self, weights, is_mtp: bool = False):
        if int(getattr(self.config, "num_experts", 0)) == 0 and bool(self.config.tie_word_embeddings):
            weights = (
                (remap_dense_tied_weight_name(name), tensor)
                for name, tensor in weights
            )
        return original_load_weights(self, weights, is_mtp=is_mtp)

    model_class.load_weights = load_weights_with_tied_head
    model_class.get_model_config_for_expert_location = classmethod(expert_config_for_dense)
    model_class._trace2skill_dense_compat = True
    logger.info("enabled dense Qwen3Next compatibility for SGLang 0.5.2")
    return True
