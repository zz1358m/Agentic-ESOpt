"""Compatibility shims for the cluster's Torch/SGLang/Transformers stack."""

from __future__ import annotations

import os
from pathlib import Path


def configure_triton_cache(root: str | os.PathLike[str], role: str) -> Path:
    """Give one long-lived model service its own writable Triton cache."""
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    if not role or any(character not in allowed for character in role):
        raise ValueError(f"invalid Triton cache role: {role!r}")
    path = Path(root).expanduser().resolve() / role
    path.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(path)
    return path


def bounded_max_new_tokens(
    *,
    response_length: int,
    max_model_len: int,
    prompt_length: int,
    requested_max_new_tokens: int | None,
) -> int:
    """Respect a per-turn cap while retaining trajectory/context limits."""
    requested = response_length if requested_max_new_tokens is None else requested_max_new_tokens
    return min(requested, response_length, max_model_len - prompt_length - 1)


def patch_runtime() -> None:
    """Apply import-time shims before importing any SGLang entrypoint."""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if not getattr(CONFIG_MAPPING.register, "_verl_vllm_compat", False):
        transformers_register = CONFIG_MAPPING.register

        def register_vllm_compat(key, value, exist_ok=False):
            return transformers_register(key, value, exist_ok=exist_ok or key == "aimv2")

        register_vllm_compat._verl_vllm_compat = True
        CONFIG_MAPPING.register = register_vllm_compat

    # SGLang 0.5 imports every optional kernel family eagerly. The Torch-2.7
    # compatible sgl-kernel does not export several newer GPTQ/GPT-OSS/Mamba
    # symbols. This BF16 Qwen3Next path uses none of them; keep imports working
    # and fail loudly if an unsupported optional path is selected.
    import sgl_kernel

    def unsupported_optional_sgl_kernel(*args, **kwargs):
        raise RuntimeError(
            "This optional SGLang kernel requires Torch 2.8 and is unavailable in the Torch 2.7 runtime"
        )

    for kernel_name in (
        "FusedSetKVBufferArg",
        "causal_conv1d_fwd",
        "causal_conv1d_update",
        "gelu_quick",
        "gptq_gemm",
        "gptq_marlin_gemm",
        "gptq_shuffle",
    ):
        if not hasattr(sgl_kernel, kernel_name):
            setattr(sgl_kernel, kernel_name, unsupported_optional_sgl_kernel)

    if os.environ.get("TRACE2SKILL_PATCH_DENSE_QWEN3NEXT") == "1":
        from verl_trace2skill.sglang_dense_qwen3next_compat import patch_sglang_dense_qwen3next

        if not patch_sglang_dense_qwen3next():
            raise RuntimeError("failed to enable dense Qwen3Next compatibility for SGLang")

    # Import this only after Ray has isolated the worker's GPU. Importing the
    # agent-loop registry from sitecustomize initializes torch too early and
    # makes every NCCL rank cache the first visible device.
    if os.environ.get("TRACE2SKILL_REGISTER_TOOL_PARSER") == "1":
        from verl.experimental.agent_loop.tool_parser import ToolParser

        if "trace2skill" not in ToolParser._registry:
            raise RuntimeError("trace2skill tool parser was not registered")
