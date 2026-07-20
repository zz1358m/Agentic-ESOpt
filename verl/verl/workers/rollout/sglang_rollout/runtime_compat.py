"""Compatibility shims for the cluster's Torch/SGLang/Transformers stack."""

from __future__ import annotations


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
