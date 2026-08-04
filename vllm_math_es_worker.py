"""vLLM worker extension for seed-replay model ES.

This module is imported inside vLLM worker processes via
``worker_extension_cls="vllm_math_es_worker.WorkerExtension"``.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Sequence

import torch


ParameterInfo = tuple[str, int, torch.nn.Parameter]


def _find_model(worker):
    if hasattr(worker, "gpu_model_runner") and hasattr(worker.gpu_model_runner, "model"):
        return worker.gpu_model_runner.model
    if hasattr(worker, "model_runner") and hasattr(worker.model_runner, "model"):
        return worker.model_runner.model
    if hasattr(worker, "model"):
        return worker.model
    raise RuntimeError("Cannot locate model inside vLLM worker.")


def _stable_tensor_id(name: str) -> int:
    digest = hashlib.blake2b(str(name).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def _mix_seed(base_seed: int, tensor_id: int) -> int:
    return (int(base_seed) ^ int(tensor_id)) & 0xFFFFFFFFFFFFFFFF


def _normalize_parameter_scope(parameter_scope: str) -> str:
    scope = str(parameter_scope or "full").strip().lower()
    if scope not in {"full", "all_linear", "lora"}:
        raise ValueError("parameter_scope must be 'full', 'all_linear', or 'lora'.")
    return scope


def _matches_target_module(module_suffix: str, target_modules: Iterable[str] | None) -> bool:
    if target_modules is None:
        return True
    return module_suffix in {str(module_name) for module_name in target_modules}


def _gather_full_parameters(model) -> list[ParameterInfo]:
    seen_ids: set[int] = set()
    parameter_infos: list[ParameterInfo] = []
    for name, parameter in model.named_parameters():
        if id(parameter) in seen_ids:
            continue
        if not torch.is_floating_point(parameter):
            continue
        seen_ids.add(id(parameter))
        parameter_infos.append((name, _stable_tensor_id(name), parameter))
    return parameter_infos


def _gather_linear_parameters(model, target_modules: Iterable[str] | None = None) -> list[ParameterInfo]:
    skip_keywords = {"embed_tokens", "lm_head", "norm"}
    seen_ids: set[int] = set()
    parameter_infos: list[ParameterInfo] = []
    for module_name, module in model.named_modules():
        if any(keyword in module_name for keyword in skip_keywords):
            continue
        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim != 2 or id(weight) in seen_ids:
            continue
        if not torch.is_floating_point(weight):
            continue
        module_suffix = module_name.split(".")[-1]
        if not _matches_target_module(module_suffix, target_modules):
            continue
        name = f"{module_name}.weight"
        seen_ids.add(id(weight))
        parameter_infos.append((name, _stable_tensor_id(name), weight))
    return parameter_infos


def _gather_lora_parameters(model) -> list[ParameterInfo]:
    parameter_infos = []
    for name, parameter in model.named_parameters():
        if "lora_" not in name:
            continue
        if not torch.is_floating_point(parameter):
            continue
        parameter_infos.append((name, _stable_tensor_id(name), parameter))
    if not parameter_infos:
        raise RuntimeError("No LoRA parameters found in the vLLM model.")
    return parameter_infos


def _iter_flat_chunks(numel: int, chunk_size: int = 8_388_608):
    for start in range(0, int(numel), int(chunk_size)):
        yield start, min(start + int(chunk_size), int(numel))


@torch.no_grad()
def _apply_seeded_noise(parameter_infos: list[ParameterInfo], *, seed: int, sigma: float) -> None:
    for _, tensor_id, parameter in parameter_infos:
        parameter_flat = parameter.view(-1)
        generator = torch.Generator(device=parameter.device)
        generator.manual_seed(_mix_seed(seed, tensor_id))
        for start, end in _iter_flat_chunks(parameter_flat.numel()):
            noise = torch.randn(
                (end - start,),
                generator=generator,
                dtype=torch.float32,
                device=parameter.device,
            )
            parameter_flat[start:end].add_(noise.to(dtype=parameter.dtype), alpha=float(sigma))


@torch.no_grad()
def _apply_dipu_update(
    parameter_infos: list[ParameterInfo],
    *,
    seeds: Sequence[int],
    weights: Sequence[float],
    eta: float,
) -> None:
    if not seeds:
        raise ValueError("ES update requires at least one seed.")
    if len(seeds) != len(weights):
        raise ValueError("ES update requires seeds and weights to have the same length.")

    scale = float(eta) / float(len(seeds))
    for _, tensor_id, parameter in parameter_infos:
        parameter_flat = parameter.view(-1)
        generators = []
        coeffs = []
        for seed, weight in zip(seeds, weights):
            generator = torch.Generator(device=parameter.device)
            generator.manual_seed(_mix_seed(int(seed), tensor_id))
            generators.append(generator)
            coeffs.append(scale * float(weight))

        for start, end in _iter_flat_chunks(parameter_flat.numel()):
            total_delta = torch.zeros((end - start,), dtype=torch.float32, device=parameter.device)
            for generator, coeff in zip(generators, coeffs):
                noise = torch.randn(
                    (end - start,),
                    generator=generator,
                    dtype=torch.float32,
                    device=parameter.device,
                )
                total_delta.add_(noise, alpha=coeff)
            parameter_flat[start:end].add_(total_delta.to(dtype=parameter.dtype))


class WorkerExtension:
    @torch.no_grad()
    def init_math_es(
        self,
        parameter_scope: str = "full",
        target_modules: Iterable[str] | None = None,
        verbose: bool = True,
    ) -> dict:
        scope = _normalize_parameter_scope(parameter_scope)
        model = _find_model(self)
        if scope == "full":
            parameter_infos = _gather_full_parameters(model)
        elif scope == "all_linear":
            parameter_infos = _gather_linear_parameters(model, target_modules=target_modules)
        else:
            parameter_infos = _gather_lora_parameters(model)

        self._math_es_parameter_infos = parameter_infos
        self._math_es_parameter_scope = scope
        self._math_es_update_history = []
        self._math_es_is_perturbed = False
        total_params = sum(parameter.numel() for _, _, parameter in parameter_infos)
        if verbose:
            print(f"[MATH_ES] scope={scope}, tensors={len(parameter_infos)}, params={total_params:,}")
        return {
            "ok": True,
            "parameter_scope": scope,
            "n_tensors": len(parameter_infos),
            "total_params": int(total_params),
        }

    def _require_initialized(self) -> list[ParameterInfo]:
        parameter_infos = getattr(self, "_math_es_parameter_infos", None)
        if parameter_infos is None:
            raise RuntimeError("Math vLLM ES is not initialized. Call init_math_es first.")
        return parameter_infos

    @torch.no_grad()
    def apply_perturbation(self, seed: int, sigma: float) -> dict:
        _apply_seeded_noise(self._require_initialized(), seed=int(seed), sigma=float(sigma))
        self._math_es_is_perturbed = True
        return {"ok": True, "seed": int(seed), "sigma": float(sigma)}

    @torch.no_grad()
    def apply_math_es(self, seed: int, sigma: float) -> dict:
        """Compatibility entrypoint used by the ml5 vLLM ES runner."""
        return self.apply_perturbation(seed=seed, sigma=sigma)

    @torch.no_grad()
    def revert_perturbation(self, seed: int, sigma: float) -> dict:
        _apply_seeded_noise(self._require_initialized(), seed=int(seed), sigma=-float(sigma))
        self._math_es_is_perturbed = False
        return {"ok": True, "seed": int(seed), "sigma": -float(sigma)}

    @torch.no_grad()
    def revert_math_es(self, seed: int, sigma: float) -> dict:
        """Compatibility entrypoint used by the ml5 vLLM ES runner."""
        return self.revert_perturbation(seed=seed, sigma=sigma)

    @torch.no_grad()
    def dipu(self, seeds: Sequence[int], weights: Sequence[float], eta: float) -> dict:
        if bool(getattr(self, "_math_es_is_perturbed", False)):
            raise RuntimeError("Math vLLM ES update requires an unperturbed model.")
        normalized_seeds = [int(seed) for seed in seeds]
        normalized_weights = [float(weight) for weight in weights]
        _apply_dipu_update(
            self._require_initialized(),
            seeds=normalized_seeds,
            weights=normalized_weights,
            eta=float(eta),
        )
        self._math_es_update_history.append(
            {"seeds": normalized_seeds, "weights": normalized_weights, "alpha": float(eta)}
        )
        weight_tensor = torch.tensor(normalized_weights, dtype=torch.float32)
        return {
            "ok": True,
            "n": len(normalized_seeds),
            "alpha": float(eta),
            "weight_norm": float(torch.linalg.norm(weight_tensor).item()) if normalized_weights else 0.0,
        }

    @torch.no_grad()
    def reset_math_es(self) -> dict:
        parameter_infos = self._require_initialized()
        history = list(getattr(self, "_math_es_update_history", []))
        for item in reversed(history):
            _apply_dipu_update(
                parameter_infos,
                seeds=item["seeds"],
                weights=item["weights"],
                eta=-float(item["alpha"]),
            )
        self._math_es_update_history = []
        self._math_es_is_perturbed = False
        return {"ok": True, "reverted_updates": len(history)}

    def status_math_es(self) -> dict:
        parameter_infos = getattr(self, "_math_es_parameter_infos", []) or []
        total_params = sum(parameter.numel() for _, _, parameter in parameter_infos)
        return {
            "ok": True,
            "initialized": bool(parameter_infos),
            "parameter_scope": getattr(self, "_math_es_parameter_scope", None),
            "n_tensors": len(parameter_infos),
            "total_params": int(total_params),
            "update_history": len(getattr(self, "_math_es_update_history", [])),
        }
