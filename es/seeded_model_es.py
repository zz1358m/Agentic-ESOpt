import hashlib
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


ParameterInfo = Tuple[str, int, torch.nn.Parameter]


class SeedReplayModelES:
    """Seed-replay ES over live torch model parameters.

    The caller owns the model server and generation loop. This class owns the
    mutable ES state: selected parameters, applied update history, and the
    deterministic noise/update math.
    """

    def __init__(self):
        self.parameter_infos: Optional[List[ParameterInfo]] = None
        self.parameter_scope: Optional[str] = None
        self.update_history: List[Dict[str, Any]] = []

    def init(
        self,
        model,
        *,
        parameter_scope: str = "full",
        target_modules: Optional[Iterable[str]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        scope = self._normalize_parameter_scope(parameter_scope)
        if scope == "full":
            parameter_infos = self._gather_full_parameters(model)
        elif scope == "all_linear":
            parameter_infos = self._gather_linear_parameters(model, target_modules=target_modules)
        else:
            parameter_infos = self._gather_lora_parameters(model)

        self.parameter_infos = parameter_infos
        self.parameter_scope = scope
        self.update_history = []
        total_params = sum(parameter.numel() for _, _, parameter in parameter_infos)
        if verbose:
            print(f"[MODEL_ES] scope={scope}, tensors={len(parameter_infos)}, params={total_params:,}")
        return {
            "ok": True,
            "parameter_scope": scope,
            "n_tensors": len(parameter_infos),
            "total_params": int(total_params),
        }

    def apply(self, *, seed: int, sigma: float) -> Dict[str, Any]:
        self._apply_seeded_noise(self._require_initialized(), seed=seed, sigma=sigma)
        return {"ok": True, "seed": int(seed), "sigma": float(sigma)}

    def revert(self, *, seed: int, sigma: float) -> Dict[str, Any]:
        self._apply_seeded_noise(self._require_initialized(), seed=seed, sigma=-float(sigma))
        return {"ok": True, "seed": int(seed), "sigma": -float(sigma)}

    def update(
        self,
        *,
        seeds: Sequence[int],
        rewards: Sequence[float],
        alpha: float,
        reward_normalization: str = "zscore",
        reward_normalization_ddof: int = 0,
        reward_normalization_eps: float = 1e-8,
    ) -> Dict[str, Any]:
        parameter_infos = self._require_initialized()
        normalized_seeds = [int(seed) for seed in seeds]
        normalized_rewards = [float(reward) for reward in rewards]
        weights = self._normalize_rewards(
            normalized_rewards,
            reward_normalization,
            ddof=reward_normalization_ddof,
            eps=reward_normalization_eps,
        )
        self._apply_dipu_update(parameter_infos, seeds=normalized_seeds, weights=weights, eta=float(alpha))
        self.update_history.append({"seeds": normalized_seeds, "weights": weights, "alpha": float(alpha)})
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        return {
            "ok": True,
            "n": len(normalized_seeds),
            "alpha": float(alpha),
            "reward_mean": float(torch.tensor(normalized_rewards, dtype=torch.float32).mean().item())
            if normalized_rewards
            else 0.0,
            "reward_best": float(max(normalized_rewards)) if normalized_rewards else 0.0,
            "weight_norm": float(torch.linalg.norm(weight_tensor).item()) if weights else 0.0,
        }

    def reset(self) -> Dict[str, Any]:
        parameter_infos = self._require_initialized()
        n_updates = len(self.update_history)
        for item in reversed(self.update_history):
            self._apply_dipu_update(
                parameter_infos,
                seeds=item["seeds"],
                weights=item["weights"],
                eta=-float(item["alpha"]),
            )
        self.update_history = []
        return {"ok": True, "reverted_updates": n_updates}

    def status(self) -> Dict[str, Any]:
        parameter_infos = self.parameter_infos or []
        total_params = sum(parameter.numel() for _, _, parameter in parameter_infos)
        return {
            "ok": True,
            "initialized": self.parameter_infos is not None,
            "parameter_scope": self.parameter_scope,
            "n_tensors": len(parameter_infos),
            "total_params": int(total_params),
            "update_history": len(self.update_history),
        }

    @staticmethod
    def _stable_tensor_id(name: str) -> int:
        digest = hashlib.blake2b(str(name).encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little", signed=False)

    @staticmethod
    def _mix_seed(base_seed: int, tensor_id: int) -> int:
        return (int(base_seed) ^ int(tensor_id)) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def _normalize_parameter_scope(parameter_scope: str) -> str:
        scope = str(parameter_scope or "full").strip().lower()
        if scope not in {"full", "all_linear", "lora"}:
            raise ValueError("parameter_scope must be 'full', 'all_linear', or 'lora'.")
        return scope

    @staticmethod
    def _matches_target_module(module_suffix: str, target_modules: Optional[Iterable[str]]) -> bool:
        if target_modules is None:
            return True
        return module_suffix in {str(module_name) for module_name in target_modules}

    def _gather_full_parameters(self, model) -> List[ParameterInfo]:
        seen_ids = set()
        parameter_infos = []
        for name, parameter in model.named_parameters():
            if id(parameter) in seen_ids:
                continue
            if not torch.is_floating_point(parameter):
                continue
            seen_ids.add(id(parameter))
            parameter_infos.append((name, self._stable_tensor_id(name), parameter))
        return parameter_infos

    def _gather_linear_parameters(self, model, target_modules=None) -> List[ParameterInfo]:
        skip_keywords = {"embed_tokens", "lm_head", "norm"}
        seen_ids = set()
        parameter_infos = []
        for module_name, module in model.named_modules():
            if any(keyword in module_name for keyword in skip_keywords):
                continue
            weight = getattr(module, "weight", None)
            if weight is None or weight.ndim != 2 or id(weight) in seen_ids:
                continue
            if not torch.is_floating_point(weight):
                continue
            module_suffix = module_name.split(".")[-1]
            if not self._matches_target_module(module_suffix, target_modules):
                continue
            name = f"{module_name}.weight"
            seen_ids.add(id(weight))
            parameter_infos.append((name, self._stable_tensor_id(name), weight))
        return parameter_infos

    def _gather_lora_parameters(self, model) -> List[ParameterInfo]:
        parameter_infos = []
        for name, parameter in model.named_parameters():
            if "lora_" not in name:
                continue
            if not torch.is_floating_point(parameter):
                continue
            parameter_infos.append((name, self._stable_tensor_id(name), parameter))
        if not parameter_infos:
            raise RuntimeError("No LoRA parameters found. Start the policy server with --enable-lora.")
        return parameter_infos

    @staticmethod
    def _iter_flat_chunks(numel: int, chunk_size: int = 8_388_608):
        for start in range(0, int(numel), int(chunk_size)):
            end = min(start + int(chunk_size), int(numel))
            yield start, end

    @torch.no_grad()
    def _apply_seeded_noise(self, parameter_infos: List[ParameterInfo], *, seed: int, sigma: float) -> None:
        for _, tensor_id, parameter in parameter_infos:
            parameter_flat = parameter.view(-1)
            generator = torch.Generator(device=parameter.device)
            generator.manual_seed(self._mix_seed(seed, tensor_id))
            for start, end in self._iter_flat_chunks(parameter_flat.numel()):
                noise = torch.randn(
                    (end - start,),
                    generator=generator,
                    dtype=torch.float32,
                    device=parameter.device,
                )
                parameter_flat[start:end].add_(noise.to(dtype=parameter.dtype), alpha=float(sigma))

    @torch.no_grad()
    def _apply_dipu_update(
        self,
        parameter_infos: List[ParameterInfo],
        *,
        seeds: Sequence[int],
        weights: Sequence[float],
        eta: float,
    ) -> None:
        n = len(seeds)
        if n == 0:
            raise ValueError("ES update requires at least one seed.")
        if len(seeds) != len(weights):
            raise ValueError("ES update requires seeds and weights to have the same length.")

        scale = float(eta) / float(n)
        for _, tensor_id, parameter in parameter_infos:
            parameter_flat = parameter.view(-1)
            generators = []
            coeffs = []
            for seed, weight in zip(seeds, weights):
                generator = torch.Generator(device=parameter.device)
                generator.manual_seed(self._mix_seed(seed, tensor_id))
                generators.append(generator)
                coeffs.append(scale * float(weight))

            for start, end in self._iter_flat_chunks(parameter_flat.numel()):
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

    @staticmethod
    def _normalize_rewards(rewards, mode, ddof=0, eps=1e-8) -> List[float]:
        tensor = torch.tensor(rewards, dtype=torch.float32)
        normalized_mode = str(mode or "none").strip().lower()
        if normalized_mode in {"none", "identity", "off"}:
            return tensor.tolist()
        if normalized_mode == "zscore":
            if tensor.numel() <= int(ddof):
                return torch.zeros_like(tensor).tolist()
            std = torch.std(tensor, unbiased=bool(ddof))
            return ((tensor - torch.mean(tensor)) / (std + float(eps))).tolist()
        if normalized_mode == "centered_rank":
            order = torch.argsort(torch.argsort(tensor))
            if tensor.numel() == 1:
                return [0.0]
            return (order.float() / (tensor.numel() - 1) - 0.5).tolist()
        raise ValueError(f"Unsupported reward_normalization: {mode}")

    def _require_initialized(self) -> List[ParameterInfo]:
        if self.parameter_infos is None:
            raise RuntimeError("Model ES is not initialized. Call /es/init first.")
        return self.parameter_infos
