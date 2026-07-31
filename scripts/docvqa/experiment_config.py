#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DOCVQA_WORLD_SIZE = 4


def _parse_devices(value: str) -> list[int]:
    try:
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"CUDA_VISIBLE_DEVICES must contain numeric ids, got {value!r}") from exc


def validate_experiment_config(
    *,
    visible_devices: str,
    train_records: int,
    train_batch_size: int,
    rollout_n: int,
    epochs: int,
    world_size: int,
    ppo_mini_batch_size: int | None = None,
    check_torch: bool = False,
    effective_visible_devices: str | None = None,
) -> dict[str, Any]:
    physical_gpu_ids = _parse_devices(visible_devices)
    if len(physical_gpu_ids) != DOCVQA_WORLD_SIZE or len(set(physical_gpu_ids)) != DOCVQA_WORLD_SIZE:
        raise ValueError(f"DocVQA requires exactly four unique physical GPUs, got {visible_devices!r}")
    for name, value in (
        ("train_records", train_records),
        ("train_batch_size", train_batch_size),
        ("rollout_n", rollout_n),
        ("epochs", epochs),
        ("world_size", world_size),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if ppo_mini_batch_size is not None and ppo_mini_batch_size <= 0:
        raise ValueError("ppo_mini_batch_size must be positive")
    if world_size != len(physical_gpu_ids):
        raise ValueError(
            f"world_size must match the {len(physical_gpu_ids)} selected GPUs, got {world_size}"
        )
    real_batch_size = train_batch_size * rollout_n
    if real_batch_size % world_size:
        raise ValueError(
            f"train_batch_size * rollout_n ({real_batch_size}) must be divisible by world_size ({world_size})"
        )
    steps_per_epoch = train_records // train_batch_size
    if steps_per_epoch <= 0:
        raise ValueError("train_records must contain at least one full train batch")
    cuda_device_names: list[str] | None = None
    effective_devices = [
        value.strip()
        for value in (effective_visible_devices or visible_devices).split(",")
        if value.strip()
    ]
    if len(effective_devices) != world_size or len(set(effective_devices)) != world_size:
        raise ValueError(
            f"effective CUDA_VISIBLE_DEVICES must contain {world_size} unique devices, "
            f"got {effective_visible_devices!r}"
        )
    if check_torch:
        import torch

        device_count = torch.cuda.device_count()
        if device_count != world_size:
            raise RuntimeError(
                f"PyTorch sees {device_count} CUDA devices, expected {world_size}"
            )
        try:
            cuda_device_names = [torch.cuda.get_device_name(index) for index in range(device_count)]
            for index in range(device_count):
                torch.empty(1, device=f"cuda:{index}")
            torch.cuda.synchronize()
        except Exception as exc:
            raise RuntimeError(
                f"PyTorch enumerated {device_count} devices but could not initialize all of them: {exc}"
            ) from exc
    total_steps = steps_per_epoch * epochs
    return {
        "physical_gpu_ids": physical_gpu_ids,
        "effective_cuda_visible_devices": effective_devices,
        "visible_world_size": world_size,
        "torch_cuda_device_names": cuda_device_names,
        "train_records": train_records,
        "train_batch_size": train_batch_size,
        "ppo_mini_batch_size": ppo_mini_batch_size,
        "rollout_n": rollout_n,
        "real_rollout_batch_size": real_batch_size,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "training_trajectories": total_steps * real_batch_size,
        "dropped_records_per_epoch": train_records % train_batch_size,
        "status": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an approved DocVQA GRPO GPU experiment config.")
    parser.add_argument(
        "--visible-devices",
        default=os.environ.get("DOCVQA_PHYSICAL_GPU_IDS", ""),
        help="Resolved four-GPU physical index list from gpu_visibility.py.",
    )
    parser.add_argument("--effective-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    parser.add_argument("--train-records", type=int, default=int(os.environ.get("DOCVQA_TRAIN_LIMIT", "50")))
    parser.add_argument("--train-batch-size", type=int, default=int(os.environ.get("TRAIN_BATCH_SIZE", "4")))
    parser.add_argument("--ppo-mini-batch-size", type=int, default=int(os.environ.get("PPO_MINI_BATCH_SIZE", "4")))
    parser.add_argument("--rollout-n", type=int, default=int(os.environ.get("ROLLOUT_N", "8")))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("TOTAL_EPOCHS", "15")))
    parser.add_argument("--world-size", type=int, default=int(os.environ.get("N_GPUS_PER_NODE", "4")))
    parser.add_argument("--check-torch", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = validate_experiment_config(
        visible_devices=args.visible_devices,
        train_records=args.train_records,
        train_batch_size=args.train_batch_size,
        ppo_mini_batch_size=args.ppo_mini_batch_size,
        rollout_n=args.rollout_n,
        epochs=args.epochs,
        world_size=args.world_size,
        check_torch=args.check_torch,
        effective_visible_devices=args.effective_visible_devices,
    )
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
