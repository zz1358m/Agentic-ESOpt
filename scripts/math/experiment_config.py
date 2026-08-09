#!/usr/bin/env python3
"""Validate and record the approved DAPO-400 Math GRPO experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


APPROVED_PHYSICAL_GPUS = (3, 4, 5, 6)


def _physical_ids(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"physical GPU ids must be numeric, got {value!r}") from exc


def _bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"expected boolean value, got {value!r}")


def validate_math_experiment_config(
    *,
    physical_gpu_ids: str,
    train_records: int,
    val_records: int,
    aime_records: int,
    train_batch_size: int,
    ppo_mini_batch_size: int,
    rollout_n: int,
    epochs: int,
    world_size: int,
    test_freq: int,
    eval_samples: int,
    ray_num_cpus: int = 32,
    max_user_turns: int = 100,
    max_assistant_turns: int = 100,
    max_response_length: int = 8192,
    max_turn_response_length: int = 512,
    save_freq: int = 20,
    rollout_data_dir: str = "",
    validation_data_dir: str = "",
    tool_config_path: str = "",
    parser_enabled: bool = True,
    dense_qwen3next_patch_enabled: bool = True,
    model_path: str = "",
    max_prompt_length: int = 4096,
    learning_rate: float = 1e-6,
    use_kl_loss: bool = True,
    kl_loss_coef: float = 0.001,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 40,
    presence_penalty: float = 2.0,
    repetition_penalty: float = 1.0,
    data_shuffle: bool = True,
    data_seed: int = 1,
    val_before_train: bool = True,
    gpu_memory_utilization: float = 0.50,
    max_num_seqs: int = 16,
    generate_timeout_seconds: float = 600.0,
    generate_max_attempts: int = 3,
    reward_timeout_seconds: float = 120.0,
    reward_max_attempts: int = 3,
    check_torch: bool = False,
) -> dict[str, Any]:
    ids = _physical_ids(physical_gpu_ids)
    if ids != APPROVED_PHYSICAL_GPUS:
        raise ValueError(f"approved Math experiment requires physical GPUs 3,4,5,6, got {physical_gpu_ids!r}")
    for name, value in (
        ("train_records", train_records),
        ("val_records", val_records),
        ("aime_records", aime_records),
        ("train_batch_size", train_batch_size),
        ("ppo_mini_batch_size", ppo_mini_batch_size),
        ("rollout_n", rollout_n),
        ("epochs", epochs),
        ("world_size", world_size),
        ("test_freq", test_freq),
        ("eval_samples", eval_samples),
        ("ray_num_cpus", ray_num_cpus),
        ("max_user_turns", max_user_turns),
        ("max_assistant_turns", max_assistant_turns),
        ("max_response_length", max_response_length),
        ("max_turn_response_length", max_turn_response_length),
        ("save_freq", save_freq),
        ("max_prompt_length", max_prompt_length),
        ("learning_rate", learning_rate),
        ("kl_loss_coef", kl_loss_coef),
        ("temperature", temperature),
        ("top_p", top_p),
        ("top_k", top_k),
        ("repetition_penalty", repetition_penalty),
        ("data_seed", data_seed),
        ("gpu_memory_utilization", gpu_memory_utilization),
        ("max_num_seqs", max_num_seqs),
        ("generate_timeout_seconds", generate_timeout_seconds),
        ("generate_max_attempts", generate_max_attempts),
        ("reward_timeout_seconds", reward_timeout_seconds),
        ("reward_max_attempts", reward_max_attempts),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if world_size != len(ids):
        raise ValueError(f"world_size must match four selected GPUs, got {world_size}")
    if train_records % train_batch_size:
        raise ValueError("train_batch_size must divide all 400 training records")
    real_rollout_batch_size = train_batch_size * rollout_n
    if real_rollout_batch_size % world_size:
        raise ValueError("train_batch_size * rollout_n must be divisible by world_size")
    if ppo_mini_batch_size % world_size:
        raise ValueError("ppo_mini_batch_size must be divisible by world_size")

    if check_torch:
        import torch

        if torch.cuda.device_count() != world_size:
            raise RuntimeError(
                f"PyTorch sees {torch.cuda.device_count()} GPUs after isolation, expected {world_size}"
            )
        for index in range(world_size):
            torch.empty(1, device=f"cuda:{index}")
        torch.cuda.synchronize()

    steps_per_epoch = train_records // train_batch_size
    total_steps = steps_per_epoch * epochs
    validation_rounds = 1 + (total_steps + test_freq - 1) // test_freq
    checkpoint_steps = list(range(save_freq, total_steps + 1, save_freq))
    if total_steps and total_steps not in checkpoint_steps:
        checkpoint_steps.append(total_steps)
    return {
        "physical_gpu_ids": list(ids),
        "effective_cuda_visible_devices": [
            item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()
        ],
        "world_size": world_size,
        "train_records": train_records,
        "val_records": val_records,
        "aime_records": aime_records,
        "train_batch_size": train_batch_size,
        "ppo_mini_batch_size": ppo_mini_batch_size,
        "rollout_n": rollout_n,
        "real_rollout_batch_size": real_rollout_batch_size,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "training_trajectories": total_steps * real_rollout_batch_size,
        "dropped_records_per_epoch": train_records % train_batch_size,
        "test_freq": test_freq,
        "validation_rounds": validation_rounds,
        "validation_trajectories": validation_rounds * val_records,
        "eval_samples": eval_samples,
        "standalone_evaluation_trajectories": (val_records + aime_records) * eval_samples * 2,
        "checkpoint_steps": checkpoint_steps,
        "limits": {
            "max_prompt_tokens": max_prompt_length,
            "max_user_turns": max_user_turns,
            "max_assistant_turns": max_assistant_turns,
            "max_response_tokens": max_response_length,
            "max_turn_response_tokens": max_turn_response_length,
        },
        "trajectory_dirs": {
            "train": rollout_data_dir,
            "validation": validation_data_dir,
        },
        "runtime": {
            "model_path": model_path,
            "ray_num_cpus": ray_num_cpus,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_num_seqs": max_num_seqs,
            "generate_timeout_seconds": generate_timeout_seconds,
            "generate_max_attempts": generate_max_attempts,
            "reward_timeout_seconds": reward_timeout_seconds,
            "reward_max_attempts": reward_max_attempts,
            "tool_config_path": tool_config_path,
            "parser_enabled": parser_enabled,
            "dense_qwen3next_patch_enabled": dense_qwen3next_patch_enabled,
        },
        "optimizer": {
            "learning_rate": learning_rate,
            "use_kl_loss": use_kl_loss,
            "kl_loss_coef": kl_loss_coef,
        },
        "sampling": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty,
        },
        "data_order": {"shuffle": data_shuffle, "seed": data_seed},
        "validation": {"before_train": val_before_train, "frequency_steps": test_freq},
        "status": "PASS",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpus", default=os.environ.get("MATH_PHYSICAL_GPU_IDS", "3,4,5,6"))
    parser.add_argument("--train-records", type=int, default=400)
    parser.add_argument("--val-records", type=int, default=100)
    parser.add_argument("--aime-records", type=int, default=30)
    parser.add_argument("--train-batch-size", type=int, default=int(os.environ.get("TRAIN_BATCH_SIZE", "20")))
    parser.add_argument("--ppo-mini-batch-size", type=int, default=int(os.environ.get("PPO_MINI_BATCH_SIZE", "20")))
    parser.add_argument("--rollout-n", type=int, default=int(os.environ.get("ROLLOUT_N", "8")))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("TOTAL_EPOCHS", "15")))
    parser.add_argument("--world-size", type=int, default=int(os.environ.get("N_GPUS_PER_NODE", "4")))
    parser.add_argument("--test-freq", type=int, default=int(os.environ.get("TEST_FREQ", "5")))
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument("--ray-num-cpus", type=int, default=int(os.environ.get("RAY_NUM_CPUS", "32")))
    parser.add_argument("--max-user-turns", type=int, default=int(os.environ.get("MAX_USER_TURNS", "100")))
    parser.add_argument("--max-assistant-turns", type=int, default=int(os.environ.get("MAX_ASSISTANT_TURNS", "100")))
    parser.add_argument("--max-response-length", type=int, default=int(os.environ.get("MAX_RESPONSE_LENGTH", "8192")))
    parser.add_argument(
        "--max-turn-response-length", type=int, default=int(os.environ.get("MAX_TURN_RESPONSE_LENGTH", "512"))
    )
    parser.add_argument("--save-freq", type=int, default=int(os.environ.get("SAVE_FREQ", "20")))
    parser.add_argument("--rollout-data-dir", default=os.environ.get("ROLLOUT_DATA_DIR", ""))
    parser.add_argument("--validation-data-dir", default=os.environ.get("VALIDATION_DATA_DIR", ""))
    parser.add_argument("--tool-config-path", default=os.environ.get("TOOL_CONFIG_PATH", ""))
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", ""))
    parser.add_argument("--max-prompt-length", type=int, default=int(os.environ.get("MAX_PROMPT_LENGTH", "4096")))
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("LR", "1e-6")))
    parser.add_argument("--use-kl-loss", default=os.environ.get("USE_KL_LOSS", "True"))
    parser.add_argument("--kl-loss-coef", type=float, default=float(os.environ.get("KL_LOSS_COEF", "0.001")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "1.0")))
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("TOP_P", "1.0")))
    parser.add_argument("--top-k", type=int, default=int(os.environ.get("TOP_K", "40")))
    parser.add_argument("--presence-penalty", type=float, default=float(os.environ.get("PRESENCE_PENALTY", "2.0")))
    parser.add_argument("--repetition-penalty", type=float, default=float(os.environ.get("REPETITION_PENALTY", "1.0")))
    parser.add_argument("--data-shuffle", default=os.environ.get("DATA_SHUFFLE", "True"))
    parser.add_argument("--data-seed", type=int, default=int(os.environ.get("DATA_SEED", "1")))
    parser.add_argument("--val-before-train", default=os.environ.get("VAL_BEFORE_TRAIN", "True"))
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.50")),
    )
    parser.add_argument("--max-num-seqs", type=int, default=int(os.environ.get("MAX_NUM_SEQS", "16")))
    parser.add_argument(
        "--generate-timeout-seconds",
        type=float,
        default=float(os.environ.get("TRACE2SKILL_GENERATE_TIMEOUT_SECONDS", "600")),
    )
    parser.add_argument(
        "--generate-max-attempts",
        type=int,
        default=int(os.environ.get("TRACE2SKILL_GENERATE_MAX_ATTEMPTS", "3")),
    )
    parser.add_argument(
        "--reward-timeout-seconds",
        type=float,
        default=float(os.environ.get("TRACE2SKILL_REWARD_TIMEOUT_SECONDS", "120")),
    )
    parser.add_argument(
        "--reward-max-attempts",
        type=int,
        default=int(os.environ.get("TRACE2SKILL_REWARD_MAX_ATTEMPTS", "3")),
    )
    parser.add_argument("--parser-enabled", type=int, choices=(0, 1), default=int(os.environ.get("TRACE2SKILL_REGISTER_TOOL_PARSER", "1")))
    parser.add_argument(
        "--dense-qwen3next-patch-enabled",
        type=int,
        choices=(0, 1),
        default=int(os.environ.get("TRACE2SKILL_PATCH_DENSE_QWEN3NEXT", "1")),
    )
    parser.add_argument("--check-torch", action="store_true")
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_math_experiment_config(
        physical_gpu_ids=args.physical_gpus,
        train_records=args.train_records,
        val_records=args.val_records,
        aime_records=args.aime_records,
        train_batch_size=args.train_batch_size,
        ppo_mini_batch_size=args.ppo_mini_batch_size,
        rollout_n=args.rollout_n,
        epochs=args.epochs,
        world_size=args.world_size,
        test_freq=args.test_freq,
        eval_samples=args.eval_samples,
        ray_num_cpus=args.ray_num_cpus,
        max_user_turns=args.max_user_turns,
        max_assistant_turns=args.max_assistant_turns,
        max_response_length=args.max_response_length,
        max_turn_response_length=args.max_turn_response_length,
        save_freq=args.save_freq,
        rollout_data_dir=args.rollout_data_dir,
        validation_data_dir=args.validation_data_dir,
        tool_config_path=args.tool_config_path,
        parser_enabled=bool(args.parser_enabled),
        dense_qwen3next_patch_enabled=bool(args.dense_qwen3next_patch_enabled),
        model_path=args.model_path,
        max_prompt_length=args.max_prompt_length,
        learning_rate=args.learning_rate,
        use_kl_loss=_bool(args.use_kl_loss),
        kl_loss_coef=args.kl_loss_coef,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        data_shuffle=_bool(args.data_shuffle),
        data_seed=args.data_seed,
        val_before_train=_bool(args.val_before_train),
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        generate_timeout_seconds=args.generate_timeout_seconds,
        generate_max_attempts=args.generate_max_attempts,
        reward_timeout_seconds=args.reward_timeout_seconds,
        reward_max_attempts=args.reward_max_attempts,
        check_torch=args.check_torch,
    )
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
