#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import LogitsProcessor

ROOT = Path(os.environ.get("ROOT", Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(ROOT / "sudoku-train-time"))

from envs.sudoku import (  # noqa: E402
    SudokuTask,
    apply_action,
    build_action_prompt,
    clone_board,
    empty_count,
    is_full,
    load_tasks,
    score_board,
)


DEFAULT_TRAIN = ROOT / "data/sudoku/train.jsonl"
DEFAULT_EVAL = ROOT / "data/sudoku/eval.jsonl"


class PresencePenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, penalty: float) -> None:
        self.penalty = float(penalty)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.penalty == 0.0:
            return scores
        adjusted = scores.clone()
        for row_idx in range(input_ids.shape[0]):
            seen = torch.unique(input_ids[row_idx])
            adjusted[row_idx, seen] -= self.penalty
        return adjusted


def set_use_cache(model: Any, enabled: bool) -> tuple[Any, bool | None, bool | None]:
    config = getattr(model, "config", None)
    generation_config = getattr(model, "generation_config", None)
    old_config = getattr(config, "use_cache", None) if config is not None else None
    old_generation = getattr(generation_config, "use_cache", None) if generation_config is not None else None
    if config is not None and hasattr(config, "use_cache"):
        config.use_cache = enabled
    if generation_config is not None and hasattr(generation_config, "use_cache"):
        generation_config.use_cache = enabled
    return model, old_config, old_generation


def restore_use_cache(state: tuple[Any, bool | None, bool | None]) -> None:
    model, old_config, old_generation = state
    config = getattr(model, "config", None)
    generation_config = getattr(model, "generation_config", None)
    if config is not None and old_config is not None and hasattr(config, "use_cache"):
        config.use_cache = old_config
    if generation_config is not None and old_generation is not None and hasattr(generation_config, "use_cache"):
        generation_config.use_cache = old_generation


def task_rows(tasks: list[SudokuTask], *, batch_size: int, step: int) -> list[int]:
    if len(tasks) <= batch_size:
        return list(range(len(tasks)))
    start = (step * batch_size) % len(tasks)
    return [(start + offset) % len(tasks) for offset in range(batch_size)]


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    size = max(1, size)
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def format_prompt(tokenizer: Any, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


@torch.no_grad()
def generate_actions(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    presence_penalty: float,
    repetition_penalty: float,
) -> list[dict[str, Any]]:
    if not prompts:
        return []
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False).to(device)
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else None,
        "top_p": top_p,
        "top_k": top_k if top_k > 0 else None,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": repetition_penalty,
        "return_dict_in_generate": True,
        "output_logits": True,
        "use_cache": True,
    }
    if min_p >= 0:
        kwargs["min_p"] = min_p
    if presence_penalty != 0.0:
        kwargs["logits_processor"] = [PresencePenaltyLogitsProcessor(presence_penalty)]
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    cache_state = set_use_cache(model, True)
    try:
        out = model.generate(**encoded, **kwargs)
    finally:
        restore_use_cache(cache_state)
    prompt_width = encoded["input_ids"].shape[1]
    raw_logits = getattr(out, "logits", None)
    if raw_logits is None:
        raise RuntimeError(
            "transformers.generate did not return raw logits; cannot cache policy logprobs without an extra forward."
        )

    results = []
    for row_idx in range(out.sequences.shape[0]):
        prompt_ids = encoded["input_ids"][row_idx][encoded["attention_mask"][row_idx].bool()].detach().cpu().tolist()
        generated_ids = out.sequences[row_idx, prompt_width:].detach().cpu().tolist()
        generated_log_probs = []
        trimmed_ids = []
        for token_idx, token_id in enumerate(generated_ids):
            if tokenizer.pad_token_id is not None and int(token_id) == int(tokenizer.pad_token_id):
                break
            if token_idx >= len(raw_logits):
                break
            step_logits = raw_logits[token_idx][row_idx].float()
            token_log_prob = F.log_softmax(step_logits, dim=-1)[int(token_id)].detach().cpu().item()
            trimmed_ids.append(int(token_id))
            generated_log_probs.append(float(token_log_prob))
            if tokenizer.eos_token_id is not None and int(token_id) == int(tokenizer.eos_token_id):
                break
        results.append(
            {
                "text": tokenizer.decode(trimmed_ids, skip_special_tokens=True).strip(),
                "prompt_token_ids": [int(item) for item in prompt_ids],
                "response_token_ids": trimmed_ids,
                "old_response_log_probs": generated_log_probs,
            }
        )
    return results


def rollout_tasks(
    *,
    model: Any,
    tokenizer: Any,
    tasks: list[SudokuTask],
    pair_items: list[tuple[int, int, int]],
    device: torch.device,
    max_turns: int,
    rollout_micro_batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    presence_penalty: float,
    repetition_penalty: float,
    collect_examples: bool,
) -> list[dict[str, Any]]:
    states = []
    for pair_index, task_index, group_index in pair_items:
        task = tasks[task_index]
        states.append(
            {
                "pair_index": pair_index,
                "task_index": task_index,
                "group_index": group_index,
                "task": task,
                "board": clone_board(task.puzzle),
                "feedback": "",
                "turn_index": 0,
                "examples": [],
                "turns": [],
            }
        )

    active = list(states)
    while active:
        next_active = []
        for batch in chunked(active, rollout_micro_batch_size):
            formatted_prompts = [
                format_prompt(
                    tokenizer,
                    build_action_prompt(
                        state["task"],
                        state["board"],
                        turn_index=int(state["turn_index"]),
                        feedback=str(state["feedback"]),
                    ),
                )
                for state in batch
            ]
            actions = generate_actions(
                model=model,
                tokenizer=tokenizer,
                prompts=formatted_prompts,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
            )
            for state, prompt, action in zip(batch, formatted_prompts, actions):
                response = str(action["text"])
                task = state["task"]
                board, info = apply_action(task.puzzle, state["board"], response)
                state["board"] = board
                row = {
                    "turn": int(state["turn_index"]),
                    "response": response,
                    "remaining_empty": empty_count(board),
                    **info,
                }
                state["turns"].append(row)
                if collect_examples and response:
                    state["examples"].append(
                        {
                            "prompt": prompt,
                            "response": response,
                            "prompt_token_ids": action["prompt_token_ids"],
                            "response_token_ids": action["response_token_ids"],
                            "old_response_log_probs": action["old_response_log_probs"],
                        }
                    )
                state["feedback"] = str(row.get("message", ""))
                state["turn_index"] = int(state["turn_index"]) + 1
                if not is_full(state["board"]) and int(state["turn_index"]) < max_turns:
                    next_active.append(state)
        active = next_active

    rows = []
    for state in states:
        task = state["task"]
        rows.append(
            {
                "pair_index": state["pair_index"],
                "task_index": state["task_index"],
                "task_id": task.id,
                "group_index": state["group_index"],
                "mask_count": task.mask_count,
                "reward": float(score_board(task.puzzle, state["board"])),
                "done": is_full(state["board"]),
                "remaining_empty": empty_count(state["board"]),
                "turn_count": int(state["turn_index"]),
                "prediction": state["board"],
                "turns": state["turns"],
                "examples": state["examples"],
            }
        )
    return rows


def pad_batch(tokenizer: Any, encoded_rows: list[dict[str, list[int]]], device: torch.device) -> dict[str, torch.Tensor]:
    max_len = max(len(row["input_ids"]) for row in encoded_rows)
    input_ids = []
    attention_mask = []
    response_mask = []
    old_log_probs = []
    ref_log_probs = []
    pad_id = tokenizer.pad_token_id
    for row in encoded_rows:
        pad = max_len - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_id] * pad)
        attention_mask.append([1] * len(row["input_ids"]) + [0] * pad)
        response_mask.append(row["response_mask"] + [0] * pad)
        old_log_probs.append(row.get("old_log_probs", [0.0] * len(row["input_ids"])) + [0.0] * pad)
        ref_log_probs.append(row.get("ref_log_probs", [0.0] * len(row["input_ids"])) + [0.0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        "response_mask": torch.tensor(response_mask, dtype=torch.float32, device=device),
        "old_log_probs": torch.tensor(old_log_probs, dtype=torch.float32, device=device),
        "ref_log_probs": torch.tensor(ref_log_probs, dtype=torch.float32, device=device),
    }


def encode_policy_examples(
    tokenizer: Any,
    examples: list[dict[str, Any]],
    *,
    max_length: int,
) -> list[dict[str, Any]]:
    encoded = []
    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    for example in examples:
        if "prompt_token_ids" in example and "response_token_ids" in example:
            prompt_ids = [int(item) for item in example["prompt_token_ids"]]
            response_ids = [int(item) for item in example["response_token_ids"]]
            old_response_log_probs = [float(item) for item in example.get("old_response_log_probs", [])]
            if len(old_response_log_probs) != len(response_ids):
                raise ValueError(
                    f"Cached old logprobs length mismatch: {len(old_response_log_probs)} logprobs for {len(response_ids)} tokens"
                )
            old_log_probs = [0.0] * len(prompt_ids) + old_response_log_probs
        else:
            prompt_ids = tokenizer(example["prompt"], add_special_tokens=False).input_ids
            response_ids = tokenizer(example["response"], add_special_tokens=False).input_ids + eos
            old_log_probs = None
        if not response_ids:
            continue
        input_ids = prompt_ids + response_ids
        response_mask = [0] * len(prompt_ids) + [1] * len(response_ids)
        if len(input_ids) > max_length:
            overflow = len(input_ids) - max_length
            input_ids = input_ids[overflow:]
            response_mask = response_mask[overflow:]
            if old_log_probs is not None:
                old_log_probs = old_log_probs[overflow:]
        if sum(response_mask) == 0:
            continue
        row = {
            "input_ids": input_ids,
            "response_mask": response_mask,
            "advantage": float(example["advantage"]),
        }
        if old_log_probs is not None:
            row["old_log_probs"] = old_log_probs
        encoded.append(row)
    return encoded


def full_token_log_probs(
    model: Any,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits[:, :-1, :].float()
    labels = batch["input_ids"][:, 1:]
    log_probs = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return F.pad(log_probs, (1, 0), value=0.0)


def finite_log_probs(log_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    active = mask > 0.0
    if not torch.isfinite(log_probs[active]).all():
        raise RuntimeError("Non-finite policy logprob found on an active response token.")
    return torch.where(active, log_probs, torch.zeros_like(log_probs))


@torch.no_grad()
def attach_ref_log_probs(
    *,
    ref_model: Any,
    tokenizer: Any,
    encoded: list[dict[str, Any]],
    device: torch.device,
    micro_batch_size: int,
) -> None:
    if not encoded:
        return
    for micro in chunked(encoded, micro_batch_size):
        missing_old = sum(1 for row in micro if "old_log_probs" not in row)
        if missing_old:
            raise RuntimeError(f"{missing_old} policy examples are missing cached old logprobs from rollout.")
        batch = pad_batch(tokenizer, micro, device)
        ref_log_probs = full_token_log_probs(ref_model, batch).detach().cpu().tolist()
        for row, ref_row in zip(micro, ref_log_probs):
            length = len(row["input_ids"])
            row["ref_log_probs"] = ref_row[:length]


def grpo_loss(
    model: Any,
    batch: dict[str, torch.Tensor],
    advantages: torch.Tensor,
    *,
    clip_epsilon: float,
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    new_log_probs = full_token_log_probs(model, batch)
    mask = batch["response_mask"] * batch["attention_mask"].float()
    mask[:, 0] = 0.0
    token_count = mask.sum(dim=1).clamp_min(1.0)

    new_log_probs = finite_log_probs(new_log_probs, mask)
    old_log_probs = finite_log_probs(batch["old_log_probs"], mask)
    ref_log_probs = finite_log_probs(batch["ref_log_probs"], mask)
    log_ratio = torch.nan_to_num(new_log_probs - old_log_probs, nan=0.0, neginf=-20.0, posinf=20.0).clamp(min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)

    advantages = advantages.view(-1, 1)
    pg_loss = torch.maximum(-advantages * ratio, -advantages * clipped_ratio)

    ref_delta = torch.nan_to_num(ref_log_probs - new_log_probs, nan=0.0, neginf=-20.0, posinf=20.0).clamp(min=-20.0, max=20.0)
    per_token_kl = torch.nan_to_num(torch.exp(ref_delta) - ref_delta - 1.0, nan=0.0, neginf=0.0, posinf=math.exp(20.0))
    per_token_loss = pg_loss + beta * per_token_kl

    per_seq_loss = (per_token_loss * mask).sum(dim=1) / token_count
    loss = per_seq_loss.mean()
    with torch.no_grad():
        clip_fraction = (((ratio - 1.0).abs() > clip_epsilon).float() * mask).sum() / mask.sum().clamp_min(1.0)
        approx_kl = (per_token_kl * mask).sum() / mask.sum().clamp_min(1.0)
    return loss, {"clip_fraction": float(clip_fraction.item()), "kl": float(approx_kl.item())}


def gathered_rows(accelerator: Any, local_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from accelerate.utils import gather_object

    gathered = gather_object(local_rows)
    rows = []
    for item in gathered:
        if isinstance(item, list):
            rows.extend(item)
        else:
            rows.append(item)
    rows.sort(key=lambda row: (int(row["pair_index"]), int(row.get("turn_count", 0))))
    return rows


def advantages_for_rows(rows: list[dict[str, Any]]) -> tuple[dict[int, float], dict[str, float]]:
    by_group: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_group[int(row["task_index"])].append(float(row["reward"]))

    result = {}
    stds = []
    zero_std_groups = 0
    all_zero_groups = 0
    all_one_groups = 0
    mixed_groups = 0
    for row in rows:
        group = by_group[int(row["task_index"])]
        mean = sum(group) / len(group)
        var = sum((reward - mean) ** 2 for reward in group) / len(group)
        std = math.sqrt(var)
        result[int(row["pair_index"])] = 0.0 if std < 1e-6 else (float(row["reward"]) - mean) / (std + 1e-6)

    for group in by_group.values():
        mean = sum(group) / len(group)
        var = sum((reward - mean) ** 2 for reward in group) / len(group)
        std = math.sqrt(var)
        stds.append(std)
        if std < 1e-6:
            zero_std_groups += 1
        if all(reward == 0.0 for reward in group):
            all_zero_groups += 1
        elif all(reward == 1.0 for reward in group):
            all_one_groups += 1
        else:
            mixed_groups += 1
    diagnostics = {
        "group_count": float(len(by_group)),
        "zero_std_groups": float(zero_std_groups),
        "all_zero_groups": float(all_zero_groups),
        "all_one_groups": float(all_one_groups),
        "mixed_groups": float(mixed_groups),
        "mean_group_std": sum(stds) / max(1, len(stds)),
    }
    return result, diagnostics


def load_causal_model(model_name_or_path: str, *, accelerator: Any, gradient_checkpointing: bool) -> Any:
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
    except Exception as exc:
        if accelerator.is_main_process:
            print(f"[setup] flash_attention_2 unavailable, falling back to default attention: {exc}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = model.generation_config.pad_token_id
    model.config.use_cache = False
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model


def summarize_eval_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    averages = [float(run["average"]) for run in runs]
    solved = [float(run["solved"]) for run in runs]
    return {
        "repeat_count": len(runs),
        "average": sum(averages) / max(1, len(averages)),
        "average_std": statistics.pstdev(averages) if len(averages) > 1 else 0.0,
        "solved_average": sum(solved) / max(1, len(solved)),
        "solved_std": statistics.pstdev(solved) if len(solved) > 1 else 0.0,
        "count": runs[0]["count"] if runs else 0,
        "runs": runs,
    }


def evaluate_repeated(
    *,
    accelerator: Any,
    model: Any,
    tokenizer: Any,
    eval_tasks: list[SudokuTask],
    args: argparse.Namespace,
    world: int,
    rank: int,
    step: int,
) -> dict[str, Any]:
    runs = []
    repeats = max(1, int(args.eval_repeats))
    for repeat_idx in range(repeats):
        torch.manual_seed(args.seed + 100000 + step * 1009 + repeat_idx * 9176 + rank)
        eval_pair_items = [(idx, idx, 0) for idx in range(len(eval_tasks)) if idx % world == rank]
        model.eval()
        local_eval = rollout_tasks(
            model=accelerator.unwrap_model(model),
            tokenizer=tokenizer,
            tasks=eval_tasks,
            pair_items=eval_pair_items,
            device=accelerator.device,
            max_turns=args.max_turns,
            rollout_micro_batch_size=args.rollout_micro_batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.eval_temperature,
            top_p=args.eval_top_p,
            top_k=args.eval_top_k,
            min_p=args.min_p,
            presence_penalty=args.presence_penalty,
            repetition_penalty=args.repetition_penalty,
            collect_examples=False,
        )
        all_eval = gathered_rows(accelerator, local_eval)
        eval_rewards = [float(row["reward"]) for row in all_eval]
        runs.append(
            {
                "repeat": repeat_idx,
                "count": len(all_eval),
                "solved": sum(1 for reward in eval_rewards if reward >= 1.0),
                "average": sum(eval_rewards) / max(1, len(eval_rewards)),
                "scores": all_eval,
            }
        )
    return summarize_eval_runs(runs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("SUDOKU_GRPO_MODEL", "/data0/zhi/meta-llama/Qwen3.5-4B"))
    parser.add_argument("--train-data", default=str(DEFAULT_TRAIN))
    parser.add_argument("--eval-data", default=str(DEFAULT_EVAL))
    parser.add_argument("--mask-count", type=int, default=int(os.environ.get("SUDOKU_TARGET_MASK_COUNT", "10")))
    parser.add_argument("--output-dir", default=os.environ.get("SUDOKU_GRPO_OUTPUT_DIR", str(ROOT / "runs/sudoku_grpo_multiturn/qwen35_4b_mask10")))
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("SUDOKU_GRPO_MAX_STEPS", "100")))
    parser.add_argument("--global-batch-size", type=int, default=int(os.environ.get("SUDOKU_GRPO_GLOBAL_BATCH", "32")))
    parser.add_argument("--num-generations", type=int, default=int(os.environ.get("SUDOKU_GRPO_NUM_GENERATIONS", "8")))
    parser.add_argument("--rollout-micro-batch-size", type=int, default=int(os.environ.get("SUDOKU_GRPO_ROLLOUT_MICRO_BATCH", "8")))
    parser.add_argument("--train-micro-batch-size", type=int, default=int(os.environ.get("SUDOKU_GRPO_TRAIN_MICRO_BATCH", "2")))
    parser.add_argument(
        "--max-policy-examples-per-step",
        type=int,
        default=int(os.environ.get("SUDOKU_GRPO_MAX_POLICY_EXAMPLES", "0")),
        help="Global policy mini-batch size; 0 removes the sample cap (only the distributed-alignment remainder may be dropped).",
    )
    parser.add_argument("--max-turns", type=int, default=int(os.environ.get("SUDOKU_GRPO_MAX_TURNS", "90")))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("SUDOKU_GRPO_MAX_NEW_TOKENS", "64")))
    parser.add_argument("--max-train-length", type=int, default=int(os.environ.get("SUDOKU_GRPO_MAX_TRAIN_LENGTH", "1400")))
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("SUDOKU_GRPO_LR", "1e-6")))
    parser.add_argument("--beta", type=float, default=float(os.environ.get("SUDOKU_GRPO_BETA", "0.001")))
    parser.add_argument("--clip-epsilon", type=float, default=float(os.environ.get("SUDOKU_GRPO_CLIP_EPSILON", "0.2")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("SUDOKU_TEMPERATURE", "0.7")))
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("SUDOKU_TOP_P", "0.8")))
    parser.add_argument("--eval-temperature", type=float, default=float(os.environ["SUDOKU_EVAL_TEMPERATURE"]) if "SUDOKU_EVAL_TEMPERATURE" in os.environ else None)
    parser.add_argument("--eval-top-p", type=float, default=float(os.environ["SUDOKU_EVAL_TOP_P"]) if "SUDOKU_EVAL_TOP_P" in os.environ else None)
    parser.add_argument("--top-k", type=int, default=int(os.environ.get("SUDOKU_TOP_K", "20")))
    parser.add_argument("--eval-top-k", type=int, default=int(os.environ["SUDOKU_EVAL_TOP_K"]) if "SUDOKU_EVAL_TOP_K" in os.environ else None)
    parser.add_argument("--min-p", type=float, default=float(os.environ.get("SUDOKU_MIN_P", "0.0")))
    parser.add_argument("--presence-penalty", type=float, default=float(os.environ.get("SUDOKU_PRESENCE_PENALTY", "1.5")))
    parser.add_argument("--repetition-penalty", type=float, default=float(os.environ.get("SUDOKU_REPETITION_PENALTY", "1.0")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SUDOKU_GRPO_SEED", "20260702")))
    parser.add_argument("--eval-before", action=argparse.BooleanOptionalAction, default=os.environ.get("SUDOKU_GRPO_EVAL_BEFORE", "0") == "1")
    parser.add_argument("--eval-after", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-interval", type=int, default=int(os.environ.get("SUDOKU_GRPO_EVAL_INTERVAL", "0")))
    parser.add_argument("--eval-repeats", type=int, default=int(os.environ.get("SUDOKU_GRPO_EVAL_REPEATS", "3")))
    parser.add_argument("--eval-limit", type=int, default=int(os.environ.get("SUDOKU_GRPO_EVAL_LIMIT", "0")))
    parser.add_argument("--log-interval", type=int, default=int(os.environ.get("SUDOKU_GRPO_LOG_INTERVAL", "1")))
    args = parser.parse_args()
    if args.eval_temperature is None:
        args.eval_temperature = args.temperature
    if args.eval_top_p is None:
        args.eval_top_p = args.top_p
    if args.eval_top_k is None:
        args.eval_top_k = args.top_k

    from accelerate import Accelerator
    from transformers import AutoTokenizer

    accelerator = Accelerator()
    rank = accelerator.process_index
    world = accelerator.num_processes
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")
    accelerator.wait_for_everyone()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_causal_model(args.model, accelerator=accelerator, gradient_checkpointing=True)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    ref_model = load_causal_model(args.model, accelerator=accelerator, gradient_checkpointing=False)
    if getattr(ref_model, "generation_config", None) is not None:
        ref_model.generation_config.pad_token_id = tokenizer.pad_token_id
    ref_model.eval()
    ref_model.requires_grad_(False)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    model, optimizer = accelerator.prepare(model, optimizer)
    ref_model = ref_model.to(accelerator.device)

    train_tasks = load_tasks(args.train_data, mask_count=args.mask_count)
    eval_tasks = load_tasks(args.eval_data, limit=args.eval_limit, mask_count=args.mask_count)
    if accelerator.is_main_process:
        print(
            f"[setup] model={args.model} mask={args.mask_count} train={len(train_tasks)} eval={len(eval_tasks)} "
            f"world={world} global_batch={args.global_batch_size} generations={args.num_generations} "
            f"train_temp={args.temperature} train_top_p={args.top_p} train_top_k={args.top_k} "
            f"eval_temp={args.eval_temperature} eval_top_p={args.eval_top_p} eval_top_k={args.eval_top_k} "
            f"beta={args.beta} clip={args.clip_epsilon} policy_logprobs=raw_logits",
            flush=True,
        )

    history = []
    if args.eval_before:
        accelerator.wait_for_everyone()
        initial_eval = evaluate_repeated(
            accelerator=accelerator,
            model=model,
            tokenizer=tokenizer,
            eval_tasks=eval_tasks,
            args=args,
            world=world,
            rank=rank,
            step=0,
        )
        if accelerator.is_main_process:
            history.append({"step": 0, "eval": initial_eval})
            (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
            print(
                f"[eval] step=0 repeats={initial_eval['repeat_count']} "
                f"solved_avg={initial_eval['solved_average']:.2f}/{initial_eval['count']} "
                f"average={initial_eval['average']:.6f} std={initial_eval['average_std']:.6f}",
                flush=True,
            )
        accelerator.wait_for_everyone()

    for step in range(args.max_steps):
        step_start = time.time()
        batch_indices = task_rows(train_tasks, batch_size=args.global_batch_size, step=step)
        pair_items = []
        pair_index = 0
        for task_index in batch_indices:
            for group_index in range(args.num_generations):
                if pair_index % world == rank:
                    pair_items.append((pair_index, task_index, group_index))
                pair_index += 1

        model.eval()
        local_rows = rollout_tasks(
            model=accelerator.unwrap_model(model),
            tokenizer=tokenizer,
            tasks=train_tasks,
            pair_items=pair_items,
            device=accelerator.device,
            max_turns=args.max_turns,
            rollout_micro_batch_size=args.rollout_micro_batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            presence_penalty=args.presence_penalty,
            repetition_penalty=args.repetition_penalty,
            collect_examples=True,
        )
        all_rows = gathered_rows(accelerator, local_rows)
        adv_by_pair, reward_diagnostics = advantages_for_rows(all_rows)

        raw_example_count = sum(len(row["examples"]) for row in all_rows)
        examples = []
        for row in all_rows:
            advantage = adv_by_pair.get(int(row["pair_index"]), 0.0)
            for example in row["examples"]:
                examples.append({**example, "advantage": advantage})
        rng = random.Random(args.seed + step * 1009)
        rng.shuffle(examples)

        policy_batch_size = int(args.max_policy_examples_per_step)
        if policy_batch_size <= 0:
            policy_batch_size = len(examples)
        unit = max(1, world * int(args.train_micro_batch_size))
        policy_batch_size = max(unit, (policy_batch_size // unit) * unit)
        usable_count = (len(examples) // policy_batch_size) * policy_batch_size if policy_batch_size > 0 else len(examples)
        examples = examples[:usable_count]
        local_policy_batch_size = max(1, policy_batch_size // world)
        local_examples = [example for idx, example in enumerate(examples) if idx % world == rank]
        encoded = encode_policy_examples(tokenizer, local_examples, max_length=args.max_train_length)

        model.eval()
        attach_ref_log_probs(
            ref_model=ref_model,
            tokenizer=tokenizer,
            encoded=encoded,
            device=accelerator.device,
            micro_batch_size=args.train_micro_batch_size,
        )

        model.train()
        loss_stats = {"kl": 0.0, "clip_fraction": 0.0}
        loss_chunks = 0
        policy_minibatches = usable_count // policy_batch_size if policy_batch_size > 0 else 0
        for policy_batch_idx in range(policy_minibatches):
            start = policy_batch_idx * local_policy_batch_size
            end = start + local_policy_batch_size
            policy_rows = encoded[start:end]
            if not policy_rows:
                continue
            optimizer.zero_grad(set_to_none=True)
            denom = max(1, len(policy_rows))
            for micro in chunked(policy_rows, args.train_micro_batch_size):
                batch = pad_batch(tokenizer, micro, accelerator.device)
                advantages = torch.tensor([row["advantage"] for row in micro], dtype=torch.float32, device=accelerator.device)
                loss, stats = grpo_loss(
                    model,
                    batch,
                    advantages,
                    clip_epsilon=args.clip_epsilon,
                    beta=args.beta,
                )
                loss = loss * (len(micro) / denom)
                loss_stats["kl"] += stats["kl"]
                loss_stats["clip_fraction"] += stats["clip_fraction"]
                loss_chunks += 1
                accelerator.backward(loss)
            optimizer.step()
        stats_tensor = torch.tensor(
            [loss_stats["kl"], loss_stats["clip_fraction"], float(loss_chunks)],
            dtype=torch.float64,
            device=accelerator.device,
        )
        stats_tensor = accelerator.reduce(stats_tensor, reduction="sum")
        global_loss_chunks = max(1.0, float(stats_tensor[2].item()))
        loss_stats = {
            "kl": float(stats_tensor[0].item()) / global_loss_chunks,
            "clip_fraction": float(stats_tensor[1].item()) / global_loss_chunks,
        }

        rewards = [float(row["reward"]) for row in all_rows]
        solved = sum(1 for reward in rewards if reward >= 1.0)
        avg_reward = sum(rewards) / max(1, len(rewards))
        turn_counts = [int(row["turn_count"]) for row in all_rows]
        record = {
            "step": step + 1,
            "avg_reward": avg_reward,
            "solved": solved,
            "count": len(all_rows),
            "avg_turns": sum(turn_counts) / max(1, len(turn_counts)),
            "examples": raw_example_count,
            "policy_examples_used": usable_count,
            "policy_examples_dropped_remainder": raw_example_count - usable_count,
            "policy_minibatches": policy_minibatches,
            "old_logprobs_cached": True,
            "old_logprobs_source": "raw_generation_logits",
            "kl": loss_stats["kl"],
            "clip_fraction": loss_stats["clip_fraction"],
            **reward_diagnostics,
            "seconds": time.time() - step_start,
        }
        if args.eval_interval > 0 and (step + 1) % args.eval_interval == 0:
            accelerator.wait_for_everyone()
            eval_result = evaluate_repeated(
                accelerator=accelerator,
                model=model,
                tokenizer=tokenizer,
                eval_tasks=eval_tasks,
                args=args,
                world=world,
                rank=rank,
                step=step + 1,
            )
            record["eval"] = eval_result
            if accelerator.is_main_process:
                print(
                    f"[eval] step={step + 1} repeats={eval_result['repeat_count']} "
                    f"solved_avg={eval_result['solved_average']:.2f}/{eval_result['count']} "
                    f"average={eval_result['average']:.6f} std={eval_result['average_std']:.6f}",
                    flush=True,
                )
        history.append(record)
        if accelerator.is_main_process and ((step + 1) % args.log_interval == 0 or step == 0):
            print(
                "[train] "
                f"step={record['step']} reward={record['avg_reward']:.4f} solved={record['solved']}/{record['count']} "
                f"avg_turns={record['avg_turns']:.1f} examples={record['examples']} used={record['policy_examples_used']} "
                f"drop={record['policy_examples_dropped_remainder']} mb={record['policy_minibatches']} "
                f"mixed={int(record['mixed_groups'])}/{int(record['group_count'])} "
                f"zero_std={int(record['zero_std_groups'])} all0={int(record['all_zero_groups'])} all1={int(record['all_one_groups'])} "
                f"std={record['mean_group_std']:.4f} kl={record['kl']:.4f} clip={record['clip_fraction']:.3f} sec={record['seconds']:.1f}",
                flush=True,
            )
            (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    if args.eval_after:
        accelerator.wait_for_everyone()
        result = evaluate_repeated(
            accelerator=accelerator,
            model=model,
            tokenizer=tokenizer,
            eval_tasks=eval_tasks,
            args=args,
            world=world,
            rank=rank,
            step=args.max_steps,
        )
        if accelerator.is_main_process:
            (output_dir / "eval.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(
                f"[eval] repeats={result['repeat_count']} solved_avg={result['solved_average']:.2f}/{result['count']} "
                f"average={result['average']:.6f} std={result['average_std']:.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
