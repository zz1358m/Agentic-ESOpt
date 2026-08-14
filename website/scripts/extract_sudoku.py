from __future__ import annotations

import re
from collections import defaultdict
from statistics import fmean
from typing import Any


SAMPLE = re.compile(r"\[sample\]\s+gen=(-?\d+)\s+idx=\d+\s+reward=([0-9.]+)")
EVAL = re.compile(
    r"\[eval\]\s+generation=(-?\d+)\s+split=(train|eval).*?average=([0-9.]+)\s+std=([0-9.]+)"
)
RL_TRAIN = re.compile(r"\[train\]\s+step=(\d+)\s+reward=([0-9.]+).*?avg_turns=([0-9.]+)")
RL_EVAL = re.compile(r"\[eval\]\s+step=(\d+).*?average=([0-9.]+)\s+std=([0-9.]+)")


def parse_training_log(text: str) -> dict[str, list[dict[str, Any]]]:
    sample_rewards: dict[int, list[float]] = defaultdict(list)
    periodic = {"train": [], "eval": []}

    for line in text.splitlines():
        sample = SAMPLE.search(line)
        if sample:
            sample_rewards[int(sample.group(1))].append(float(sample.group(2)))
            continue
        evaluation = EVAL.search(line)
        if evaluation:
            generation, split, value, std = evaluation.groups()
            periodic[split].append(
                {
                    "generation": int(generation),
                    "value": float(value),
                    "std": float(std),
                }
            )

    return {
        "trainCurve": [
            {"generation": generation, "value": fmean(values)}
            for generation, values in sorted(sample_rewards.items())
        ],
        "periodicTrain": periodic["train"],
        "periodicEval": periodic["eval"],
    }


def parse_rl_training_log(text: str) -> dict[str, list[dict[str, Any]]]:
    """Parse the retained PPO/GRPO-style step log without remapping its step axis."""
    train = []
    evaluation = []
    for line in text.splitlines():
        train_match = RL_TRAIN.search(line)
        if train_match:
            step, reward, average_turns = train_match.groups()
            train.append({"generation": int(step), "value": float(reward), "averageTurns": float(average_turns)})
            continue
        eval_match = RL_EVAL.search(line)
        if eval_match:
            step, value, std = eval_match.groups()
            evaluation.append({"generation": int(step), "value": float(value), "std": float(std)})
    return {"trainCurve": train, "periodicEval": evaluation}
