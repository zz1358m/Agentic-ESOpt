from __future__ import annotations

import json
import re
from collections import defaultdict

import numpy as np
import torch
from verl import DataProto
from verl.workers.reward_manager import register


def _latest_sudoku_state(text: str) -> dict | None:
    matches = re.findall(r"<sudoku_state>(.*?)</sudoku_state>", text, flags=re.DOTALL)
    for item in reversed(matches):
        try:
            return json.loads(item)
        except json.JSONDecodeError:
            continue
    return None


@register("sudoku_binary")
class SudokuBinaryRewardManager:
    name = "sudoku_binary"

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict=False):
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        printed = 0
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            state = _latest_sudoku_state(response_str) or {}
            reward = 1.0 if float(state.get("reward", 0.0)) >= 1.0 and bool(state.get("done", False)) else 0.0
            reward_tensor[i, max(int(valid_response_length) - 1, 0)] = reward
            reward_extra_info["score"].append(reward)
            reward_extra_info["accuracy"].append(reward)
            reward_extra_info["remaining_empty"].append(float(state.get("remaining_empty", 81)))
            reward_extra_info["valid_final_state"].append(1.0 if state else 0.0)
            if printed < self.num_examine:
                printed += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[sudoku_state]", state)
                print("[score]", reward)
        for key, values in list(reward_extra_info.items()):
            if values and isinstance(values[0], (int, float)):
                reward_extra_info[f"{key}_mean"] = [float(np.mean(values))] * len(data)
        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": dict(sorted(reward_extra_info.items()))}
        return reward_tensor
