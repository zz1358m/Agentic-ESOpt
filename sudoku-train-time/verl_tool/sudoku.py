from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from .base import BaseTool, register_tool


def _add_dynamic_agent_root() -> None:
    root = os.environ.get("DYNAMIC_AGENT_ROOT") or os.environ.get("ROOT")
    if root:
        sudoku_root = Path(root) / "sudoku-train-time"
        if str(sudoku_root) not in sys.path:
            sys.path.insert(0, str(sudoku_root))


_add_dynamic_agent_root()

from envs.sudoku import (  # noqa: E402
    apply_action,
    clone_board,
    empty_count,
    feedback_for_board,
    format_board,
    is_full,
    parse_action,
    score_board,
)


def _json_default(value: Any):
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def _extract_create_kwargs(extra_field: dict) -> dict:
    if not extra_field:
        return {}
    if "create_kwargs" in extra_field:
        return extra_field["create_kwargs"] or {}
    tools_kwargs = extra_field.get("tools_kwargs") or {}
    sudoku_kwargs = tools_kwargs.get("sudoku") or tools_kwargs.get("sudoku_tool") or {}
    if "create_kwargs" in sudoku_kwargs:
        return sudoku_kwargs["create_kwargs"] or {}
    return extra_field


def _compact_observation(payload: dict) -> str:
    return "<sudoku_state>" + json.dumps(payload, separators=(",", ":"), default=_json_default) + "</sudoku_state>"


@register_tool
class SudokuTool(BaseTool):
    tool_type = "sudoku"

    def get_usage_inst(self):
        return (
            "Fill one Sudoku cell per turn. Use exactly: set <row> <col> <value>. "
            "Rows and columns are 1-indexed."
        )

    def parse_action(self, action: str):
        parsed = parse_action(action)
        return parsed, parsed is not None

    def load_env(self, trajectory_id):
        env = self.env_cache.get(trajectory_id)
        if env is None:
            env = {
                "trajectory_id": trajectory_id,
                "metadata": {"turns": 0},
                "previous_obs": [],
                "puzzle": None,
                "solution": None,
                "board": None,
                "task_id": trajectory_id,
                "mask_count": None,
            }
        return env

    def _ensure_env_initialized(self, env: dict, extra_field: dict) -> None:
        if env.get("board") is not None:
            return
        create_kwargs = _extract_create_kwargs(extra_field)
        puzzle = create_kwargs.get("puzzle")
        solution = create_kwargs.get("solution")
        if puzzle is None or solution is None:
            reward_model = create_kwargs.get("reward_model") or extra_field.get("reward_model", {})
            ground_truth = reward_model.get("ground_truth", {}) if isinstance(reward_model, dict) else {}
            puzzle = puzzle if puzzle is not None else ground_truth.get("puzzle")
            solution = solution if solution is not None else ground_truth.get("solution")
        if puzzle is None or solution is None:
            raise ValueError("SudokuTool requires puzzle and solution in extra_field/create_kwargs.")
        env["puzzle"] = puzzle
        env["solution"] = solution
        env["board"] = clone_board(puzzle)
        env["task_id"] = str(create_kwargs.get("task_id", env["trajectory_id"]))
        env["mask_count"] = int(create_kwargs.get("mask_count", empty_count(puzzle)))

    def conduct_action(self, trajectory_id, action, extra_field):
        env = self.load_env(trajectory_id)
        self._ensure_env_initialized(env, extra_field or {})
        parsed_action, is_parse_valid = self.parse_action(action)
        board, info = apply_action(env["puzzle"], env["board"], action)
        env["board"] = board
        done = is_full(board)
        reward = score_board(env["puzzle"], board) if done else 0.0
        payload = {
            "task_id": env["task_id"],
            "mask_count": env["mask_count"],
            "turn": env["metadata"]["turns"],
            "valid": bool(info.get("valid", False) and is_parse_valid),
            "done": done,
            "reward": reward,
            "remaining_empty": empty_count(board),
            "message": info.get("message", ""),
            "board": format_board(board),
            "feedback": feedback_for_board(env["puzzle"], board),
        }
        observation = _compact_observation(payload)
        self.update_env(trajectory_id, env, parsed_action, payload["valid"], extra_field or {}, observation, reward=reward)
        self.save_env(trajectory_id, env)
        return observation, done, payload["valid"]
