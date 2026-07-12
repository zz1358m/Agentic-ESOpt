from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib import request


DIGITS = set(range(1, 10))


@dataclass(frozen=True)
class SudokuTask:
    id: str
    puzzle: list[list[int]]
    solution: list[list[int]]
    mask_count: int
    source: str = "generated"


@dataclass(frozen=True)
class SudokuAction:
    row: int
    col: int
    value: int


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_tasks(path: str | Path, limit: int = 0, mask_count: int | None = None) -> list[SudokuTask]:
    tasks = []
    for idx, row in enumerate(read_jsonl(Path(path))):
        tasks.append(
            SudokuTask(
                id=str(row.get("id", idx)),
                puzzle=normalize_grid(row["puzzle"], allow_zero=True),
                solution=normalize_grid(row["solution"], allow_zero=False),
                mask_count=int(row.get("mask_count", 0)),
                source=str(row.get("source", "generated")),
            )
        )
    if mask_count is not None and mask_count >= 0:
        tasks = [task for task in tasks if task.mask_count == mask_count]
    if limit > 0:
        tasks = tasks[:limit]
    if not tasks:
        suffix = f" with mask_count={mask_count}" if mask_count is not None and mask_count >= 0 else ""
        raise RuntimeError(f"No Sudoku tasks loaded from {path}{suffix}")
    return tasks


def normalize_grid(value: object, *, allow_zero: bool) -> list[list[int]]:
    if isinstance(value, str):
        chars = [ch for ch in value if ch.isdigit() or ch == "."]
        if len(chars) != 81:
            raise ValueError("Sudoku grid string must contain 81 cells")
        rows = [chars[i : i + 9] for i in range(0, 81, 9)]
    else:
        rows = value
    grid = []
    for row in rows:  # type: ignore[assignment]
        if len(row) != 9:
            raise ValueError("Sudoku rows must have 9 cells")
        parsed = []
        for cell in row:
            if cell == ".":
                number = 0
            else:
                number = int(cell)
            if number == 0 and allow_zero:
                parsed.append(number)
            elif number in DIGITS:
                parsed.append(number)
            else:
                raise ValueError(f"Invalid Sudoku cell: {cell!r}")
        grid.append(parsed)
    if len(grid) != 9:
        raise ValueError("Sudoku grid must have 9 rows")
    return grid


def board_to_lines(board: list[list[int]], *, blank: str = ".") -> list[str]:
    return [" ".join(blank if value == 0 else str(value) for value in row) for row in board]


def format_board(board: list[list[int]], *, blank: str = ".") -> str:
    return "\n".join(board_to_lines(board, blank=blank))


def format_board_with_coords(board: list[list[int]], *, blank: str = ".") -> str:
    header = "      c1 c2 c3 | c4 c5 c6 | c7 c8 c9"
    lines = [header]
    for row_idx, row in enumerate(board, start=1):
        values = [blank if value == 0 else str(value) for value in row]
        cells = " ".join(values[:3]) + " | " + " ".join(values[3:6]) + " | " + " ".join(values[6:])
        lines.append(f"r{row_idx}:   {cells}")
        if row_idx in {3, 6}:
            lines.append("      --------+----------+--------")
    return "\n".join(lines)


def empty_cells_text(board: list[list[int]]) -> str:
    cells = [f"r{row_idx}c{col_idx}" for row_idx, row in enumerate(board, start=1) for col_idx, value in enumerate(row, start=1) if value == 0]
    return ", ".join(cells) if cells else "none"


def clone_board(board: list[list[int]]) -> list[list[int]]:
    return [row[:] for row in board]


def empty_count(board: list[list[int]]) -> int:
    return sum(1 for row in board for value in row if value == 0)


def is_full(board: list[list[int]]) -> bool:
    return empty_count(board) == 0


def completion_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/completions") or endpoint.endswith("/v1/chat/completions"):
        return endpoint
    return f"{endpoint}/completions"


def post_json(url: str, payload: dict, timeout: int = 900) -> dict:
    data = json.dumps(payload).encode()
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def call_completion(
    endpoint: str,
    prompt: str,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    min_p: float | None,
    presence_penalty: float,
    repetition_penalty: float,
    timeout: int,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": temperature > 0,
        "presence_penalty": presence_penalty,
        "repetition_penalty": repetition_penalty,
    }
    if top_k is not None:
        payload["top_k"] = top_k
    if min_p is not None:
        payload["min_p"] = min_p
    response = post_json(completion_url(endpoint), payload, timeout=timeout)
    return first_completion_text(response)


def first_completion_text(response: dict) -> str:
    if isinstance(response.get("content"), list):
        return str(response["content"][0])
    if response.get("content") is not None:
        return str(response["content"])
    choices = response.get("choices") or []
    if choices:
        choice = choices[0]
        if isinstance(choice.get("message"), dict):
            return str(choice["message"].get("content", ""))
        return str(choice.get("text", ""))
    return ""


def completion_texts(response: dict) -> list[str]:
    if isinstance(response.get("content"), list):
        return [str(item) for item in response["content"]]
    return [first_completion_text(response)]


def call_completion_batch(
    endpoint: str,
    prompts: list[str],
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    min_p: float | None,
    presence_penalty: float,
    repetition_penalty: float,
    timeout: int,
) -> list[str]:
    if not prompts:
        return []
    payload = {
        "model": model,
        "prompt": prompts,
        "max_tokens": max_tokens,
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": temperature > 0,
        "presence_penalty": presence_penalty,
        "repetition_penalty": repetition_penalty,
    }
    if top_k is not None:
        payload["top_k"] = top_k
    if min_p is not None:
        payload["min_p"] = min_p
    texts = completion_texts(post_json(completion_url(endpoint), payload, timeout=timeout))
    if len(texts) != len(prompts):
        raise RuntimeError(f"Expected {len(prompts)} completions, got {len(texts)}")
    return texts


def extract_board(text: str) -> list[list[int]] | None:
    candidates = []
    tagged = re.findall(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    fenced = re.findall(r"```(?:text|sudoku)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    candidates.extend(tagged)
    candidates.extend(fenced)
    candidates.append(text)
    for candidate in candidates:
        compact = re.sub(r"[^1-9]", "", candidate)
        if len(compact) == 81:
            return [[int(ch) for ch in compact[i : i + 9]] for i in range(0, 81, 9)]
        rows = []
        for line in candidate.splitlines():
            digits = re.sub(r"[^1-9]", "", line)
            if len(digits) == 9:
                rows.append([int(ch) for ch in digits])
        if len(rows) >= 9:
            return rows[-9:]
    return None


def parse_action(text: str) -> SudokuAction | None:
    candidates = []
    json_match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
            row = int(payload.get("row", payload.get("r")))
            col = int(payload.get("col", payload.get("c")))
            value = int(payload.get("value", payload.get("v")))
            return SudokuAction(row=row, col=col, value=value)
        except Exception:
            pass
    for line in text.splitlines():
        if re.search(r"\b(set|fill|place|action)\b", line, flags=re.IGNORECASE):
            candidates.append(line)
    candidates.append(text)
    for candidate in candidates:
        cell_match = re.search(r"\br\s*([1-9])\s*c\s*([1-9])\b\D*([1-9])\b", candidate, flags=re.IGNORECASE)
        if cell_match:
            return SudokuAction(row=int(cell_match.group(1)), col=int(cell_match.group(2)), value=int(cell_match.group(3)))
        numbers = [int(item) for item in re.findall(r"\b[1-9]\b", candidate)]
        if len(numbers) >= 3:
            return SudokuAction(row=numbers[0], col=numbers[1], value=numbers[2])
    return None


def parse_action_sequence(text: str, *, max_actions: int = 81) -> list[SudokuAction]:
    actions = []
    for line in text.splitlines():
        action = parse_action(line)
        if action is not None:
            actions.append(action)
        if len(actions) >= max_actions:
            return actions
    if actions:
        return actions
    numbers = [int(item) for item in re.findall(r"\b[1-9]\b", text)]
    for idx in range(0, len(numbers) - 2, 3):
        actions.append(SudokuAction(row=numbers[idx], col=numbers[idx + 1], value=numbers[idx + 2]))
        if len(actions) >= max_actions:
            break
    return actions


def unit_score(values: list[int]) -> float:
    filled = [value for value in values if value in DIGITS]
    unique = len(set(filled))
    return unique / 9.0


def invalid_units(board: list[list[int]]) -> list[str]:
    bad = []
    for row_idx, row in enumerate(board):
        if set(row) != DIGITS:
            bad.append(f"row {row_idx + 1}")
    for col_idx in range(9):
        col = [board[row_idx][col_idx] for row_idx in range(9)]
        if set(col) != DIGITS:
            bad.append(f"column {col_idx + 1}")
    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            box = [board[r][c] for r in range(box_row, box_row + 3) for c in range(box_col, box_col + 3)]
            if set(box) != DIGITS:
                bad.append(f"box {box_row // 3 + 1},{box_col // 3 + 1}")
    return bad


def givens_match(puzzle: list[list[int]], board: list[list[int]]) -> tuple[int, int]:
    total = 0
    matched = 0
    for row_idx in range(9):
        for col_idx in range(9):
            given = puzzle[row_idx][col_idx]
            if given:
                total += 1
                if board[row_idx][col_idx] == given:
                    matched += 1
    return matched, total


def score_board(puzzle: list[list[int]], board: list[list[int]] | None) -> float:
    if board is None or len(board) != 9 or any(len(row) != 9 for row in board):
        return 0.0
    matched, total = givens_match(puzzle, board)
    if matched != total:
        return 0.0
    return 0.0 if invalid_units(board) else 1.0


def feedback_for_board(puzzle: list[list[int]], board: list[list[int]] | None) -> str:
    if board is None:
        return "No complete 9x9 grid was found. Return exactly 9 rows of 9 digits."
    matched, total = givens_match(puzzle, board)
    bad = invalid_units(board)
    parts = []
    if matched != total:
        parts.append(f"{total - matched} given cells were changed")
    if bad:
        parts.append("invalid " + ", ".join(bad[:8]))
    return "; ".join(parts) if parts else "The grid is valid."


def apply_action(
    puzzle: list[list[int]],
    board: list[list[int]],
    action_text: str,
) -> tuple[list[list[int]], dict]:
    next_board = clone_board(board)
    action = parse_action(action_text)
    if action is None:
        return next_board, {
            "valid": False,
            "action": None,
            "message": "Invalid action format. Use exactly: set <row> <col> <value>.",
        }
    if not (1 <= action.row <= 9 and 1 <= action.col <= 9 and 1 <= action.value <= 9):
        return next_board, {
            "valid": False,
            "action": action.__dict__,
            "message": "Row, column, and value must all be integers from 1 to 9.",
        }
    row_idx = action.row - 1
    col_idx = action.col - 1
    if puzzle[row_idx][col_idx] != 0:
        return next_board, {
            "valid": False,
            "action": action.__dict__,
            "message": f"Cell r{action.row}c{action.col} is a fixed given and cannot be changed.",
        }
    if board[row_idx][col_idx] != 0:
        return next_board, {
            "valid": False,
            "action": action.__dict__,
            "message": f"Cell r{action.row}c{action.col} is already filled.",
        }
    next_board[row_idx][col_idx] = action.value
    return next_board, {
        "valid": True,
        "action": action.__dict__,
        "message": f"Filled r{action.row}c{action.col} with {action.value}.",
    }


def replay_actions(
    puzzle: list[list[int]],
    action_text: str,
    *,
    max_actions: int = 81,
) -> tuple[list[list[int]], list[dict]]:
    board = clone_board(puzzle)
    trace = []
    for turn_index, action in enumerate(parse_action_sequence(action_text, max_actions=max_actions)):
        if is_full(board):
            break
        board, info = apply_action(puzzle, board, f"set {action.row} {action.col} {action.value}")
        info["turn"] = turn_index
        trace.append(info)
    return board, trace


def build_action_prompt(task: SudokuTask, board: list[list[int]], *, turn_index: int = 0, feedback: str = "") -> str:
    remaining = empty_count(board)
    prefix = (
        "You are an agent solving Sudoku one action at a time. "
        "At each turn, fill exactly one empty cell. Rows and columns are 1-indexed. "
        "The board is split by | and horizontal lines into nine 3x3 boxes. "
        "Every row, every column, and every 3x3 box must contain digits 1 through 9 exactly once. "
        "Choose the cell only from the Current empty cells list. "
        "Keep all original givens fixed. Your entire response must be exactly one line "
        "containing one action and nothing else. Do not explain, reason aloud, wrap the "
        "answer in code fences, or output any extra words. Use exactly this format:\n"
        "set <row> <col> <value>\n"
    )
    if feedback:
        prefix += f"\nLast environment feedback: {feedback}\n"
    return (
        f"{prefix}\nOriginal puzzle, mask_count={task.mask_count}:\n{format_board_with_coords(task.puzzle)}\n\n"
        f"Current board, turn={turn_index}, remaining_empty={remaining}:\n{format_board_with_coords(board)}\n\n"
        f"Current empty cells: {empty_cells_text(board)}\n"
    )


def build_prompt(task: SudokuTask, *, turn_index: int = 0, feedback: str = "") -> str:
    board = clone_board(task.puzzle)
    return build_action_prompt(task, board, turn_index=turn_index, feedback=feedback)


def build_grpo_prompt(task: SudokuTask) -> str:
    return (
        "You are an agent solving Sudoku with actions. Produce a sequence of one-cell actions, "
        "one per line, until the board is full. Use exactly this action format:\n"
        "set <row> <col> <value>\n\n"
        f"Original puzzle, mask_count={task.mask_count}:\n{format_board(task.puzzle)}\n"
    )


class SudokuEnv:
    def __init__(self, data_path: str | Path, *, limit: int = 0, mask_count: int | None = None) -> None:
        self.data_path = Path(data_path)
        self.mask_count = mask_count
        self.tasks = load_tasks(self.data_path, limit, mask_count=mask_count)

    def evaluate_response(self, task: SudokuTask, response: str) -> dict:
        board, action_trace = replay_actions(task.puzzle, response, max_actions=task.mask_count + 20)
        if not action_trace:
            board = extract_board(response)
        score = score_board(task.puzzle, board)
        return {
            "task_id": task.id,
            "score": score,
            "mask_count": task.mask_count,
            "prediction": board,
            "action_trace": action_trace,
            "filled": 81 - empty_count(board) if board is not None else 0,
            "done": is_full(board) if board is not None else False,
            "feedback": feedback_for_board(task.puzzle, board),
            "response": response,
        }

    def evaluate_task(
        self,
        *,
        endpoint: str,
        task: SudokuTask,
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int | None,
        min_p: float | None,
        presence_penalty: float,
        repetition_penalty: float,
        timeout: int,
        max_turns: int = 1,
    ) -> dict:
        turns = []
        feedback = ""
        board = clone_board(task.puzzle)
        for turn_index in range(max(1, max_turns)):
            if is_full(board):
                break
            prompt = build_action_prompt(task, board, turn_index=turn_index, feedback=feedback)
            try:
                response = call_completion(
                    endpoint,
                    prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    presence_penalty=presence_penalty,
                    repetition_penalty=repetition_penalty,
                    timeout=timeout,
                )
                board, info = apply_action(task.puzzle, board, response)
                row = {
                    "task_id": task.id,
                    "mask_count": task.mask_count,
                    "response": response,
                    "board": board,
                    "remaining_empty": empty_count(board),
                    **info,
                }
            except Exception as exc:
                row = {"task_id": task.id, "score": -1.0, "mask_count": task.mask_count, "error": repr(exc)}
            row["turn"] = turn_index
            turns.append(row)
            feedback = str(row.get("message", row.get("error", "")))
        score = score_board(task.puzzle, board)
        result = {
            "task_id": task.id,
            "score": score,
            "mask_count": task.mask_count,
            "prediction": board,
            "filled": 81 - empty_count(board),
            "remaining_empty": empty_count(board),
            "done": is_full(board),
            "feedback": feedback_for_board(task.puzzle, board),
        }
        result["turns"] = turns
        return result
