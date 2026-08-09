import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "webarena" / "replay_es_history_and_eval.py"
SPEC = importlib.util.spec_from_file_location("webarena_es_replay", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_history(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_load_update_history_ignores_eval_records(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    write_history(
        path,
        [
            {"generation": -1, "kind": "initial_base_eval"},
            {"generation": 0, "seeds": [1, 2], "rewards": [0.0, 1.0]},
            {"generation": 1, "seeds": [3, 4], "rewards": [1.0, 0.0]},
        ],
    )
    assert [row["generation"] for row in MODULE.load_update_history(path)] == [0, 1]


def test_load_update_history_rejects_generation_gaps(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    write_history(path, [{"generation": 1, "seeds": [1], "rewards": [1.0]}])
    with pytest.raises(ValueError, match="not contiguous"):
        MODULE.load_update_history(path)
