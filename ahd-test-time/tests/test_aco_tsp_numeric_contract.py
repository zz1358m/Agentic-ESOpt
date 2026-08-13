from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
EOH_SRC = ROOT / "ahd-test-time" / "methods" / "eoh" / "original" / "eoh" / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EOH_SRC))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


from eoh.problems.optimization.tsp_aco import run as TRAIN


FINAL = load_module(
    "aco_tsp_final_evaluator_for_contract_test",
    ROOT / "ahd-test-time" / "results" / "eval_aco_results.py",
)
VALIDATION_GATE = load_module(
    "aco_tsp_validation_gate_for_contract_test",
    ROOT / "scripts" / "ahd" / "aco_tsp_validation_gate.py",
)
RUNNER = load_module(
    "run_eoh_ahd_for_numeric_contract_test",
    ROOT / "ahd-test-time" / "scripts" / "run_eoh_ahd.py",
)


def _training_solve(points: np.ndarray, heuristic) -> float:
    evaluator = object.__new__(TRAIN.TSPACO)
    evaluator.n_iterations = 1
    evaluator.n_ants = 2
    return evaluator.solve(points, heuristic)


def _final_solve(points: np.ndarray, heuristic) -> float:
    return FINAL.solve_tsp(heuristic, points, n_iterations=1, n_ants=2)


def test_training_and_final_fail_closed_when_float64_heuristic_overflows_float32() -> None:
    points = np.array([[0.0, 0.0], [0.01, 0.0], [1.0, 0.0]], dtype=np.float64)

    def finite_float64_but_not_float32(distance_matrix: np.ndarray) -> np.ndarray:
        return np.exp(1.0 / distance_matrix)

    for solve in (_training_solve, _final_solve):
        with pytest.raises(ValueError, match="float32"):
            solve(points, finite_float64_but_not_float32)


def test_training_evaluator_is_stable_for_the_same_candidate_and_instances() -> None:
    evaluator = object.__new__(TRAIN.TSPACO)
    evaluator.problem_size = 4
    evaluator.n_iterations = 2
    evaluator.n_ants = 4
    evaluator.node_positions = np.array(
        [
            [[0.0, 0.0], [0.2, 0.1], [0.9, 0.2], [0.4, 0.8]],
            [[0.1, 0.0], [0.3, 0.7], [0.8, 0.4], [0.9, 0.9]],
        ],
        dtype=np.float64,
    )
    code = "def heuristics(distance_matrix):\n    return 1.0 / distance_matrix\n"

    assert evaluator.evaluate(code) == evaluator.evaluate(code)


def test_validation_provenance_gate_rejects_instance_overlap_with_test(tmp_path: Path) -> None:
    import hashlib

    expected_validation = {}
    expected_test = {}
    for size in (20, 50, 100):
        validation = np.arange(2 * size * 2, dtype=np.float64).reshape(2, size, 2)
        test = validation.copy() if size == 50 else validation + 10_000.0
        validation_path = tmp_path / f"val{size}_dataset.npy"
        test_path = tmp_path / f"test{size}_dataset.npy"
        np.save(validation_path, validation, allow_pickle=False)
        np.save(test_path, test, allow_pickle=False)
        expected_validation[validation_path.name] = hashlib.sha256(
            validation_path.read_bytes()
        ).hexdigest()
        expected_test[test_path.name] = hashlib.sha256(test_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="validation.*test.*overlap"):
        VALIDATION_GATE.audit_frozen_validation_datasets(
            tmp_path,
            expected_validation_sha256=expected_validation,
            expected_test_sha256=expected_test,
        )


def test_validation_provenance_rejects_translation_equivalent_test_instances(
    tmp_path: Path,
) -> None:
    import hashlib

    expected_validation = {}
    expected_test = {}
    for size in (20, 50, 100):
        validation = np.stack(
            [np.column_stack([np.linspace(0.0, 1.0, size), np.linspace(1.0, 2.0, size)])]
        )
        test = validation + np.array([500.0, -250.0])
        validation_path = tmp_path / f"val{size}_dataset.npy"
        test_path = tmp_path / f"test{size}_dataset.npy"
        np.save(validation_path, validation, allow_pickle=False)
        np.save(test_path, test, allow_pickle=False)
        expected_validation[validation_path.name] = hashlib.sha256(
            validation_path.read_bytes()
        ).hexdigest()
        expected_test[test_path.name] = hashlib.sha256(test_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="validation.*test.*overlap"):
        VALIDATION_GATE.audit_frozen_validation_datasets(
            tmp_path,
            expected_validation_sha256=expected_validation,
            expected_test_sha256=expected_test,
        )


def test_frozen_n20_n50_n100_validation_is_stable_and_reports_only_validation(
    tmp_path: Path,
) -> None:
    import hashlib

    expected_validation = {}
    expected_test = {}
    for size in (20, 50, 100):
        validation = np.stack(
            [
                np.column_stack(
                    [np.linspace(0.0, 1.0, size), np.linspace(index, index + 0.5, size)]
                )
                for index in (0.0, 2.0)
            ]
        )
        test = validation * 3.0 + np.array([10_000.0, -10_000.0])
        validation_path = tmp_path / f"val{size}_dataset.npy"
        test_path = tmp_path / f"test{size}_dataset.npy"
        np.save(validation_path, validation, allow_pickle=False)
        np.save(test_path, test, allow_pickle=False)
        expected_validation[validation_path.name] = hashlib.sha256(
            validation_path.read_bytes()
        ).hexdigest()
        expected_test[test_path.name] = hashlib.sha256(test_path.read_bytes()).hexdigest()

    code_path = tmp_path / "candidate.py"
    code_path.write_text(
        "import random\n"
        "import numpy as np\n"
        "def heuristics(distance_matrix):\n"
        "    return np.array([[random.random() + 0.1 for _ in row] for row in distance_matrix])\n",
        encoding="utf-8",
    )

    first = VALIDATION_GATE.evaluate_frozen_validation(
        code_path,
        tmp_path,
        expected_validation_sha256=expected_validation,
        expected_test_sha256=expected_test,
        seed=20260814,
        n_iterations=1,
        n_ants=2,
    )
    second = VALIDATION_GATE.evaluate_frozen_validation(
        code_path,
        tmp_path,
        expected_validation_sha256=expected_validation,
        expected_test_sha256=expected_test,
        seed=20260814,
        n_iterations=1,
        n_ants=2,
    )

    assert first == second
    assert first["status"] == "PASS"
    assert first["stability_repeats"] == 2
    assert [(row["split"], row["size"], row["valid_count"]) for row in first["results"]] == [
        ("validation", 20, 2),
        ("validation", 50, 2),
        ("validation", 100, 2),
    ]


def test_training_and_final_select_the_same_supported_heuristic_function() -> None:
    function = lambda distance_matrix: distance_matrix  # noqa: E731
    candidate = SimpleNamespace(heuristics_v4=function)

    assert TRAIN._get_heuristic_name(candidate) == "heuristics_v4"
    assert FINAL.get_heuristics(candidate) is function


def test_training_and_final_return_exactly_the_same_seeded_objective() -> None:
    points = np.array(
        [[0.0, 0.0], [0.2, 0.9], [0.8, 0.3], [1.0, 1.0]],
        dtype=np.float64,
    )

    def heuristic(distance_matrix: np.ndarray) -> np.ndarray:
        return 1.0 / distance_matrix

    TRAIN.seed_aco_random_stream(20260814)
    training = _training_solve(points, heuristic)
    TRAIN.seed_aco_random_stream(20260814)
    final = _final_solve(points, heuristic)

    assert training == final


def test_finite_float32_weights_fail_closed_when_row_mass_cannot_form_simplex() -> None:
    points = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]], dtype=np.float64)

    def unnormalizable_weights(distance_matrix: np.ndarray) -> np.ndarray:
        return np.full(distance_matrix.shape, np.finfo(np.float32).max, dtype=np.float32)

    for solve in (_training_solve, _final_solve):
        with pytest.raises(ValueError, match="positive simplex"):
            solve(points, unnormalizable_weights)


def test_final_candidate_is_frozen_from_population_without_modifying_code(tmp_path: Path) -> None:
    import json

    code = "import numpy as np\ndef heuristics(distance_matrix):\n    return 1 / distance_matrix\n"
    population = tmp_path / "population_generation_50.json"
    population.write_text(json.dumps({"code": code, "objective": 6.25}), encoding="utf-8")
    frozen = tmp_path / "final_best_code.py"

    result = VALIDATION_GATE.freeze_candidate_from_population(population, frozen)

    assert frozen.read_text(encoding="utf-8") == code
    assert result["train_objective"] == 6.25
    assert result["source_population"] == str(population.resolve())


def test_training_cli_exposes_the_frozen_evaluation_seed(monkeypatch) -> None:
    monkeypatch.setenv("AHD_EVALUATION_SEED", "1234")
    monkeypatch.setattr(sys, "argv", ["run_eoh_ahd.py"])

    assert RUNNER.parse_args().evaluation_seed == 1234
