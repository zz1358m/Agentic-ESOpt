#!/usr/bin/env python3
"""Fail-closed frozen validation gate for AHD ACO-TSP finalists."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.ahd.aco_tsp_evaluator import (
    pairwise_distances_float64,
    seed_aco_random_stream,
    select_heuristic_function,
    solve_tsp_instance,
)


FROZEN_SIZES = (20, 50, 100)
FROZEN_VALIDATION_SHA256 = {
    "val20_dataset.npy": "29a45a2737cbd5365d9b492c110bbec3419b89d407fea85ddd7848025c34c911",
    "val50_dataset.npy": "9fd047f64a38417a059592a864a6fd3be32035fdc0352b2478bd97b407540135",
    "val100_dataset.npy": "cc1e619b6d65237c4d75d554b909e8da7f1bd2c5ce2d39c2e8c4ce1640195a5c",
}
FROZEN_TEST_SHA256 = {
    "test20_dataset.npy": "84cca88c7bcc34dd7fe076101cc7cf8cc6de505f23aeb7f8388d29842f13d74f",
    "test50_dataset.npy": "d17c6c87ee622dbc75f031f2ad64bcf3c1d660e6179a92fc20ab9c443f99d405",
    "test100_dataset.npy": "c2ef5e1fa54afce93d90e7fec0920438617f2b17e2f23e12c73987d606b1eff0",
}
FROZEN_SEED = 1234
FROZEN_ITERATIONS = 100
FROZEN_ANTS = 30


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _instance_fingerprint(instance: np.ndarray) -> str:
    evaluator_input = pairwise_distances_float64(instance)
    array = np.ascontiguousarray(evaluator_input.astype(np.float32))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def freeze_candidate_from_population(population_path: Path, destination: Path) -> dict:
    """Copy the selected code byte-for-byte from a population JSON artifact."""
    source = Path(population_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("final population artifact must be a JSON object")
    code = payload.get("code")
    objective = payload.get("objective")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("final population artifact has no non-empty code")
    if isinstance(objective, bool) or not isinstance(objective, (int, float)):
        raise ValueError("final population artifact has no numeric objective")
    if not math.isfinite(float(objective)):
        raise ValueError("final population objective is non-finite")
    target = Path(destination).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(code, encoding="utf-8")
    temporary.replace(target)
    code_sha256 = hashlib.sha256(code.encode("utf-8")).hexdigest()
    if _sha256(target) != code_sha256:
        raise RuntimeError("frozen candidate differs from population code bytes")
    return {
        "source_population": str(source),
        "source_population_sha256": _sha256(source),
        "frozen_candidate": str(target),
        "frozen_candidate_sha256": code_sha256,
        "train_objective": float(objective),
    }


def audit_frozen_validation_datasets(
    dataset_root: Path,
    *,
    expected_validation_sha256: Mapping[str, str],
    expected_test_sha256: Mapping[str, str],
) -> dict:
    root = Path(dataset_root).resolve()
    expected_names = {f"val{size}_dataset.npy" for size in FROZEN_SIZES}
    expected_test_names = {f"test{size}_dataset.npy" for size in FROZEN_SIZES}
    if set(expected_validation_sha256) != expected_names:
        raise ValueError("frozen validation manifest must contain exactly N20/N50/N100")
    if set(expected_test_sha256) != expected_test_names:
        raise ValueError("frozen test manifest must contain exactly N20/N50/N100")

    rows = []
    for size in FROZEN_SIZES:
        validation_path = root / f"val{size}_dataset.npy"
        test_path = root / f"test{size}_dataset.npy"
        for path, expected in (
            (validation_path, expected_validation_sha256[validation_path.name]),
            (test_path, expected_test_sha256[test_path.name]),
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = _sha256(path)
            if actual != expected:
                raise ValueError(f"frozen dataset SHA-256 mismatch for {path.name}")

        validation = np.load(validation_path, allow_pickle=False)
        test = np.load(test_path, allow_pickle=False)
        expected_tail = (size, 2)
        if validation.ndim != 3 or validation.shape[1:] != expected_tail:
            raise ValueError(f"validation N{size} shape is not (*,{size},2): {validation.shape}")
        if test.ndim != 3 or test.shape[1:] != expected_tail:
            raise ValueError(f"test N{size} shape is not (*,{size},2): {test.shape}")
        validation_fingerprints = {_instance_fingerprint(item) for item in validation}
        test_fingerprints = {_instance_fingerprint(item) for item in test}
        overlap = validation_fingerprints & test_fingerprints
        if overlap:
            raise ValueError(
                f"validation and test instance overlap at N{size}: {len(overlap)} fingerprint(s)"
            )
        rows.append(
            {
                "size": size,
                "validation_file": validation_path.name,
                "validation_sha256": expected_validation_sha256[validation_path.name],
                "validation_count": len(validation),
                "validation_unique_instances": len(validation_fingerprints),
                "test_file": test_path.name,
                "test_sha256": expected_test_sha256[test_path.name],
                "test_count": len(test),
                "test_unique_instances": len(test_fingerprints),
                "validation_test_overlap_count": 0,
            }
        )
    return {"status": "PASS", "sizes": list(FROZEN_SIZES), "datasets": rows}


def evaluate_frozen_validation(
    code_path: Path,
    dataset_root: Path,
    *,
    expected_validation_sha256: Mapping[str, str],
    expected_test_sha256: Mapping[str, str],
    seed: int = 1234,
    n_iterations: int = 100,
    n_ants: int = 30,
) -> dict:
    provenance = audit_frozen_validation_datasets(
        dataset_root,
        expected_validation_sha256=expected_validation_sha256,
        expected_test_sha256=expected_test_sha256,
    )
    candidate = Path(code_path).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    if seed < 0 or n_iterations <= 0 or n_ants <= 0:
        raise ValueError("validation seed must be non-negative and ACO settings must be positive")

    def load_heuristic(repeat_index: int):
        module_name = f"aco_tsp_validation_candidate_{repeat_index}_{_sha256(candidate)[:12]}"
        spec = importlib.util.spec_from_file_location(module_name, candidate)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load validation candidate from {candidate}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return select_heuristic_function(module)[1]

    def evaluate_once(repeat_index: int) -> list[dict]:
        heuristic = load_heuristic(repeat_index)
        rows = []
        for size in FROZEN_SIZES:
            validation_path = Path(dataset_root).resolve() / f"val{size}_dataset.npy"
            instances = np.load(validation_path, allow_pickle=False)
            values = []
            for instance_index, instance in enumerate(instances):
                instance_seed = (int(seed) + size * 1_000 + instance_index) % (2**32)
                seed_aco_random_stream(instance_seed)
                value = float(
                    solve_tsp_instance(
                        heuristic,
                        instance,
                        n_iterations=n_iterations,
                        n_ants=n_ants,
                    )
                )
                if not math.isfinite(value):
                    raise ValueError(
                        f"non-finite validation objective at N{size} instance {instance_index}"
                    )
                values.append(value)
            value_array = np.asarray(values, dtype=np.float64)
            rows.append(
                {
                    "split": "validation",
                    "size": size,
                    "count": len(instances),
                    "valid_count": len(values),
                    "failure_count": 0,
                    "mean": float(value_array.mean()),
                    "std": float(value_array.std()),
                    "min": float(value_array.min()),
                    "max": float(value_array.max()),
                    "setting_seed": (int(seed) + size * 1_000) % (2**32),
                    "instance_objectives": values,
                }
            )
        return rows

    first = evaluate_once(1)
    second = evaluate_once(2)
    if first != second:
        raise ValueError("frozen validation is unstable across two seeded repeats")
    canonical = json.dumps(
        first,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "status": "PASS",
        "contract": "aco_tsp_float32_fail_closed_v1",
        "candidate": str(candidate),
        "candidate_sha256": _sha256(candidate),
        "seed": int(seed),
        "n_iterations": int(n_iterations),
        "n_ants": int(n_ants),
        "sizes": list(FROZEN_SIZES),
        "stability_repeats": 2,
        "stable_results_sha256": hashlib.sha256(canonical).hexdigest(),
        "provenance": provenance,
        "results": first,
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--code", type=Path, help="Already-frozen final candidate Python file.")
    source.add_argument(
        "--population-json",
        type=Path,
        help="Final population JSON; its code is copied exactly, never patched.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Fail-closed validation report path.",
    )
    args = parser.parse_args()
    freeze = None
    code_path = args.code
    if args.population_json is not None:
        code_path = args.output.resolve().parent / "final_best_code.py"
        freeze = freeze_candidate_from_population(args.population_json, code_path)
    assert code_path is not None
    dataset_root = ROOT / "data" / "ahd" / "datasets" / "tsp_aco"
    report = evaluate_frozen_validation(
        code_path,
        dataset_root,
        expected_validation_sha256=FROZEN_VALIDATION_SHA256,
        expected_test_sha256=FROZEN_TEST_SHA256,
        seed=FROZEN_SEED,
        n_iterations=FROZEN_ITERATIONS,
        n_ants=FROZEN_ANTS,
    )
    if freeze is not None:
        report["candidate_freeze"] = freeze
    _write_json_atomic(args.output, report)
    print(
        f"PASS: frozen validation N20/N50/N100, two stable repeats, report={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
