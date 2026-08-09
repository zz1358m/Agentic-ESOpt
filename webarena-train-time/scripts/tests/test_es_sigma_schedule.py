import importlib.util
from pathlib import Path


RUNNER = Path(__file__).resolve().parents[1] / "run_webrl_lite_full_es_train.py"
SPEC = importlib.util.spec_from_file_location("webarena_es_runner", RUNNER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def sigma_values(start: float, end: float) -> list[float]:
    return [
        MODULE.sigma_for_generation(
            sigma_start=start,
            sigma_end=end,
            generation=generation,
            generations=70,
            schedule="cosine",
            warmup_steps=0,
        )
        for generation in range(70)
    ]


def test_equal_cosine_endpoints_are_constant() -> None:
    assert sigma_values(1.5e-3, 1.5e-3) == [1.5e-3] * 70


def test_cosine_schedule_reaches_both_endpoints() -> None:
    values = sigma_values(1.75e-3, 1.25e-3)
    assert values[0] == 1.75e-3
    assert values[-1] == 1.25e-3
    assert all(left >= right for left, right in zip(values, values[1:]))
