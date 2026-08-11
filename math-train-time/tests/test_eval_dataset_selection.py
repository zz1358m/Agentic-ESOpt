from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "math-train-time" / "scripts" / "run_math_es_vllm_train.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_math_es_vllm_train_dataset_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_eval_datasets_preserves_default_order() -> None:
    module = load_module()
    assert module.parse_eval_datasets("dapo,aime") == ("dapo", "aime")


@pytest.mark.parametrize(
    ("value", "expected"),
    [("dapo", ("dapo",)), ("aime", ("aime",)), ("aime,dapo", ("dapo", "aime"))],
)
def test_parse_eval_datasets_supports_independent_formal_runs(value: str, expected: tuple[str, ...]) -> None:
    module = load_module()
    assert module.parse_eval_datasets(value) == expected


@pytest.mark.parametrize("value", ["", "unknown", "dapo,unknown"])
def test_parse_eval_datasets_rejects_empty_or_unknown_values(value: str) -> None:
    module = load_module()
    with pytest.raises(ValueError, match="eval datasets"):
        module.parse_eval_datasets(value)


def test_trajectory_seed_is_stable_per_row_and_sample() -> None:
    module = load_module()
    assert module.trajectory_seed(20260627, row_index=0, sample_index=0) == 20260627
    assert module.trajectory_seed(20260627, row_index=7, sample_index=2) == 22260640


def test_load_eval_key_seeds_preserves_frozen_slot_seeds(tmp_path: Path) -> None:
    module = load_module()
    path = tmp_path / "key_seeds.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"key":"q0:sample00","seed":20260627}',
                '{"key":"q0:sample01","seed":21260630}',
                '{"key":"q0:sample02","seed":23260636}',
                '{"key":"q0:sample03","seed":21260631}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    seeds = module.load_eval_key_seeds(path)
    assert seeds["q0:sample02"] == 23260636
    assert seeds["q0:sample03"] == 21260631


def test_load_eval_key_seeds_rejects_duplicate_keys(tmp_path: Path) -> None:
    module = load_module()
    path = tmp_path / "duplicate.jsonl"
    path.write_text(
        '{"key":"q0:sample00","seed":1}\n{"key":"q0:sample00","seed":2}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate key"):
        module.load_eval_key_seeds(path)


def test_load_eval_key_seeds_rejects_wrong_frozen_sha(tmp_path: Path) -> None:
    module = load_module()
    path = tmp_path / "key_seeds.jsonl"
    path.write_text('{"key":"q0:sample00","seed":1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        module.load_eval_key_seeds(path, expected_sha256="0" * 64)


def test_validate_formal_eval_summary_accepts_exact_matrix() -> None:
    module = load_module()
    tasks = [
        module.MathTask(id=f"q{i}", question="question", answer="answer", source="test")
        for i in range(100)
    ]
    rows = [
        {
            "key": f"q{row_index}:sample{sample_index:02d}",
            "task_id": f"q{row_index}",
            "row_index": row_index,
            "sample_index": sample_index,
            "seed": module.trajectory_seed(20260627, row_index=row_index, sample_index=sample_index),
            "engine_index": (row_index * 4 + sample_index) % 4,
        }
        for row_index in range(100)
        for sample_index in range(4)
    ]
    module.validate_formal_eval_summary("dapo", {"scores": rows}, tasks=tasks, samples=4)


def test_validate_formal_eval_summary_accepts_frozen_seed_mapping() -> None:
    module = load_module()
    tasks = [module.MathTask(id=f"q{i}", question="q", answer="a", source="test") for i in range(30)]
    frozen_seeds = {
        f"q{row_index}:sample{sample_index:02d}": 10_000_000 * sample_index + row_index
        for row_index in range(30)
        for sample_index in range(4)
    }
    rows = [
        {
            "key": key,
            "task_id": key.split(":", 1)[0],
            "row_index": row_index,
            "sample_index": sample_index,
            "seed": frozen_seeds[key],
            "engine_index": (row_index * 4 + sample_index) % 4,
        }
        for row_index in range(30)
        for sample_index in range(4)
        for key in [f"q{row_index}:sample{sample_index:02d}"]
    ]
    module.validate_formal_eval_summary(
        "aime",
        {"scores": rows},
        tasks=tasks,
        samples=4,
        expected_seeds=frozen_seeds,
    )


@pytest.mark.parametrize("missing_field", ["seed", "engine_index"])
def test_validate_formal_eval_summary_requires_audit_fields(missing_field: str) -> None:
    module = load_module()
    tasks = [module.MathTask(id=f"q{i}", question="q", answer="a", source="test") for i in range(30)]
    rows = [
        {
            "key": f"q{row_index}:sample{sample_index:02d}",
            "task_id": f"q{row_index}",
            "row_index": row_index,
            "sample_index": sample_index,
            "seed": module.trajectory_seed(20260627, row_index=row_index, sample_index=sample_index),
            "engine_index": (row_index * 4 + sample_index) % 4,
        }
        for row_index in range(30)
        for sample_index in range(4)
    ]
    rows[0].pop(missing_field)
    with pytest.raises(ValueError, match=missing_field):
        module.validate_formal_eval_summary("aime", {"scores": rows}, tasks=tasks, samples=4)


def test_validate_formal_eval_summary_rejects_incomplete_key_set() -> None:
    module = load_module()
    tasks = [module.MathTask(id=f"q{i}", question="q", answer="a", source="test") for i in range(30)]
    rows = [
        {
            "key": f"q{row_index}:sample{sample_index:02d}",
            "task_id": f"q{row_index}",
            "row_index": row_index,
            "sample_index": sample_index,
            "seed": module.trajectory_seed(20260627, row_index=row_index, sample_index=sample_index),
            "engine_index": (row_index * 4 + sample_index) % 4,
        }
        for row_index in range(30)
        for sample_index in range(4)
    ][:-1]
    with pytest.raises(ValueError, match="expected 120 rows"):
        module.validate_formal_eval_summary("aime", {"scores": rows}, tasks=tasks, samples=4)


def formal_options(module):
    return SimpleNamespace(
        formal_eval=True,
        eval_only=True,
        generations=0,
        skip_initial_eval=False,
        reuse_initial_eval_history="",
        resume_history="",
        num_engines=4,
        gpu_fraction=1.0,
        gpu_memory_utilization=0.85,
        max_model_len=131072,
        dtype="bfloat16",
        gdn_prefill_backend="triton",
        enforce_eager=True,
        inference_batch_size=16,
        rollout_token_budget=131072,
        max_total_tokens=0,
        eval_samples=4,
        eval_limit=100,
        aime_limit=30,
        max_turns=50,
        max_tokens=4096,
        vllm_default_max_tokens=4096,
        trim_context=False,
        temperature=1.0,
        top_p=1.0,
        top_k=40,
        min_p=0.0,
        presence_penalty=2.0,
        repetition_penalty=1.0,
        parameter_scope="full",
        seed=20260627,
        eval_seed=20260627,
        eval_data=str(module.DEFAULT_EVAL),
        aime_data=str(module.DEFAULT_AIME),
        eval_key_seed_file="/tmp/original_eval4_key_seed.jsonl",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("skip_initial_eval", True),
        ("reuse_initial_eval_history", "/tmp/reuse.json"),
        ("resume_history", "/tmp/resume.json"),
    ],
)
def test_formal_eval_rejects_unvalidated_shortcuts(field: str, value: object) -> None:
    module = load_module()
    options = formal_options(module)
    setattr(options, field, value)
    with pytest.raises(ValueError, match=field.replace("_", "-")):
        module.validate_formal_eval_options(options, ("dapo",))


@pytest.mark.parametrize(
    ("field", "value"),
    [("num_engines", 8), ("max_turns", 49), ("eval_samples", 3), ("trim_context", True)],
)
def test_formal_eval_rejects_wrong_core_settings(field: str, value: object) -> None:
    module = load_module()
    options = formal_options(module)
    setattr(options, field, value)
    with pytest.raises(ValueError, match=field):
        module.validate_formal_eval_options(options, ("dapo",))


def test_formal_eval_accepts_original_ray_vllm_settings() -> None:
    module = load_module()
    module.validate_formal_eval_options(formal_options(module), ("dapo", "aime"))


def test_formal_eval_requires_frozen_key_seed_file() -> None:
    module = load_module()
    options = formal_options(module)
    options.eval_key_seed_file = ""
    with pytest.raises(ValueError, match="eval-key-seed-file"):
        module.validate_formal_eval_options(options, ("dapo",))


def test_validate_engine_topologies_accepts_one_actor_per_gpu() -> None:
    module = load_module()
    topologies = [
        {
            "engine_index": index,
            "pid": 1000 + index,
            "ray_gpu_ids": [index],
            "actor_cuda_visible_devices": str(index),
            "worker_pid": 2000 + index,
            "worker_cuda_visible_devices": str(index),
            "worker_torch_device_count": 1,
            "worker_torch_device_name": "NVIDIA A100-SXM4-80GB",
            "worker_torch_device_uuid": f"uuid-{index}",
        }
        for index in range(4)
    ]
    module.validate_engine_topologies(topologies, expected_engines=4)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows.pop(),
        lambda rows: rows[1].update(actor_cuda_visible_devices="0"),
        lambda rows: rows[2].update(worker_torch_device_count=2),
        lambda rows: rows[3].update(worker_torch_device_uuid="uuid-0"),
    ],
)
def test_validate_engine_topologies_rejects_incomplete_or_shared_gpu_mapping(mutation) -> None:
    module = load_module()
    topologies = [
        {
            "engine_index": index,
            "pid": 1000 + index,
            "ray_gpu_ids": [index],
            "actor_cuda_visible_devices": str(index),
            "worker_pid": 2000 + index,
            "worker_cuda_visible_devices": str(index),
            "worker_torch_device_count": 1,
            "worker_torch_device_name": "NVIDIA A100-SXM4-80GB",
            "worker_torch_device_uuid": f"uuid-{index}",
        }
        for index in range(4)
    ]
    mutation(topologies)
    with pytest.raises(ValueError, match="engine topology"):
        module.validate_engine_topologies(topologies, expected_engines=4)


def test_worker_topology_reports_vllm_worker_physical_uuid(monkeypatch) -> None:
    import vllm_math_es_worker

    properties = SimpleNamespace(
        name="NVIDIA A100-SXM4-80GB",
        uuid="physical-uuid",
        pci_domain_id=0,
        pci_bus_id=57,
        pci_device_id=0,
    )
    monkeypatch.setattr(vllm_math_es_worker.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(vllm_math_es_worker.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        vllm_math_es_worker.torch.cuda,
        "get_device_properties",
        lambda _index: properties,
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")

    topology = vllm_math_es_worker.WorkerExtension().topology_math_es()

    assert topology["pid"] == os.getpid()
    assert topology["cuda_visible_devices"] == "3"
    assert topology["torch_device_count"] == 1
    assert topology["torch_device_uuid"] == "physical-uuid"
    assert topology["torch_device_pci"] == "0000:39:00"
