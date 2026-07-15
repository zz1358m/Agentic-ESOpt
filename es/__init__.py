"""Repository-level Dynamic-Agent method implementation.

HTTP and torch integrations are loaded lazily so schedule/history utilities can
be used by data checks and ``--help`` commands without optional runtime wheels.
"""

from .registry import ES_METHOD_ALIASES, ES_METHOD_REGISTRY, ESMethodSpec, get_es_method_spec, list_es_methods
from .run_state import (
    SUPPORTED_SIGMA_SCHEDULES,
    atomic_write_history,
    completed_update_records,
    history_prefix_through_updates,
    history_output_path,
    map_endpoint_serial,
    normalize_sigma_schedule,
    read_history,
    replay_http_updates,
    resolve_warmup_steps,
    sigma_at_step,
    validate_es_run_shape,
    validate_seed_sequence,
)


def __getattr__(name: str):
    if name == "ModelESClient":
        from .model_es_client import ModelESClient

        return ModelESClient
    if name == "SeedReplayModelES":
        from .seeded_model_es import SeedReplayModelES

        return SeedReplayModelES
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ES_METHOD_REGISTRY",
    "ES_METHOD_ALIASES",
    "ESMethodSpec",
    "ModelESClient",
    "SeedReplayModelES",
    "SUPPORTED_SIGMA_SCHEDULES",
    "atomic_write_history",
    "completed_update_records",
    "history_prefix_through_updates",
    "get_es_method_spec",
    "history_output_path",
    "list_es_methods",
    "map_endpoint_serial",
    "normalize_sigma_schedule",
    "read_history",
    "replay_http_updates",
    "resolve_warmup_steps",
    "sigma_at_step",
    "validate_es_run_shape",
    "validate_seed_sequence",
]
