from dataclasses import dataclass


@dataclass(frozen=True)
class ESMethodSpec:
    name: str
    status: str
    module: str
    target_env_families: tuple[str, ...]
    built_on: tuple[str, ...]
    notes: str = ""


ES_METHOD_REGISTRY = {
    "dynamic_agent": ESMethodSpec(
        name="dynamic_agent",
        status="maintained",
        module="es.model_es_client",
        target_env_families=(
            "sudoku",
            "math",
            "docvqa",
            "webarena",
            "ahd_test_time",
        ),
        built_on=("seed_replay_model_updates", "eoh"),
        notes="Dynamic-Agent model-weight optimization with explicit sigma schedules and deterministic history replay.",
    ),
}

ES_METHOD_ALIASES = {
    "model_weight_es": "dynamic_agent",
}


def list_es_methods():
    return list(ES_METHOD_REGISTRY.values())


def get_es_method_spec(name: str) -> ESMethodSpec:
    key = str(name).strip().lower().replace("-", "_")
    key = ES_METHOD_ALIASES.get(key, key)
    if key not in ES_METHOD_REGISTRY:
        supported = ", ".join(sorted((*ES_METHOD_REGISTRY.keys(), *ES_METHOD_ALIASES.keys())))
        raise KeyError(f"Unknown Dynamic-Agent method {name!r}. Supported methods: {supported}")
    return ES_METHOD_REGISTRY[key]
