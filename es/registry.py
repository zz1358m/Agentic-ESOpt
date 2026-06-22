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
        status="experimental",
        module="es.model_es_client",
        target_env_families=(
            "ahd_construct",
            "ahd_aco",
            "agent_interactive_text",
            "agent_web",
        ),
        built_on=("eoh", "jitrl"),
        notes="Dynamic-Agent: seed-replay model-weight update built on top of setting methods. Concrete support is enabled per setting.",
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
