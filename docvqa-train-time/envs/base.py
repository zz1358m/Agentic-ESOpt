from dataclasses import dataclass
from typing import Any, Dict, Protocol


@dataclass(frozen=True)
class EnvSpec:
    name: str
    family: str
    status: str
    adapter: str
    notes: str = ""


class Environment(Protocol):
    def reset(self, task: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def evaluate(self) -> Dict[str, Any]:
        ...
