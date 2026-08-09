from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DocVQATask:
    id: str
    question: str
    answers: list[str]
    image: str
    source: str


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def resolve_image_path(value: str, data_path: Path) -> Path:
    """Resolve dataset image paths written on either Windows or Linux."""

    image = Path(str(value).replace("\\", "/")).expanduser()
    if image.is_absolute():
        return image.resolve()
    data_path = data_path.expanduser().resolve()
    candidates = [data_path.parent / image, Path.cwd() / image]
    candidates.extend(parent / image for parent in data_path.parents)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (Path.cwd() / image).resolve()


def load_tasks(path: Path, limit: int = 0) -> list[DocVQATask]:
    path = path.expanduser().resolve()
    tasks = [
        DocVQATask(
            id=str(row.get("id", idx)),
            question=str(row.get("question", "")),
            answers=[str(answer) for answer in row.get("answers", [])],
            image=str(resolve_image_path(str(row.get("image", "")), path)),
            source=str(row.get("source", "")),
        )
        for idx, row in enumerate(read_jsonl(path))
    ]
    if limit > 0:
        tasks = tasks[:limit]
    if not tasks:
        raise RuntimeError(f"No DocVQA tasks loaded from {path}")
    return tasks

def normalize_text(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if ca == cb else 1),
                )
            )
        previous = current
    return previous[-1]


def anls(prediction: str, answers: list[str]) -> float:
    pred = normalize_text(prediction)
    if not pred or not answers:
        return 0.0
    scores = []
    for answer in answers:
        ans = normalize_text(answer)
        if not ans:
            continue
        distance = levenshtein(pred, ans)
        norm = distance / max(len(pred), len(ans), 1)
        scores.append(1.0 - norm if norm < 0.5 else 0.0)
    return max(scores) if scores else 0.0


class DocVQAEnv:
    def __init__(self, data_path: str | Path, *, limit: int = 0, skill_file: str | Path | None = None) -> None:
        self.data_path = Path(data_path)
        self.tasks = load_tasks(self.data_path, limit)
        self.skill = ""
        if skill_file:
            path = Path(skill_file)
            if path.exists():
                self.skill = path.read_text(encoding="utf-8")
