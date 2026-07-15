from __future__ import annotations

import base64
import json
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from urllib import request


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


def post_json(url: str, payload: dict, timeout: int = 900) -> dict:
    data = json.dumps(payload).encode()
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def completion_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/completions") or endpoint.endswith("/v1/chat/completions"):
        return endpoint
    return f"{endpoint}/completions"


def chat_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/v1/chat/completions"):
        return endpoint
    if endpoint.endswith("/completions"):
        return endpoint[: -len("/completions")] + "/v1/chat/completions"
    return f"{endpoint}/v1/chat/completions"


def encode_image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def response_text(response: dict) -> str:
    if isinstance(response.get("content"), list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in response["content"]
        )
    if response.get("content") is not None:
        return str(response["content"])
    choices = response.get("choices") or []
    if choices:
        choice = choices[0]
        if isinstance(choice.get("message"), dict):
            return str(choice["message"].get("content", ""))
        return str(choice.get("text", ""))
    return ""


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


def extract_answer(text: str) -> str:
    matches = re.findall(r"(?:final answer|answer)\s*[:：]\s*([^\n]+)", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def build_prompt(task: DocVQATask, skill: str = "") -> str:
    skill_block = f"\nSkill instructions:\n{skill.strip()}\n" if skill.strip() else ""
    return (
        "Answer the document visual question. Inspect the referenced document "
        "image carefully. Return only the short answer.\n"
        f"{skill_block}\n"
        f"Image path: {task.image}\n"
        f"Question: {task.question}\n"
        "Final answer:"
    )


def call_model(
    endpoint: str,
    *,
    task: DocVQATask,
    prompt: str,
    model: str,
    endpoint_mode: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: int,
) -> str:
    if endpoint_mode == "completion":
        response = post_json(
            completion_url(endpoint),
            {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": temperature > 0,
            },
            timeout=timeout,
        )
        return response_text(response)

    if endpoint_mode == "openai_chat":
        response = post_json(
            chat_url(endpoint),
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
            timeout=timeout,
        )
        return response_text(response)

    if endpoint_mode == "openai_vision_chat":
        image_path = Path(task.image)
        if not image_path.is_absolute():
            image_path = Path.cwd() / image_path
        content: list[dict] = [{"type": "text", "text": prompt}]
        if image_path.exists():
            content.append({"type": "image_url", "image_url": {"url": encode_image_data_url(image_path)}})
        response = post_json(
            chat_url(endpoint),
            {
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
            timeout=timeout,
        )
        return response_text(response)

    raise ValueError(f"Unsupported endpoint mode: {endpoint_mode}")


class DocVQAEnv:
    def __init__(self, data_path: str | Path, *, limit: int = 0, skill_file: str | Path | None = None) -> None:
        self.data_path = Path(data_path)
        self.tasks = load_tasks(self.data_path, limit)
        self.skill = ""
        if skill_file:
            path = Path(skill_file)
            if path.exists():
                self.skill = path.read_text(encoding="utf-8")

    def evaluate_task(
        self,
        *,
        endpoint: str,
        task: DocVQATask,
        model: str,
        endpoint_mode: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        timeout: int,
    ) -> dict:
        prompt = build_prompt(task, self.skill)
        try:
            response = call_model(
                endpoint,
                task=task,
                prompt=prompt,
                model=model,
                endpoint_mode=endpoint_mode,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                timeout=timeout,
            )
            prediction = extract_answer(response)
            score = anls(prediction, task.answers)
            return {
                "task_id": task.id,
                "score": score,
                "answers": task.answers,
                "prediction": prediction,
                "response": response,
                "image": task.image,
            }
        except Exception as exc:
            return {"task_id": task.id, "score": -1.0, "error": repr(exc), "image": task.image}
