"""HTTP adapter for the local WebArena policy-model service."""

from __future__ import annotations

import os
from typing import Any

import requests


def _response_text(data: Any) -> str:
    if isinstance(data, dict) and data.get("choices"):
        choice = data["choices"][0]
        if isinstance(choice, dict):
            message = choice.get("message")
            if isinstance(message, dict):
                return str(message.get("content", ""))
            return str(choice.get("text", ""))
    if isinstance(data, dict) and isinstance(data.get("content"), list):
        return str(data["content"][0]) if data["content"] else ""
    if isinstance(data, dict):
        return str(data.get("text", "") or data.get("response", ""))
    return str(data)


def _chat_endpoint(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/v1/chat/completions") or endpoint.endswith("/chat/completions"):
        return endpoint
    if endpoint.endswith("/completions"):
        return endpoint[: -len("/completions")] + "/v1/chat/completions"
    if endpoint.endswith("/v1"):
        return endpoint + "/chat/completions"
    return endpoint + "/v1/chat/completions"


def generate_from_local_completion(
    prompt: str,
    endpoint: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    top_k: int | None = None,
    min_p: float | None = None,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    stop_token: str | None = None,
) -> str:
    params: dict[str, Any] = {
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "presence_penalty": presence_penalty,
        "repetition_penalty": repetition_penalty,
        "do_sample": temperature > 0,
        "use_chat_template": False,
    }
    if stop_token:
        params["stop"] = [stop_token]
    response = requests.post(
        endpoint,
        json={"prompt": prompt, "params": params, "repeat_prompt": 1},
        timeout=240,
    )
    response.raise_for_status()
    return _response_text(response.json())


def generate_from_local_chat_completion(
    messages: list[Any],
    endpoint: str,
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    top_k: int | None = None,
    min_p: float | None = None,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
) -> str:
    params: dict[str, Any] = {
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "presence_penalty": presence_penalty,
        "repetition_penalty": repetition_penalty,
        "do_sample": temperature > 0,
    }
    enable_thinking = os.environ.get("WEBRL_LOCAL_ENABLE_THINKING")
    if enable_thinking is not None:
        params["enable_thinking"] = enable_thinking
    response = requests.post(
        _chat_endpoint(endpoint),
        json={
            "model": model,
            "messages": messages,
            "params": params,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        },
        timeout=240,
    )
    response.raise_for_status()
    return _response_text(response.json())
