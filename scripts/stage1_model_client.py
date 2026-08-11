"""Shared HTTP and prompt helpers for Stage 1 checkpoint smoke runners."""

from __future__ import annotations

import inspect
import json
from urllib import request


def render_prompt(tokenizer: object, messages: list[dict]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        signature = inspect.signature(tokenizer.apply_chat_template)
        supports_enable_thinking = "enable_thinking" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
    except (TypeError, ValueError):
        supports_enable_thinking = False
    if supports_enable_thinking:
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


def build_sampling_payload(*, model: str, prompt: str, max_tokens: int) -> dict:
    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "max_new_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 40,
        "min_p": 0.0,
        "presence_penalty": 2.0,
        "repetition_penalty": 1.0,
        "do_sample": True,
        "use_chat_template": False,
    }


def call_completion(endpoint: str, payload: dict, *, timeout: int) -> str:
    body = json.dumps(payload).encode()
    req = request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        result = json.loads(response.read().decode())
    content = result.get("content")
    if isinstance(content, list):
        return str(content[0]) if content else ""
    if content is not None:
        return str(content)
    choices = result.get("choices") or []
    return str(choices[0].get("text", "")) if choices else ""
