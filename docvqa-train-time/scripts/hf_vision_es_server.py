#!/usr/bin/env python3
"""OpenAI-compatible Hugging Face vision server with Dynamic-Agent routes."""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, help="HF model id or local vision-language checkpoint.")
    parser.add_argument("--d", nargs="+", default=["0"], help="CUDA device ids, or 'cpu'.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11013)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--model-class",
        choices=["auto", "image-text-to-text", "vision2seq"],
        default="auto",
        help="Transformers auto-model family. 'auto' tries both multimodal families.",
    )
    parser.add_argument("--attn-implementation", default="")
    parser.add_argument("--max-image-bytes", type=int, default=32 * 1024 * 1024)
    args = parser.parse_args()
    if args.max_image_bytes <= 0:
        parser.error("--max-image-bytes must be positive")
    return args


def decode_data_image(url: str, *, image_module: Any, max_bytes: int):
    if not str(url).startswith("data:image/") or "," not in str(url):
        raise ValueError("Vision requests must use a data:image/...;base64 URL.")
    header, encoded = str(url).split(",", 1)
    if ";base64" not in header.lower():
        raise ValueError("Vision image data URL must be base64 encoded.")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("Invalid base64 image payload.") from exc
    if len(payload) > int(max_bytes):
        raise ValueError(f"Image payload exceeds the {int(max_bytes)} byte limit.")
    image = image_module.open(io.BytesIO(payload))
    image.load()
    return image.convert("RGB")


def normalize_messages(
    messages: list[dict[str, Any]],
    *,
    image_module: Any,
    max_image_bytes: int,
) -> tuple[list[dict[str, Any]], list[Any]]:
    normalized_messages: list[dict[str, Any]] = []
    images: list[Any] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        if isinstance(content, str):
            normalized_messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            normalized_messages.append({"role": role, "content": str(content)})
            continue
        normalized_content: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                normalized_content.append({"type": "text", "text": str(part)})
                continue
            part_type = str(part.get("type", "text"))
            if part_type == "text":
                normalized_content.append({"type": "text", "text": str(part.get("text", ""))})
                continue
            if part_type == "image_url":
                image_value = part.get("image_url", {})
                url = image_value.get("url", "") if isinstance(image_value, dict) else image_value
                image = decode_data_image(
                    str(url),
                    image_module=image_module,
                    max_bytes=max_image_bytes,
                )
                images.append(image)
                # The decoded PIL image is passed separately to the processor.
                # Keeping only the content marker makes this work across chat
                # templates that cannot serialize a PIL object.
                normalized_content.append({"type": "image"})
                continue
            raise ValueError(f"Unsupported OpenAI content part type: {part_type!r}")
        normalized_messages.append({"role": role, "content": normalized_content})
    return normalized_messages, images


def torch_dtype(torch_module: Any, name: str):
    if name == "auto":
        return "auto"
    return getattr(torch_module, name)


def model_input_device(model: Any, torch_module: Any):
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch_module.device("cuda:0" if torch_module.cuda.is_available() else "cpu")


def load_model(transformers_module: Any, args: argparse.Namespace, model_kwargs: dict[str, Any]):
    class_names = {
        "image-text-to-text": ["AutoModelForImageTextToText"],
        "vision2seq": ["AutoModelForVision2Seq"],
        "auto": ["AutoModelForImageTextToText", "AutoModelForVision2Seq"],
    }[args.model_class]
    errors: list[str] = []
    for class_name in class_names:
        model_class = getattr(transformers_module, class_name, None)
        if model_class is None:
            errors.append(f"{class_name}: unavailable in installed transformers")
            continue
        try:
            return model_class.from_pretrained(args.path, **model_kwargs)
        except Exception as exc:
            errors.append(f"{class_name}: {exc!r}")
    raise RuntimeError("Could not load the vision-language model:\n" + "\n".join(errors))


def create_runtime(args: argparse.Namespace):
    if args.d and args.d != ["cpu"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.d)

    import torch
    import transformers
    from flask import Flask, jsonify, request
    from flask_cors import CORS
    from PIL import Image
    from transformers import AutoProcessor

    from algorithms.es.seeded_model_es import SeedReplayModelES

    processor = AutoProcessor.from_pretrained(
        args.path,
        trust_remote_code=args.trust_remote_code,
    )
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype(torch, args.dtype),
        "device_map": "auto",
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = load_model(transformers, args, model_kwargs)
    model.eval()
    model_es = SeedReplayModelES()

    app = Flask(__name__)
    CORS(app)

    def generate(messages: list[dict[str, Any]], payload: dict[str, Any]) -> tuple[list[str], int, int]:
        normalized, images = normalize_messages(
            messages,
            image_module=Image,
            max_image_bytes=args.max_image_bytes,
        )
        prompt = processor.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=True,
        )
        processor_kwargs: dict[str, Any] = {
            "text": [prompt],
            "padding": True,
            "return_tensors": "pt",
        }
        if images:
            processor_kwargs["images"] = images
        inputs = processor(**processor_kwargs)
        device = model_input_device(model, torch)
        inputs = inputs.to(device)

        temperature = float(payload.get("temperature", 0.0) or 0.0)
        max_new_tokens = int(payload.get("max_tokens", payload.get("max_completion_tokens", 512)))
        if max_new_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
        }
        if generation_kwargs["do_sample"]:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = float(payload.get("top_p", 1.0))
            if payload.get("top_k") is not None:
                generation_kwargs["top_k"] = int(payload["top_k"])

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_kwargs)
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        generated_ids = (
            output_ids
            if bool(getattr(model.config, "is_encoder_decoder", False))
            else output_ids[:, prompt_tokens:]
        )
        texts = [text.strip() for text in processor.batch_decode(generated_ids, skip_special_tokens=True)]
        completion_tokens = int(generated_ids.numel())
        return texts, prompt_tokens, completion_tokens

    @app.errorhandler(ValueError)
    def handle_value_error(exc):
        return jsonify({"ok": False, "error": str(exc)}), 400

    @app.errorhandler(Exception)
    def handle_error(exc):
        app.logger.exception("request failed")
        return jsonify({"ok": False, "error": repr(exc)}), 500

    @app.get("/health")
    def health():
        return jsonify({"ok": True, "model": args.path, "vision": True})

    @app.post("/v1/chat/completions")
    def chat_completions():
        payload = request.get_json(force=True) or {}
        texts, prompt_tokens, completion_tokens = generate(payload.get("messages") or [], payload)
        return jsonify(
            {
                "id": "chatcmpl-local-vision",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": payload.get("model", args.path),
                "choices": [
                    {
                        "index": index,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                    for index, text in enumerate(texts)
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
        )

    @app.post("/completions")
    def completions():
        payload = request.get_json(force=True) or {}
        texts, _, _ = generate(
            [{"role": "user", "content": str(payload.get("prompt", ""))}],
            payload,
        )
        return jsonify({"content": texts})

    @app.post("/es/init")
    def es_init():
        payload = request.get_json(force=True) or {}
        return jsonify(
            model_es.init(
                model,
                parameter_scope=payload.get("parameter_scope", "full"),
                target_modules=payload.get("target_modules"),
                verbose=payload.get("verbose", True),
            )
        )

    @app.post("/es/apply")
    def es_apply():
        payload = request.get_json(force=True) or {}
        return jsonify(model_es.apply(seed=int(payload["seed"]), sigma=float(payload["sigma"])))

    @app.post("/es/revert")
    def es_revert():
        payload = request.get_json(force=True) or {}
        return jsonify(model_es.revert(seed=int(payload["seed"]), sigma=float(payload["sigma"])))

    @app.post("/es/update")
    def es_update():
        payload = request.get_json(force=True) or {}
        return jsonify(
            model_es.update(
                seeds=payload["seeds"],
                rewards=payload["rewards"],
                alpha=float(payload["alpha"]),
                reward_normalization=payload.get("reward_normalization", "zscore"),
                reward_normalization_ddof=int(payload.get("reward_normalization_ddof", 0)),
                reward_normalization_eps=float(payload.get("reward_normalization_eps", 1e-8)),
            )
        )

    @app.post("/es/reset")
    def es_reset():
        return jsonify(model_es.reset())

    @app.route("/es/status", methods=["GET", "POST"])
    def es_status():
        return jsonify(model_es.status())

    return app


def main() -> None:
    args = parse_args()
    app = create_runtime(args)
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
