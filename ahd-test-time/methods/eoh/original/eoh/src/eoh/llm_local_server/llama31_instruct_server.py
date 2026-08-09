import gc
import inspect
import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path


default_model_path = "meta-llama/Llama-3.1-8B-Instruct"

parser = ArgumentParser()
parser.add_argument("--d", nargs="+", default=["0"], help="CUDA device ids, for example: --d 0 1 2 3")
parser.add_argument("--path", type=str, default=default_model_path, help="HF model id or local model path")
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=11012)
parser.add_argument("--quantization", default=False, action="store_true", help="Load the model in 8-bit")
parser.add_argument("--load-in-4bit", default=False, action="store_true", help="Load the model in 4-bit")
parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16"], default="float16")
parser.add_argument("--max-repeat-prompt", type=int, default=8)
parser.add_argument("--trust-remote-code", default=False, action="store_true")
parser.add_argument("--enable-lora", default=False, action="store_true")
parser.add_argument("--lora-r", type=int, default=8)
parser.add_argument("--lora-alpha", type=int, default=16)
parser.add_argument("--lora-dropout", type=float, default=0.0)
parser.add_argument(
    "--lora-target-modules",
    nargs="*",
    default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
parser.add_argument(
    "--chat-template-enable-thinking",
    choices=["auto", "true", "false"],
    default="auto",
    help="Qwen3-style thinking switch for chat templates. Use false for concise code-generation prompts.",
)
args = parser.parse_args()

if args.d and args.d != ["cpu"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.d)

import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LogitsProcessor

_repo_root = next(
    (
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "es").is_dir() and (parent / "PROJECT_LAYOUT.md").is_file()
    ),
    None,
)
if _repo_root is not None and str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
elif _repo_root is not None:
    sys.path.remove(str(_repo_root))
    sys.path.insert(0, str(_repo_root))

from algorithms.es import SeedReplayModelES

if args.enable_lora:
    from peft import LoraConfig, TaskType, get_peft_model


class PresencePenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, penalty):
        self.penalty = float(penalty)

    def __call__(self, input_ids, scores):
        if self.penalty == 0.0:
            return scores
        adjusted = scores.clone()
        for row_idx in range(input_ids.shape[0]):
            seen = torch.unique(input_ids[row_idx])
            adjusted[row_idx, seen] -= self.penalty
        return adjusted


def _torch_dtype(dtype_name):
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float16


def _quantization_config():
    if args.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    if args.quantization:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def _model_input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _eos_token_ids(tokenizer):
    eos_ids = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id, int) and eot_id >= 0 and eot_id not in eos_ids:
        eos_ids.append(eot_id)
    return eos_ids or None


def _parse_optional_bool(value):
    normalized = str(value).strip().lower()
    if normalized in {"auto", "none", ""}:
        return None
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Unsupported boolean value: {value!r}")


pretrained_model_path = args.path
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_path,
    trust_remote_code=args.trust_remote_code,
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_path,
    torch_dtype=_torch_dtype(args.dtype),
    quantization_config=_quantization_config(),
    device_map="auto",
    trust_remote_code=args.trust_remote_code,
)
if args.enable_lora:
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
model.eval()

app = Flask(__name__)
CORS(app)
model_es = SeedReplayModelES()


def _format_prompt(prompt, params):
    use_chat_template = params.get("use_chat_template", True)
    if not use_chat_template:
        return prompt

    system_prompt = params.get("system_prompt", "")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    enable_thinking = _parse_optional_bool(
        params.get("enable_thinking", args.chat_template_enable_thinking)
    )
    if enable_thinking is not None:
        try:
            signature = inspect.signature(tokenizer.apply_chat_template)
            supports_enable_thinking = "enable_thinking" in signature.parameters or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
        except (TypeError, ValueError):
            supports_enable_thinking = False
        if supports_enable_thinking:
            template_kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(messages, **template_kwargs)


def _completion_params_from_payload(payload):
    params = dict(payload.get("params", {}) or {})
    if "max_tokens" in payload and "max_new_tokens" not in params:
        params["max_new_tokens"] = payload["max_tokens"]
    if "max_completion_tokens" in payload and "max_new_tokens" not in params:
        params["max_new_tokens"] = payload["max_completion_tokens"]
    for key in (
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "do_sample",
        "stop",
        "repetition_penalty",
        "presence_penalty",
        "enable_thinking",
        "use_chat_template",
        "system_prompt",
    ):
        if key in payload and key not in params:
            params[key] = payload[key]
    return params


def _generate_from_formatted_prompt(formatted_prompt, params, repeat_prompt):
    print("========================================== Prompt ==========================================")
    print(formatted_prompt)
    print("============================================================================================")

    max_new_tokens = params.get("max_new_tokens", 768)
    do_sample = params.get("do_sample", True)
    temperature = params.get("temperature", 0.6)
    top_k = params.get("top_k", None)
    top_p = params.get("top_p", 0.9)
    min_p = params.get("min_p", None)
    repetition_penalty = params.get("repetition_penalty", None)
    presence_penalty = params.get("presence_penalty", 0.0)
    num_return_sequences = params.get("num_return_sequences", 1)
    eos_token_id = params.get("eos_token_id", _eos_token_ids(tokenizer))
    pad_token_id = params.get("pad_token_id", tokenizer.pad_token_id)

    while True:
        input_texts = [formatted_prompt] * repeat_prompt
        inputs = tokenizer(input_texts, return_tensors="pt", padding=True)
        inputs = {name: value.to(_model_input_device(model)) for name, value in inputs.items()}

        if temperature is not None and float(temperature) <= 0:
            do_sample = False

        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": num_return_sequences,
            "pad_token_id": pad_token_id,
        }
        if eos_token_id is not None:
            generate_kwargs["eos_token_id"] = eos_token_id
        if do_sample:
            if temperature is not None:
                generate_kwargs["temperature"] = temperature
            if top_k is not None:
                generate_kwargs["top_k"] = top_k
            if top_p is not None:
                generate_kwargs["top_p"] = top_p
            if min_p is not None:
                generate_kwargs["min_p"] = min_p
        if repetition_penalty is not None:
            generate_kwargs["repetition_penalty"] = repetition_penalty
        if float(presence_penalty or 0.0) != 0.0:
            generate_kwargs["logits_processor"] = [PresencePenaltyLogitsProcessor(presence_penalty)]

        try:
            with torch.inference_mode():
                output = model.generate(**inputs, **generate_kwargs)
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            if torch.cuda.device_count() > 0:
                torch.cuda.empty_cache()
            if repeat_prompt == 1:
                raise
            repeat_prompt = max(repeat_prompt // 2, 1)
            continue

        prompt_len = inputs["input_ids"].shape[1]
        content = [
            tokenizer.decode(out[prompt_len:], skip_special_tokens=True).strip()
            for out in output
        ]

        print("======================================== Response Content ========================================")
        print(content)
        print("==================================================================================================")

        gc.collect()
        if torch.cuda.device_count() > 0:
            torch.cuda.empty_cache()

        return content, int(inputs["input_ids"].numel()), int(output.numel() - inputs["input_ids"].numel())


@app.route("/completions", methods=["POST"])
def completions():
    payload = request.get_json(force=True)
    prompt = payload["prompt"]
    params = _completion_params_from_payload(payload)
    repeat_prompt = min(int(payload.get("repeat_prompt", 1)), args.max_repeat_prompt)

    formatted_prompt = _format_prompt(prompt, params)
    content, _, _ = _generate_from_formatted_prompt(formatted_prompt, params, repeat_prompt)
    return jsonify({"content": content})

@app.route("/v1/chat/completions", methods=["POST"])
def openai_chat_completions():
    payload = request.get_json(force=True)
    messages = payload.get("messages") or []
    params = _completion_params_from_payload(payload)
    repeat_prompt = min(int(payload.get("repeat_prompt", 1)), args.max_repeat_prompt)
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    enable_thinking = _parse_optional_bool(
        params.get("enable_thinking", args.chat_template_enable_thinking)
    )
    if enable_thinking is not None:
        try:
            signature = inspect.signature(tokenizer.apply_chat_template)
            supports_enable_thinking = "enable_thinking" in signature.parameters or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
        except (TypeError, ValueError):
            supports_enable_thinking = False
        if supports_enable_thinking:
            template_kwargs["enable_thinking"] = enable_thinking
    formatted_prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
    content, prompt_tokens, completion_tokens = _generate_from_formatted_prompt(
        formatted_prompt,
        params,
        repeat_prompt,
    )
    return jsonify(
        {
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model", args.path),
            "choices": [
                {
                    "index": index,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
                for index, text in enumerate(content)
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    )

@app.route("/es/init", methods=["POST"])
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


@app.route("/es/apply", methods=["POST"])
def es_apply():
    payload = request.get_json(force=True) or {}
    seed = int(payload["seed"])
    sigma = float(payload["sigma"])
    return jsonify(model_es.apply(seed=seed, sigma=sigma))


@app.route("/es/revert", methods=["POST"])
def es_revert():
    payload = request.get_json(force=True) or {}
    seed = int(payload["seed"])
    sigma = float(payload["sigma"])
    return jsonify(model_es.revert(seed=seed, sigma=sigma))


@app.route("/es/update", methods=["POST"])
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


@app.route("/es/reset", methods=["POST"])
def es_reset():
    return jsonify(model_es.reset())


@app.route("/es/status", methods=["POST"])
def es_status():
    return jsonify(model_es.status())


if __name__ == "__main__":
    app.run(host=args.host, port=args.port, threaded=False)
