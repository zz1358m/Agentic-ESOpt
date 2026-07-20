#!/usr/bin/env python3
"""Convert Qwen3.5 multimodal HF checkpoint to a text-only Qwen3Next checkpoint.

The released Qwen3.5 checkpoint uses model_type=qwen3_5, which this GRPO
environment's Transformers build does not register. The language backbone uses
Qwen3Next-compatible text_config and stores its weights under
``model.language_model.*``. This script writes a CausalLM checkpoint with
model_type=qwen3_next and weights renamed to ``model.*``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


COPY_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "README.md",
    "LICENSE",
]


def build_text_config(src: Path) -> dict:
    source_config = json.loads((src / "config.json").read_text(encoding="utf-8"))
    text_config = dict(source_config["text_config"])
    text_config["model_type"] = "qwen3_next"
    text_config["architectures"] = ["Qwen3NextForCausalLM"]
    text_config["tie_word_embeddings"] = source_config.get(
        "tie_word_embeddings", text_config.get("tie_word_embeddings", True)
    )
    # Qwen3.5-4B is dense. Qwen3NextConfig otherwise fills in MoE defaults
    # (512 experts), which silently constructs a ~67B model.
    text_config["num_experts"] = 0
    text_config["num_experts_per_tok"] = 0
    text_config["decoder_sparse_step"] = 1
    text_config["mlp_only_layers"] = list(range(int(text_config["num_hidden_layers"])))
    text_config.pop("mtp_num_hidden_layers", None)
    text_config.pop("mtp_use_dedicated_embeddings", None)

    rope_parameters = text_config.pop("rope_parameters", None)
    if rope_parameters is not None:
        if "rope_theta" in rope_parameters:
            text_config["rope_theta"] = rope_parameters["rope_theta"]
        if "partial_rotary_factor" in rope_parameters:
            text_config["partial_rotary_factor"] = rope_parameters["partial_rotary_factor"]

    return text_config


def _fuse_linear_attention(parts: dict[str, torch.Tensor], config: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack Qwen3.5's separate qkv/z and b/a projections as Qwen3Next expects."""
    qkv = parts["qkv"]
    z = parts["z"]
    b = parts["b"]
    a = parts["a"]
    num_k_heads = int(config["linear_num_key_heads"])
    num_v_heads = int(config["linear_num_value_heads"])
    key_dim = int(config["linear_key_head_dim"])
    value_dim = int(config["linear_value_head_dim"])
    value_heads_per_key = num_v_heads // num_k_heads

    q_size = num_k_heads * key_dim
    k_size = num_k_heads * key_dim
    v_size = num_v_heads * value_dim
    if qkv.shape[0] != q_size + k_size + v_size:
        raise ValueError(f"Unexpected in_proj_qkv shape: {tuple(qkv.shape)}")
    q, k, v = torch.split(qkv, [q_size, k_size, v_size], dim=0)
    q = q.reshape(num_k_heads, key_dim, qkv.shape[1])
    k = k.reshape(num_k_heads, key_dim, qkv.shape[1])
    v = v.reshape(num_k_heads, value_heads_per_key * value_dim, qkv.shape[1])
    z = z.reshape(num_k_heads, value_heads_per_key * value_dim, qkv.shape[1])
    qkvz = torch.cat((q, k, v, z), dim=1).reshape(-1, qkv.shape[1]).contiguous()

    b = b.reshape(num_k_heads, value_heads_per_key, b.shape[1])
    a = a.reshape(num_k_heads, value_heads_per_key, a.shape[1])
    ba = torch.cat((b, a), dim=1).reshape(-1, b.shape[2]).contiguous()
    return qkvz, ba


def convert_shard(shard: Path, out_shard: Path, config: dict) -> dict[str, str]:
    tensors = {}
    index_entries = {}
    linear_parts: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(shard, framework="pt", device="cpu") as reader:
        for key in reader.keys():
            if not key.startswith("model.language_model."):
                continue
            new_key = "model." + key[len("model.language_model.") :]
            matched = False
            for suffix, part in (
                (".linear_attn.in_proj_qkv.weight", "qkv"),
                (".linear_attn.in_proj_z.weight", "z"),
                (".linear_attn.in_proj_b.weight", "b"),
                (".linear_attn.in_proj_a.weight", "a"),
            ):
                if new_key.endswith(suffix):
                    base = new_key[: -len(suffix)] + ".linear_attn"
                    linear_parts.setdefault(base, {})[part] = reader.get_tensor(key)
                    matched = True
                    break
            if not matched:
                tensors[new_key] = reader.get_tensor(key)

    for base, parts in linear_parts.items():
        missing = {"qkv", "z", "b", "a"} - set(parts)
        if missing:
            raise ValueError(f"Linear-attention projections split across shards for {base}: missing {sorted(missing)}")
        qkvz, ba = _fuse_linear_attention(parts, config)
        tensors[f"{base}.in_proj_qkvz.weight"] = qkvz
        tensors[f"{base}.in_proj_ba.weight"] = ba

    for key in tensors:
        index_entries[key] = out_shard.name
    if tensors:
        save_file(tensors, out_shard)
    return index_entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        default=os.environ.get("QWEN35_SOURCE_MODEL"),
        help="Source Qwen3.5 checkpoint (or set QWEN35_SOURCE_MODEL).",
    )
    parser.add_argument(
        "--dst",
        default=os.environ.get("QWEN35_TEXT_MODEL"),
        help="Destination checkpoint; defaults to <src>-text (or set QWEN35_TEXT_MODEL).",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Refresh config/tokenizer metadata without rewriting model shards.",
    )
    args = parser.parse_args()

    if not args.src:
        parser.error("--src is required unless QWEN35_SOURCE_MODEL is set")
    src = Path(args.src).expanduser().resolve()
    dst = (
        Path(args.dst).expanduser().resolve()
        if args.dst
        else src.with_name(f"{src.name}-text")
    )
    dst.mkdir(parents=True, exist_ok=True)

    text_config = build_text_config(src)
    (dst / "config.json").write_text(json.dumps(text_config, indent=2) + "\n", encoding="utf-8")
    for name in COPY_FILES:
        path = src / name
        if path.exists():
            shutil.copy2(path, dst / name)

    # Transformers 4.57 detects the legacy Mistral/Qwen2 pre-tokenizer regex
    # and otherwise warns that tokenization is incorrect. Persist the fix in
    # model metadata so both verl and the independent SGLang tokenizer agree.
    tokenizer_config_path = dst / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        tokenizer_config["fix_mistral_regex"] = True
        tokenizer_config_path.write_text(json.dumps(tokenizer_config, indent=2) + "\n", encoding="utf-8")

    if args.metadata_only:
        print(json.dumps({"dst": str(dst), "metadata_only": True}, indent=2))
        return

    weight_map = {}
    for shard in sorted(src.glob("model*.safetensors")):
        shard_map = convert_shard(shard, dst / shard.name, text_config)
        weight_map.update(shard_map)

    total_size = sum((dst / shard).stat().st_size for shard in set(weight_map.values()))
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dst": str(dst), "weights": len(weight_map), "total_size": total_size}, indent=2))


if __name__ == "__main__":
    main()
