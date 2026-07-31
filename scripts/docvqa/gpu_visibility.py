#!/usr/bin/env python3
"""Resolve the requested physical NVIDIA indices to stable CUDA UUIDs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import NamedTuple


DOCVQA_GPU_COUNT = 4


class GpuIdentity(NamedTuple):
    index: int
    uuid: str
    name: str


def _parse_physical_ids(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"physical GPU ids must be numeric, got {value!r}") from exc


def validate_docvqa_physical_ids(value: str) -> tuple[int, ...]:
    ids = _parse_physical_ids(value)
    if len(ids) != DOCVQA_GPU_COUNT or len(set(ids)) != DOCVQA_GPU_COUNT:
        raise ValueError(f"DocVQA requires exactly four unique physical GPU ids, got {value!r}")
    if any(index < 0 for index in ids):
        raise ValueError("physical GPU ids must be non-negative")
    return ids


def last_four_physical_ids(available: dict[int, GpuIdentity]) -> tuple[int, ...]:
    """Select the highest four physical indices reported by NVIDIA."""
    indices = sorted(available)
    if len(indices) < DOCVQA_GPU_COUNT:
        raise ValueError(f"DocVQA requires at least four GPUs, found {len(indices)}")
    return tuple(indices[-DOCVQA_GPU_COUNT:])


def parse_nvidia_smi_query(output: str) -> dict[int, GpuIdentity]:
    identities: dict[int, GpuIdentity] = {}
    for line_number, raw_line in enumerate(output.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) != 3:
            raise ValueError(f"invalid nvidia-smi row {line_number}: {raw_line!r}")
        try:
            index = int(fields[0])
        except ValueError as exc:
            raise ValueError(f"invalid physical GPU index on row {line_number}: {fields[0]!r}") from exc
        if index in identities:
            raise ValueError(f"duplicate physical GPU index {index} in nvidia-smi output")
        uuid = fields[1]
        if not uuid.startswith("GPU-"):
            raise ValueError(f"invalid GPU UUID on row {line_number}: {uuid!r}")
        identities[index] = GpuIdentity(index=index, uuid=uuid, name=fields[2])
    return identities


def query_nvidia_smi() -> str:
    process = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout


def _parse_expected_uuids(value: str, *, expected_count: int) -> tuple[str, ...]:
    uuids = tuple(part.strip() for part in value.split(",") if part.strip())
    if len(uuids) != expected_count or len(set(uuids)) != len(uuids):
        raise ValueError(f"expected UUIDs must contain exactly {expected_count} unique GPUs")
    if any(not uuid.startswith("GPU-") for uuid in uuids):
        raise ValueError("every expected GPU UUID must start with GPU-")
    return uuids


def resolve_physical_gpus(
    value: str,
    *,
    query_output: str | None = None,
    expected_uuids: str | None = None,
) -> list[GpuIdentity]:
    available = parse_nvidia_smi_query(query_nvidia_smi() if query_output is None else query_output)
    requested = (
        last_four_physical_ids(available)
        if value.strip().lower() in ("", "auto")
        else validate_docvqa_physical_ids(value)
    )
    if expected_uuids:
        requested_uuids = _parse_expected_uuids(
            expected_uuids,
            expected_count=len(requested),
        )
        available_by_uuid = {identity.uuid: identity for identity in available.values()}
        missing = [uuid for uuid in requested_uuids if uuid not in available_by_uuid]
        if missing:
            raise ValueError(f"expected DocVQA GPU UUIDs were not found: {missing}")
        # UUID selection remains stable if container-visible indices compact
        # after a restart. Preserve the requested labels in the manifest.
        return [
            GpuIdentity(index=physical_id, uuid=uuid, name=available_by_uuid[uuid].name)
            for physical_id, uuid in zip(requested, requested_uuids, strict=True)
        ]
    resolved = []
    for index in requested:
        if index not in available:
            raise ValueError(f"physical GPU {index} was not found in nvidia-smi output")
        resolved.append(available[index])
    uuids = [identity.uuid for identity in resolved]
    if len(set(uuids)) != len(uuids):
        raise ValueError("nvidia-smi returned duplicate UUIDs for the requested physical GPUs")
    return resolved


def cuda_visible_devices(identities: list[GpuIdentity]) -> str:
    return ",".join(identity.uuid for identity in identities)


def manifest(identities: list[GpuIdentity]) -> dict[str, object]:
    return {
        "physical_gpu_ids": [identity.index for identity in identities],
        "cuda_visible_devices": cuda_visible_devices(identities),
        "gpus": [identity._asdict() for identity in identities],
        "status": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--physical-devices",
        default=os.environ.get("DOCVQA_PHYSICAL_GPU_IDS", "auto"),
        help="Four comma-separated indices, or 'auto' to select the last four GPUs.",
    )
    parser.add_argument("--expected-uuids", default="")
    parser.add_argument("--format", choices=("cuda", "physical", "json"), default="cuda")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    identities = resolve_physical_gpus(
        args.physical_devices,
        expected_uuids=args.expected_uuids or None,
    )
    payload = manifest(identities)
    if args.format == "cuda":
        text = cuda_visible_devices(identities)
    elif args.format == "physical":
        text = ",".join(str(identity.index) for identity in identities)
    else:
        text = json.dumps(payload, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
