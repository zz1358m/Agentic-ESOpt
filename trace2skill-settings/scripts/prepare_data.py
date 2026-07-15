#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("ROOT", Path(__file__).resolve().parents[2])).resolve()
DEFAULT_MATH_OUT = ROOT / "data/trace2skill/math_reasoning"
DEFAULT_DOCVQA_OUT = ROOT / "data/trace2skill/docvqa"
DEFAULT_DAPO_DATASET = "BytedTsinghua-SIA/DAPO-Math-17k"
DEFAULT_AIME_DATASET = "MathArena/aime_2026"
DEFAULT_DOCVQA_DATASET = "lmms-lab/DocVQA"
DEFAULT_DOCVQA_CONFIG = "DocVQA"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no} is {type(value).__name__}, expected object")
        rows.append(value)
    return rows


def read_json(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if isinstance(value, list):
        rows = value
    elif isinstance(value, dict):
        for key in ("data", "items", "examples", "rows"):
            if isinstance(value.get(key), list):
                rows = value[key]
                break
        else:
            rows = [value]
    else:
        raise ValueError(f"{path} is {type(value).__name__}, expected array or object")
    bad = [type(row).__name__ for row in rows if not isinstance(row, dict)]
    if bad:
        raise ValueError(f"{path} contains non-object rows: {bad[:3]}")
    return rows


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(path)
    if suffix == ".json":
        return read_json(path)
    if suffix == ".csv":
        return read_csv(path)
    raise ValueError(f"Unsupported input format for {path}; use .jsonl, .json, or .csv")


def first_value(row: dict[str, Any], paths: tuple[str, ...]) -> Any:
    for path in paths:
        value: Any = row
        found = True
        for part in path.split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                found = False
                break
        if found and value not in (None, ""):
            return value
    return ""


def question_from_messages(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    return text_from_messages(messages)


def text_from_messages(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").lower()
        if role and role not in {"user", "human"}:
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
    return "\n".join(parts)


def stringify(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def normalize_math_rows(rows: list[dict[str, Any]], *, split_name: str) -> list[dict[str, str]]:
    tasks: list[dict[str, str]] = []
    for idx, row in enumerate(rows):
        raw_question = first_value(
            row,
            (
                "question",
                "problem",
                "prompt",
                "instruction",
                "query",
                "input",
            ),
        )
        question = text_from_messages(raw_question) if isinstance(raw_question, list) else stringify(raw_question)
        if not question:
            question = question_from_messages(row)
        answer = stringify(
            first_value(
                row,
                (
                    "answer",
                    "final_answer",
                    "ground_truth",
                    "target",
                    "label",
                    "reward_model.ground_truth",
                    "reward_model.answer",
                ),
            )
        )
        task_id = stringify(first_value(row, ("id", "task_id", "question_id", "uid"))) or f"{split_name}_{idx}"
        source = stringify(first_value(row, ("source", "dataset", "data_source", "source_dataset"))) or split_name
        if question and answer:
            tasks.append({"id": task_id, "question": question, "answer": answer, "source": source})
    if not tasks:
        raise ValueError(
            f"No usable math rows found for {split_name}. "
            "Expected question/problem/prompt plus answer/final_answer/ground_truth fields."
        )
    return tasks


def load_hf_dataset_rows(dataset_name: str, *, split: str = "") -> list[dict[str, Any]]:
    try:
        from datasets import DatasetDict, load_dataset
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("Python package 'datasets' is required for automatic HuggingFace loading.") from exc

    if split:
        dataset = load_dataset(dataset_name, split=split)
        return [dict(row) for row in dataset]

    dataset = load_dataset(dataset_name)
    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            selected = dataset["train"]
        else:
            first_key = next(iter(dataset.keys()))
            selected = dataset[first_key]
    else:
        selected = dataset
    return [dict(row) for row in selected]


def limited(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    return rows[:limit] if limit > 0 else rows


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def jsonl_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def prepare_math_reasoning(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    targets = {
        "train": output_dir / "dapo_evolve.jsonl",
        "eval": output_dir / "dapo_test.jsonl",
        "aime": output_dir / "aime_2026.jsonl",
    }
    sources = {
        "train": Path(args.train_source).resolve() if args.train_source else None,
        "eval": Path(args.eval_source).resolve() if args.eval_source else None,
        "aime": Path(args.aime_source).resolve() if args.aime_source else None,
    }

    if all(jsonl_count(path) > 0 for path in targets.values()) and not any(sources.values()):
        print(f"[skip] math_reasoning: existing non-empty splits under {output_dir}")
        return

    need_train = sources["train"] is not None or not targets["train"].exists()
    need_eval = sources["eval"] is not None or not targets["eval"].exists()
    need_aime = sources["aime"] is not None or not targets["aime"].exists()

    if (need_train or need_eval) and sources["train"] is None and sources["eval"] is None:
        print(f"[load] dapo: {args.dapo_dataset}")
        dapo_rows = normalize_math_rows(load_hf_dataset_rows(args.dapo_dataset), split_name="dapo")
        if args.seed >= 0:
            import random

            rng = random.Random(args.seed)
            rng.shuffle(dapo_rows)
        required = args.dapo_evolve_count + args.dapo_test_count
        if len(dapo_rows) < required:
            raise RuntimeError(f"DAPO rows={len(dapo_rows)} is less than required={required}")
        write_jsonl(targets["train"], limited(dapo_rows[: args.dapo_evolve_count], args.limit))
        write_jsonl(targets["eval"], limited(dapo_rows[args.dapo_evolve_count : required], args.limit))
        print(f"[write] train: {min(args.dapo_evolve_count, args.limit or args.dapo_evolve_count)} rows -> {targets['train']}")
        print(f"[write] eval: {min(args.dapo_test_count, args.limit or args.dapo_test_count)} rows -> {targets['eval']}")
        sources["train"] = None
        sources["eval"] = None

    if need_aime and sources["aime"] is None:
        print(f"[load] aime: {args.aime_dataset}")
        aime_rows = normalize_math_rows(load_hf_dataset_rows(args.aime_dataset), split_name="aime")
        write_jsonl(targets["aime"], limited(aime_rows[: args.aime_count], args.limit))
        print(f"[write] aime: {min(args.aime_count, args.limit or args.aime_count)} rows -> {targets['aime']}")
        sources["aime"] = None

    missing_targets = [name for name, source in sources.items() if source is None and not targets[name].exists()]
    if missing_targets:
        missing = ", ".join(missing_targets)
        raise SystemExit(
            f"Missing source data for {missing}. Pass the matching --*-source path "
            f"or let the script load the default HuggingFace datasets into {output_dir}."
        )

    for name, target in targets.items():
        source = sources[name]
        if source is None:
            print(f"[skip] {name}: existing {target}")
            continue
        if not source.exists():
            raise FileNotFoundError(source)
        rows = normalize_math_rows(read_rows(source), split_name=name)
        rows = limited(rows, args.limit)
        write_jsonl(target, rows)
        print(f"[write] {name}: {len(rows)} rows -> {target}")

    write_json(
        output_dir / "manifest.json",
        {
            "setting": "math_reasoning",
            "seed": args.seed,
            "counts": {name: jsonl_count(path) for name, path in targets.items()},
            "files": {name: path.name for name, path in targets.items()},
        },
    )


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _resolve_source_image(value: str, source_dir: Path) -> Path:
    raw = Path(str(value).replace("\\", "/")).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    candidates = [source_dir / raw, ROOT / raw, Path.cwd() / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"DocVQA source image not found: {value}")


def _save_docvqa_image(value: Any, *, task_id: str, image_dir: Path, source_dir: Path) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id).strip("._") or "document"
    image_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(value, (str, os.PathLike)):
        source = _resolve_source_image(str(value), source_dir)
        suffix = source.suffix.lower() or ".png"
        target = image_dir / f"{safe_id}{suffix}"
        if source != target.resolve():
            shutil.copy2(source, target)
        return target.resolve()

    if isinstance(value, dict):
        if value.get("bytes") is not None:
            target = image_dir / f"{safe_id}.png"
            target.write_bytes(bytes(value["bytes"]))
            return target.resolve()
        if value.get("path"):
            return _save_docvqa_image(value["path"], task_id=task_id, image_dir=image_dir, source_dir=source_dir)

    if hasattr(value, "save"):
        target = image_dir / f"{safe_id}.png"
        value.save(target, format="PNG")
        return target.resolve()
    raise TypeError(f"Unsupported DocVQA image value for {task_id}: {type(value).__name__}")


def normalize_docvqa_rows(
    rows: Any,
    *,
    output_dir: Path,
    source_dir: Path,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    image_dir = output_dir / "images"
    for idx, row in enumerate(rows):
        row = dict(row)
        question = stringify(first_value(row, ("question", "query", "prompt")))
        raw_answers = first_value(row, ("answers", "answer", "ground_truth"))
        if isinstance(raw_answers, list):
            answers = [stringify(answer) for answer in raw_answers if stringify(answer)]
        else:
            answer = stringify(raw_answers)
            answers = [answer] if answer else []
        raw_id = stringify(first_value(row, ("questionId", "question_id", "id", "task_id"))) or str(idx)
        task_id = raw_id if raw_id.startswith("docvqa_") else f"docvqa_{raw_id}"
        image_value = first_value(row, ("image", "image_path", "document"))
        if not question or not answers or image_value in (None, ""):
            continue
        image_path = _save_docvqa_image(
            image_value,
            task_id=task_id,
            image_dir=image_dir,
            source_dir=source_dir,
        )
        tasks.append(
            {
                "id": task_id,
                "source": stringify(first_value(row, ("source", "dataset"))) or DEFAULT_DOCVQA_DATASET,
                "config": DEFAULT_DOCVQA_CONFIG,
                "question": question,
                "answers": answers,
                "image": _portable_path(image_path),
                "doc_id": first_value(row, ("docId", "doc_id", "document_id")),
                "question_types": first_value(row, ("question_types", "question_type")) or [],
                "split_source": "validation",
            }
        )
    if not tasks:
        raise ValueError("No usable DocVQA rows found; expected question, answers, and image fields.")
    return tasks


def prepare_docvqa(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    evolve_path = output_dir / "evolve.jsonl"
    test_path = output_dir / "test.jsonl"
    existing_ready = jsonl_count(evolve_path) > 0 and jsonl_count(test_path) > 0
    if not args.docvqa_source_jsonl and existing_ready:
        print(f"[skip] docvqa: existing non-empty splits under {output_dir}")
        return

    if args.docvqa_source_jsonl:
        source_path = Path(args.docvqa_source_jsonl).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        rows: Any = read_rows(source_path)
        source_dir = source_path.parent
        source_description = str(source_path)
    else:
        try:
            from datasets import load_dataset
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("Python package 'datasets' is required to prepare DocVQA.") from exc
        print(
            f"[load] docvqa: {args.docvqa_dataset} "
            f"config={args.docvqa_config} split={args.docvqa_split}"
        )
        rows = load_dataset(
            args.docvqa_dataset,
            args.docvqa_config,
            split=args.docvqa_split,
        )
        source_dir = ROOT
        source_description = f"{args.docvqa_dataset}/{args.docvqa_config}:{args.docvqa_split}"

    tasks = normalize_docvqa_rows(rows, output_dir=output_dir, source_dir=source_dir)
    random.Random(args.seed).shuffle(tasks)
    if args.limit > 0:
        tasks = tasks[: args.limit]
    if len(tasks) <= args.docvqa_evolve_count:
        raise ValueError(
            f"DocVQA has {len(tasks)} usable rows, but needs more than "
            f"--docvqa-evolve-count={args.docvqa_evolve_count} to create a held-out split."
        )
    evolve_rows = tasks[: args.docvqa_evolve_count]
    test_rows = tasks[args.docvqa_evolve_count :]
    write_jsonl(evolve_path, evolve_rows)
    write_jsonl(test_path, test_rows)
    write_json(
        output_dir / "manifest.json",
        {
            "setting": "docvqa",
            "source": source_description,
            "seed": args.seed,
            "counts": {"evolve": len(evolve_rows), "test": len(test_rows)},
            "files": {"evolve": evolve_path.name, "test": test_path.name, "images": "images"},
        },
    )
    print(f"[write] docvqa evolve: {len(evolve_rows)} rows -> {evolve_path}")
    print(f"[write] docvqa test: {len(test_rows)} rows -> {test_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", required=True, choices=["all", "math_reasoning", "docvqa"])
    parser.add_argument("--output-dir", default="", help="Override output for one setting; not valid with --setting all.")
    parser.add_argument("--train-source", default="")
    parser.add_argument("--eval-source", default="")
    parser.add_argument("--aime-source", default="")
    parser.add_argument("--dapo-dataset", default=DEFAULT_DAPO_DATASET)
    parser.add_argument("--aime-dataset", default=DEFAULT_AIME_DATASET)
    parser.add_argument("--docvqa-source-jsonl", default="")
    parser.add_argument("--docvqa-dataset", default=DEFAULT_DOCVQA_DATASET)
    parser.add_argument("--docvqa-config", default=DEFAULT_DOCVQA_CONFIG)
    parser.add_argument("--docvqa-split", default="validation")
    parser.add_argument("--docvqa-evolve-count", type=int, default=50)
    parser.add_argument("--dapo-evolve-count", type=int, default=400)
    parser.add_argument("--dapo-test-count", type=int, default=100)
    parser.add_argument("--aime-count", type=int, default=30)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.setting == "all" and args.output_dir:
        parser.error("--output-dir can only be used with one setting.")
    if args.setting in {"all", "math_reasoning"}:
        args.output_dir = args.output_dir or str(DEFAULT_MATH_OUT)
        prepare_math_reasoning(args)
    if args.setting in {"all", "docvqa"}:
        args.output_dir = args.output_dir if args.setting == "docvqa" and args.output_dir else str(DEFAULT_DOCVQA_OUT)
        prepare_docvqa(args)


if __name__ == "__main__":
    main()
