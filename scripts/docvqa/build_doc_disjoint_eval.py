#!/usr/bin/env python3
"""Build a fixed DocVQA eval set with no train question, document, or image overlap."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def image_digest(path: str) -> str:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_answers(value: str) -> list[str]:
    parsed: Any
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = value
    values = parsed if isinstance(parsed, list) else [parsed]
    return [str(item) for item in values if str(item).strip()]


def load_metadata(path: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    return rows, {str(row["questionId"]): row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--preferred-eval", type=Path, required=True)
    parser.add_argument("--items-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()

    train_rows = read_jsonl(args.train)
    preferred_rows = read_jsonl(args.preferred_eval)
    item_rows, metadata = load_metadata(args.items_csv)

    train_ids = {str(row["id"]) for row in train_rows}
    train_docs = {str(metadata[task_id]["docId"]) for task_id in train_ids}
    train_hashes = {image_digest(str(row["image"])) for row in train_rows}

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    excluded: list[dict[str, str]] = []

    def add(row: dict[str, Any], *, origin: str) -> bool:
        task_id = str(row["id"])
        meta = metadata.get(task_id)
        if meta is None:
            excluded.append({"id": task_id, "reason": "missing_metadata", "origin": origin})
            return False
        document_id = str(meta["docId"])
        digest = image_digest(str(row["image"]))
        if task_id in train_ids:
            reason = "question_overlap"
        elif document_id in train_docs:
            reason = "document_overlap"
        elif digest in train_hashes:
            reason = "image_overlap"
        elif task_id in selected_ids:
            reason = "duplicate_question"
        else:
            selected.append(row)
            selected_ids.add(task_id)
            return True
        excluded.append({"id": task_id, "reason": reason, "origin": origin})
        return False

    for row in preferred_rows:
        if len(selected) >= args.count:
            break
        add(row, origin="preferred")

    for item in item_rows:
        if len(selected) >= args.count:
            break
        task_id = str(item["questionId"])
        add(
            {
                "id": task_id,
                "question": str(item["question"]),
                "answers": parse_answers(str(item["answer"])),
                "image": str(Path(item["image_path"]).expanduser().resolve()),
                "source": "lmms-lab/DocVQA:validation",
            },
            origin="replacement",
        )

    if len(selected) != args.count:
        raise RuntimeError(f"Only found {len(selected)} clean eval rows; requested {args.count}.")

    eval_docs = {str(metadata[str(row["id"])]["docId"]) for row in selected}
    eval_hashes = {image_digest(str(row["image"])) for row in selected}
    if train_ids & selected_ids or train_docs & eval_docs or train_hashes & eval_hashes:
        raise RuntimeError("Generated eval set is not train-disjoint.")

    output = args.output.expanduser().resolve()
    write_jsonl(output, selected)
    manifest = {
        "train": str(args.train.expanduser().resolve()),
        "preferred_eval": str(args.preferred_eval.expanduser().resolve()),
        "items_csv": str(args.items_csv.expanduser().resolve()),
        "output": str(output),
        "count": len(selected),
        "train_questions": len(train_ids),
        "train_documents": len(train_docs),
        "eval_documents": len(eval_docs),
        "question_overlap": 0,
        "document_overlap": 0,
        "image_hash_overlap": 0,
        "excluded": excluded,
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
