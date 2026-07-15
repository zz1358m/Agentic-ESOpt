#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Any


MATH_SYSTEM = """You are a math reasoning agent. Solve the problem using a command-line Python ReAct loop.

You are not allowed to answer from the problem alone. First use the bash tool to run command-line Python for calculation, checking, symbolic manipulation, or search over cases. Then finish with the final answer.

Available action:

Action:
{"name": "bash", "arguments": {"command": "<shell command>"}}

Use command-line Python deliberately, for example python -c "...", for arithmetic, algebraic verification, brute force checks, or symbolic computation. When finished, output exactly:

Final answer: \\boxed{<answer>}

Do not include tool outputs in the final answer."""

DOCVQA_SYSTEM = """You are a DocVQA agent. You answer questions about document images using a command-line and Python ReAct loop.

You are not allowed to answer from the question alone. You must inspect or process the local image file using command-line tools and Python commands, then answer from the textual observations you produced.

Available action:

Action:
{"name": "bash", "arguments": {"command": "<shell command>"}}

The bash action runs in the image directory. Use shell commands and command-line Python, for example python -c "...", to inspect or process the provided image path.
Tool observations are text only. Do not expect the image to be displayed back to you.
When finished, output exactly:

Final answer: <short answer>

Return only the requested short answer after the Final answer prefix. Do not include reasoning in the final answer."""


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    try:
        import datasets
    except ImportError as exc:
        raise RuntimeError(
            "Preparing VERL parquet files requires the 'datasets' package."
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    datasets.Dataset.from_list(rows).to_parquet(str(path))


def _math_rows(records: list[dict[str, Any]], split: str, cwd: Path) -> list[dict[str, Any]]:
    rows = []
    for idx, item in enumerate(records):
        question = str(item["question"])
        answer = str(item["answer"])
        user = (
            "Task: Solve the following math problem.\n\n"
            f"{question}\n\n"
            "You must call the bash action at least once before giving the final answer."
        )
        rows.append(
            {
                "data_source": "trace2skill_math_dapo",
                "prompt": [
                    {"role": "system", "content": MATH_SYSTEM},
                    {"role": "user", "content": user},
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "id": item.get("id", f"math-{split}-{idx}"),
                    "question": question,
                    "answer": answer,
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "bash": {
                            "create_kwargs": {"cwd": str(cwd)},
                        }
                    },
                },
            }
        )
    return rows


def _resolve_doc_image(docvqa_root: Path, image: str) -> str:
    image_path = Path(str(image).replace("\\", "/"))
    if image_path.is_absolute():
        resolved = image_path.resolve()
    else:
        resolved = (docvqa_root / image_path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"DocVQA image not found: {resolved}")
    return str(resolved)


def _docvqa_rows(records: list[dict[str, Any]], split: str, docvqa_root: Path) -> list[dict[str, Any]]:
    rows = []
    for idx, item in enumerate(records):
        question = str(item["question"])
        image = _resolve_doc_image(docvqa_root, str(item["image"]))
        answers = [str(x) for x in item.get("answers", [])]
        user = (
            "Task: Answer the document visual question.\n"
            f"Image path: {image}\n"
            f"Question: {question}\n"
            "You must call at least one bash action before giving the final answer."
        )
        rows.append(
            {
                "data_source": "trace2skill_docvqa",
                "prompt": [
                    {"role": "system", "content": DOCVQA_SYSTEM},
                    {"role": "user", "content": user},
                ],
                "ability": "docvqa",
                "reward_model": {"style": "rule", "ground_truth": answers},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "id": item.get("id", f"docvqa-{split}-{idx}"),
                    "question": question,
                    "answers": answers,
                    "image": image,
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "bash": {
                            "create_kwargs": {"cwd": str(Path(image).parent)},
                        }
                    },
                },
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["all", "math", "docvqa"], default="all")
    parser.add_argument("--math-train", default="data/trace2skill/math_reasoning/dapo_evolve.jsonl")
    parser.add_argument("--math-val", default="data/trace2skill/math_reasoning/dapo_test.jsonl")
    parser.add_argument("--docvqa-root", default=".")
    parser.add_argument("--docvqa-train", default="data/trace2skill/docvqa/evolve.jsonl")
    parser.add_argument("--docvqa-val", default="data/trace2skill/docvqa/test.jsonl")
    parser.add_argument("--out-dir", default="data/trace2skill/verl")
    parser.add_argument("--math-train-limit", type=int, default=400)
    parser.add_argument("--docvqa-train-limit", type=int, default=50)
    parser.add_argument("--docvqa-val-limit", type=int, default=500)
    args = parser.parse_args()

    root = Path(os.environ.get("ROOT", Path.cwd())).resolve()
    def resolve_from_root(value: str) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    out_dir = resolve_from_root(args.out_dir)
    docvqa_root = resolve_from_root(args.docvqa_root)
    math_tool_cwd = Path(os.environ.get("TRACE2SKILL_MATH_TOOL_CWD", root)).expanduser().resolve()
    math_tool_cwd.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict[str, int]] = {}
    if args.task in {"all", "math"}:
        math_train = _read_jsonl(resolve_from_root(args.math_train), args.math_train_limit)
        math_val = _read_jsonl(resolve_from_root(args.math_val), None)
        _write_parquet(_math_rows(math_train, "train", math_tool_cwd), out_dir / "math" / "train.parquet")
        _write_parquet(_math_rows(math_val, "val", math_tool_cwd), out_dir / "math" / "val.parquet")
        manifest["math"] = {"train": len(math_train), "val": len(math_val)}

    if args.task in {"all", "docvqa"}:
        doc_train = _read_jsonl(resolve_from_root(args.docvqa_train), args.docvqa_train_limit)
        doc_val = _read_jsonl(resolve_from_root(args.docvqa_val), args.docvqa_val_limit)
        if not doc_val:
            raise ValueError(
                "DocVQA validation data is empty. Prepare the full DocVQA split before GRPO training."
            )
        _write_parquet(_docvqa_rows(doc_train, "train", docvqa_root), out_dir / "docvqa" / "train.parquet")
        _write_parquet(_docvqa_rows(doc_val, "val", docvqa_root), out_dir / "docvqa" / "val.parquet")
        manifest["docvqa"] = {"train": len(doc_train), "val": len(doc_val)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
