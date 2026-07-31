#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl_trace2skill.docvqa_protocol import build_docvqa_messages

MATH_SYSTEM = """You are a math reasoning agent. Solve the problem using the provided bash tool.

You are not allowed to answer from the problem alone. Your very first assistant turn must consist only of one bash tool call. Do not reason, solve, or write any text before that first tool call. After receiving its observation, continue solving and call bash again when useful.

Call bash by emitting exactly this XML shape, with no text after the closing tool_call tag:

<tool_call>
<function=bash>
<parameter=command>
python -c "print(1 + 1)"
</parameter>
</function>
</tool_call>

The command inside <parameter=command> may be replaced with any shell command needed for the problem. After the tool observation is returned, continue reasoning with the complete conversation history. You must make at least one such bash call before answering.

Use command-line Python deliberately, for example python -c "...", for arithmetic, algebraic verification, brute force checks, or symbolic computation. When finished, output exactly:

Final answer: \\boxed{<answer>}

Do not include tool outputs in the final answer."""

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


def _validate_agent_routing(rows: list[dict[str, Any]], expected: str) -> None:
    values = {str(row.get("agent_name")) for row in rows}
    if values != {expected}:
        raise RuntimeError(f"expected agent_name={expected!r}, got {sorted(values)!r}")


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
                # Required by VERL's async agent-loop router. Without this
                # field VERL silently uses single_turn_agent even when
                # rollout.multi_turn.enable=True.
                "agent_name": "tool_agent",
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
        rows.append(
            {
                "data_source": "trace2skill_docvqa",
                "agent_name": "paper_react_cli_agent",
                "prompt": build_docvqa_messages(question),
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
                            "create_kwargs": {"image_path": image},
                        }
                    },
                },
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["all", "math", "docvqa"], default="all")
    parser.add_argument(
        "--math-train",
        default=os.environ.get(
            "TRACE2SKILL_MATH_TRAIN",
            "data/trace2skill/math_reasoning/dapo_evolve.jsonl",
        ),
    )
    parser.add_argument(
        "--math-val",
        default=os.environ.get(
            "TRACE2SKILL_MATH_VAL",
            "data/trace2skill/math_reasoning/dapo_test.jsonl",
        ),
    )
    parser.add_argument("--docvqa-root", default=os.environ.get("DOCVQA_ROOT", "."))
    parser.add_argument(
        "--docvqa-train",
        default=os.environ.get("DOCVQA_TRAIN", "data/trace2skill/docvqa/evolve.jsonl"),
    )
    parser.add_argument(
        "--docvqa-val",
        default=os.environ.get("DOCVQA_VAL", "data/trace2skill/docvqa/test.jsonl"),
    )
    parser.add_argument(
        "--out-dir",
        default=os.environ.get("TRACE2SKILL_VERL_DATA_DIR", "data/trace2skill/verl"),
    )
    parser.add_argument("--math-train-limit", type=int, default=400)
    parser.add_argument(
        "--docvqa-train-limit",
        type=int,
        default=int(os.environ.get("DOCVQA_TRAIN_LIMIT", "50")),
    )
    parser.add_argument(
        "--docvqa-val-limit",
        type=int,
        default=int(os.environ.get("DOCVQA_VAL_LIMIT", "100")),
    )
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
        math_train_rows = _math_rows(math_train, "train", math_tool_cwd)
        math_val_rows = _math_rows(math_val, "val", math_tool_cwd)
        _validate_agent_routing(math_train_rows, "tool_agent")
        _validate_agent_routing(math_val_rows, "tool_agent")
        _write_parquet(math_train_rows, out_dir / "math" / "train.parquet")
        _write_parquet(math_val_rows, out_dir / "math" / "val.parquet")
        manifest["math"] = {"train": len(math_train), "val": len(math_val)}

    if args.task in {"all", "docvqa"}:
        doc_train = _read_jsonl(resolve_from_root(args.docvqa_train), args.docvqa_train_limit)
        doc_val = _read_jsonl(resolve_from_root(args.docvqa_val), args.docvqa_val_limit)
        if not doc_val:
            raise ValueError(
                "DocVQA validation data is empty. Prepare the full DocVQA split before GRPO training."
            )
        doc_train_rows = _docvqa_rows(doc_train, "train", docvqa_root)
        doc_val_rows = _docvqa_rows(doc_val, "val", docvqa_root)
        _validate_agent_routing(doc_train_rows, "paper_react_cli_agent")
        _validate_agent_routing(doc_val_rows, "paper_react_cli_agent")
        _write_parquet(doc_train_rows, out_dir / "docvqa" / "train.parquet")
        _write_parquet(doc_val_rows, out_dir / "docvqa" / "val.parquet")
        manifest["docvqa"] = {"train": len(doc_train), "val": len(doc_val)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
