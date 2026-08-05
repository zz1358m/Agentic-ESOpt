#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path("/home/zhi/Dynamic-Agent")
sys.path.insert(0, str(ROOT / "webarena-train-time/scripts"))

from run_webrl_lite_full_es_train import run_episode  # noqa: E402


ENDPOINTS = [
    "http://127.0.0.1:12013",
    "http://127.0.0.1:12014",
    "http://127.0.0.1:12015",
    "http://127.0.0.1:12016",
]
RESULT_ROOT = ROOT / "runs/webrl_lite_eval"
CONFIG_DIR = ROOT / "data/webarena/vab-lite/config_files/wa/test_webarena_lite"
SPLIT_PATH = ROOT / "data/webarena/vab_lite_split/items.json"
INSTRUCTION_PATH = "agent/prompts/jsons/p_webrl_chat_qwen_action.json"
SKILL_PATH = ROOT / "webarena-train-time/skills/webarena_default_skill_v2.md"


def task_ids() -> list[int]:
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"{SPLIT_PATH} not found. Run "
            "webarena-train-time/scripts/prepare_vab_webarena_lite_split.py first."
        )
    items = json.loads(SPLIT_PATH.read_text())
    return [int(item["task_id"]) for item in items]


def cleanup_artifacts(run_name: str, task_id: int) -> None:
    task_dir = RESULT_ROOT / run_name / f"task_{task_id}"
    for pattern in ("traces/*.zip", "render_*.html"):
        for path in task_dir.glob(pattern):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    screenshots = task_dir / "screehshots"
    if screenshots.exists():
        shutil.rmtree(screenshots, ignore_errors=True)


def read_score(run_name: str, task_id: int) -> float | None:
    task_dir = RESULT_ROOT / run_name / f"task_{task_id}"
    actions = sorted((task_dir / "actions").glob("*.json"))
    if not (task_dir / "run.log").exists() or not actions:
        return None
    try:
        return float(json.loads(actions[-1].read_text()).get("score", -1.0))
    except Exception:
        return -1.0


def summarize(run_name: str, ids: list[int]) -> dict:
    scores = []
    for task_id in ids:
        score = read_score(run_name, task_id)
        scores.append({"task_id": task_id, "score": -1.0 if score is None else score})
    valid = [row["score"] for row in scores if row["score"] >= 0.0]
    summary = {
        "run_name": run_name,
        "count": len(scores),
        "valid_count": len(valid),
        "average": sum(valid) / len(valid) if valid else -1.0,
        "max": max(valid) if valid else -1.0,
        "scores": scores,
    }
    out_dir = RESULT_ROOT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_eval(label: str, skill_file: Path | None) -> None:
    ids = task_ids()
    env_name = os.environ.get(f"QWEN35_{label.upper()}_RUN_NAME")
    run_name = env_name or f"qwen35_27b_fixed_{label}_lite165_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}"
    print(f"[start] {run_name} skill={skill_file}", flush=True)
    scores: dict[int, float] = {}
    with ThreadPoolExecutor(max_workers=len(ENDPOINTS)) as pool:
        futures = {}
        for index, task_id in enumerate(ids):
            previous_score = read_score(run_name, task_id)
            if previous_score is not None:
                scores[task_id] = previous_score
                continue
            endpoint = ENDPOINTS[index % len(ENDPOINTS)]
            future = pool.submit(
                run_episode,
                endpoint=endpoint,
                task_id=task_id,
                config_dir=CONFIG_DIR,
                result_root=RESULT_ROOT,
                run_name=run_name,
                skill_file=skill_file,
                instruction_path=INSTRUCTION_PATH,
                model_name="Qwen3.5-27B",
                mode="chat",
                stop_token="",
            )
            futures[future] = (task_id, endpoint)
        for future in as_completed(futures):
            task_id, endpoint = futures[future]
            try:
                score = float(future.result())
            except Exception as exc:
                score = -1.0
                print(
                    f"[error] {run_name} task={task_id} endpoint={endpoint} "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            cleanup_artifacts(run_name, task_id)
            scores[task_id] = score
            valid = [value for value in scores.values() if value >= 0.0]
            avg = sum(valid) / len(valid) if valid else -1.0
            print(
                f"[eval] {run_name} task={task_id} endpoint={endpoint} "
                f"score={score} completed={len(scores)}/165 valid={len(valid)} avg={avg}",
                flush=True,
            )
    summary = summarize(run_name, ids)
    print(
        f"[done] {run_name} avg={summary['average']} "
        f"valid={summary['valid_count']}/165 summary={RESULT_ROOT / run_name / 'summary.json'}",
        flush=True,
    )


def main() -> None:
    run_eval("noskill", None)
    run_eval("llmskill", SKILL_PATH)


if __name__ == "__main__":
    main()
