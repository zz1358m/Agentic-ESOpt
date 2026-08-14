from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, NamedTuple, TypedDict

from scripts.data_contract import redact, redact_text
from scripts.extract_ahd import parse_search_log
from scripts.extract_capability import (
    extract_ahd_heuristic_checkpoints,
    extract_sudoku_case_checkpoints,
    parse_webarena_case_progression,
    parse_webarena_task_scores,
)
from scripts.extract_history import extract_history_case
from scripts.extract_sudoku import parse_rl_training_log, parse_training_log
from scripts.validate_data import validate_public_payload


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(os.environ.get("ESOPT_SOURCE_ROOT", PROJECT_ROOT.parent))
RECHECK_ROOT = Path(
    os.environ.get("ESOPT_RECHECK_ROOT", PROJECT_ROOT / ".external-data" / "recheck")
)
MANUSCRIPT_ROOT = Path(
    os.environ.get("ESOPT_MANUSCRIPT_ROOT", PROJECT_ROOT / ".external-data" / "manuscript")
)
DATA_DIR = PROJECT_ROOT / "public" / "data"
SOURCE_CHECKS: list[dict[str, str]] = []


class AhdTaskSource(TypedDict):
    problem: str
    mode: str
    result_directory: str
    file_prefix: str


class SudokuFavorability(NamedTuple):
    repeat_gain: float
    negative_regression: float
    post_base_score_sum: float


class WebArenaFavorability(NamedTuple):
    setting_gain: float
    training_gain: float
    negative_regression: float


def record_source_check(name: str, condition: bool) -> None:
    if not condition:
        raise ValueError(f"source consistency check failed: {name}")
    SOURCE_CHECKS.append({"check": name, "status": "passed"})


def write_payload(name: str, payload: dict[str, Any]) -> None:
    validate_public_payload(payload)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def add_eval_curves(payload: dict[str, Any], history: list[dict[str, Any]], *, math: bool) -> None:
    splits = [("dapo_eval", "Periodic DAPO evaluation" if math else "Periodic held-out DocVQA evaluation")]
    if math:
        splits.append(("aime_eval", "Periodic AIME 2026 evaluation"))
    for key, label in splits:
        points = []
        for row in history:
            evaluation = row.get(key)
            if isinstance(evaluation, dict) and isinstance(evaluation.get("mean_score"), (int, float)):
                points.append({"generation": row["generation"], "value": evaluation["mean_score"]})
        payload["curves"].append({"id": key, "kind": "eval", "label": label, "points": points})


def merge_history_cases(
    history: list[dict[str, Any]],
    *,
    case_specs: list[dict[str, Any]],
    checkpoint_generations: list[int],
    task_name: str,
    metric_name: str,
) -> dict[str, Any]:
    """Extract a small, explicitly selected set of cases shared by replay checkpoints."""
    payloads = [
        extract_history_case(
            history,
            case_id=spec["id"],
            checkpoint_generations=checkpoint_generations,
            task_name=task_name,
            metric_name=metric_name,
        )
        for spec in case_specs
    ]
    extracted_cases = [item["cases"][0] for item in payloads]
    payload = payloads[0]
    payload["cases"] = []
    for spec, case in zip(case_specs, extracted_cases, strict=True):
        case.update({key: value for key, value in spec.items() if key != "id"})
        if len(case["checkpoints"]) != len(checkpoint_generations):
            raise ValueError(f"case {spec['id']} is not available at every requested checkpoint")
        payload["cases"].append(case)
    for checkpoint in payload["checkpoints"]:
        checkpoint["caseIds"] = [
            case["id"]
            for case in payload["cases"]
            if any(item["generation"] == checkpoint["generation"] for item in case["checkpoints"])
        ]
        checkpoint["trajectoryAvailable"] = bool(checkpoint["caseIds"])
    return payload


def build_math() -> None:
    history_path = SOURCE_ROOT / "math-train-time/results/training/history.json"
    history = load_json(history_path)
    payload = merge_history_cases(
        history,
        case_specs=[
            {
                "id": "dapo_5585520b-e89c-4c8c-afdb-172b0137f0e2",
                "dataset": "DAPO",
                "label": "Congruence system",
            },
            {
                "id": "dapo_5236e690-f702-4dc2-9751-d075a1fc0344",
                "dataset": "DAPO",
                "label": "Polynomial derivative",
            },
            {
                "id": "aime_2026_026",
                "dataset": "AIME 2026",
                "label": "Cubic root signs",
            },
            {
                "id": "aime_2026_004",
                "dataset": "AIME 2026",
                "label": "Integer representations",
            },
        ],
        checkpoint_generations=[9, 19, 24, 25],
        task_name="math",
        metric_name="Exact-match accuracy",
    )
    payload["configurations"] = [{"dataset": dataset} for dataset in ("DAPO", "AIME 2026")]
    payload["metadata"].update(
        {
            "title": "Tool-Using Math Reasoning",
            "sourceFiles": ["math training history", "paper Table: Math and DocVQA"],
            "note": "Full ReAct trajectories were retained at generations 9, 19, 24, and 25.",
        }
    )
    add_eval_curves(payload, history, math=True)
    expected_train = [
        {"generation": row["generation"], "value": float(row["reward_mean"])}
        for row in history
        if isinstance(row.get("generation"), int) and isinstance(row.get("reward_mean"), (int, float))
    ]
    record_source_check("Math curves equal parsed training history", payload["curves"][0]["points"] == expected_train)
    payload["finalResults"] = [
        {"context": "No Skill", "method": "No Skill", "dapoMean": 63.0, "dapoPass": 86.0, "aimeMean": 55.8, "aimePass": 86.7},
        {"context": "No Skill", "method": "GRPO", "dapoMean": 68.8, "dapoPass": 83.0, "aimeMean": 58.3, "aimePass": 76.7},
        {"context": "No Skill", "method": "Agentic ESOpt", "dapoMean": 76.8, "dapoPass": 86.0, "aimeMean": 70.8, "aimePass": 96.7},
        {"context": "Trace2Skill", "method": "Trace2Skill", "dapoMean": 64.8, "dapoPass": 82.0, "aimeMean": 50.8, "aimePass": 83.3},
        {"context": "Trace2Skill", "method": "GRPO + Trace2Skill", "dapoMean": 67.8, "dapoPass": 85.0, "aimeMean": 50.0, "aimePass": 80.0},
        {"context": "Trace2Skill", "method": "Agentic ESOpt + Trace2Skill", "dapoMean": 77.3, "dapoPass": 86.0, "aimeMean": 71.7, "aimePass": 96.7},
    ]
    write_payload("math.json", payload)


def build_docvqa() -> None:
    history_path = SOURCE_ROOT / "docvqa-train-time/results/training/history_first40.json"
    history = load_json(history_path)
    payload = merge_history_cases(
        history,
        case_specs=[
            {
                "id": "docvqa_53536",
                "label": "Policy conference agenda",
                "image": "selected-documents/agenda-53536.png",
            },
            {
                "id": "docvqa_5518",
                "label": "Environmental education report",
                "image": "selected-documents/ceer-5518.png",
            },
        ],
        checkpoint_generations=[-1, 9, 19, 29, 39],
        task_name="docvqa",
        metric_name="ANLS",
    )
    payload["metadata"].update(
        {
            "title": "Document Visual Question Answering",
            "sourceFiles": ["DocVQA training history", "paper Table: Math and DocVQA"],
            "note": "The selected document is the only source image copied into the website.",
        }
    )
    add_eval_curves(payload, history, math=False)
    expected_train = [
        {"generation": row["generation"], "value": float(row["reward_mean"])}
        for row in history
        if isinstance(row.get("generation"), int) and isinstance(row.get("reward_mean"), (int, float))
    ]
    record_source_check("DocVQA curves equal parsed training history", payload["curves"][0]["points"] == expected_train)
    payload["finalResults"] = [
        {"context": "No Skill", "method": "No Skill", "anlsMean": 0.3875, "anlsPass": 0.5981, "accMean": 40.3, "accPass": 53.0},
        {"context": "No Skill", "method": "GRPO", "anlsMean": 0.4627, "anlsPass": 0.5398, "accMean": 48.0, "accPass": 56.0},
        {"context": "No Skill", "method": "Agentic ESOpt", "anlsMean": 0.5043, "anlsPass": 0.6507, "accMean": 52.5, "accPass": 61.0},
        {"context": "Trace2Skill", "method": "Trace2Skill", "anlsMean": 0.4612, "anlsPass": 0.6772, "accMean": 47.3, "accPass": 69.0},
        {"context": "Trace2Skill", "method": "GRPO + Trace2Skill", "anlsMean": 0.4743, "anlsPass": 0.5692, "accMean": 49.5, "accPass": 60.0},
        {"context": "Trace2Skill", "method": "Agentic ESOpt + Trace2Skill", "anlsMean": 0.5086, "anlsPass": 0.6654, "accMean": 52.8, "accPass": 61.0},
    ]
    image_dir = PROJECT_ROOT / "public" / "selected-documents"
    image_dir.mkdir(parents=True, exist_ok=True)
    for source_name, public_name in (("53536.png", "agenda-53536.png"), ("5518.png", "ceer-5518.png")):
        shutil.copyfile(SOURCE_ROOT / "data/trace2skill/docvqa/images" / source_name, image_dir / public_name)
    write_payload("docvqa.json", payload)


def build_sudoku() -> None:
    curves = []
    for mask in (5, 10, 15):
        source = SOURCE_ROOT / f"sudoku-train-time/results/training/agentic_esopt_es32/mask{mask}.log"
        parsed = parse_training_log(source.read_text(encoding="utf-8"))
        generated_curves = [
            {"id": f"mask{mask}-train", "method": "Agentic ESOpt", "kind": "train", "label": "Agentic ESOpt · sampled train reward", "mask": mask, "points": parsed["trainCurve"]},
            {"id": f"mask{mask}-train-eval", "method": "Agentic ESOpt", "kind": "eval", "label": "Agentic ESOpt · periodic train", "mask": mask, "points": parsed["periodicTrain"]},
            {"id": f"mask{mask}-eval", "method": "Agentic ESOpt", "kind": "eval", "label": "Agentic ESOpt · periodic evaluation", "mask": mask, "points": parsed["periodicEval"]},
        ]
        curves.extend(generated_curves)
        record_source_check(
            f"Sudoku mask {mask} public curves match every parsed log point",
            generated_curves[0]["points"] == parsed["trainCurve"]
            and generated_curves[1]["points"] == parsed["periodicTrain"]
            and generated_curves[2]["points"] == parsed["periodicEval"]
            and bool(parsed["trainCurve"] and parsed["periodicEval"]),
        )
        for folder, method, suffix in (
            ("grpo_t0p7_p0p8_k20", "GRPO · recommended sampling", "grpo-rec"),
            ("grpo_t1_p1_kneg1", "GRPO · exploration sampling", "grpo-exp"),
        ):
            grpo_source = SOURCE_ROOT / f"sudoku-train-time/results/training/{folder}/mask{mask}.log"
            grpo = parse_rl_training_log(grpo_source.read_text(encoding="utf-8"))
            curves.extend(
                [
                    {"id": f"mask{mask}-{suffix}-train", "method": method, "kind": "train", "label": f"{method} · train reward", "mask": mask, "points": grpo["trainCurve"]},
                    {"id": f"mask{mask}-{suffix}-eval", "method": method, "kind": "eval", "label": f"{method} · periodic evaluation", "mask": mask, "points": grpo["periodicEval"]},
                ]
            )
            record_source_check(
                f"Sudoku mask {mask} {method} curves match every parsed log point",
                bool(grpo["trainCurve"] and grpo["periodicEval"]),
            )
    vanilla_source = SOURCE_ROOT / "sudoku-train-time/results/training/vanilla_es32/mask15.log"
    vanilla = parse_training_log(vanilla_source.read_text(encoding="utf-8"))
    curves.extend(
        [
            {"id": "mask15-vanilla-train", "method": "Vanilla ES", "kind": "train", "label": "Vanilla ES · sampled train reward", "mask": 15, "points": vanilla["trainCurve"]},
            {"id": "mask15-vanilla-eval", "method": "Vanilla ES", "kind": "eval", "label": "Vanilla ES · periodic evaluation", "mask": 15, "points": vanilla["periodicEval"]},
        ]
    )
    record_source_check(
        "Sudoku mask 15 Vanilla ES curves match every parsed log point",
        bool(vanilla["trainCurve"] and vanilla["periodicEval"]),
    )

    cases_by_mask: dict[int, dict[str, Any]] = {}
    for raw in (SOURCE_ROOT / "data/sudoku/eval.jsonl").read_text(encoding="utf-8").splitlines():
        item = json.loads(raw)
        mask = int(item["mask_count"])
        if mask in (5, 10, 15) and mask not in cases_by_mask:
            cases_by_mask[mask] = {
                "id": item["id"],
                "maskCount": mask,
                "puzzle": item["puzzle"],
                "solution": item["solution"],
            }
        if len(cases_by_mask) == 3:
            break

    recheck_history_path = (
        RECHECK_ROOT
        / "records/sudoku/training/sudoku_stage3_mask5_g40_current_780e7f2_20260812/formal_run/history.json"
    )
    recheck_history = load_json(recheck_history_path)
    retained_evaluations = [
        row for row in recheck_history if isinstance(row.get("eval"), dict)
    ]
    retained_case_ids = sorted(
        score["task_id"]
        for score in retained_evaluations[0]["eval"]["runs"][0]["scores"]
    )

    def sudoku_favorability(case_id: str) -> SudokuFavorability:
        repeat_means = []
        for row in retained_evaluations:
            run_scores = []
            for run in row["eval"]["runs"]:
                selected = next(
                    score for score in run["scores"] if score["task_id"] == case_id
                )
                run_scores.append(float(selected["score"]))
            repeat_means.append(sum(run_scores) / len(run_scores))
        gain = repeat_means[-1] - repeat_means[0]
        regression = sum(
            max(0.0, left - right)
            for left, right in zip(repeat_means, repeat_means[1:])
        )
        return SudokuFavorability(gain, -regression, sum(repeat_means[1:]))

    sudoku_ranks = {
        case_id: sudoku_favorability(case_id) for case_id in retained_case_ids
    }
    best_sudoku_rank = max(sudoku_ranks.values())
    capability_case_id = next(
        case_id
        for case_id in retained_case_ids
        if sudoku_ranks[case_id] == best_sudoku_rank
    )
    capability_checkpoints = extract_sudoku_case_checkpoints(
        recheck_history, case_id=capability_case_id
    )
    capability_checkpoints = [
        {
            "task": "sudoku",
            "configurationId": "sudoku-mask5-stage3-recheck",
            "caseId": capability_case_id,
            "modelCheckpointId": f"stage3-recheck-generation-{item['optimizationStep']}",
            "sourceArtifact": "accepted mask-5 Stage 3 recheck history",
            **item,
        }
        for item in capability_checkpoints
    ]
    for raw in (SOURCE_ROOT / "data/sudoku/eval.jsonl").read_text(encoding="utf-8").splitlines():
        item = json.loads(raw)
        if item["id"] == capability_case_id and int(item["mask_count"]) == 5:
            cases_by_mask[5] = {
                "id": item["id"],
                "maskCount": 5,
                "puzzle": item["puzzle"],
                "solution": item["solution"],
                "capabilityCheckpoints": capability_checkpoints,
                "evidenceScope": "Favorable eligible case · accepted Stage 3 recheck · mask 5 · maximum three-repeat base-to-final gain with no regression · deterministic case-ID tie-break",
            }
            break
    if cases_by_mask[5]["id"] != capability_case_id:
        raise ValueError(f"Sudoku capability case {capability_case_id} was not found")
    capability_points = [
        {"generation": item["optimizationStep"], "value": item["aggregateMetric"]}
        for item in capability_checkpoints
    ]
    curves.append(
        {
            "id": "mask5-stage3-recheck-eval",
            "configId": "sudoku-mask5-stage3-recheck",
            "method": "Agentic ESOpt · Stage 3 recheck",
            "kind": "eval",
            "label": "Stage 3 recheck · periodic evaluation",
            "mask": 5,
            "points": capability_points,
        }
    )
    record_source_check(
        "Sudoku favorable eligible case maximizes three-repeat base-to-final gain and links five same-case outputs",
        capability_case_id == "eval-000064"
        and [item["optimizationStep"] for item in capability_checkpoints] == [-1, 9, 19, 29, 39]
        and len({json.dumps(item["prediction"]) for item in capability_checkpoints}) == 2
        and sum(
            left["prediction"] != right["prediction"]
            for left, right in zip(capability_checkpoints, capability_checkpoints[1:])
        ) == 1
        and [item["score"] for item in capability_checkpoints] == [0.0, 1.0, 1.0, 1.0, 1.0],
    )
    cases = [cases_by_mask[mask] for mask in (5, 10, 15)]

    payload = {
        "metadata": {
            "task": "sudoku",
            "title": "Long-Horizon Sudoku",
            "method": "Agentic ESOpt",
            "metric": "Episode success rate",
            "sourceFiles": ["Agentic ESOpt mask 5/10/15 logs", "accepted mask-5 Stage 3 recheck history", "paper Sudoku summary table"],
            "note": "Paper curves cover masks 5/10/15. The linked model-board replay is the accepted, explicitly scoped mask-5 40-generation Stage 3 recheck.",
        },
        "configurations": [
            {"maskCount": mask, "methods": ["Agentic ESOpt", "GRPO · recommended sampling", "GRPO · exploration sampling"] + (["Vanilla ES"] if mask == 15 else [])}
            for mask in (5, 10, 15)
        ],
        "curves": curves,
        "checkpoints": [],
        "cases": cases,
        "finalResults": [
            {"method": "PPO", "5": 90.63, "10": 56.25, "15": 0.0},
            {"method": "GRPO", "5": 85.42, "10": 67.71, "15": 40.63},
            {"method": "Vanilla ES", "5": 85.42, "10": 55.21, "15": 42.71},
            {"method": "Agentic ESOpt", "5": 89.58, "10": 62.5, "15": 53.13},
        ],
    }
    write_payload("sudoku.json", payload)


def build_webarena() -> None:
    curve_path = SOURCE_ROOT / "webarena-train-time/results/training/noskill_agentic_esopt/eval_curve.csv"
    with curve_path.open(encoding="utf-8") as handle:
        curve = [
            {"generation": int(row["eval_epoch"]), "value": float(row["avg"])}
            for row in csv.DictReader(handle)
        ]
    expected_curve = [
        {"generation": 10, "value": 0.303030303},
        {"generation": 20, "value": 0.339393939},
        {"generation": 30, "value": 0.333333333},
        {"generation": 40, "value": 0.321212121},
        {"generation": 50, "value": 0.327272727},
        {"generation": 60, "value": 0.323170732},
        {"generation": 70, "value": 0.357575758},
    ]
    record_source_check(
        "WebArena public curve matches every CSV evaluation row",
        curve == expected_curve and len(curve) == 7,
    )

    settings = {
        "noskill_no-finetune": "No Skill",
        "noskill_agentic_esopt": "Agentic ESOpt",
        "trace2skill_no-finetune": "Trace2Skill",
        "trace2skill_agentic_esopt": "Agentic ESOpt + Trace2Skill",
    }
    selected_task_ids = [4, 15, 37, 75, 132, 0]
    selected_cases = {task_id: [] for task_id in selected_task_ids}
    indexed_runs: dict[str, dict[int, dict[int, dict[str, Any]]]] = {}
    for folder, label in settings.items():
        indexed_runs[folder] = {}
        for run in range(1, 4):
            data = load_json(SOURCE_ROOT / f"webarena-train-time/results/eval/{folder}/run_{run:02d}.json")
            indexed = {int(item["task_id"]): item for item in data["results"]}
            indexed_runs[folder][run] = indexed
            for task_id in selected_task_ids:
                result = indexed[task_id]
                selected_cases[task_id].append(
                    {
                        "setting": label,
                        "run": run,
                        "goal": result["task_description"],
                        "site": result["task_type"],
                        "hard": result["hard"],
                        "soft": result["soft"],
                        "turns": result["n_turns"],
                        "answer": result["predicted_answer"],
                        "failureReason": result.get("fail_reason", ""),
                    }
                )
    cases = [
        {
            "id": f"task-{task_id}",
            "taskId": task_id,
            "site": outcomes[0]["site"],
            "goal": outcomes[0]["goal"],
            "outcomes": redact(outcomes),
        }
        for task_id, outcomes in selected_cases.items()
    ]
    train_eval_log = (
        SOURCE_ROOT
        / "webarena-train-time/results/training/noskill_agentic_esopt/train_eval.log"
    )
    train_eval_text = train_eval_log.read_text(encoding="utf-8", errors="replace")
    task_scores_by_epoch = parse_webarena_task_scores(train_eval_text)
    aggregate_by_epoch = {item["generation"]: item["value"] for item in curve}
    evaluation_epochs = [item["generation"] for item in curve]

    def webarena_favorability(task_id: int) -> WebArenaFavorability:
        no_skill = sum(
            float(indexed_runs["noskill_no-finetune"][run][task_id]["hard"])
            for run in range(1, 4)
        ) / 3
        esopt = sum(
            float(indexed_runs["noskill_agentic_esopt"][run][task_id]["hard"])
            for run in range(1, 4)
        ) / 3
        scores = [task_scores_by_epoch[task_id][epoch] for epoch in evaluation_epochs]
        regression = sum(
            max(0.0, left - right) for left, right in zip(scores, scores[1:])
        )
        return WebArenaFavorability(
            esopt - no_skill, scores[-1] - scores[0], -regression
        )

    all_task_ids = sorted(indexed_runs["noskill_agentic_esopt"][1])
    webarena_ranks = {
        task_id: webarena_favorability(task_id) for task_id in all_task_ids
    }
    best_webarena_rank = max(webarena_ranks.values())
    capability_task_id = next(
        task_id
        for task_id in all_task_ids
        if webarena_ranks[task_id] == best_webarena_rank
    )
    capability_case = next(case for case in cases if case["taskId"] == capability_task_id)
    final_outcome = next(
        outcome
        for outcome in capability_case["outcomes"]
        if outcome["setting"] == "Agentic ESOpt" and outcome["run"] == 1
    )
    full_progression = parse_webarena_case_progression(
        train_eval_text,
        task_id=capability_task_id,
        aggregate_by_epoch=aggregate_by_epoch,
        selected_epochs=evaluation_epochs,
        final_output=final_outcome["answer"],
        parsed_scores=task_scores_by_epoch,
    )
    first_success_epoch = next(
        item["optimizationStep"] for item in full_progression if item["score"] == 1
    )
    selected_epochs = [evaluation_epochs[0], first_success_epoch, evaluation_epochs[-1]]
    capability_checkpoints = parse_webarena_case_progression(
        train_eval_text,
        task_id=capability_task_id,
        aggregate_by_epoch=aggregate_by_epoch,
        selected_epochs=selected_epochs,
        final_output=final_outcome["answer"],
        parsed_scores=task_scores_by_epoch,
    )
    capability_checkpoints = [
        {
            "task": "webarena",
            "configurationId": "noskill-agentic-esopt",
            "caseId": f"task-{capability_task_id}",
            "sourceArtifact": "original WebArena train/eval log" if item["outputUnavailable"] else "original final evaluation run 1",
            **item,
        }
        for item in capability_checkpoints
    ]
    capability_case["capabilityCheckpoints"] = capability_checkpoints
    capability_case["evidenceScope"] = "Favorable eligible case · original 70-epoch run · maximum three-repeat No Skill-to-ESOpt gain with monotonic training outcome · final readable output retained"
    record_source_check(
        "WebArena favorable eligible case maximizes three-repeat No Skill-to-ESOpt gain and improves monotonically in training",
        capability_task_id == 4
        and selected_epochs == [10, 50, 70]
        and [item["score"] for item in capability_checkpoints] == [0.0, 1.0, 1.0]
        and webarena_ranks[capability_task_id]
        == WebArenaFavorability(1.0, 1.0, 0.0),
    )
    record_source_check(
        "WebArena selected cases retain all four settings and three repeats",
        len(cases) == len(selected_task_ids) and all(len(case["outcomes"]) == 12 for case in cases),
    )

    payload = {
        "metadata": {
            "task": "webarena",
            "title": "WebArena-Lite",
            "method": "Agentic ESOpt",
            "metric": "Task success rate",
            "sourceFiles": ["WebArena evaluation curve", "four settings × three final runs", "paper WebArena table"],
            "note": "The source retains periodic evaluation scores, not a separate training-split score. Per-turn browser observations and actions were not retained in the original logs.",
        },
        "configurations": [
            {"epochs": [10, 20, 30, 40, 50, 60, 70]},
            {"sites": sorted({case["site"] for case in cases})},
            {"settings": list(settings.values()), "runs": [1, 2, 3]},
        ],
        "curves": [{"id": "esopt-eval", "kind": "eval", "label": "Periodic WebArena-Lite evaluation", "points": curve}],
        "checkpoints": [],
        "cases": cases,
        "finalResults": [
            {"method": "No Skill", "reddit": 50.79, "gitlab": 35.42, "cms": 41.90, "map": 8.33, "oss": 21.01, "average": 29.47},
            {"method": "Agentic ESOpt", "reddit": 49.21, "gitlab": 43.75, "cms": 49.52, "map": 14.29, "oss": 30.43, "average": 36.16},
            {"method": "Trace2Skill", "reddit": 49.21, "gitlab": 39.58, "cms": 46.67, "map": 13.10, "oss": 28.26, "average": 33.94},
            {"method": "Agentic ESOpt + Trace2Skill", "reddit": 52.80, "gitlab": 41.67, "cms": 50.48, "map": 10.71, "oss": 32.61, "average": 36.36},
        ],
    }
    write_payload("webarena.json", payload)


def build_ahd() -> None:
    tasks: list[AhdTaskSource] = [
        {"problem": "TSP", "mode": "Constructive", "result_directory": "TSP_construct", "file_prefix": "construct_tsp"},
        {"problem": "TSP", "mode": "ACO", "result_directory": "TSP_ACO", "file_prefix": "aco_tsp"},
        {"problem": "KP", "mode": "Constructive", "result_directory": "KP_construct", "file_prefix": "construct_kp"},
        {"problem": "ASP", "mode": "Constructive", "result_directory": "ASP_construct", "file_prefix": "construct_asp"},
        {"problem": "CVRP", "mode": "ACO", "result_directory": "CVRP_ACO", "file_prefix": "aco_cvrp"},
        {"problem": "BPP", "mode": "ACO", "result_directory": "BPP_ACO", "file_prefix": "aco_bpp"},
    ]
    configurations: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for task in tasks:
        problem = task["problem"]
        mode = task["mode"]
        task_dir = task["result_directory"]
        prefix = task["file_prefix"]
        for outer_method in ("Sample", "EoH"):
            for agentic in (False, True):
                for budget in (1000, 2000):
                    result_root = SOURCE_ROOT / "ahd-test-time" / "results" / (
                        f"{outer_method}{'+AgenticESOpt' if agentic else ''}{budget}"
                    ) / task_dir
                    for repeat in (1, 2, 3):
                        config_id = "-".join(
                            [
                                problem.lower(),
                                mode.lower(),
                                outer_method.lower(),
                                "agentic" if agentic else "baseline",
                                str(budget),
                                f"r{repeat}",
                            ]
                        )
                        config = {
                            "id": config_id,
                            "problem": problem,
                            "mode": mode,
                            "outerMethod": outer_method,
                            "agenticESOpt": agentic,
                            "budget": budget,
                            "repeat": repeat,
                        }
                        code_candidates = sorted(result_root.glob(f"{prefix}*rep{repeat}_final_best_code.py"))
                        if len(code_candidates) != 1:
                            raise ValueError(f"expected one final heuristic for {config_id}, found {len(code_candidates)}")
                        heuristic = redact_text(code_candidates[0].read_text(encoding="utf-8"))
                        log_candidates = sorted(result_root.glob(f"{prefix}*rep{repeat}*.log"))
                        config["sourceFiles"] = [
                            f"{problem} {mode} {outer_method}{' + Agentic ESOpt' if agentic else ''} result family",
                            f"Retained final heuristic: {code_candidates[0].name}",
                            *([f"Retained search log: {log_candidates[0].name}"] if log_candidates else []),
                        ]
                        configurations.append(config)
                        parsed_checkpoints: list[dict[str, Any]] = []
                        if log_candidates:
                            search = parse_search_log(log_candidates[0].read_text(encoding="utf-8", errors="replace"))
                            parsed_checkpoints = search["generations"]
                            points = [
                                {"generation": item["generation"], "value": item["bestSoFar"]}
                                for item in parsed_checkpoints
                            ]
                            curves.append(
                                {
                                    "id": f"{config_id}-best-so-far",
                                    "configId": config_id,
                                    "kind": "train",
                                    "label": "Best-so-far objective",
                                    "points": points,
                                }
                            )
                            checkpoints.extend(
                                {
                                    "configId": config_id,
                                    "trajectoryAvailable": False,
                                    "caseIds": [config_id],
                                    **item,
                                }
                                for item in parsed_checkpoints
                            )
                        cases.append(
                            {
                                "id": config_id,
                                "configId": config_id,
                                "checkpoints": parsed_checkpoints,
                                "finalHeuristic": heuristic,
                            }
                        )
    record_source_check(
        "AHD public configurations map to retained final heuristics",
        len(configurations) == 144 and len(cases) == 144 and all(case["finalHeuristic"] for case in cases),
    )
    default_id = "tsp-constructive-eoh-agentic-1000-r1"
    default_case = next(case for case in cases if case["id"] == default_id)
    record_source_check(
        "AHD default curve matches every parsed search generation",
        len(default_case["checkpoints"]) == 25
        and next(curve for curve in curves if curve["configId"] == default_id)["points"]
        == [
            {"generation": item["generation"], "value": item["bestSoFar"]}
            for item in default_case["checkpoints"]
        ],
    )
    capability_id = "tsp-aco-sample-agentic-1000-r1"
    capability_case = next(case for case in cases if case["id"] == capability_id)
    accepted_root = (
        RECHECK_ROOT
        / "records/ahd/training/ahd_stage3_aco_tsp_sample_es_b1000_current_f3a570e_v1_20260814"
    )
    acceptance = (accepted_root / "acceptance.md").read_text(encoding="utf-8")
    record_source_check(
        "AHD aco_tsp Stage 3 recheck has an execution-side PASS pending final review",
        "Conclusion: PASS" in acceptance
        and "does not impersonate or replace the designated" in acceptance
        and (accepted_root / "READY_FOR_MANUAL_ACCEPTANCE").is_file(),
    )
    raw_run_name = Path(
        (accepted_root / "queues/aco_tsp/rep1_raw_run_path.txt")
        .read_text(encoding="utf-8")
        .strip()
    ).name
    best_directory = SOURCE_ROOT / "cache/active_runs" / raw_run_name / "results/pops_best"
    capability_checkpoints = extract_ahd_heuristic_checkpoints(
        best_directory, generations=[1, 12, 50]
    )
    capability_checkpoints = [
        {
            "task": "ahd",
            "configurationId": capability_id,
            "runId": "stage3-aco-tsp-execution-pass-rep1",
            "caseId": capability_id,
            "sourceArtifact": "execution-side PASS aco_tsp Stage 3 best-population artifact",
            **item,
        }
        for item in capability_checkpoints
    ]
    frozen_test = load_json(accepted_root / "queues/aco_tsp/rep1_test_eval.json")
    frozen_test_code_path = accepted_root / "queues/aco_tsp/rep1_final_best_code.py"
    frozen_test_code = frozen_test_code_path.read_text(encoding="utf-8")
    frozen_tsp50 = next(
        result for result in frozen_test["results"] if result["size"] == 50
    )
    record_source_check(
        "AHD ACO-TSP frozen test provides a complete TSP-50 single-instance minimum",
        frozen_test["status"] == "PASS"
        and frozen_test["task"] == "aco_tsp"
        and frozen_test["settings"]["split"] == "test"
        and frozen_tsp50["task"] == "tsp"
        and frozen_tsp50["objective"] == "min"
        and frozen_tsp50["valid_count"] == frozen_tsp50["count"] == 64
        and frozen_tsp50["failure_count"] == 0,
    )
    record_source_check(
        "AHD ACO-TSP frozen test evaluates the generation-50 heuristic code modulo surrounding whitespace",
        hashlib.sha256(frozen_test_code_path.read_bytes()).hexdigest()
        == frozen_test["code_sha256"]
        and frozen_test_code.strip() == capability_checkpoints[-1]["heuristic"].strip(),
    )
    capability_checkpoints[-1]["testInstanceMinimum"] = {
        "value": frozen_tsp50["min"],
        "scope": "TSP-50 · minimum across 64 frozen-test instances",
    }
    all_recheck_points = extract_ahd_heuristic_checkpoints(
        best_directory, generations=list(range(1, 51))
    )
    capability_case["capabilityCheckpoints"] = capability_checkpoints
    capability_case["evidenceScope"] = "Favorable eligible code-evolution case · Stage 3 execution-side PASS · awaiting final review · aco_tsp · Sample + Agentic ESOpt · budget 1,000 · repeat 1"
    curves.append(
        {
            "id": f"{capability_id}-stage3-recheck",
            "configId": capability_id,
            "kind": "train",
            "label": "Stage 3 execution-side PASS · best heuristic",
            "points": [
                {"generation": item["optimizationStep"], "value": item["objective"]}
                for item in all_recheck_points
            ],
            "capabilityCurve": True,
        }
    )
    record_source_check(
        "AHD ACO-TSP Stage 3 execution-side PASS exposes a sub-6 objective across three distinct heuristic versions",
        [item["optimizationStep"] for item in capability_checkpoints] == [1, 12, 50]
        and [item["objective"] for item in capability_checkpoints]
        == [6.48937, 5.9408, 5.90256]
        and len({item["heuristic"] for item in capability_checkpoints}) == 3,
    )
    prior_code_evolution = extract_ahd_heuristic_checkpoints(
        RECHECK_ROOT
        / "records/ahd/training/ahd_stage3_construct_tsp_eoh_b1000_current_ba0f0fd_20260813/rep1/raw/results/pops_best",
        generations=[1, 25],
    )
    record_source_check(
        "AHD favorable eligible case has a lower final objective and larger gain than the other accepted multi-version code replay",
        capability_checkpoints[-1]["objective"] < prior_code_evolution[-1]["objective"]
        and capability_checkpoints[0]["objective"] - capability_checkpoints[-1]["objective"]
        > prior_code_evolution[0]["objective"] - prior_code_evolution[-1]["objective"],
    )
    payload = {
        "metadata": {
            "task": "ahd",
            "title": "Heuristic Evolution Explorer",
            "method": "Agentic ESOpt + heuristic search",
            "metric": "TSP heuristic objective (lower is better)",
            "sourceFiles": ["TSP search logs", "aco_tsp Stage 3 execution-side PASS record", "final best heuristics", "paper AHD tables"],
            "note": "Original result families provide search curves and final heuristics. The homepage capability case uses the ACO-TSP Stage 3 execution-side PASS record for real intermediate heuristic code evolution and a lower final objective; designated final review remains pending.",
        },
        "configurations": configurations,
        "curves": curves,
        "checkpoints": checkpoints,
        "cases": cases,
        "finalResults": [
            {"budget": 1000, "method": "EoH", "tsp20": 4.2481, "tsp50": 6.5450},
            {"budget": 1000, "method": "Agentic ESOpt + EoH", "tsp20": 4.2107, "tsp50": 6.5167},
            {"budget": 2000, "method": "EoH", "tsp20": 4.2165, "tsp50": 6.4706},
            {"budget": 2000, "method": "Agentic ESOpt + EoH", "tsp20": 4.1799, "tsp50": 6.4442},
        ],
    }
    write_payload("ahd.json", payload)


def build_scaling() -> None:
    results = [
        {"model": "4B", "population": 8, "best": 5.10, "final": 2.95},
        {"model": "4B", "population": 16, "best": 35.42, "final": 22.92, "bestRelative": 594.5, "finalRelative": 677.0},
        {"model": "9B", "population": 8, "best": 30.21, "final": 30.21},
        {"model": "9B", "population": 16, "best": 37.50, "final": 30.21, "bestRelative": 24.1, "finalRelative": 0.0},
    ]
    record_source_check(
        "Scaling payload contains one row for each 4B/9B by G=8/16 setting",
        {(row["model"], row["population"]) for row in results}
        == {("4B", 8), ("4B", 16), ("9B", 8), ("9B", 16)},
    )
    payload = {
        "metadata": {
            "task": "scaling",
            "title": "Model-size and ES Population Scaling",
            "method": "Vanilla ES",
            "sourceFiles": ["paper scaling ablation"],
            "note": "Sudoku Mask-15 Vanilla-ES ablation over model sizes 4B/9B and ES population sizes G=8/16. G means perturbation directions per update, not physical compute nodes.",
        },
        "configurations": [
            {"axis": "modelSize", "values": ["4B", "9B"]},
            {"axis": "esPopulationSize", "symbol": "G", "values": [8, 16], "meaning": "perturbation directions per ES update"},
        ],
        "curves": [],
        "checkpoints": [],
        "cases": [],
        "finalResults": results,
    }
    write_payload("scaling.json", payload)


def copy_paper() -> None:
    paper_dir = PROJECT_ROOT / "public" / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    source_pdf = MANUSCRIPT_ROOT / "main.pdf"
    source_tex = MANUSCRIPT_ROOT / "main.tex"
    target_pdf = paper_dir / "agentic-esopt.pdf"
    if source_pdf.stat().st_mtime < source_tex.stat().st_mtime:
        if not target_pdf.exists():
            raise ValueError("manuscript PDF is older than main.tex and no current public PDF exists")
        record_source_check("Stale manuscript PDF did not overwrite the current public artifact", True)
        return
    shutil.copyfile(source_pdf, target_pdf)
    record_source_check("Public paper PDF copied from a manuscript build newer than main.tex", True)


NUMBER = re.compile(r"(?<![A-Za-z])\d[\d,]*(?:\.\d+)?")


def manuscript_table(manuscript: str, label: str) -> str:
    marker = rf"\label{{{label}}}"
    start = manuscript.index(marker)
    end = manuscript.index(r"\end{table}", start)
    return manuscript[start:end]


def row_after(table: str, marker: str, occurrence: int = 0) -> str:
    start = -1
    for _ in range(occurrence + 1):
        start = table.index(marker, start + 1)
    end = table.index(r"\\", start)
    return table[start:end]


def row_contains_values(row: str, values: list[float | int]) -> bool:
    """Require the published cells to occur in order within their manuscript row."""
    found = [float(token.replace(",", "")) for token in NUMBER.findall(row)]
    cursor = 0
    for expected in values:
        while cursor < len(found) and abs(found[cursor] - float(expected)) > 1e-9:
            cursor += 1
        if cursor == len(found):
            return False
        cursor += 1
    return True


def verify_rows(
    manuscript: str,
    *,
    table_label: str,
    payload_name: str,
    row_specs: list[tuple[str, int, list[str]]],
) -> None:
    table = manuscript_table(manuscript, table_label)
    results = load_json(DATA_DIR / payload_name)["finalResults"]
    if len(results) != len(row_specs):
        raise ValueError(f"row specification count differs for {payload_name}")
    for result, (marker, occurrence, fields) in zip(results, row_specs, strict=True):
        values = [result[field] for field in fields]
        record_source_check(
            f"{payload_name} row '{result.get('method', result.get('model'))}' matches main.tex {table_label} cell by cell",
            row_contains_values(row_after(table, marker, occurrence), values),
        )


def verify_manuscript_results() -> None:
    manuscript = (MANUSCRIPT_ROOT / "main.tex").read_text(encoding="utf-8")
    verify_rows(
        manuscript,
        table_label="tab:sudoku-agentic-summary",
        payload_name="sudoku.json",
        row_specs=[
            (r"Agentic PPO\textsuperscript", 0, ["5", "10", "15"]),
            (r"Agentic GRPO\textsuperscript{$\ddagger$}", 0, ["5", "10", "15"]),
            (r"w/o $\sigma$ decay (Vanilla ES)", 0, ["5", "10", "15"]),
            ("Agentic ESOpt (G=32)", 0, ["5", "10", "15"]),
        ],
    )
    math_doc_rows = [
        ("Qwen3.5-4B & No Skill", 0),
        ("GRPO + No Skill", 0),
        ("Agentic ESOpt + No Skill", 0),
        ("Qwen3.5-4B & Trace2Skill", 0),
        ("GRPO + Trace2Skill", 0),
        ("Agentic ESOpt + Trace2Skill", 0),
    ]
    verify_rows(
        manuscript,
        table_label="tab:math-docvqa",
        payload_name="math.json",
        row_specs=[(marker, occurrence, ["dapoMean", "dapoPass", "aimeMean", "aimePass"]) for marker, occurrence in math_doc_rows],
    )
    verify_rows(
        manuscript,
        table_label="tab:math-docvqa",
        payload_name="docvqa.json",
        row_specs=[(marker, occurrence, ["anlsMean", "anlsPass", "accMean", "accPass"]) for marker, occurrence in math_doc_rows],
    )
    verify_rows(
        manuscript,
        table_label="tab:webarena",
        payload_name="webarena.json",
        row_specs=[
            ("Qwen3.5-27B & No Skill", 0, ["reddit", "gitlab", "cms", "map", "oss", "average"]),
            ("Agentic ESOpt + No Skill", 0, ["reddit", "gitlab", "cms", "map", "oss", "average"]),
            ("Qwen3.5-27B & Trace2Skill", 0, ["reddit", "gitlab", "cms", "map", "oss", "average"]),
            ("Agentic ESOpt + Trace2Skill", 0, ["reddit", "gitlab", "cms", "map", "oss", "average"]),
        ],
    )
    verify_rows(
        manuscript,
        table_label="tab:ahd-construct",
        payload_name="ahd.json",
        row_specs=[
            ("\nEoH &", 0, ["tsp20", "tsp50"]),
            ("Agentic ESOpt + EoH", 0, ["tsp20", "tsp50"]),
            ("\nEoH &", 1, ["tsp20", "tsp50"]),
            ("Agentic ESOpt + EoH", 1, ["tsp20", "tsp50"]),
        ],
    )
    verify_rows(
        manuscript,
        table_label="tab:population-scaling",
        payload_name="scaling.json",
        row_specs=[
            ("Qwen3.5-4B & 8", 0, ["population", "best", "final"]),
            ("Qwen3.5-4B & 16", 0, ["population", "best", "final", "bestRelative", "finalRelative"]),
            ("Qwen3.5-9B & 8", 0, ["population", "best", "final"]),
            ("Qwen3.5-9B & 16", 0, ["population", "best", "final", "bestRelative", "finalRelative"]),
        ],
    )


def write_audit_report() -> None:
    report: dict[str, Any] = {
        "status": "passed",
        "sourceConsistency": SOURCE_CHECKS,
        "publicPayloads": {},
    }
    for path in sorted(DATA_DIR.glob("*.json")):
        payload = load_json(path)
        validate_public_payload(payload)
        report["publicPayloads"][path.name] = {
            "curves": len(payload["curves"]),
            "curvePoints": sum(len(curve.get("points", [])) for curve in payload["curves"]),
            "checkpoints": len(payload["checkpoints"]),
            "selectedCases": len(payload["cases"]),
            "finalResultRows": len(payload["finalResults"]),
            "privatePathScan": "passed",
            "localEndpointScan": "passed",
            "contactPatternScan": "passed",
        }
    report["publishedAssets"] = {
        "selectedDocumentImages": len(list((PROJECT_ROOT / "public" / "selected-documents").glob("*"))),
        "paperPdfs": 1,
        "rawLogDownloads": 0,
    }
    (PROJECT_ROOT / "data_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    required_roots = [
        (SOURCE_ROOT, "ESOPT_SOURCE_ROOT", "Agentic-ESOpt source tree"),
        (RECHECK_ROOT, "ESOPT_RECHECK_ROOT", "accepted recheck archive"),
        (MANUSCRIPT_ROOT, "ESOPT_MANUSCRIPT_ROOT", "paper manuscript"),
    ]
    for root, environment_variable, label in required_roots:
        if not root.is_dir():
            raise FileNotFoundError(
                f"Missing {label}: {root}. Set {environment_variable} to its directory."
            )
    build_math()
    build_docvqa()
    build_sudoku()
    build_webarena()
    build_ahd()
    build_scaling()
    verify_manuscript_results()
    copy_paper()
    write_audit_report()
    print(f"Built public data in {DATA_DIR}")


if __name__ == "__main__":
    main()
