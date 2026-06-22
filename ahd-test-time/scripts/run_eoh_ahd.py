#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EOH_SRC = REPO_ROOT / "ahd-test-time" / "methods" / "eoh" / "original" / "eoh" / "src"
DATA_ROOT = REPO_ROOT / "data" / "ahd" / "datasets"
SETTING_ROOT = REPO_ROOT / "data" / "ahd" / "settings"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EOH_SRC))

from eoh import eoh  # noqa: E402
from eoh.utils.getParas import Paras  # noqa: E402


TASKS = {
    "construct_tsp": {
        "problem": "tsp_construct",
        "data": DATA_ROOT / "tsp_constructive",
        "cfg": SETTING_ROOT / "cfg" / "problem" / "tsp_constructive.yaml",
        "prompt": SETTING_ROOT / "prompts" / "tsp_constructive",
        "implemented": True,
    },
    "construct_kp": {
        "problem": "kp_constructive",
        "data": DATA_ROOT / "kp_constructive",
        "cfg": SETTING_ROOT / "cfg" / "problem" / "kp_constructive.yaml",
        "prompt": SETTING_ROOT / "prompts" / "kp_constructive",
        "implemented": True,
    },
    "construct_asp": {
        "problem": "asp_constructive",
        "data": None,
        "cfg": SETTING_ROOT / "cfg" / "problem" / "asp_constructive.yaml",
        "prompt": SETTING_ROOT / "prompts" / "asp_constructive",
        "implemented": True,
    },
    "aco_tsp": {
        "problem": "tsp_aco",
        "data": DATA_ROOT / "tsp_aco",
        "cfg": SETTING_ROOT / "cfg" / "problem" / "tsp_aco.yaml",
        "prompt": SETTING_ROOT / "prompts" / "tsp_aco",
        "implemented": True,
    },
    "aco_cvrp": {
        "problem": "cvrp_aco",
        "data": DATA_ROOT / "cvrp_aco",
        "cfg": SETTING_ROOT / "cfg" / "problem" / "cvrp_aco.yaml",
        "prompt": SETTING_ROOT / "prompts" / "cvrp_aco",
        "implemented": True,
    },
    "aco_bpp": {
        "problem": "bpp_offline_aco",
        "data": DATA_ROOT / "bpp_offline_aco",
        "cfg": SETTING_ROOT / "cfg" / "problem" / "bpp_offline_aco.yaml",
        "prompt": SETTING_ROOT / "prompts" / "bpp_offline_aco",
        "implemented": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), default="construct_tsp")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--method", choices=["eoh", "es"], default="eoh")
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID", "run1"))
    parser.add_argument("--llm-local-url", default=os.environ.get("LLM_LOCAL_URL", "http://127.0.0.1:11012/completions"))
    parser.add_argument(
        "--llm-local-timeout",
        type=float,
        default=float(os.environ.get("LLM_LOCAL_TIMEOUT", "600")),
    )
    parser.add_argument("--es-engine-urls", default=os.environ.get("ES_ENGINE_URLS", ""))
    parser.add_argument("--es-operators", default=os.environ.get("ES_OPERATORS", "e1,e2,m1,m2"))
    parser.add_argument("--es-sigma", type=float, default=float(os.environ.get("ES_SIGMA", "1e-3")))
    parser.add_argument("--es-alpha", type=float, default=float(os.environ.get("ES_ALPHA", "5e-4")))
    parser.add_argument("--eva-timeout", type=float, default=float(os.environ.get("EVA_TIMEOUT", "0")))
    return parser.parse_args()


def require_setting(task: dict) -> None:
    if task["data"] is not None and not task["data"].exists():
        raise FileNotFoundError(task["data"])
    if not task["cfg"].exists():
        raise FileNotFoundError(task["cfg"])
    if not task["prompt"].exists():
        raise FileNotFoundError(task["prompt"])


def main() -> None:
    args = parse_args()
    task = TASKS[args.task]
    require_setting(task)

    print(f"[ahd] task={args.task} split={args.split} method={args.method}", flush=True)
    print(f"[ahd] data={task['data'] or 'none'}", flush=True)
    print(f"[ahd] cfg={task['cfg']}", flush=True)
    print(f"[ahd] prompt={task['prompt']}", flush=True)

    if not task["implemented"]:
        raise NotImplementedError(
            f"{args.task} data/settings are present, but the EoH runner is not wired for this task yet."
        )

    paras = Paras()
    if args.method == "es":
        es_engine_urls = [url.strip() for url in args.es_engine_urls.split(",") if url.strip()]
        es_operators = [op.strip() for op in args.es_operators.split(",") if op.strip()]
        if not es_operators:
            raise ValueError("ES operator list is empty.")
        if not es_engine_urls:
            es_engine_urls = [
                "http://127.0.0.1:11013/completions",
                "http://127.0.0.1:11014/completions",
                "http://127.0.0.1:11015/completions",
                "http://127.0.0.1:11016/completions",
            ]
        paras.set_paras(
            method="eoh",
            problem=task["problem"],
            llm_use_local=True,
            llm_local_url=es_engine_urls[0],
            llm_es_enabled=True,
            llm_es_engine_urls=es_engine_urls,
            llm_es_operators=es_operators,
            llm_es_directions=10,
            llm_es_sigma=args.es_sigma,
            llm_es_alpha=args.es_alpha,
            llm_es_reward_mode="improvement",
            llm_es_reward_normalization="zscore",
            llm_es_parameter_scope="full",
            llm_es_reward_floor=-1.0,
            llm_local_timeout=args.llm_local_timeout,
            ec_pop_size=10,
            ec_n_pop=25,
            ec_operator_attempts=1,
            exp_n_proc=4,
            data_split=args.split,
            problem_data_root=str(task["data"]) if task["data"] is not None else None,
            exp_output_path=(
                f"./cache/active_runs/{args.task}_{args.split}_es_"
                f"sigma{args.es_sigma:g}_alpha{args.es_alpha:g}_{args.run_id}"
            ),
            eva_invalid_objective=float("inf"),
            exp_debug_mode=False,
        )
    else:
        paras.set_paras(
            method="eoh",
            problem=task["problem"],
            llm_use_local=True,
            llm_local_url=args.llm_local_url,
            llm_es_enabled=False,
            ec_pop_size=10,
            ec_n_pop=25,
            ec_operator_attempts=1,
            exp_n_proc=4,
            data_split=args.split,
            problem_data_root=str(task["data"]) if task["data"] is not None else None,
            exp_output_path=f"./cache/active_runs/{args.task}_{args.split}_eoh_{args.run_id}",
            eva_invalid_objective=float("inf"),
            llm_local_timeout=args.llm_local_timeout,
            exp_debug_mode=False,
        )

    if args.eva_timeout > 0:
        paras.eva_timeout = args.eva_timeout

    evolution = eoh.EVOL(paras)
    evolution.run()


if __name__ == "__main__":
    main()
