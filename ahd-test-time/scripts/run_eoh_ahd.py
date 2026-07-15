#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EOH_SRC = REPO_ROOT / "ahd-test-time" / "methods" / "eoh" / "original" / "eoh" / "src"
DATA_ROOT = REPO_ROOT / "data" / "ahd" / "datasets"
SETTING_ROOT = REPO_ROOT / "data" / "ahd" / "settings"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EOH_SRC))


def env_bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


TASKS = {
    "construct_tsp": {
        "problem": "tsp_construct",
        "data": DATA_ROOT / "tsp_constructive",
        "cfg": SETTING_ROOT / "cfg" / "problem" / "tsp_constructive.yaml",
        "prompt": SETTING_ROOT / "prompts" / "tsp_constructive",
    },
    "construct_kp": {
        "problem": "kp_constructive",
        "data": DATA_ROOT / "kp_constructive",
        "cfg": SETTING_ROOT / "cfg" / "problem" / "kp_constructive.yaml",
        "prompt": SETTING_ROOT / "prompts" / "kp_constructive",
    },
    "construct_asp": {
        "problem": "asp_constructive",
        "data": None,
        "cfg": SETTING_ROOT / "cfg" / "problem" / "asp_constructive.yaml",
        "prompt": SETTING_ROOT / "prompts" / "asp_constructive",
    },
    "aco_tsp": {
        "problem": "tsp_aco",
        "data": DATA_ROOT / "tsp_aco",
        "cfg": SETTING_ROOT / "cfg" / "problem" / "tsp_aco.yaml",
        "prompt": SETTING_ROOT / "prompts" / "tsp_aco",
    },
    "aco_cvrp": {
        "problem": "cvrp_aco",
        "data": DATA_ROOT / "cvrp_aco",
        "cfg": SETTING_ROOT / "cfg" / "problem" / "cvrp_aco.yaml",
        "prompt": SETTING_ROOT / "prompts" / "cvrp_aco",
    },
    "aco_bpp": {
        "problem": "bpp_offline_aco",
        "data": DATA_ROOT / "bpp_offline_aco",
        "cfg": SETTING_ROOT / "cfg" / "problem" / "bpp_offline_aco.yaml",
        "prompt": SETTING_ROOT / "prompts" / "bpp_offline_aco",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), default="construct_tsp")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--method", choices=["eoh", "es"], default="eoh")
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID", "run1"))
    parser.add_argument(
        "--eoh-k",
        "--ec-m1m2-multiplier",
        dest="eoh_k",
        type=float,
        default=float(os.environ.get("EC_M1M2_MULTIPLIER", "1")),
        help="Offspring multiplier k for EoH m1/m2 operators.",
    )
    parser.add_argument("--ec-pop-size", type=int, default=int(os.environ.get("EC_POP_SIZE", "10")))
    parser.add_argument("--ec-generations", type=int, default=int(os.environ.get("EC_GENERATIONS", "25")))
    parser.add_argument("--llm-local-url", default=os.environ.get("LLM_LOCAL_URL", "http://127.0.0.1:11012/completions"))
    parser.add_argument(
        "--llm-local-timeout",
        type=float,
        default=float(os.environ.get("LLM_LOCAL_TIMEOUT", "600")),
    )
    parser.add_argument("--es-engine-urls", default=os.environ.get("ES_ENGINE_URLS", ""))
    parser.add_argument("--es-operators", default=os.environ.get("ES_OPERATORS", "e1,e2,m1,m2"))
    parser.add_argument("--es-directions", type=int, default=int(os.environ.get("ES_DIRECTIONS", "10")))
    parser.add_argument(
        "--es-sigma-start",
        dest="es_sigma_start",
        type=float,
        default=float(os.environ.get("ES_SIGMA_START", "1e-3")),
    )
    parser.add_argument(
        "--es-sigma-end",
        type=float,
        default=float(os.environ.get("ES_SIGMA_END", os.environ.get("ES_SIGMA_START", "1e-3"))),
        help="Perturbation scale at the final EoH generation.",
    )
    parser.add_argument(
        "--es-sigma-schedule",
        choices=["constant", "linear", "cosine"],
        default=os.environ.get("ES_SIGMA_SCHEDULE", "constant"),
    )
    parser.add_argument(
        "--es-sigma-warmup-steps",
        type=int,
        default=int(os.environ.get("ES_SIGMA_WARMUP_STEPS", "0")),
    )
    parser.add_argument("--es-alpha", type=float, default=float(os.environ.get("ES_ALPHA", "5e-4")))
    parser.add_argument("--es-seed", type=int, default=int(os.environ.get("ES_SEED", "2024")))
    parser.add_argument("--es-parameter-scope", default=os.environ.get("ES_PARAMETER_SCOPE", "full"))
    parser.add_argument("--es-target-modules", default=os.environ.get("ES_TARGET_MODULES", ""))
    parser.add_argument("--es-disable-update", action="store_true", default=env_bool("ES_DISABLE_UPDATE"))
    parser.add_argument("--es-history-file", default=os.environ.get("ES_HISTORY_FILE", ""))
    parser.add_argument("--resume-history", default=os.environ.get("ES_RESUME_HISTORY", ""))
    parser.add_argument("--continue-path", default=os.environ.get("AHD_CONTINUE_PATH", ""))
    parser.add_argument("--continue-id", type=int, default=int(os.environ.get("AHD_CONTINUE_ID", "0")))
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

    if args.ec_pop_size <= 0 or args.ec_generations <= 0:
        raise ValueError("--ec-pop-size and --ec-generations must be positive.")
    if args.eoh_k <= 0:
        raise ValueError("--eoh-k must be positive.")
    es_sigma_end = args.es_sigma_end
    if (
        not math.isfinite(args.es_sigma_start)
        or not math.isfinite(es_sigma_end)
        or args.es_sigma_start < 0
        or es_sigma_end < 0
    ):
        raise ValueError("ES sigma endpoints must be finite and non-negative.")
    if not math.isfinite(args.es_alpha) or args.es_alpha < 0:
        raise ValueError("--es-alpha must be finite and non-negative.")

    from eoh import eoh
    from eoh.utils.getParas import Paras

    paras = Paras()
    if args.method == "es":
        es_engine_urls = [url.strip() for url in args.es_engine_urls.split(",") if url.strip()]
        es_operators = [op.strip() for op in args.es_operators.split(",") if op.strip()]
        es_target_modules = [module.strip() for module in args.es_target_modules.split(",") if module.strip()]
        if not es_target_modules:
            es_target_modules = None
        if not es_operators:
            raise ValueError("ES operator list is empty.")
        if not es_engine_urls:
            es_engine_urls = [
                "http://127.0.0.1:11013/completions",
                "http://127.0.0.1:11014/completions",
                "http://127.0.0.1:11015/completions",
                "http://127.0.0.1:11016/completions",
            ]
        output_path = REPO_ROOT / "cache" / "active_runs" / (
            f"{args.task}_{args.split}_es_"
            f"sigma{args.es_sigma_start:g}_alpha{args.es_alpha:g}_{args.run_id}"
        )
        paras.set_paras(
            method="eoh",
            problem=task["problem"],
            llm_use_local=True,
            llm_local_url=es_engine_urls[0],
            llm_es_enabled=True,
            llm_es_engine_urls=es_engine_urls,
            llm_es_operators=es_operators,
            llm_es_directions=args.es_directions,
            llm_es_sigma_start=args.es_sigma_start,
            llm_es_sigma_end=es_sigma_end,
            llm_es_sigma_schedule=args.es_sigma_schedule,
            llm_es_sigma_warmup_steps=args.es_sigma_warmup_steps,
            llm_es_alpha=args.es_alpha,
            llm_es_seed=args.es_seed,
            llm_es_reward_mode="improvement",
            llm_es_reward_normalization="zscore",
            llm_es_parameter_scope=args.es_parameter_scope,
            llm_es_target_modules=es_target_modules,
            llm_es_reward_floor=-1.0,
            llm_es_disable_update=args.es_disable_update,
            llm_es_history_path=args.es_history_file or None,
            llm_es_resume_history=args.resume_history or None,
            llm_local_timeout=args.llm_local_timeout,
            ec_pop_size=args.ec_pop_size,
            ec_n_pop=args.ec_generations,
            ec_operator_attempts=1,
            ec_m1m2_multiplier=args.eoh_k,
            exp_n_proc=4,
            data_split=args.split,
            problem_data_root=str(task["data"]) if task["data"] is not None else None,
            exp_output_path=str(output_path),
            exp_use_continue=bool(args.continue_path),
            exp_continue_path=args.continue_path,
            exp_continue_id=args.continue_id,
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
            ec_pop_size=args.ec_pop_size,
            ec_n_pop=args.ec_generations,
            ec_operator_attempts=1,
            ec_m1m2_multiplier=args.eoh_k,
            exp_n_proc=4,
            data_split=args.split,
            problem_data_root=str(task["data"]) if task["data"] is not None else None,
            exp_output_path=str(
                REPO_ROOT / "cache" / "active_runs" / f"{args.task}_{args.split}_eoh_{args.run_id}"
            ),
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
