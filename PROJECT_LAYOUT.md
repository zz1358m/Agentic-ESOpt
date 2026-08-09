# Project layout

The repository is organized around five maintained tasks. The root `README.md`
is the complete reproduction guide; this file is the compact directory map.

```text
Agentic-ESOpt/
|-- algorithms/               shared optimization and training implementations
|   |-- es/                   seeded model ES, schedules, history, and replay
|   |-- trace2skill-settings/ Trace2Skill configs, prompts, scripts, and skills
|   |-- verl/                 bundled VERL used by Math/DocVQA GRPO
|   `-- verl_trace2skill/     VERL tools, parsers, rewards, sandbox, and tests
|-- sudoku-train-time/        Sudoku env, ES/GRPO runners, profiles, curated results
|-- math-train-time/          Math ReAct env, vLLM ES runner, and tests
|-- docvqa-train-time/        DocVQA bash/OCR env, vLLM/GRPO runners, and tests
|-- webarena-train-time/      WebArena env, data tools, ES/Trace2Skill/SkillOpt/JITRL
|-- ahd-test-time/            four AHD methods, six tasks, evaluators, curated programs
|-- scripts/                  portable task launchers and data validation
|-- data/                     stable task data/settings contract
|-- vllm_math_es_worker.py    in-process Math vLLM ES worker
|-- README.md                 full structure and experiment commands
|-- PROJECT_LAYOUT.md         this compact map
`-- LICENSE
```

The portable user-facing entry points are:

```text
scripts/sudoku/{run_es,run_grpo,run_grpo_t1}.sh
scripts/es_skill_workflow.sh {math,docvqa} {es-train,eval,distill-skill,skill-eval}
scripts/math/{run_vllm_es_4gpu,run_grpo,run_trace2skill}.sh
scripts/docvqa/{run_grpo,run_react_verl_grpo,run_trace2skill,run_four_gpu_experiment_pipeline}.sh
scripts/webarena/run.sh
scripts/ahd/{start_llama31_8b_servers,run,run_four_method_ahd}.sh
```

The effective defaults for every maintained launcher are listed in
[`scripts/RUN_HYPERPARAMETERS.md`](scripts/RUN_HYPERPARAMETERS.md).

`ahd-test-time/methods/eoh/` is the only AHD runtime. Its canonical four-method
runner is `ahd-test-time/scripts/run_eoh_ahd.py`; the similarly named
`run_ahd_four_methods.py` is a compatibility forwarder. The exact eight
curated result folders are documented in the root README.

Task-independent machine settings belong in `scripts/settings.local.env`.
Ordinary run outputs belong in `runs/` or `cache/active_runs/` and remain
ignored. Every Agentic-ESOpt run writes a replayable `history.json`. Curated
AHD programs, Math/DocVQA bundles, WebArena ES results, and Sudoku
heatmaps/training logs are intentional result-archive exceptions.
