# Project layout

The repository is organized around five maintained tasks.

```text
Dynamic-Agent/
|-- es/                       shared model-weight ES and run-state code
|-- sudoku-train-time/        Sudoku environment and Dynamic-Agent runner
|-- math-train-time/          Math environment and HTTP/vLLM runners
|-- docvqa-train-time/        DocVQA environment and Dynamic-Agent runner
|-- webarena-train-time/      WebArena and Trace2Skill integration
|-- ahd-test-time/            test-time EoH and Dynamic-Agent
|-- verl_trace2skill/         VERL tool, parser, and reward functions
|-- trace2skill-settings/     Math/DocVQA Trace2Skill adapters
|-- scripts/                  canonical task launchers and data checks
`-- data/                     stable task data paths (large files ignored)
```

The user-facing entrypoints are:

```text
scripts/sudoku/{run_es,run_grpo}.sh
scripts/math/{run,run_vllm_es_4gpu,run_grpo,run_trace2skill,run_trace2skill_es}.sh
scripts/docvqa/{start_vision_server,run,run_grpo,run_trace2skill,run_trace2skill_es}.sh
scripts/webarena/run.sh
scripts/ahd/run.sh
```

Task-independent machine settings belong in `scripts/settings.local.env`.
Run outputs belong in `runs/` or `cache/active_runs/`; neither is source code.
Every Dynamic-Agent run writes a replayable `history.json` beside its outputs.

Paper sources, figures, plotting utilities, result archives, machine-local
external checkouts, and obsolete code are deliberately excluded from the core
GitHub tree.
