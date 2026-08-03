# Dynamic-Agent

This repository maintains one model-weight evolution method, **Dynamic-Agent**,
across five agentic tasks, together with the baselines needed for comparison.
The GitHub tree is task-oriented and contains only runnable code, small source
data, data contracts, and documentation; paper assets and obsolete experiments
are not part of the core repository.

## Maintained matrix

| Task | Dynamic-Agent | Multi-turn GRPO | Trace2Skill | Trace2Skill + Dynamic-Agent | EoH |
| --- | --- | --- | --- | --- | --- |
| Sudoku | Yes | Yes | — | — | — |
| Math | Yes | Yes | Yes | Yes | — |
| DocVQA | Yes | Yes | Yes | Yes | — |
| WebArena | Yes | — | Yes | Yes | — |
| AHD (test-time) | Yes | — | — | — | Yes, configurable `k` |

`es` in script and HTTP route names refers to the optimization mechanism. The
method implemented by these runners is Dynamic-Agent; it is not the paper's
vanilla-ES baseline.

## Quick start

Copy the machine-local settings template and edit model paths, endpoint URLs,
and GPU settings:

```bash
cp scripts/settings.example.env scripts/settings.local.env
```

Check the local data contract before launching a job:

```bash
python scripts/check_data.py
python scripts/check_data.py --task math --strict
```

Large datasets, external checkouts, checkpoints, and run outputs are ignored by
git. Their stable locations and preparation commands are documented in
[`data/README.md`](data/README.md).

The launchers target Linux GPU environments. Install the shared EoH/runtime
dependencies with:

```bash
python -m pip install -e 'ahd-test-time/methods/eoh/original/eoh[all]'
python -m pip install pillow datasets pandas pyarrow math-verify
python -m pip install -e ./verl
```

Math's in-process runner additionally needs `vllm`. Multi-turn GRPO uses an
included, locally adapted `verl` tree for Math and DocVQA. Sudoku GRPO still
uses a separate `verl-tool` checkout. WebArena and the standalone Trace2Skill
pipeline use the external checkouts listed in `data/README.md`; those
fast-moving projects are kept out of a single pinned root environment.

Dynamic-Agent policy servers expose an OpenAI-compatible generation route plus:

```text
/es/init  /es/apply  /es/revert  /es/update  /es/reset  /es/status
```

The reference model-server integration is in
`ahd-test-time/methods/eoh/original/eoh/src/eoh/llm_local_server/llama31_instruct_server.py`.
Start four local endpoints with:

```bash
MODEL=/path/to/model scripts/ahd/start_llama31_8b_servers.sh
```

Set `GPUS`, `PORTS`, or `MODEL_SERVER_EXTRA_ARGS` when the defaults do not fit
the machine. These text endpoints serve Sudoku, Math, WebArena, and AHD.
DocVQA uses the vision-language server documented in its section.

## Sigma schedules and replay

Every maintained Dynamic-Agent runner supports an explicit perturbation
schedule from `sigma_start` to `sigma_end`:

```text
--sigma-start 0.001 --sigma-end 0.0001
--sigma-schedule constant|linear|cosine
--sigma-warmup-steps 0
```

With at least two generations, the first scheduled generation uses
`sigma_start` and the final generation uses exactly `sigma_end`. `constant`
always uses `sigma_start`; a one-generation run also uses `sigma_start`. AHD
uses the same flags with the `--es-` prefix.

Each update is atomically written to `history.json`. On a fresh server,
`--resume-history` deterministically replays past seed/reward updates before new
rollouts begin. The replay validates the original seed stream and population,
so an incompatible history fails rather than silently creating a different run.

| Task | Default history |
| --- | --- |
| Sudoku | `runs/sudoku_es/<run-id>/history.json` |
| Math HTTP | `runs/math_es/<run-id>/history.json` |
| Math vLLM | `runs/math_es_vllm/<run-id>/history.json` |
| DocVQA | `runs/docvqa_es/<run-id>/history.json` |
| WebArena | `runs/webrl_lite_full_es/<run-id>/history.json` |
| AHD | `cache/active_runs/<run>/results/es/history.json` |

Set a different output with `--history-file` (`--es-history-file` for AHD).
The task launchers expose matching `*_HISTORY_FILE` and `*_RESUME_HISTORY`
environment variables.

## Sudoku

Generate controlled-mask puzzles if needed:

```bash
python sudoku-train-time/scripts/generate_sudoku_data.py \
  --output-dir data/sudoku --train-size 128 --eval-size 128 \
  --mask-counts 5,10,15,20
```

Run Dynamic-Agent with an explicit decay:

```bash
SUDOKU_TARGET_MASK_COUNT=15 \
SUDOKU_ES_SIGMA_START=1e-3 SUDOKU_ES_SIGMA_END=1e-4 \
SUDOKU_ES_SIGMA_SCHEDULE=cosine scripts/sudoku/run_es.sh
```

Resume model updates:

```bash
SUDOKU_ES_RESUME_HISTORY=runs/sudoku_es/old/history.json \
RUN_ID=sudoku_resumed scripts/sudoku/run_es.sh
```

Run the maintained asynchronous multi-turn GRPO baseline. This baseline uses a
`verl-tool` checkout and the Sudoku tool/reward adapters in this repository:

```bash
VERL_TOOL_ROOT=/path/to/verl-tool \
SUDOKU_TARGET_MASK_COUNT=15 scripts/sudoku/run_grpo.sh
```

## Math

Run Dynamic-Agent against already-running HTTP model servers:

```bash
MATH_ES_SIGMA_START=5e-4 MATH_ES_SIGMA_END=1e-4 \
MATH_ES_SIGMA_SCHEDULE=linear scripts/math/run.sh
```

The in-process multi-GPU vLLM variant has the same schedule and replay
semantics:

```bash
MODEL_PATH=Qwen/Qwen3.5-4B scripts/math/run_vllm_es_4gpu.sh
```

Run multi-turn GRPO using the repository's `verl_trace2skill` bash tool,
reward, and parser implementation:

```bash
scripts/math/run_grpo.sh
```

For the complete fixed long-running DAPO-400 experiment (baseline, training,
post-evaluation, trajectory export, and report), use the persistent pipeline:

```bash
python scripts/math/run_experiment_until_complete.py
```

Its atomic state and append-only control log are written to
`runs/multiturn_grpo/reports/qwen35-4b-math-grpo-dapo400-e15-seed1/`.
The pipeline waits rather than starting a second process when GPUs 3–6 are
occupied, and every evaluation/training stage resumes its existing outputs.

For the training stage alone, use the persistent training watchdog:

```bash
python scripts/math/run_training_until_complete.py
```

It waits until all approved physical GPUs `3,4,5,6` are free, restarts
non-capacity failures at the same limits from VERL's latest checkpoint, and
applies the approved turn/token fallback tiers only after explicit
OOM/context-capacity evidence. Attempts are recorded in the run report tree.

The Math launcher is fixed to physical GPUs `3,4,5,6`, the checked-in
DAPO-Math-17k 400/100 split, batch size 20, `rollout.n=8`, and 15 epochs. It
uses GRPO (DAPO names the dataset only), writes immutable train/validation
JSONL under `runs/multiturn_grpo/trajectories/`, and resumes from the latest
checkpoint automatically. Bash actions use the text protocol
`Action: {"name":"bash",...}` and execute in a run-specific tool workspace.

Run the matched four-replica DAPO-100/AIME 2026 evaluation with:

```bash
python scripts/math/run_four_gpu_eval.py \
  --model-path /path/to/hf_model --out-dir /path/to/eval --resume
```

The ReAct evaluator uses 50 turns, 4096 generated tokens per assistant request,
and a 262144-token context. For the four-sample table comparison, add
`--samples 4 --profile repo-react-v1-50x4096`.

Validate the fixed source data and export replayable trajectory JSONL plus
per-epoch/step success/failure Markdown with:

```bash
python scripts/math/validate_data.py --out /path/to/data_manifest.json
python scripts/trace2skill/export_math_trajectories.py --help
```

The launcher defaults to the bundled `verl/`; set `VERL_ROOT` only to test a
different checkout. Evaluation and PBS helper commands are documented in
[`scripts/trace2skill/README.md`](scripts/trace2skill/README.md).

Run Trace2Skill on existing success/failure traces:

```bash
TRACE_LOGS=runs/math_es/base/trace_logs/dapo_eval \
RUN_ID=math_t2s scripts/math/run_trace2skill.sh
```

Evolve the skill and then optimize the model with Dynamic-Agent:

```bash
TRACE_LOGS=runs/math_es/base/trace_logs/dapo_eval \
RUN_ID=math_t2s_dynamic scripts/math/run_trace2skill_es.sh
```

## DocVQA

Start the included Hugging Face vision-language server, then run
Dynamic-Agent in vision-chat mode:

```bash
DOCVQA_MODEL_PATH=/path/to/vision-language-model \
scripts/docvqa/start_vision_server.sh

DOCVQA_ENDPOINT_MODE=openai_vision_chat \
DOCVQA_ES_ENDPOINTS=http://127.0.0.1:11013 \
DOCVQA_ES_SIGMA_START=5e-4 DOCVQA_ES_SIGMA_END=1e-4 \
DOCVQA_ES_SIGMA_SCHEDULE=cosine scripts/docvqa/run.sh
```

Run multi-turn GRPO:

```bash
scripts/docvqa/run_grpo.sh
```

Run Trace2Skill or Trace2Skill followed by Dynamic-Agent:

```bash
TRACE_LOGS=runs/docvqa_es/base/trace_logs/eval_initial \
RUN_ID=docvqa_t2s scripts/docvqa/run_trace2skill.sh

TRACE_LOGS=runs/docvqa_es/base/trace_logs/eval_initial \
RUN_ID=docvqa_t2s_dynamic scripts/docvqa/run_trace2skill_es.sh
```

The VERL data converter rejects an empty DocVQA validation split. A tiny local
smoke subset is therefore not mistaken for the full baseline dataset.

## WebArena

Required local components include the VAB/WebArena-Lite checkout, the shared
non-Lite train/validation split, the held-out 165-task test split, and the two
external runtime checkouts listed in the data contract. Prepare the split files
with:

```bash
python webarena-train-time/scripts/prepare_webarena_nonlite_split.py
python webarena-train-time/scripts/prepare_vab_webarena_lite_split.py
```

Run the standalone Trace2Skill baseline:

```bash
scripts/webarena/run.sh trace2skill train
scripts/webarena/run.sh trace2skill test
```

Run Dynamic-Agent without or with a Trace2Skill skill:

```bash
scripts/webarena/run.sh no_skill_es train
scripts/webarena/run.sh trace2skill_es train
```

The combination defaults to the skill produced by the `webarena_trace2skill`
run. Set `TRACE2SKILL_RUN_ID` or `TRACE2SKILL_SKILL_FILE` to select another
skill.

For generation-by-generation skill/model co-optimization:

```bash
WEBARENA_TRACE2SKILL_EVERY_GENERATION=1 \
WEBARENA_ES_SIGMA_START=5e-4 WEBARENA_ES_SIGMA_END=1e-4 \
WEBARENA_ES_SIGMA_SCHEDULE=cosine \
scripts/webarena/run.sh trace2skill_es train
```

This mode initializes a missing skill file and evolves it after each
Dynamic-Agent generation.

Additional runner flags can follow the stage, for example
`scripts/webarena/run.sh trace2skill_es train --generations 10`.

## AHD (test-time)

The EoH baseline supports different offspring multipliers `k`:

```bash
EOH_K=1 scripts/ahd/run.sh construct_tsp train eoh
EOH_K=3 scripts/ahd/run.sh construct_tsp train eoh
```

Run Dynamic-Agent + EoH:

```bash
EOH_K=1 ES_SIGMA_START=1e-3 ES_SIGMA_END=1e-4 \
ES_SIGMA_SCHEDULE=cosine scripts/ahd/run.sh construct_tsp train es
```

AHD has two independent continuation states:

- `ES_RESUME_HISTORY` replays model-weight updates.
- `AHD_CONTINUE_PATH` and `AHD_CONTINUE_ID` restore the EoH population.

Use both when continuing a complete Dynamic-Agent+AHD run:

```bash
ES_RESUME_HISTORY=/old/run/results/es/history.json \
AHD_CONTINUE_PATH=/old/run/results/pops/population_generation_10.json \
AHD_CONTINUE_ID=10 scripts/ahd/run.sh construct_tsp train es
```

Supported tasks are `construct_tsp`, `construct_kp`, `construct_asp`,
`aco_tsp`, `aco_cvrp`, and `aco_bpp`.

## Repository layout

- `es/`: shared seeded model updates, sigma schedules, and history utilities.
- `{sudoku,math,docvqa,webarena}-train-time/`: task environments and runners.
- `ahd-test-time/`: test-time EoH and Dynamic-Agent integration.
- `verl/`: bundled VERL runtime with the required multi-turn compatibility changes.
- `verl_trace2skill/`: maintained VERL multi-turn tool, parser, and rewards.
- `trace2skill-settings/`: Math/DocVQA data and skill-evolution adapters.
- `scripts/<task>/`: canonical user-facing launchers.
- `data/`: stable data contract; large contents are ignored.

More detail is in [`PROJECT_LAYOUT.md`](PROJECT_LAYOUT.md).

## Checks

Fast local checks do not require a model server:

```bash
python -m unittest es.test_run_state -v
python -m unittest es.test_seeded_model_es -v
python -m unittest discover math-train-time/tests -v
python -m unittest discover docvqa-train-time/tests -v
python -m unittest verl_trace2skill.test_reward -v
python scripts/check_data.py
```
