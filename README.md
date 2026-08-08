# Agentic-ESOpt / Dynamic-Agent

This repository maintains one model-weight evolution method, **Dynamic-Agent**,
across five agentic tasks, together with the baselines needed for comparison.
The GitHub tree is task-oriented and contains runnable code, small source data,
data contracts, documentation, and the currently curated AHD programs; paper
assets and obsolete experiments are not part of the core repository.

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

## Experiment methods

| Method | What changes during the run | Purpose |
| --- | --- | --- |
| Dynamic-Agent / Agentic ESOpt | The language-model weights are perturbed with replayable random seeds; evaluated rewards produce a model-weight ES update. | Main method. |
| Multi-turn GRPO | The policy is optimized with multi-turn reinforcement learning and task tools/rewards. | RL baseline for Sudoku, Math, and DocVQA. |
| Trace2Skill | Success/failure traces are converted into an explicit reusable skill file; model weights stay fixed. | Prompt/skill-evolution baseline. |
| Trace2Skill + Dynamic-Agent | A skill is evolved first or between generations, then model weights are optimized with Dynamic-Agent. | Tests whether skill and model evolution are complementary. |
| EoH | A fixed LLM evolves executable heuristics through `i1/e1/e2/m1/m2`; model weights stay fixed. | AHD evolutionary-search baseline. |
| Sample | A fixed LLM repeatedly performs independent `i1` generations with no parent population in the prompt. | AHD search-free sampling baseline. |
| EoH + Agentic ESOpt | EoH evolves heuristics while evaluated offspring also update the LLM through the shared ES layer. | Dynamic AHD method with population feedback. |
| Sample + Agentic ESOpt | Independent `i1` candidates are generated in batches and each evaluated batch updates the LLM. | Separates model evolution from EoH population evolution. |

Canonical entry points and default run roots are:

| Task | Methods | Entry point | Default output root |
| --- | --- | --- | --- |
| Sudoku | Dynamic-Agent, multi-turn GRPO | `scripts/sudoku/` | `runs/sudoku_es/` or the configured VERL output |
| Math | Dynamic-Agent vLLM, raw/skill evaluation, Trace2Skill, GRPO | `scripts/es_skill_workflow.sh math <action>`; compatibility launchers in `scripts/math/` | `runs/math_es_vllm/`, `runs/trace2skill_extra/`, or VERL output |
| DocVQA | Dynamic-Agent vLLM, raw/skill evaluation, Trace2Skill, GRPO | `scripts/es_skill_workflow.sh docvqa <action>`; compatibility launchers in `scripts/docvqa/` | `runs/docvqa_es_vllm/`, `runs/trace2skill_extra/`, or VERL output |
| WebArena | Dynamic-Agent, Trace2Skill, combined | `scripts/webarena/run.sh` | `runs/webrl_lite_full_es/` or `runs/trace2skill_webarena_sft/` |
| AHD | EoH, Sample, and both Agentic ESOpt variants | `scripts/ahd/run.sh` and `scripts/ahd/run_four_method_ahd.sh` | `cache/active_runs/`; curated code is under `ahd-test-time/results/` |

## File and directory structure

The repository is split by task. The root-level files and directories have the
following roles:

| Path | Purpose |
| --- | --- |
| `es/` | Shared Agentic ESOpt implementation used by every Dynamic-Agent runner. `seeded_model_es.py` applies/reverts seeded weight perturbations and performs updates; `model_es_client.py` calls model-server ES routes; `run_state.py` owns sigma schedules, atomic history, and replay; `registry.py` exposes server-side ES instances; `test_*.py` are CPU state-machine tests. |
| `sudoku-train-time/` | Sudoku environment, controlled-mask data generator, ES training runner, and adapters for the external `verl-tool` GRPO baseline. |
| `math-train-time/` | Math reasoning environment plus HTTP-server and in-process vLLM Dynamic-Agent runners; its `tests/` checks rollout accounting and `results/` holds curated training, distillation, skill, and raw/skill evaluation artifacts. |
| `docvqa-train-time/` | DocVQA environment, Hugging Face vision server, HTTP/vLLM runners, data validator, pipeline tests, and curated artifacts under `results/`. |
| `webarena-train-time/` | WebArena/WebRL environment; data preparation and ES runners; Trace2Skill, SkillOpt, and JITRL integrations; versioned skills and local-model templates. |
| `ahd-test-time/` | Test-time heuristic design for six constructive/ACO tasks. It contains the unified EoH runtime, four-method runner, task evaluators, curated result programs, and ES-adapter regression test. |
| `scripts/` | User-facing launchers grouped by task. `es_skill_workflow.sh` is the canonical four-action Math/DocVQA interface; `settings.example.env` is the machine-local settings template and `check_data.py` validates the data contract. |
| `data/` | Stable data locations. `data/ahd/settings/` holds AHD YAML/prompt definitions, `data/ahd/datasets/` holds its small datasets, and `data/trace2skill/` holds versioned Math manifests/data. Large local datasets remain ignored. |
| `trace2skill-settings/` | Math and DocVQA Trace2Skill configs, analysis/evolution prompts, trajectory preparation/aggregation/compression helpers, trace-to-skill script, and current `SKILL.md` files. |
| `verl/` | Bundled VERL source with the multi-turn compatibility changes required by the Math and DocVQA GRPO launchers. It is a dependency tree, not an experiment output. |
| `verl_trace2skill/` | Maintained local bash tool, parsers, reward functions, DocVQA sandbox/protocol, dense-model compatibility shims, and their unit tests. |
| `vllm_math_es_worker.py` | Worker used by the four-GPU in-process Math vLLM experiment. |
| `README.md` | Main reproduction guide and map of the repository. |
| `PROJECT_LAYOUT.md` | Short structure/entry-point reference. |
| `LICENSE` | Repository license. |
| `.gitignore` | Separates source and curated AHD programs from checkpoints, caches, logs, and generated reports. |

The complete maintained source/artifact layout is:

```text
Agentic-ESOpt/
|-- es/                            shared seeded ES, schedules, history, replay
|-- scripts/
|   |-- es_skill_workflow.sh       canonical Math/DocVQA four-action interface
|   |-- es_skill_workflow.example.env
|   |-- README_ES_SKILL_WORKFLOW.md
|   |-- math/                      legacy/compatibility Math launchers
|   |-- docvqa/                    legacy/compatibility DocVQA launchers
|   |-- sudoku/                    Sudoku launchers
|   |-- webarena/                  WebArena launchers
|   |-- ahd/                       AHD launchers and experiment provenance
|   `-- trace2skill/               VERL data/eval/PBS helpers
|-- math-train-time/
|   |-- envs/                      prompts, parsing, rewards
|   |-- scripts/                   HTTP and in-process vLLM runners
|   |-- tests/                     rollout regressions
|   |-- results/
|   |   |-- training/              ES history and training log
|   |   |-- distillation/          selection manifests and distillation logs
|   |   |-- skill/                 evaluated SKILL.md
|   |   `-- eval/                  no-skill and skill replay results/logs
|   `-- README.md
|-- docvqa-train-time/
|   |-- envs/                      prompts, ANLS reward, data contracts
|   |-- scripts/                   vision HTTP and text-backbone vLLM runners
|   |-- tests/                     pipeline regressions
|   |-- results/
|   |   |-- training/              generation-0..39 ES history and summary
|   |   |-- distillation/          trajectory manifest and distillation logs
|   |   |-- skill/                 evaluated SKILL.md
|   |   `-- eval/{raw,skill}/      paired replay results and logs
|   `-- README.md
|-- trace2skill-settings/
|   |-- configs/                   task/model settings
|   |-- prompts/                   analysis/evolution prompts
|   |-- skills/                    maintained starting skills
|   `-- scripts/                   prepare, select, aggregate, evolve, compress
|-- verl_trace2skill/              bash tool, protocols, rewards, sandbox, tests
|-- verl/                          bundled multi-turn RL dependency
|-- sudoku-train-time/             Sudoku environment and runners
|-- webarena-train-time/            WebArena environment and runners
|-- ahd-test-time/                 EoH/Agentic-ESOpt runtime and curated results
|-- data/                          stable small-data contracts and manifests
|-- vllm_math_es_worker.py         in-process Math vLLM worker
|-- README.md                      main reproduction and structure guide
|-- PROJECT_LAYOUT.md              compact entry-point map
`-- LICENSE
```

Generated checkpoints and the full per-trajectory Markdown trees remain
outside git. The curated `results/` directories keep the replayable histories,
selection manifests, final evaluated skills, aggregate JSON results, and logs
needed to audit the reported Math and DocVQA comparisons.

The important AHD subtree is:

```text
ahd-test-time/
|-- envs/                         six task adapters used by migrated utilities
|-- methods/eoh/
|   |-- README.md
|   `-- original/eoh/             single installable EoH runtime
|       |-- setup.py
|       `-- src/eoh/
|           |-- eoh.py            top-level evolution/sampling loop
|           |-- methods/eoh/      operators, evaluation, selection, ES bridge
|           |-- llm/              local/general LLM clients
|           |-- llm_local_server/ reference text model server with ES routes
|           |-- problems/         EoH problem/prompt interfaces
|           `-- utils/            parameters and output helpers
|-- scripts/
|   |-- run_eoh_ahd.py            canonical runner for all four AHD methods
|   `-- run_ahd_four_methods.py   compatibility forwarder to the same runner
|-- results/                       eight curated 1000/2000 result families
|-- tests/test_four_method_es_adapter.py
`-- README.md
```

There is no second `eoh_four_methods` runtime: the sampling and Agentic ESOpt
adaptation lives in the original `methods/eoh/` tree. Under `scripts/ahd/`,
`run.sh`, `run_four_method_ahd.sh`, and `start_llama31_8b_servers.sh` are the
portable entry points. The other dated/queued scripts are retained experiment
provenance: task grids, reruns, chained jobs, archiving/evaluation helpers, and
their TSV plans. They may contain historical machine defaults, so new runs
should start from the portable entry points.

`scripts/math/`, `scripts/docvqa/`, `scripts/sudoku/`, and
`scripts/webarena/` likewise hold the shell entry points described below.
`scripts/trace2skill/` contains shared VERL data preparation, evaluation, PBS
submission, and monitoring helpers. More compact detail is available in
[`PROJECT_LAYOUT.md`](PROJECT_LAYOUT.md).

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

Math's in-process runner additionally needs `vllm`. DocVQA's bash/OCR protocol
requires `bubblewrap` and `tesseract-ocr` (or equivalents available through
`DOCVQA_TOOL_PREFIX`). Multi-turn GRPO uses an
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

The canonical Math workflow exposes exactly four actions:

```bash
scripts/es_skill_workflow.sh math es-train
scripts/es_skill_workflow.sh math eval
scripts/es_skill_workflow.sh math distill-skill
scripts/es_skill_workflow.sh math skill-eval
```

`es-train` is always no-skill and records trajectories. `eval` replays the
saved ES history without a skill, `distill-skill` selects success/failure
traces and evolves `SKILL.md`, and `skill-eval` replays the same history with
only that skill added. Copy the needed values from
`scripts/es_skill_workflow.example.env` into `scripts/settings.local.env`.

See [`scripts/README_ES_SKILL_WORKFLOW.md`](scripts/README_ES_SKILL_WORKFLOW.md)
for every setting. Historical HTTP, GRPO, and experiment launchers remain in
`scripts/math/` as compatibility/provenance code. Curated completed artifacts
are in [`math-train-time/results/`](math-train-time/results/README.md).

Run multi-turn GRPO using the repository's `verl_trace2skill` bash tool,
reward, and parser implementation:

```bash
scripts/math/run_grpo.sh
```

The launcher defaults to the bundled `verl/`; set `VERL_ROOT` only to test a
different checkout. Evaluation and PBS helper commands are documented in
[`scripts/trace2skill/README.md`](scripts/trace2skill/README.md).

## DocVQA

DocVQA uses the same four-action interface:

```bash
scripts/es_skill_workflow.sh docvqa es-train
scripts/es_skill_workflow.sh docvqa eval
scripts/es_skill_workflow.sh docvqa distill-skill
scripts/es_skill_workflow.sh docvqa skill-eval
```

Training and both evaluations share the in-process text-backbone vLLM runner,
131072-token context, bash/OCR ReAct protocol, and ANLS scorer. Raw and skill
evaluation replay the same ES history; skill evaluation only adds the distilled
skill to the system prompt. The legacy direct-image HTTP/vision launchers remain
in `scripts/docvqa/` for provenance but are not the canonical comparison path.
Curated completed artifacts are in
[`docvqa-train-time/results/`](docvqa-train-time/results/README.md).

Run multi-turn GRPO:

```bash
scripts/docvqa/run_grpo.sh
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

### What the four methods mean

All four methods now run through `ahd-test-time/scripts/run_eoh_ahd.py` and the
single runtime at `ahd-test-time/methods/eoh/`:

| CLI method | Batch launcher name | Meaning |
| --- | --- | --- |
| `eoh` | `eoh` | Original EoH. A fixed LLM evolves a heuristic population with `i1`, `e1`, `e2`, `m1`, and `m2`. This remains both a valuable fixed-model baseline and the population-search engine used by the dynamic variant. |
| `sample` | `sample` | A fixed LLM generates independent `i1` candidates. No parent population is placed in the prompt and no model update occurs. |
| `es` | `dynamic-eoh` | EoH population search plus Agentic ESOpt model-weight updates from evaluated offspring. |
| `sample_es` | `dynamic-sample` | Independent `i1` batches plus one Agentic ESOpt model update per evaluated batch. |

For each dynamic generation, a seeded perturbation is installed on one model
engine, one candidate is generated, and the perturbation is reverted before
candidate evaluation. The same seed/reward update is then broadcast to every
engine. The bridge calls the root `es/` implementation for strict perturbation
state, schedules, history, and replay; it is adapted to this repository's ES
environment rather than carrying an incompatible second ES implementation.

Supported tasks are `construct_tsp`, `construct_kp`, `construct_asp`,
`aco_tsp`, `aco_cvrp`, and `aco_bpp`. Their settings and prompts are in
`data/ahd/settings/`; task datasets are in `data/ahd/datasets/` (constructive
ASP creates its instances internally).

### Start the model services

Install the runtime once and start the reference four-endpoint server:

```bash
python -m pip install -e 'ahd-test-time/methods/eoh/original/eoh[all]'
MODEL=/path/to/Llama-3.1-8B-Instruct \
PY=/path/to/python scripts/ahd/start_llama31_8b_servers.sh
```

Override `GPUS`, `PORTS`, or `MODEL_SERVER_EXTRA_ARGS` when needed. Fixed-model
methods use `LLM_LOCAL_URL`; dynamic methods use the comma-separated
`ES_ENGINE_URLS`.

### Run one task

With population size 10 and 25 generations, an EoH run makes
`25 * 2 * 10 * (1 + k)` evolutionary operator calls, excluding initial
population construction. Thus `k=1` corresponds to the archived 1000 budget
and `k=3` to 2000:

```bash
# EoH1000 and EoH2000
EOH_K=1 AHD_POP_SIZE=10 AHD_GENERATIONS=25 \
scripts/ahd/run.sh construct_tsp train eoh

EOH_K=3 AHD_POP_SIZE=10 AHD_GENERATIONS=25 \
scripts/ahd/run.sh construct_tsp train eoh
```

Fixed-model independent sampling uses the requested candidate count directly:

```bash
# Sample1000 and Sample2000
SAMPLE_TOTAL=1000 SAMPLE_BATCH_SIZE=20 \
scripts/ahd/run.sh construct_tsp train sample

SAMPLE_TOTAL=2000 SAMPLE_BATCH_SIZE=20 \
scripts/ahd/run.sh construct_tsp train sample
```

Run EoH + Agentic ESOpt with the same `k=1`/`k=3` budget mapping:

```bash
# EoH+AgenticESOpt1000; use EOH_K=3 for 2000
EOH_K=1 AHD_POP_SIZE=10 AHD_GENERATIONS=25 \
ES_SIGMA_START=1e-3 ES_SIGMA_END=0 ES_SIGMA_SCHEDULE=cosine \
scripts/ahd/run.sh construct_tsp train es
```

Run independent Sample + Agentic ESOpt. Candidate budget is batch size times
the number of model-update generations:

```bash
# Sample+AgenticESOpt1000
SAMPLE_BATCH_SIZE=20 SAMPLE_GENERATIONS=50 \
ES_SIGMA_START=1e-3 ES_SIGMA_END=0 ES_SIGMA_SCHEDULE=cosine \
scripts/ahd/run.sh construct_tsp train sample_es

# Sample+AgenticESOpt2000
SAMPLE_BATCH_SIZE=20 SAMPLE_GENERATIONS=100 \
ES_SIGMA_START=1e-3 ES_SIGMA_END=0 ES_SIGMA_SCHEDULE=cosine \
scripts/ahd/run.sh construct_tsp train sample_es
```

`RUN_ID` changes the run name. Outputs are written beneath
`cache/active_runs/`; a dynamic run stores replayable model state at
`<run>/results/es/history.json`.

### Run all six tasks and three repetitions

The batch launcher accepts four names and defaults to all six tasks,
repetitions `1 2 3`, batch size 20, and budget 2000:

```bash
# Fixed EoH; EOH_K is inferred as 1 for budget 1000 and 3 for budget 2000.
BUDGET=1000 bash scripts/ahd/run_four_method_ahd.sh eoh
BUDGET=2000 bash scripts/ahd/run_four_method_ahd.sh eoh

BUDGET=1000 bash scripts/ahd/run_four_method_ahd.sh sample
BUDGET=2000 bash scripts/ahd/run_four_method_ahd.sh sample

BUDGET=1000 bash scripts/ahd/run_four_method_ahd.sh dynamic-eoh
BUDGET=2000 bash scripts/ahd/run_four_method_ahd.sh dynamic-eoh

BUDGET=1000 bash scripts/ahd/run_four_method_ahd.sh dynamic-sample
BUDGET=2000 bash scripts/ahd/run_four_method_ahd.sh dynamic-sample
```

Use `TASKS="construct_tsp construct_kp"` or `REPS="1"` for a subset. For a
non-standard EoH budget, set `EOH_K` explicitly; `AHD_POP_SIZE` and
`AHD_GENERATIONS` control the other two terms in the budget formula. Other
important settings are `PY`, `LLM_LOCAL_URL`, `ES_ENGINE_URLS`, `BATCH_SIZE`,
`ES_SIGMA_START`, `ES_SIGMA_END`, `ES_SIGMA_SCHEDULE`, `ES_ALPHA`, `ES_SEED`,
and `ES_INVALID_REWARD_STRATEGY`.

### Continue an interrupted run

AHD has two independent continuation states:

- `ES_RESUME_HISTORY` replays model-weight updates.
- `AHD_CONTINUE_PATH` and `AHD_CONTINUE_ID` restore the EoH population.
- `SAMPLE_RESUME_PATH` appends fixed-model sampling to an existing sample run.

Use both when continuing a complete Dynamic-Agent+AHD run:

```bash
ES_RESUME_HISTORY=/old/run/results/es/history.json \
AHD_CONTINUE_PATH=/old/run/results/pops/population_generation_10.json \
AHD_CONTINUE_ID=10 scripts/ahd/run.sh construct_tsp train es
```

### What the current result folders contain

`ahd-test-time/results/` currently contains exactly eight curated result
families and two evaluators. Every family has six task subdirectories
(`TSP_construct`, `KP_construct`, `ASP_construct`, `TSP_ACO`, `CVRP_ACO`, and
`BPP_ACO`) with three repetitions per task.

| Existing folder | Method and generation budget | Contents currently retained |
| --- | --- | --- |
| `EoH1000/` | Fixed-model EoH, 1000 evolutionary operator calls (`k=1`). | 18 `final_best_code.py` programs and 18 search logs. |
| `EoH2000/` | Fixed-model EoH, 2000 evolutionary operator calls (`k=3`). | 18 final programs and 18 search logs. |
| `Sample1000/` | Fixed-model independent `i1`, 1000 candidates. | 18 final programs only. |
| `Sample2000/` | Fixed-model independent `i1`, 2000 candidates. | 18 final programs only. |
| `EoH+AgenticESOpt1000/` | EoH plus model-weight updates, 1000 operator calls. | 18 final programs and 18 search logs. |
| `EoH+AgenticESOpt2000/` | EoH plus model-weight updates, 2000 operator calls. | 18 final programs and 18 search logs. |
| `Sample+AgenticESOpt1000/` | Independent generation plus model-weight updates, 1000 candidates. | 18 final programs only. |
| `Sample+AgenticESOpt2000/` | Independent generation plus model-weight updates, 2000 candidates. | 18 final programs only. |

Those are archives of selected programs, not resumable run directories; ES
history and EoH populations remain under the original `cache/active_runs/`
run while it exists. No other deleted or older result directory is assumed by
this README.

Evaluate all retained constructive programs or ACO programs with:

```bash
python ahd-test-time/results/eval_construct_results.py --tasks tsp,kp,asp
python ahd-test-time/results/eval_aco_results.py --tasks tsp,cvrp,bpp --split test
```

For a quick smoke check add `--max-instances 1`. The scripts discover the
current `*/<task>/*final_best_code.py` files automatically and write local
JSON/CSV reports under `ahd-test-time/results/`; generated reports are ignored
unless deliberately curated.

## Checks

Fast local checks do not require a model server:

```bash
python -m unittest es.test_run_state -v
python -m unittest es.test_seeded_model_es -v
PYTHONPATH=.:ahd-test-time/methods/eoh/original/eoh/src \
python -m unittest -v ahd-test-time/tests/test_four_method_es_adapter.py
python -m unittest discover math-train-time/tests -v
python -m unittest discover docvqa-train-time/tests -v
python -m unittest verl_trace2skill.test_reward -v
python scripts/check_data.py
```
