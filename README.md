# Dynamic-Agent

This repository contains the minimal code and configuration used for the
Dynamic-Agent paper experiments. The public repo is intentionally scoped to the
reported AHD and WebArena-Lite settings; historical Jericho, JITRL, SkillOpt,
large datasets, external source checkouts, and run outputs are kept out of git.

## Scope

Tracked experiment code:

- `ahd-test-time/`: EoH-based automatic heuristic design with optional model ES.
- `webarena-train-time/`: WebArena-Lite wrappers for Trace2Skill and model ES.
- `es/`: shared ES client and seeded perturbation utilities.
- `data/ahd/settings/`: AHD problem cfgs and prompts used by the paper.
- `scripts/`: compact launchers and machine-local configuration template.

External data and sources are expected locally and are ignored by git:

```text
data/ahd/datasets/
data/webarena/
webarena-train-time/methods/trace2skill/source/
```

Copy `scripts/settings.example.env` to `scripts/settings.local.env` and edit
machine-specific paths, ports, and model locations there.

## AHD

Supported paper settings:

```text
construct_tsp
construct_kp
construct_asp
aco_tsp
aco_cvrp
```

Tracked settings:

```text
data/ahd/settings/cfg/problem/
data/ahd/settings/prompts/
```

External datasets should be placed under:

```text
data/ahd/datasets/
```

Run examples:

```bash
scripts/ahd/run.sh construct_tsp train eoh
scripts/ahd/run.sh construct_tsp train es
scripts/ahd/grid.sh
```

Paper AHD defaults:

```text
Backbone: Llama-3.1-8B-Instruct
Population: 10
Generations: 25
Workers: 4
ES directions: 10
ES operators: e1,e2,m1,m2
ES scope: full
sigma: 1e-3
alpha: 5e-4
Reward normalization: z-score
Reward mode: improvement
```

## WebArena-Lite

Supported paper settings:

```text
No skill baseline
Dynamic-Agent + No skill
Trace2Skill baseline
Dynamic-Agent + Trace2Skill
```

Required local sources/data:

```text
data/webarena/vab-lite/
data/webarena/vab_lite_split/items.json
data/webarena/skillopt_splits/{train,val,test}/
webarena-train-time/methods/trace2skill/source/
```

The `skillopt_splits` path name is retained as the shared split interface used
by the wrappers; SkillOpt method code itself is not part of the public scope.

Prepare the VAB/WebArena-Lite split:

```bash
python webarena-train-time/scripts/prepare_vab_webarena_lite_split.py
```

Run Trace2Skill:

```bash
METHOD=trace2skill STAGE=train scripts/webarena/run.sh
METHOD=trace2skill STAGE=test scripts/webarena/run.sh
METHOD=trace2skill STAGE=train_test scripts/webarena/run.sh
```

Run Dynamic-Agent ES:

```bash
METHOD=no_skill_es STAGE=train scripts/webarena/run.sh
METHOD=trace2skill_es STAGE=train scripts/webarena/run.sh
METHOD=no_skill_es STAGE=test scripts/webarena/run.sh
METHOD=trace2skill_es STAGE=test scripts/webarena/run.sh
```

Paper WebArena-Lite defaults:

```text
Backbone: Qwen3.5-27B
Population: 8
Case batch: 8
Generations: 1
sigma: 5e-4
alpha: 5e-4
ES scope: full
Max steps: 30
Decoding temperature: 0.0
Max tokens: 2048
Action format: WebRL id actions
Trace2Skill max skill lines: 20
Generated skill file: webarena-train-time/skills/dynamic_agent_trace2skill_generation.md
```

The top-level WebArena launcher reads these environment variables:

```text
WEBARENA_ES_ENDPOINTS
WEBARENA_ES_POPULATION
WEBARENA_ES_CASE_BATCH
WEBARENA_ES_GENERATIONS
WEBARENA_ES_SIGMA
WEBARENA_ES_ALPHA
WEBARENA_ES_SCOPE
WEBARENA_MODEL_NAME
WEBARENA_HOST
TRACE2SKILL_MAX_SKILL_LINES
TRACE2SKILL_SKILL_FILE
```

## ES Endpoints

The local policy servers must expose OpenAI-compatible completion/chat routes
and the ES control routes:

```text
/es/init
/es/apply
/es/revert
/es/update
/es/reset
/es/status
```

The shared ES client lives in `es/`. See `es/README.md` for the perturbation and
update contract.

## Repository Hygiene

Not uploaded to GitHub:

- local secrets and machine settings
- downloaded external repos
- AHD/WebArena datasets
- run outputs, caches, logs, tensorboard files
- Jericho/JITRL/SkillOpt historical code
- unused AHD BPP settings
- the paper source directory
