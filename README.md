# Dynamic-Agent

This repository studies applying evolution strategies (ES) to existing
test-time and train-time agents.

The active scope follows `EXPERIMENT_README.md` and is organized around three
settings:

- AHD test-time optimization.
- Jericho test-time agents.
- WebArena/WebRL train-time agents.

## Directory Layout

```text
data/
ahd-test-time/
jericho-test-time/
webarena-train-time/
es/
scripts/
```

### `data/`

Vendored source snapshots, cached datasets, and prepared split data.

Data is split by active setting:

- `data/ahd`
- `data/jericho`
- `data/webarena`

### `ahd-test-time/`

AHD test-time experiments built around EoH plus model ES.

The active AHD scope is:

```text
construct/tsp -> data/ahd/datasets/tsp_constructive
construct/kp  -> data/ahd/datasets/kp_constructive
construct/asp -> data/ahd/settings/prompts/asp_constructive
aco/tsp       -> data/ahd/datasets/tsp_aco
aco/cvrp      -> data/ahd/datasets/cvrp_aco
aco/bpp       -> data/ahd/datasets/bpp_offline_aco
```

MCTS-AHD problem source code is not duplicated here. Only active datasets and
settings are extracted into the working layout.

AHD cfg and prompt settings live in:

```text
data/ahd/settings/
```

### `jericho-test-time/`

Jericho test-time agent wrappers, examples, and notes.

Current Jericho experiment status and historical run details are tracked in
`EXPERIMENT_README.md`.

### `webarena-train-time/`

WebArena/WebRL train-time ES harness, WebArena eval scripts, SkillOpt and
Trace2Skill method slots, skill prompts, and local model server templates.

Key scripts live in:

```text
webarena-train-time/scripts/
```

Skill prompt files live in:

```text
webarena-train-time/skills/
```

### `es/`

Shared ES implementation and model-server client utilities.

This is the common location for:

- ES method registry.
- ES model client.
- Full/all-linear parameter perturbation flow.
- LoRA-capable ES flow.

### `scripts/`

Shell launchers for running the setting matrix. Machine-specific paths should
go in `scripts/settings.local.env`, using `scripts/settings.example.env` as the
template.

Launchers are split by setting:

```text
scripts/ahd/
scripts/jericho/
scripts/webarena/
```

Examples:

```text
scripts/ahd/run.sh construct_tsp train eoh
scripts/ahd/run.sh construct_asp test es
scripts/jericho/run.sh library memory
METHOD=skillopt STAGE=train_test scripts/webarena/run.sh
METHOD=trace2skill STAGE=test scripts/webarena/run.sh
```

## Active Settings

The active settings are:

```text
construct_tsp
construct_kp
construct_asp
aco_tsp
aco_cvrp
aco_bpp
jericho
webarena
```

Packaged methods:

```text
AHD      -> EoH; ES builds on EoH
Jericho  -> EvoTest; ES builds on EvoTest
WebArena -> SkillOpt and Trace2Skill; ES builds on WebArena methods where enabled
```

## Detailed Flow

### 1. AHD Test-Time

Code location:

```text
ahd-test-time/
  methods/eoh/
  envs/
  scripts/run_eoh_ahd.py
```

Data and settings:

```text
data/ahd/
  datasets/
  settings/
```

Supported AHD settings:

```text
construct_tsp
construct_kp
construct_asp
aco_tsp
aco_cvrp
aco_bpp
```

Top-level AHD entry:

```text
scripts/ahd/run.sh <task> <train|test> <eoh|es>
scripts/ahd/grid.sh
scripts/ahd/start_llama31_8b_servers.sh
```

Flow without ES:

```text
data/ahd datasets/settings
  -> ahd-test-time/scripts/run_eoh_ahd.py
  -> ahd-test-time/methods/eoh
  -> EoH proposes heuristic code
  -> AHD evaluator scores candidate
```

Flow with ES:

```text
data/ahd datasets/settings
  -> EoH proposes heuristic code
  -> es applies model perturbation through /es/apply
  -> AHD evaluator scores candidate
  -> es reverts perturbation through /es/revert
  -> rewards update model through /es/update
```

EoH is the method ES builds on. It is not an ES baseline directory.

### 2. Jericho Test-Time

Code and data:

```text
data/jericho/
  source/
  jitrl/
```

Packaged method:

```text
EvoTest
```

Packaged tasks:

```text
detective
library
ludicorp
balances
```

Top-level Jericho entry:

```text
scripts/jericho/run.sh <game> <agent_type>
```

Flow:

```text
data/jericho/jitrl/main.py
  -> EvoTest-style memory agent
  -> local policy /completions endpoint
  -> Jericho game ROM
  -> rollout reward/score
```

Jericho/EvoTest calls a local policy model server exposing `/completions`. The
script defaults to:

```text
http://127.0.0.1:11013/completions
```

Override it with:

```text
POLICY_COMPLETIONS_URL=http://host:port/completions scripts/jericho/run.sh library memory
```

With ES, ES builds on EvoTest:

```text
EvoTest rollout segments
  -> /es/apply LoRA perturbation on policy endpoint
  -> segment reward/advantage
  -> /es/revert
  -> episode-level /es/update
```

### 3. WebArena Train-Time

Code and data:

```text
data/webarena/
  source/
  jitrl/
  vab-lite/
  webrl/

webarena-train-time/
  methods/skillopt/
  methods/trace2skill/
  scripts/
  skills/
```

Runtime source locations:

```text
data/webarena/source                 # official WebArena source
data/webarena/jitrl                  # JitRL/WebArena source
data/webarena/lite                   # prepared WebArena-Lite 165-task test set
data/webarena/webrl                  # WebRL SFT/experience source and derived train/val
data/webarena/skillopt_splits        # train/val/test interface for SkillOpt and Trace2Skill
data/webarena/vab-lite               # optional VAB/WebArena-Lite runtime source
webarena-train-time/methods/skillopt/source
webarena-train-time/methods/trace2skill/source
```

External source locations can be overridden with `VAB_ROOT`, `SKILLOPT_ROOT`,
and `TRACE2SKILL_ROOT`.

Packaged methods:

```text
SkillOpt
Trace2Skill
```

Top-level WebArena entry:

```text
METHOD=skillopt STAGE=train scripts/webarena/run.sh
METHOD=skillopt STAGE=test scripts/webarena/run.sh
METHOD=skillopt STAGE=train_test scripts/webarena/run.sh

METHOD=trace2skill STAGE=train scripts/webarena/run.sh
METHOD=trace2skill STAGE=test scripts/webarena/run.sh
METHOD=trace2skill STAGE=train_test scripts/webarena/run.sh
```

SkillOpt flow:

```text
data/webarena/skillopt_splits/train + val
  -> SkillOpt method wrapper
  -> generated/selected skill prompt
  -> WebArena-Lite 165 test runner
  -> validation/test scores
```

SkillOpt training and validation use WebRL SFT/experience data. Final testing
uses `data/webarena/lite`, the WebArena-Lite 165-task benchmark.

The SkillOpt launcher keeps the policy/rollout model local and targets it
through the OpenAI-compatible route:

```text
http://127.0.0.1:11013/v1/chat/completions
```

The attribution/optimizer side may use a stronger OpenAI-compatible model via
`SKILLOPT_OPTIMIZER_BACKEND` and `SKILLOPT_OPTIMIZER_MODEL`. The rollout policy
side remains controlled by `SKILLOPT_TARGET_BACKEND`, `SKILLOPT_TARGET_MODEL`,
and `SKILLOPT_TARGET_BASE_URL`.

WebArena website URLs are configured through `WEBARENA_HOST` or explicit
`SHOPPING`, `SHOPPING_ADMIN`, `REDDIT`, `GITLAB`, `MAP`, `WIKIPEDIA`, and
`HOMEPAGE`. JitRL also accepts the `WA_*` equivalents; the WebArena environment
adapter mirrors both forms.

Trace2Skill flow:

```text
data/webarena/skillopt_splits train/val/test
  -> Trace2Skill method source
  -> skill extraction
  -> WebArena-Lite 165 test runner
  -> train/test scores
```

Trace2Skill source is expected at:

```text
webarena-train-time/methods/trace2skill/source
```

or via:

```bash
TRACE2SKILL_ROOT=/path/to/Trace2Skill
```

With ES, ES builds on the WebArena method/prompt policy:

```text
WebArena train batch
  -> /es/apply model perturbation
  -> evaluate batch reward
  -> /es/revert
  -> /es/update
```

Current WebArena ES runners support environment-interaction ES: a perturbed
policy is evaluated in executable WebArena browser tasks and the resulting
scores are used as rewards. Offline ES training directly on WebRL SFT
trajectories is intentionally not enabled yet; the runner exposes
`--train-source webrl_sft` only as a guarded interface and raises until a real
trajectory-scoring objective is implemented.

### 4. ES

Shared ES code:

```text
es/
  model_es_client.py
  seeded_model_es.py
  registry.py
  README.md
```

`model_es_client.py` is intentionally short: it only sends HTTP requests to the
local model server. The executable ES logic lives in `seeded_model_es.py`; it
selects model parameters, applies deterministic seeded Gaussian noise, replays
the same seeds for ES updates, and resets accumulated ES updates.

The model server endpoints are:

```text
/es/init
/es/apply
/es/revert
/es/update
/es/reset
/es/status
```

Supported parameter scopes:

```text
full
all_linear
lora
```

The detailed ES perturbation/update process is documented in `es/README.md`.

## Experiment Log

Use `EXPERIMENT_README.md` for run status, current WebArena/WebRL split
decisions, ES settings, and completed result notes.

Use `PROJECT_LAYOUT.md` for a shorter structure summary.
