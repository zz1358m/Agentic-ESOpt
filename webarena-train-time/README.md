# WebArena 🌐

This directory maintains the four paper settings on Qwen3.5-27B:

| Setting | Model state | Skill |
| --- | --- | --- |
| NoSkill-NoFT | base | none |
| NoSkill-Agentic-ESOpt | replayed ES updates | none |
| Trace2Skill-NoFT | base | distilled from trajectories |
| Trace2Skill-Agentic-ESOpt | replayed ES updates | the same distilled skill |

The common entry point is `scripts/webarena/run.sh`. Install the external
Trace2Skill source and the SkillOpt WebArena rollout runtime as described in
`data/README.md`, then validate the external data with:

```bash
python scripts/check_data.py --task webarena --strict
```

Start four model replicas before training or evaluation:

```bash
MODEL=/path/to/Qwen3.5-27B \
webarena-train-time/scripts/start_webarena_es_servers.sh
```

## Reproduce the trajectory-to-skill pipeline

First run Agentic-ESOpt. Its run directory contains both `history.json` and
the `gen_*` browser trajectories used for distillation:

```bash
RUN_ID=webarena_noskill_es \
scripts/webarena/run.sh noskill_agentic_esopt train
```

Distill a skill from the last 10 completed generations of those trajectories:

```bash
WEBARENA_TRAJECTORY_RUN=runs/webrl_lite_full_es/webarena_noskill_es \
TRACE2SKILL_RUN_ID=webarena_trace2skill \
scripts/webarena/run.sh trace2skill_noft distill
```

Run the four clean final evaluations. Every `test` resets and initializes the
model servers first; Agentic-ESOpt tests then replay every update in the given
history before evaluation.

```bash
scripts/webarena/run.sh noskill_noft test

WEBARENA_ES_HISTORY_FILE=runs/webrl_lite_full_es/webarena_noskill_es/history.json \
scripts/webarena/run.sh noskill_agentic_esopt test

TRACE2SKILL_RUN_ID=webarena_trace2skill \
scripts/webarena/run.sh trace2skill_noft test

WEBARENA_ES_HISTORY_FILE=runs/webrl_lite_full_es/webarena_noskill_es/history.json \
TRACE2SKILL_RUN_ID=webarena_trace2skill \
scripts/webarena/run.sh trace2skill_agentic_esopt test
```

## Default parameters

Agentic-ESOpt uses 70 generations, population 8, case batch 8, alpha
`2.5e-4`, z-score reward normalization, and full-parameter updates. Noise is
represented uniformly as a cosine schedule from `1.5e-3` to `1.5e-3` with no
warmup, so it is numerically constant.

Trajectory distillation uses the last 10 generations with no trace cap,
12,000-character HTML truncation, `gpt-5.4-mini`, 16 analysis workers, medium
reasoning effort for analysis/evolution/consolidation, an empty initial skill,
the committed WebArena success/error prompts, and no skill line/token/reference
cap.

Final evaluation uses three runs with temperature `0.7`, top-p `0.8`, top-k
`20`, min-p `0.0`, presence penalty `1.5`, repetition penalty `1.0`, and 30
browser steps. Train/eval splits are the non-Lite train metadata and the held
out 165-task WebArena-Lite split respectively.

For standalone rollout-and-distill Trace2Skill, use
`scripts/webarena/run.sh trace2skill_noft train`. For joint online skill
evolution during ES, set `WEBARENA_TRACE2SKILL_EVERY_GENERATION=1` when running
`trace2skill_agentic_esopt train`.

Curated logs, detailed three-run evaluations, and evaluated skills are under
`results/`. The Trace2Skill implementation is under `methods/trace2skill/`;
`third_party/skillopt/` is only the vendored WebArena rollout runtime.
