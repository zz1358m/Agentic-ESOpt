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

## Dataset and released split

The experiments use the WebArena configs shipped with
[VAB-WebArena-Lite at `9055fc2`](https://github.com/THUDM/VisualAgentBench/tree/9055fc299c366ef34700d1710215fb60a0d8c35e/VAB-WebArena-Lite),
not VAB's separately generated training trajectories. Two config directories
have different ID spaces:

| Raw task source used by the experiments | Canonical JSON SHA-256 |
| --- | --- |
| [VisualWebArena 812-task WebArena config at `ad57aae`](https://github.com/web-arena-x/visualwebarena/blob/ad57aae4dad71531504726900b80db02e0526158/config_files/wa/test_webarena.raw.json) | `d35a86509d117021744a58c735eeb61e34356a42163475d8c2535f65ba9c0d33` |
| [VAB 165-task Lite config at `9055fc2`](https://github.com/THUDM/VisualAgentBench/blob/9055fc299c366ef34700d1710215fb60a0d8c35e/VAB-WebArena-Lite/new/test_webarena_lite.raw.json) | `92cef9ca77065d28ad3cac19ccf7f27c2a3784a19bed14905467f71b003846bf` |

The hashes are computed after parsing JSON and serializing with sorted keys,
so whitespace does not affect them. The two generated runtime directories then
have these ID meanings:

| Configs | Count | ID meaning |
| --- | ---: | --- |
| `config_files/wa/test_webarena` | 812 | original WebArena `task_id`, 0–811 |
| `config_files/wa/test_webarena_lite` | 165 | new Lite `task_id`, 0–164, plus `old_task_id` pointing back to the original task |

The paper split is constructed as follows:

1. Read all 812 original configs in numeric `task_id` order.
2. Remove the 165 original IDs named by the Lite configs' `old_task_id` fields.
   This is the leakage boundary; removing original IDs 0–164 would be wrong.
3. Site-stratify and deterministically interleave the remaining 647 tasks with
   Python seed `20260605`.
4. Put the first `round(647 × 0.1) = 65` ordered tasks in validation and the
   remaining 582 in training.

This produces the only maintained partitions:

| Partition | Path | Used by |
| --- | --- | --- |
| train (582) | `data/webarena/vab_nonlite_split/train/items.json` | Agentic-ESOpt rollouts and trajectory collection |
| validation (65) | `data/webarena/vab_nonlite_split/val/items.json` | standalone Trace2Skill validation |
| held-out test (165) | `data/webarena/vab_lite_split/items.json` | periodic read-only evaluation and all three-run final evaluations |

`vab_nonlite_split/test/items.json` is intentionally empty. The 165 Lite
tasks are never added to the update reward or trajectory-to-skill input;
periodic Lite evaluations only log progress. All six sites must remain enabled,
including `wikipedia`: seven training items and four Lite items require it.

Generate and verify the split with:

```bash
python webarena-train-time/scripts/prepare_webarena_nonlite_split.py
python webarena-train-time/scripts/prepare_vab_webarena_lite_split.py
python scripts/check_data.py --task webarena --strict
```

The checker verifies counts, disjointness, full coverage of original task IDs
0–811, and these ordered-ID fingerprints:

| Ordered IDs | SHA-256 |
| --- | --- |
| train `task_id` | `c0a433f7ca57809442f97c1042f4d4154fd2cb26bd049e805ab567d28058c271` |
| validation `task_id` | `5b772371a33363601a6fe094208b39ad732c1824e3cebdb09fe02f3d9e12f49b` |
| Lite `old_task_id` mapping | `79e446fc5738d4a616d5b11f5d804e7d339c8b1932b9f09627394a0977bc7642` |

These are not newly chosen splits: the generated order matches the split used
by the released experiments, and all 560 case positions consumed by the
70-update Agentic-ESOpt run (generations 0–69, batch size 8) match the archived
training log exactly.

## Reproduce the trajectory-to-skill pipeline

First run Agentic-ESOpt. Its run directory contains both `history.json` and
the `gen_*` browser trajectories used for distillation:

```bash
RUN_ID=webarena_noskill_es \
scripts/webarena/run.sh noskill_agentic_esopt train
```

Distill a skill from every trajectory in all completed ES generations:

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

Trajectory distillation uses all completed ES generations and every available
trajectory, with 12,000-character HTML truncation, `gpt-5.4-nano`, 16 analysis
workers, medium reasoning effort for analysis/evolution/consolidation, an empty
initial skill, the committed WebArena success/error prompts, and no skill
line/token/reference cap. In the CLI, both `--generations 0` and
`--max-traces 0` mean unlimited.

Final evaluation uses three runs over all 165 held-out tasks with temperature
`0.7`, top-p `0.8`, top-k `20`, min-p `0.0`, presence penalty `1.5`, repetition
penalty `1.0`, and 30 browser steps.

For standalone rollout-and-distill Trace2Skill, use
`scripts/webarena/run.sh trace2skill_noft train`. For joint online skill
evolution during ES, set `WEBARENA_TRACE2SKILL_EVERY_GENERATION=1` when running
`trace2skill_agentic_esopt train`.

Curated logs, detailed three-run evaluations, and evaluated skills are under
`results/`. The Trace2Skill implementation is under `methods/trace2skill/`;
`third_party/skillopt/` is only the vendored WebArena rollout runtime.
