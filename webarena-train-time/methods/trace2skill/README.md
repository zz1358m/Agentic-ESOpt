# Trace2Skill

The maintained wrappers turn WebArena browser trajectories into the Markdown
success/failure records consumed by the official Trace2Skill analysis and
skill-evolution code. The official source is installed at `source/`; browser
rollouts use the runtime under `../../third_party/skillopt/`.

The paper path distills an existing Agentic-ESOpt trajectory run:

```bash
WEBARENA_TRAJECTORY_RUN=runs/webrl_lite_full_es/<source-run> \
TRACE2SKILL_RUN_ID=<skill-run> \
scripts/webarena/run.sh trace2skill_noft distill
```

This reads completed `gen_*_sample_*/task_*` trajectories, preserves their
success/failure labels, runs Trace2Skill error and success analysis, and writes
the resulting skill to `runs/trace2skill_webarena_sft/<skill-run>/skill/SKILL.md`.
The exact source trajectory list and distillation parameters are recorded in
that run's `manifest.json` and `source_traces.json`.

`run_trace2skill_webarena_sft.py` additionally supports standalone iterative
rollout-and-distill training. The committed WebArena success/error prompts
under `prompts/` are the default. Pass `--official-prompts` only to reproduce
an upstream-prompt variant.
