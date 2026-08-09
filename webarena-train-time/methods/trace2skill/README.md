# Trace2Skill

The maintained wrappers turn WebArena browser trajectories into the Markdown
success/failure records consumed by the official Trace2Skill analysis and
skill-evolution code. The official source is installed at `source/`; browser
rollouts use the runtime under `../../third_party/skillopt/`.

Trace2Skill-No-Finetune rolls out the fixed base model and evolves only its
skill:

```bash
scripts/webarena/run.sh trace2skill_no-finetune distill
```

Trace2Skill-Agentic-ESOpt distills its separate skill from an existing NoSkill
Agentic-ESOpt trajectory run:

```bash
WEBARENA_TRAJECTORY_RUN=runs/webrl_lite_full_es/<source-run> \
scripts/webarena/run.sh trace2skill_agentic_esopt distill
```

This reads every available `gen_*_sample_*/task_*` trajectory from every
completed ES generation, preserves the success/failure labels, and runs
Trace2Skill error and success analysis with `gpt-5.4-nano`. The resulting skill
is written to the Agentic-ESOpt Trace2Skill run under
`runs/trace2skill_webarena_sft/`. The
exact source trajectory list and distillation parameters are recorded in that
run's `manifest.json` and `source_traces.json`.

The two skills are intentionally different and are evaluated only with their
matching model state. The committed WebArena success/error prompts under
`prompts/` are the defaults. Neither path fine-tunes model weights after skill
distillation.
