# Trace2Skill

The maintained wrapper adapts WebArena trajectories to the official
Trace2Skill Markdown trace/evolution interface. The official source checkout
is external and lives at `source/`; the WebArena rollout adapter is imported
from `../../third_party/skillopt`.

Install both ignored checkouts as described in `data/README.md`, then run:

```bash
scripts/webarena/run.sh trace2skill train
scripts/webarena/run.sh trace2skill test
```

Training uses `data/webarena/vab_nonlite_split/{train,val}/items.json`. Testing
always uses the official 165 WebArena-Lite tasks. Outputs and the current skill
are stored under `runs/trace2skill_webarena_sft/<run-id>/`.

WebArena-specific success/error distillation templates are versioned under
`prompts/`. Select them with `--no-official-prompts` in the standalone runner,
or `--no-trace2skill-official-prompts` in the distributed ES runner. The
provided WebArena Trace2Skill+ES launcher enables these templates.

`scripts/webarena/run.sh trace2skill_es train` uses that learned skill for
Agentic-ESOpt. `WEBARENA_TRACE2SKILL_EVERY_GENERATION=1` enables joint
generation-by-generation skill and model-weight updates.
