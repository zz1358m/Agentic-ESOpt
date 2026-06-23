# Trace2Skill

Trace2Skill is a required WebArena train-time method slot, but its source is
not tracked in this repository.

Expected install location:

```text
webarena-train-time/methods/trace2skill/source
```

The top-level launcher will call:

```text
webarena-train-time/methods/trace2skill/source/run_traintest.sh
```

or you can set:

```bash
TRACE2SKILL_ROOT=/path/to/Trace2Skill
```

Top-level entries:

```bash
METHOD=trace2skill STAGE=train scripts/webarena/run.sh
METHOD=trace2skill STAGE=test scripts/webarena/run.sh
METHOD=trace2skill STAGE=train_test scripts/webarena/run.sh
```

The repository passes standard split paths through environment variables:

```text
TRACE2SKILL_TRAIN_SPLIT=data/webarena/skillopt_splits/train/items.json
TRACE2SKILL_VAL_SPLIT=data/webarena/skillopt_splits/val/items.json
TRACE2SKILL_TEST_SPLIT=data/webarena/skillopt_splits/test/items.json
TRACE2SKILL_MAX_SKILL_LINES=20
```

The paper run uses a 20-line cap for the generated skill document. The final
generated skill included in this repository is:

```text
webarena-train-time/skills/dynamic_agent_trace2skill_generation.md
```
