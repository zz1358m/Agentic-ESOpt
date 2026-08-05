# SkillOpt

SkillOpt is one of the WebArena train-time methods used by this repository.

The current launcher expects the external SkillOpt source at:

```text
webarena-train-time/methods/skillopt/source
```

Override with `SKILLOPT_ROOT=/path/to/SkillOpt` if the source lives elsewhere.
The VAB/WebArena-Lite environment is expected at `data/webarena/vab-lite` or
via `VAB_ROOT=/path/to/VAB-WebArena-Lite`.

Standard data splits:

```text
data/webarena/skillopt_splits/train/items.json  # WebRL SFT/experience
data/webarena/skillopt_splits/val/items.json    # heldout WebRL validation
data/webarena/skillopt_splits/test/items.json   # WebArena-Lite 165
```

Generate them with:

```bash
python webarena-train-time/scripts/prepare_standard_webarena_data.py
```

The training/eval wrapper is:

```text
webarena-train-time/scripts/train_skillopt_webagent_epoch_eval.sh
```

The wrapper defaults to a local target model server at
`http://127.0.0.1:11013/v1`. That endpoint is provided by the local model
server's OpenAI-compatible `/v1/chat/completions` route. Override model backends
with `SKILLOPT_OPTIMIZER_BACKEND`, `SKILLOPT_TARGET_BACKEND`,
`SKILLOPT_OPTIMIZER_MODEL`, `SKILLOPT_TARGET_MODEL`, and
`SKILLOPT_TARGET_BASE_URL`.

Top-level entries:

```bash
scripts/webarena/skillopt_train.sh
scripts/webarena/skillopt_test.sh
scripts/webarena/skillopt_train_test.sh
```
