# WebArena

WebArena maintains three paths: Dynamic-Agent without a skill, the standalone
Trace2Skill baseline, and Trace2Skill + Dynamic-Agent. The distributed
Dynamic-Agent runner can also evolve the skill after every ES generation.

Required external data and source locations are listed in `data/README.md`.
Validate them with:

```bash
python scripts/check_data.py --task webarena --strict
```

Both Dynamic-Agent and Trace2Skill train on
`data/webarena/vab_nonlite_split/train/items.json`, validate on the matching
non-Lite validation split, and reserve `vab_lite_split/items.json` for the
165-task test benchmark.

Canonical commands:

```bash
scripts/webarena/run.sh no_skill_es train
scripts/webarena/run.sh trace2skill train
scripts/webarena/run.sh trace2skill_es train

WEBARENA_TRACE2SKILL_EVERY_GENERATION=1 \
scripts/webarena/run.sh trace2skill_es train
```

Dynamic-Agent histories live under
`runs/webrl_lite_full_es/<run-id>/history.json`. Set
`WEBARENA_ES_RESUME_HISTORY` to replay one on fresh endpoints.

The active implementation is in `scripts/run_webrl_lite_*_es_train.py` and
`scripts/run_trace2skill_*`. Trace2Skill reuses SkillOpt's WebArena rollout
environment as a vendored runtime dependency under `third_party/skillopt`;
SkillOpt itself is not a maintained baseline; its historical training launchers
are not included in the core repository.
