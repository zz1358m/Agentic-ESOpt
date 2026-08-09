# VERL multi-turn components

This package is the maintained local integration used by the Math and DocVQA
GRPO baselines:

- `local_bash_tool.py`: per-trajectory native bash tool;
- `local_bash_tool_config.yaml`: VERL tool declaration;
- `sitecustomize.py`: registers the Trace2Skill text-ReAct parser; and
- `reward.py`: task-aware Math and DocVQA rewards.

The shared launcher adds this package and its `sitecustomize.py` directly to
`PYTHONPATH` so Ray workers register the parser during startup:

```bash
scripts/math/run_grpo.sh
scripts/docvqa/run_grpo.sh
```

The launchers use the bundled `algorithms/verl/` runtime by default. Set `VERL_ROOT` only
to test another checkout. The importable integration directory uses an
underscore (`verl_trace2skill`) even if the source folder was originally named
with a hyphen. Data conversion additionally requires `datasets` and `pyarrow`.

All former machine-specific absolute defaults have been replaced by
repository-relative paths or explicit environment variables.
