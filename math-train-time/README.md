# Math

## Canonical ES/skill workflow

The maintained public interface has four actions:

```bash
scripts/es_skill_workflow.sh math es-train
scripts/es_skill_workflow.sh math eval
scripts/es_skill_workflow.sh math distill-skill
scripts/es_skill_workflow.sh math skill-eval
```

The ES run is no-skill and writes the trajectories consumed by
`distill-skill`. Both evaluation actions replay the same ES history; the only
prompt difference is the distilled skill passed to `skill-eval`. Configure
machine paths through `scripts/settings.local.env`; see
`scripts/es_skill_workflow.example.env` and
`scripts/README_ES_SKILL_WORKFLOW.md`. Completed reference artifacts are under
`math-train-time/results/`.

The maintained Math setting uses DAPO train/held-out problems and AIME 2026 OOD
evaluation. It supports Dynamic-Agent, multi-turn GRPO, Trace2Skill, and their
Trace2Skill + Dynamic-Agent composition.

Prepare or validate data:

```bash
python algorithms/trace2skill-settings/scripts/prepare_data.py --setting math_reasoning
python scripts/check_data.py --task math --strict
```

Run Dynamic-Agent through HTTP endpoints or in-process multi-GPU vLLM:

```bash
MATH_ES_SIGMA_START=5e-4 MATH_ES_SIGMA_END=1e-4 \
MATH_ES_SIGMA_SCHEDULE=cosine scripts/math/run.sh

MODEL_PATH=Qwen/Qwen3.5-4B scripts/math/run_vllm_es_4gpu.sh
```

Both runners write Trace2Skill-compatible `*_SUCCEED.md` and `*_FAILED.md`
logs and a replayable history. The HTTP history is under `runs/math_es/`; the
vLLM history is under `runs/math_es_vllm/`.

Run multi-turn GRPO with the local `algorithms/verl_trace2skill` package:

```bash
scripts/math/run_grpo.sh
```

This uses the bundled `algorithms/verl/` by default. `VERL_ROOT` remains available as an
explicit override.

Run Trace2Skill alone or feed its evolved skill into Dynamic-Agent:

```bash
TRACE_LOGS=/path/to/trace_logs RUN_ID=math_t2s scripts/math/run_trace2skill.sh
TRACE_LOGS=/path/to/trace_logs RUN_ID=math_combo scripts/math/run_trace2skill_es.sh
```
