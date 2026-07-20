# Math

The maintained Math setting uses DAPO train/held-out problems and AIME 2026 OOD
evaluation. It supports Dynamic-Agent, multi-turn GRPO, Trace2Skill, and their
Trace2Skill + Dynamic-Agent composition.

Prepare or validate data:

```bash
python trace2skill-settings/scripts/prepare_data.py --setting math_reasoning
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

Run multi-turn GRPO with the local `verl_trace2skill` package:

```bash
scripts/math/run_grpo.sh
```

This uses the bundled `verl/` by default. `VERL_ROOT` remains available as an
explicit override.

Run Trace2Skill alone or feed its evolved skill into Dynamic-Agent:

```bash
TRACE_LOGS=/path/to/trace_logs RUN_ID=math_t2s scripts/math/run_trace2skill.sh
TRACE_LOGS=/path/to/trace_logs RUN_ID=math_combo scripts/math/run_trace2skill_es.sh
```
