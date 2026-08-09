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

## Published evaluation protocol

The published DAPO/AIME evaluations use the following aligned defaults for the
tool-using No Skill/Trace2Skill, GRPO, and Agentic-ESOpt comparisons:

| Setting | Value |
| --- | --- |
| Evaluation data | DAPO held-out 100; AIME 2026 30 |
| Samples per problem | 4 |
| Maximum ReAct turns | 50 |
| Maximum output per assistant turn | 4096 tokens |
| Total trajectory response cap | Uncapped (`0`); still bounded by the model context |
| Sampling | temperature `1`, top-p `1`, top-k `40`, min-p `0` |
| Penalties | presence `2`, repetition `1` |
| Thinking mode | Disabled; the explicit bash ReAct loop owns the reasoning turns |

The 50-turn limit applies to tool-using ReAct evaluation. A direct No Skill
profile has no external ReAct loop, but uses the same final-answer scorer.

## Answer extraction and scoring

The ReAct prompt requires at least one parsed `bash` action before accepting a
final response. The expected terminal format is:

```text
Final answer: \boxed{<answer>}
```

Evaluation strips `<think>...</think>`, selects the last `Final answer:` or
`Answer:` line (English or Chinese colon), and extracts the contents of the
last balanced `\boxed{...}`. If no final-answer line exists, an explicit boxed
answer can still be scored. The `prediction` field stored for inspection may
fall back to the last number or non-empty line, but that display-only fallback
is never credited by the scorer: no explicit final/boxed answer receives zero
with `missing_final_answer`.

Scoring first uses `math_verify` for symbolic/LaTeX equivalence. If parsing is
unavailable or inconclusive, the exact fallback removes boxes, outer braces,
commas, whitespace, and a trailing period, then compares lowercase strings.
Numeric fractions and decimals are also compared with absolute tolerance
`1e-8`. ReAct rows without a bash call, and request/context failures, are never
counted as successful.

For four-sample reporting, `Mean4` is the mean of every binary sample score.
`Pass4` first takes the maximum binary score across the four samples for each
problem, then averages over problems; a problem therefore passes when at least
one sample is correct. The canonical implementation is in
`math-train-time/envs/math_reasoning.py` and
`algorithms/verl_trace2skill/reward.py`.

The maintained Math setting uses DAPO train/held-out problems and AIME 2026 OOD
evaluation. It supports Agentic-ESOpt, multi-turn GRPO, Trace2Skill, and their
Trace2Skill + Agentic-ESOpt composition.

Prepare or validate data:

```bash
python algorithms/trace2skill-settings/scripts/prepare_data.py --setting math_reasoning
python scripts/check_data.py --task math --strict
```

Run Agentic-ESOpt through HTTP endpoints or in-process multi-GPU vLLM:

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

Run Trace2Skill alone or feed its evolved skill into Agentic-ESOpt:

```bash
TRACE_LOGS=/path/to/trace_logs RUN_ID=math_t2s scripts/math/run_trace2skill.sh
TRACE_LOGS=/path/to/trace_logs RUN_ID=math_combo scripts/math/run_trace2skill_es.sh
```
