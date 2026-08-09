# Trace2Skill: Math and DocVQA

This directory adds the two non-WebArena Trace2Skill settings used in the
Trace2Skill paper:

- `math_reasoning`: DAPO Math skill creation, with held-out DAPO and AIME 2026
  evaluation.
- `docvqa`: DocVQA skill creation from 50 validation examples, evaluated on the
  remaining validation examples.

The adapters prepare the data layout and Markdown trace format expected by the
Trace2Skill skill-evolution code. Large datasets stay outside git under
`data/trace2skill/`.

## Data

Download and normalize both settings:

```bash
python algorithms/trace2skill-settings/scripts/prepare_data.py --setting all
```

Useful smoke tests:

```bash
python algorithms/trace2skill-settings/scripts/prepare_data.py --setting math_reasoning --limit 20
python algorithms/trace2skill-settings/scripts/prepare_data.py --setting docvqa --limit 60
```

If the local Hugging Face/Arrow stack cannot materialize the full DocVQA split,
prepare a JSONL elsewhere and normalize it with:

```bash
python algorithms/trace2skill-settings/scripts/prepare_data.py \
  --setting docvqa \
  --docvqa-source-jsonl /path/to/docvqa_validation.jsonl
```

Output layout:

```text
data/trace2skill/math_reasoning/
  dapo_evolve.jsonl
  dapo_test.jsonl
  aime_2026.jsonl
  manifest.json

data/trace2skill/docvqa/
  evolve.jsonl
  test.jsonl
  images/
  manifest.json
```

The paper setting uses:

```text
Math: 400 DAPO evolution questions, 100 held-out DAPO questions, AIME 2026 OOD.
DocVQA: 50 DocVQA evolution examples, remaining 5299 validation examples held out.
```

## Trace2Skill Evolution

Point `TRACE2SKILL_ROOT` to the official Trace2Skill checkout, or keep the source
at `webarena-train-time/methods/trace2skill/source`.

```bash
python algorithms/trace2skill-settings/scripts/evolve_from_trace_logs.py \
  --setting math_reasoning \
  --trace-logs runs/math_reasoning/logs \
  --run-id math_reasoning_t2s
```

The script expects Markdown logs with names ending in `_FAILED.md` or
`_SUCCEED.md`, matching the Trace2Skill analysis entrypoints. Canonical wrappers
are:

```bash
TRACE_LOGS=/path/to/logs RUN_ID=math_t2s scripts/math/run_trace2skill.sh
TRACE_LOGS=/path/to/logs RUN_ID=docvqa_t2s scripts/docvqa/run_trace2skill.sh
```

The maintained paper default for analysis and skill evolution is
`gpt-5.4-nano`.

The maintained composition workflow evolves the skill and evaluates it on the
same replayed Agentic-ESOpt checkpoint:

```bash
scripts/es_skill_workflow.sh math distill-skill
scripts/es_skill_workflow.sh math skill-eval
scripts/es_skill_workflow.sh docvqa distill-skill
scripts/es_skill_workflow.sh docvqa skill-eval
```

Multi-turn GRPO is separate from skill evolution and uses the checked-in
`algorithms/verl_trace2skill` package:

```bash
VERL_ROOT=/path/to/verl scripts/math/run_grpo.sh
VERL_ROOT=/path/to/verl scripts/docvqa/run_grpo.sh
```
