# DocVQA results

This directory contains the curated generation-40 DocVQA ES/Trace2Skill result.

- `training/`: the replayable 40-update ES history prefix and a compact training
  summary derived from it.
- `distillation/`: the exact no-skill trajectory-selection manifest and the
  Trace2Skill distillation logs/manifest.
- `skill/`: the exact skill used by the completed skill evaluation.
- `eval/raw/`: complete no-skill 100-question × 4-sample evaluation and stdout.
- `eval/skill/`: complete skill-conditioned 100-question × 4-sample evaluation
  and stdout.

Both evaluations use the same generation-40 replay, seed, sampling parameters,
131072-token context, 50-turn bash/OCR ReAct protocol, and ANLS scorer. Paths
embedded in historical records refer to the original machine and are
provenance, not runtime requirements.
