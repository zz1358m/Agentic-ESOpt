# Math results

This directory contains the curated generation-25 Math ES/Trace2Skill result.

- `training/`: the 25-update replayable ES history and training stdout log.
- `distillation/`: the exact no-skill trajectory-selection manifest and the
  Trace2Skill distillation log/manifest.
- `skill/`: the exact R1 skill used by the selected evaluation.
- `eval/no_skill_baseline_eval4.json`: four no-skill DAPO/AIME samples.
- `eval/trace2skill_selected_eval4.json`: four selected skill-conditioned
  DAPO/AIME samples.
- `eval/selection_pass4_86.json`: selection rule and aggregate metrics.

The selected skill result has DAPO overall 0.7725 and pass@4 0.86; the no-skill
baseline overall is 0.7675. The JSON records preserve the full completions and
tool trajectories. Paths embedded in historical records refer to the original
machine and are provenance, not runtime requirements.
