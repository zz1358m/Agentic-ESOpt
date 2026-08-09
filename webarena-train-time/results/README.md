# WebArena results

All four settings use Qwen3.5-27B and the 165-task WebArena-Lite evaluation
split. Each reported result is a three-run mean with temperature 0.7, top-p
0.8, and top-k 20.

| Setting | Three-run mean | Reported gain |
|---|---:|---:|
| NoSkill-NoFT | 29.5% | — |
| NoSkill-Agentic-ESOpt | 36.16% | +6.69 points |
| Trace2Skill-NoFT | 33.9% | — |
| Trace2Skill-Agentic-ESOpt | 36.36% | +2.4 points |

The NoSkill-NoFT mean includes the initial 46/165 (27.88%) evaluation from the
first no-skill run.

## Training archive

`training/noskill_agentic_esopt/` contains the two retained training artifacts
from the 70-update Agentic-ESOpt run:

- `train_eval.log`: the full training stream, including candidate rollouts,
  model updates, and evaluations every 10 generations.
- `eval_curve.csv`: evaluation success counts and accuracy every 10
  generations.

The run uses 70 generations, population 8, case batch 8, alpha `2.5e-4`,
z-score reward normalization, and full-parameter updates. Sigma is expressed
as a cosine schedule from `1.5e-3` to `1.5e-3` with zero warmup, so the noise
is constant throughout training. Each generation takes the next eight ordered
tasks from the released 582-task non-Lite train split; all 560 logged case
positions for generations 0–69 match the split fingerprint documented in
`../README.md`.

## Final evaluations and skills

`eval/<setting>/run_01.json` through `run_03.json` use the same detailed schema:
task metadata, hard and soft scores, turn count, final answer, failure reason,
runner status, and wall time. Each file's `run`, `repeat`, and `run_name`
fields match its outer `run_0N.json` number. The two evaluated Trace2Skill files
are under `skills/trace2skill_noft/` and `skills/trace2skill_agentic_esopt/`.

The WebArena-specific Trace2Skill success and error distillation prompts are
versioned under `../methods/trace2skill/prompts/`. Browser screenshots and
rendered DOM trees are omitted because the compact task-level results retain
the scores and final agent answers without hundreds of megabytes per run.
