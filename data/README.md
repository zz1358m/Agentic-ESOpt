# Data contract

Small reproducibility datasets are versioned; licensed datasets and external
checkouts stay outside git. Active code always uses the stable paths below. Run
`python scripts/check_data.py` for a readable inventory, or add `--strict` in a
job preflight.

| Task | Maintained data |
| --- | --- |
| Sudoku | `data/sudoku/train.jsonl` and `eval.jsonl` |
| Math | `data/trace2skill/math_reasoning/{dapo_evolve,dapo_test,aime_2026}.jsonl` |
| DocVQA | `data/trace2skill/docvqa/{evolve,test}.jsonl` plus `images/` |
| WebArena | VAB at `data/webarena/vab-lite`, non-Lite train/val metadata in `vab_nonlite_split/`, and the 165-task test metadata in `vab_lite_split/` |
| AHD | arrays under `data/ahd/datasets/` and task configs/prompts under `data/ahd/settings/` |

Prepare Sudoku with:

```bash
python sudoku-train-time/scripts/generate_sudoku_data.py --output-dir data/sudoku
```

Prepare Math and DocVQA with:

```bash
python trace2skill-settings/scripts/prepare_data.py --setting math_reasoning
python trace2skill-settings/scripts/prepare_data.py --setting docvqa
```

Do not substitute a tiny DocVQA smoke subset for the full setting. A missing or
zero-row `test.jsonl` is invalid; the data checker and VERL converter reject it.

Install the two ignored external WebArena runtime checkouts:

```bash
git clone https://github.com/Qwen-Applications/Trace2Skill.git \
  webarena-train-time/methods/trace2skill/source
git clone https://github.com/microsoft/SkillOpt.git \
  webarena-train-time/third_party/skillopt
```

Place VAB/WebArena-Lite at `data/webarena/vab-lite`, then prepare the shared
non-Lite train/validation split and the held-out 165-task test metadata:

```bash
python webarena-train-time/scripts/prepare_webarena_nonlite_split.py
python webarena-train-time/scripts/prepare_vab_webarena_lite_split.py
```

AHD arrays and task settings are versioned under `data/ahd/`. Do not mix runtime
histories with datasets: model-update histories live under each run directory
(or the explicit `--history-file` path).
