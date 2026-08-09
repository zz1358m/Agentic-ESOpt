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
| WebArena | VAB-WebArena-Lite runtime at `data/webarena/vab-lite`, released 582/65 non-Lite train/val metadata in `vab_nonlite_split/`, and the held-out 165-task Lite metadata in `vab_lite_split/` |
| AHD | arrays under `data/ahd/datasets/` and task configs/prompts under `data/ahd/settings/` |

Prepare Sudoku with:

```bash
python sudoku-train-time/scripts/generate_sudoku_data.py --output-dir data/sudoku
```

Prepare Math and DocVQA with:

```bash
python algorithms/trace2skill-settings/scripts/prepare_data.py --setting math_reasoning
python algorithms/trace2skill-settings/scripts/prepare_data.py --setting docvqa
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

Place the
[VAB-WebArena-Lite at `9055fc2`](https://github.com/THUDM/VisualAgentBench/tree/9055fc299c366ef34700d1710215fb60a0d8c35e/VAB-WebArena-Lite)
runtime at `data/webarena/vab-lite`, including the generated 812-task
`config_files/wa/test_webarena` directory and 165-task
`config_files/wa/test_webarena_lite` directory. Then prepare the released
split metadata:

```bash
python webarena-train-time/scripts/prepare_webarena_nonlite_split.py
python webarena-train-time/scripts/prepare_vab_webarena_lite_split.py
```

The 165 Lite configs use a new `task_id` range 0–164. Their `old_task_id`
fields identify which tasks to remove from the original 812; the project does
not treat original IDs 0–164 as the test set. After exclusion, seed `20260605`
and the released site-stratified ordering produce 582 train and 65 validation
tasks. The preparation scripts first verify the exact canonical raw-config
hashes from VisualWebArena `ad57aae` and VAB `9055fc2`.
`python scripts/check_data.py --task webarena --strict` checks those source
hashes, the ordered-ID hashes, and train/validation/test disjointness. See
`webarena-train-time/README.md` for the full split protocol and experiment
alignment evidence.

AHD arrays and task settings are versioned under `data/ahd/`. Do not mix runtime
histories with datasets: model-update histories live under each run directory.
Training continuation uses `--replay-history`; clean final evaluation uses
`scripts/webarena/replay_es_history_and_eval.py --source-history`.
