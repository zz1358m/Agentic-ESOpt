# Agentic-ESOpt 🚀

[简体中文](README_zh-CN.md)

Agentic-ESOpt optimizes language-model weights from agent rollouts using
seeded, replayable evolution strategies. This repository contains the shared
optimizer, task runners, comparison baselines, Trace2Skill workflows, and
curated experiment logs for five agentic settings.

## What is included 🧭

| Task | Maintained workflows |
| --- | --- |
| Sudoku | Agentic-ESOpt and multi-turn GRPO |
| Math | Agentic-ESOpt, multi-turn GRPO, Trace2Skill, and Trace2Skill + Agentic-ESOpt |
| DocVQA | Agentic-ESOpt, multi-turn GRPO, Trace2Skill, and Trace2Skill + Agentic-ESOpt |
| WebArena | Agentic-ESOpt, Trace2Skill, and Trace2Skill + Agentic-ESOpt |
| AHD | EoH, independent sampling, and their Agentic-ESOpt variants |

The shared implementation is under [`algorithms/es/`](algorithms/es/). Every maintained
Agentic-ESOpt runner records seeded perturbations, rewards, schedules, and
updates in `history.json`, which can be replayed on a fresh model server.

## Released checkpoints 🤗

The task-specific Agentic-ESOpt model weights are collected in the
[Agentic-ESOpt Checkpoints Collection](https://huggingface.co/collections/zz1358m/agentic-esopt-checkpoints-collection):

| Task | Checkpoint |
| --- | --- |
| Math | [`zz1358m/Qwen3.5-4B-MATH-ReAct-Agentic-ESOpt`](https://huggingface.co/zz1358m/Qwen3.5-4B-MATH-ReAct-Agentic-ESOpt) |
| DocVQA | [`zz1358m/Qwen3.5-4B-DocVQA-ReAct-Agentic-ESOpt`](https://huggingface.co/zz1358m/Qwen3.5-4B-DocVQA-ReAct-Agentic-ESOpt) |
| WebArena | [`zz1358m/Qwen3.5-27B-WebArena-Agentic-ESOpt`](https://huggingface.co/zz1358m/Qwen3.5-27B-WebArena-Agentic-ESOpt) |
| Sudoku Mask15 | [`zz1358m/Qwen3.5-4B-Sudoku-Mask15-Agentic-ESOpt`](https://huggingface.co/zz1358m/Qwen3.5-4B-Sudoku-Mask15-Agentic-ESOpt) |

Pass a repository ID directly to a compatible Transformers/vLLM launcher, or
download it before serving:

```bash
hf download zz1358m/Qwen3.5-4B-MATH-ReAct-Agentic-ESOpt \
  --local-dir checkpoints/math-agentic-esopt
```

Use that repository ID or local directory wherever the task instructions ask
for `MODEL_PATH` or `MODEL`. The corresponding evaluation logs, skills, and
exact launcher settings remain in this repository's task directories.

## Repository map 📦

```text
algorithms/                 optimization and training implementations
  es/                       shared Agentic-ESOpt implementation
  trace2skill-settings/     prompts, configs, scripts, and skills
  verl/                     bundled VERL source used by GRPO
  verl_trace2skill/         multi-turn tools, parsers, rewards, and tests
scripts/                    user-facing launchers and data checks
sudoku-train-time/          Sudoku environment, runners, and logs
math-train-time/            Math environment, runners, and logs
docvqa-train-time/          DocVQA environment, runners, and logs
webarena-train-time/        WebArena environment, integrations, and logs
ahd-test-time/              EoH/AHD runtime, evaluators, and programs
data/                       stable data contracts and small source data
```

See [`PROJECT_LAYOUT.md`](PROJECT_LAYOUT.md) for the compact entry-point map
and [`scripts/RUN_HYPERPARAMETERS.md`](scripts/RUN_HYPERPARAMETERS.md) for the
effective defaults of every maintained run launcher.

## Requirements 🧰

- Linux with NVIDIA GPUs for training and model serving.
- Python `>=3.10`; Python 3.10 or 3.11 is recommended.
- CUDA 12.x and a matching PyTorch build.
- `bubblewrap` and `tesseract-ocr` for the DocVQA bash/OCR environment.
- Local model checkpoints and full task datasets are not committed.

The following are known-working profiles, not the only supported versions:

| Workflow | Python | PyTorch | Transformers | vLLM / VERL |
| --- | --- | --- | --- | --- |
| Qwen3.5 Agentic-ESOpt and evaluation | 3.10 | 2.10.0 | 4.57.6 | vLLM 0.19.1 |
| Multi-turn GRPO | 3.11 | 2.6.0 | 4.51.1 | bundled VERL |
| CPU checks and AHD utilities | >=3.10 | optional | compatible recent release | not required |

Use separate environments for Qwen3.5 inference and GRPO: the bundled VERL
stack and the newer Qwen3.5 vLLM stack have different dependency constraints.
Install the PyTorch build appropriate for the machine before the remaining
packages.

```bash
# Agentic-ESOpt / evaluation environment
python3.10 -m venv .venv-es
source .venv-es/bin/activate
python -m pip install --upgrade pip
python -m pip install transformers==4.57.6 vllm==0.19.1 \
  accelerate datasets pillow pandas pyarrow math-verify openai tiktoken
python -m pip install -e 'ahd-test-time/methods/eoh/original/eoh[all]'

# GRPO environment (create and activate a separate Python 3.11 environment)
python -m pip install torch==2.6.0 transformers==4.51.1
python -m pip install -e ./algorithms/verl
```

For CUDA-specific VERL images and optional backends, see
[`algorithms/verl/docker/`](algorithms/verl/docker/) and
[`algorithms/verl/README.md`](algorithms/verl/README.md).

## Quick start ⚡

```bash
git clone https://github.com/zz1358m/Agentic-ESOpt.git
cd Agentic-ESOpt
cp scripts/settings.example.env scripts/settings.local.env
```

Edit `scripts/settings.local.env` with model paths, GPUs, ports, and endpoints.
Prepare the datasets described in [`data/README.md`](data/README.md), then
validate them before starting an experiment:

```bash
python scripts/check_data.py
python scripts/check_data.py --task math --strict
```

Generated checkpoints and ordinary run outputs are ignored by git. Use a
unique `RUN_ID` for each experiment and keep the complete command and local
settings with the run.

## Reproduce a workflow 🧪

### Sudoku

The ES launcher generates the default controlled-mask dataset when it is
missing. Both launchers read `scripts/settings.local.env`.

```bash
SUDOKU_TARGET_MASK_COUNT=15 RUN_ID=sudoku_es_m15 \
  scripts/sudoku/run_es.sh

SUDOKU_TARGET_MASK_COUNT=15 SUDOKU_GRPO_MODEL=/path/to/Qwen3.5-4B \
  scripts/sudoku/run_grpo.sh

SUDOKU_TARGET_MASK_COUNT=15 SUDOKU_GRPO_MODEL=/path/to/Qwen3.5-4B \
  scripts/sudoku/run_grpo_t1.sh
```

`run_grpo.sh` uses rollout temperature 0.7, top-p 0.8, and top-k 20;
`run_grpo_t1.sh` uses rollout temperature 1, top-p 1, and top-k -1. Both use
temperature 0.7, top-p 0.8, and top-k 20 for evaluation. The complete
ES/GRPO hyperparameters are
documented in [`sudoku-train-time/README.md`](sudoku-train-time/README.md).

### Math and DocVQA

The canonical four-stage workflow is identical for both tasks:

```bash
scripts/es_skill_workflow.sh math es-train
scripts/es_skill_workflow.sh math eval
scripts/es_skill_workflow.sh math distill-skill
scripts/es_skill_workflow.sh math skill-eval

# Replace math with docvqa for the DocVQA workflow.
```

Trajectory distillation is task-specific. Math scans every candidate trajectory
from all ES generations, keeps at most one `FAILED` trace per training problem,
and excludes every successful trace. With 25 generations × 16 problems, its
400-problem training set therefore contributes at most 400 traces. DocVQA keeps
at most one `FAILED` and one `SUCCEED` trace from each of the exact final 50
task occurrences. The selection manifest records any unavailable outcome, and
`skill-eval` evaluates the resulting skill after replaying the same
Agentic-ESOpt history. All paper Trace2Skill analysis and skill-evolution calls
use `gpt-5.4-nano`.

Set `MODEL_PATH`, `TRAIN_RUN_ID`, dataset paths, and GPU settings in
`scripts/settings.local.env`. Detailed variables are listed in
[`scripts/README_ES_SKILL_WORKFLOW.md`](scripts/README_ES_SKILL_WORKFLOW.md).

The React-VERL wrappers expose the merged GRPO training and four-replica
evaluation paths:

```bash
scripts/math/run_react_verl_grpo.sh train
MATH_GRPO_EVAL_MODEL_PATH=/path/to/hf_checkpoint \
  scripts/math/run_react_verl_grpo.sh eval

scripts/docvqa/run_react_verl_grpo.sh train
DOCVQA_GRPO_EVAL_MODEL_PATH=/path/to/hf_checkpoint \
  scripts/docvqa/run_react_verl_grpo.sh eval
```

Training writes checkpoints, raw trajectories, validation trajectories, and
logs under `runs/multiturn_grpo/`. Math evaluation defaults to the four-sample
`repo-react-v1-50x4096` profile. Set the `*_PHYSICAL_GPU_IDS`, `*_EVAL_OUT`,
and `*_EVAL_SAMPLES` variables to reproduce a different machine layout.

For the fixed DAPO-400 Math watchdog that resumes baseline evaluation,
training, post-evaluation, trajectory export, and reporting, run:

```bash
python scripts/math/run_experiment_until_complete.py
```

### WebArena

Prepare the configured train and held-out evaluation splits first:

```bash
python webarena-train-time/scripts/prepare_webarena_nonlite_split.py
python webarena-train-time/scripts/prepare_vab_webarena_lite_split.py
```

The released protocol removes the 165 Lite configs' `old_task_id` values from
the original 812 WebArena tasks, then uses seed `20260605` to create 582 train
and 65 validation tasks. The Lite `task_id` values 0–164 are new evaluation
indices, not the original IDs to exclude. Exact hashes and the full split
definition are in [`webarena-train-time/README.md`](webarena-train-time/README.md).

WebArena has two distinct trajectory-to-skill paths. No-Finetune rolls out the
base model and evolves only its skill; Agentic-ESOpt distills a different skill
from the completed NoSkill ES run:

```bash
scripts/webarena/run.sh trace2skill_no-finetune distill

RUN_ID=webarena_noskill_es \
scripts/webarena/run.sh noskill_agentic_esopt train

WEBARENA_TRAJECTORY_RUN=runs/webrl_lite_full_es/webarena_noskill_es \
scripts/webarena/run.sh trace2skill_agentic_esopt distill
```

The Agentic-ESOpt distillation consumes every trajectory from every completed
ES generation; it does not select only the final generations or impose a trace
cap. Both paths use `gpt-5.4-nano` for analysis and skill evolution.

After its distillation, Trace2Skill-Agentic-ESOpt replays the already trained
NoSkill ES history and injects the ES-trajectory skill only for final
evaluation. It never starts a second ES run or updates model weights after
distillation.

The same entry point evaluates all four NoSkill/Trace2Skill ×
No-Finetune/Agentic-ESOpt settings. Exact commands, defaults, external
checkouts, and service setup are listed in
[`data/README.md`](data/README.md) and
[`webarena-train-time/README.md`](webarena-train-time/README.md).

Of the 165 WebArena-Lite tasks, 40 use the benchmark's GPT fuzzy/semantic
grader. The released protocol fixes that judge to `gpt-4.1-mini` at
temperature `0`; it is separate from the local Qwen browser policy. Use
`scripts/webarena/run_final_eval_suite.sh` to reproduce all four three-run
evaluations. Judge API failures are retried and are never silently scored as
incorrect answers; an incomplete repeat aborts instead of rerunning a
state-changing task.

### AHD

Install the EoH runtime, start the model endpoints, and run one task:

```bash
MODEL=/path/to/Llama-3.1-8B-Instruct \
  scripts/ahd/start_llama31_8b_servers.sh

bash scripts/ahd/run_ahd_1000.sh eoh
bash scripts/ahd/run_ahd_1000.sh agentic-esopt-eoh

bash scripts/ahd/run_ahd_2000.sh eoh
bash scripts/ahd/run_ahd_2000.sh agentic-esopt-eoh
```

The six supported tasks, sampling workflows, budgets, and continuation flags
are documented in [`ahd-test-time/README.md`](ahd-test-time/README.md).

## Resume and outputs ♻️

Agentic-ESOpt writes an atomic `history.json`. Pass the corresponding
`*_RESUME_HISTORY` variable to replay completed updates before continuing. For
example:

```bash
SUDOKU_ES_RESUME_HISTORY=/path/to/history.json \
RUN_ID=sudoku_resumed scripts/sudoku/run_es.sh
```

New runs are written under `runs/` or `cache/active_runs/`. Curated logs and
evaluation artifacts live in each task's `*-train-time/results/` directory.
Those directories contain experiment evidence rather than a duplicated result
summary; inspect the task README and raw logs together.

## Checks ✅

Fast checks do not require a running model server:

```bash
python -m unittest algorithms.es.test_run_state -v
python -m unittest algorithms.es.test_seeded_model_es -v
python -m unittest discover math-train-time/tests -v
python -m unittest discover docvqa-train-time/tests -v
python -m unittest algorithms.verl_trace2skill.test_reward -v
python scripts/check_data.py
```

## License

See [`LICENSE`](LICENSE).
