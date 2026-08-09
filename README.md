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
and each task directory for task-specific parameters.

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
  accelerate datasets pillow pandas pyarrow math-verify
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
```

The two maintained GRPO sampling profiles and the ES/GRPO hyperparameters are
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

Then run the required method and stage:

```bash
scripts/webarena/run.sh trace2skill train
scripts/webarena/run.sh trace2skill test
scripts/webarena/run.sh no_skill_es train
scripts/webarena/run.sh no_skill_es test
scripts/webarena/run.sh trace2skill_es train
scripts/webarena/run.sh trace2skill_es test
```

WebArena requires the external checkouts and services listed in
[`data/README.md`](data/README.md) and
[`webarena-train-time/README.md`](webarena-train-time/README.md).

### AHD

Install the EoH runtime, start the model endpoints, and run one task:

```bash
MODEL=/path/to/Llama-3.1-8B-Instruct \
  scripts/ahd/start_llama31_8b_servers.sh

EOH_K=1 AHD_POP_SIZE=10 AHD_GENERATIONS=25 \
  scripts/ahd/run.sh construct_tsp train eoh

EOH_K=1 AHD_POP_SIZE=10 AHD_GENERATIONS=25 \
  scripts/ahd/run.sh construct_tsp train es
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
python -m unittest es.test_run_state -v
python -m unittest es.test_seeded_model_es -v
python -m unittest discover math-train-time/tests -v
python -m unittest discover docvqa-train-time/tests -v
python -m unittest algorithms.verl_trace2skill.test_reward -v
python scripts/check_data.py
```

## License

See [`LICENSE`](LICENSE).
