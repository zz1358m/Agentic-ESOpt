# VERL and Trace2Skill launchers

Math and DocVQA multi-turn GRPO share `run_verl_agentic_rl.sh`. The task-level
wrappers select the task and use the bundled `algorithms/verl/` runtime:

```bash
scripts/math/run_grpo.sh
scripts/docvqa/run_grpo.sh
```

Set `MODEL_PATH`, `CONDA_ENV`, and the usual batch/rollout variables as needed.
Use `CONDA_ENV=''` to keep the current environment, or set `VERL_ROOT` to
intentionally override the bundled source. The launcher prepares only the
selected task, verifies every Parquet record routes to `tool_agent`, and writes
checkpoints and logs under `runs/multiturn_grpo/` by default.

The complete fixed DAPO-400 Math pipeline should be launched through
`scripts/math/run_experiment_until_complete.py`. It serializes the baseline,
training, post-evaluation, trajectory export, and final report while preserving
and validating each stage's resumable outputs. For training alone,
`scripts/math/run_training_until_complete.py` waits for physical GPUs 3–6,
preserves the same tier for ordinary restarts, resumes the latest checkpoint
automatically, and progressively reduces trajectory turn/token limits only when
explicit OOM/sequence-capacity evidence is present. Dataset size, epochs,
`rollout.n`, and raw trajectory dumping remain fixed across tiers.

The standalone 16-sample evaluation client targets an OpenAI-compatible vLLM
server and evaluates DAPO-100, AIME 2026, and DocVQA:

```bash
python scripts/trace2skill/run_trace2skill_vllm_eval16.py \
  --base-url http://127.0.0.1:18080/v1 \
  --model /path/to/model \
  --samples 16 --resume
```

Set `DOCVQA_ROOT` to the directory containing
`data/trace2skill/docvqa/test.jsonl` and its images. Outputs default to
`runs/trace2skill_vllm/`; `TRACE2SKILL_VLLM_OUT` or `--out-dir` overrides that
location.

Additional trajectory and fixed-skill helpers:

| Script | Purpose |
| --- | --- |
| `eval16_react_4gpu_vllm.sh` | Starts four local vLLM replicas, then runs the shared Math/DocVQA ReAct evaluator with optional skill files. |
| `collect_noskill_trajectories_4gpu_vllm.sh` | Collects 16 no-skill trajectories for every configured Math and DocVQA evolution item and writes Trace2Skill Markdown logs. |
| `queue_noskill_trajectory_collection.sh` | Waits for an existing evaluator and its ports, then starts the no-skill collection run. |
| `queue_distilled_skill_eval4.sh` | Optionally waits for distillation processes, audits the two skill files, and launches the fixed four-sample skill evaluation. |
| `run_docvqa_fixed_then_resume_trajectory.sh` | Runs the fixed DocVQA evaluation before resuming an explicitly named trajectory collection run. |
| `consolidate_math_task_units.py` | Consolidates one evidence patch per Math task into a compact skill with `gpt-5.4-nano`. |

Set `MODEL_PATH` before using the four-GPU vLLM launcher. Machine-specific
paths, process IDs, and validators are supplied through environment variables;
none are embedded in the maintained scripts.

For the fixed Math before/after comparison, use
`scripts/math/run_four_gpu_eval.py`. It starts four SGLang TP=1 replicas on
physical GPUs 3–6 and refuses incomplete, duplicate-key, or request-error
outputs. The matched profile requires 16 samples for each of 100 DAPO and 30
AIME 2026 items; the table-alignment profile requires four. Math ReAct
evaluation is fixed to 50 turns and 4096 generated tokens per assistant
request, with a 262144-token server context by default. Use `--samples 4
--profile repo-react-v1-50x4096` for the table-alignment run. Its command-line
bash workspaces live below the evaluation output directory.

For the Qwen3.5 text-backbone compatibility conversion:

```bash
python scripts/trace2skill/convert_qwen35_to_text_qwen3next.py \
  --src /path/to/Qwen3.5-4B --dst /path/to/Qwen3.5-4B-text
```

The `submit_*.sh` files are portable PBS front ends, not site-specific job
descriptions. Set `JOB` (or `EVAL_JOB`) to a local `.pbs` file when using them.
For example:

```bash
JOB=/cluster/jobs/run_verl.pbs \
  scripts/trace2skill/submit_verl_agentic_rl_4b.sh --both

EVAL_JOB=/cluster/jobs/run_eval.pbs \
  scripts/trace2skill/submit_verl_checkpoint_eval16.sh math /path/to/hf_model
```

`monitor_trace2skill_job.sh JOB_ID` follows a PBS job and discovers logs below
`runs/trace2skill_vllm/`. Set `TRACE2SKILL_VLLM_RUN_ROOT`, `PBS_SERVER_NAME`, or
`PBS_STDOUT` when a cluster uses different locations or naming.
