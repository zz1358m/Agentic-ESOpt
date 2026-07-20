# VERL and Trace2Skill launchers

Math and DocVQA multi-turn GRPO share `run_verl_agentic_rl.sh`. The task-level
wrappers select the task and use the bundled `verl/` runtime:

```bash
scripts/math/run_grpo.sh
scripts/docvqa/run_grpo.sh
```

Set `MODEL_PATH`, `CONDA_ENV`, and the usual batch/rollout variables as needed.
Use `CONDA_ENV=''` to keep the current environment, or set `VERL_ROOT` to
intentionally override the bundled source. The launcher prepares only the
selected task, verifies every Parquet record routes to `tool_agent`, and writes
checkpoints and logs under `runs/multiturn_grpo/` by default.

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
