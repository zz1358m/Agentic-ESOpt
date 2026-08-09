# Four-GPU DocVQA GRPO

## Canonical ES/skill workflow

The maintained public ES/Trace2Skill interface has four actions:

```bash
scripts/es_skill_workflow.sh docvqa es-train
scripts/es_skill_workflow.sh docvqa eval
scripts/es_skill_workflow.sh docvqa distill-skill
scripts/es_skill_workflow.sh docvqa skill-eval
```

The ES run is no-skill and writes the trajectories consumed by
`distill-skill`. Both evaluations replay the same ES history with the same
four-GPU vLLM, 131072-token context, bash/OCR ReAct protocol and ANLS scorer;
`skill-eval` additionally injects the distilled skill into the system prompt.
Configure machine paths through `scripts/settings.local.env`; see
`scripts/es_skill_workflow.example.env` and
`scripts/README_ES_SKILL_WORKFLOW.md`. Completed reference artifacts are under
`docvqa-train-time/results/`.

## Published evaluation protocol

The published DocVQA comparisons use one aligned paper-style bash/OCR ReAct
protocol:

| Setting | Value |
| --- | --- |
| Evaluation data | Fixed first 100 held-out documents |
| Samples per question | 4 |
| Maximum ReAct turns | 50 |
| Maximum output per assistant turn | 512 tokens |
| Total trajectory response cap | 32768 tokens |
| Sampling | temperature `1`, top-p `1`, top-k `40`, min-p `0` |
| Penalties | presence `2`, repetition `1` |
| Thinking mode | Disabled; the explicit bash/OCR ReAct loop owns the reasoning turns |

## Answer extraction and scoring

Each document image is exposed inside the isolated tool workspace as
`/workspace/document.png`. The model must invoke at least one parsed `bash`
action to inspect or OCR the image; tool observations are returned as text.
The expected terminal format is:

```text
Final answer: <short answer>
```

Evaluation takes the last `Final answer:` or `Answer:` line (English or Chinese
colon). A final response issued before any bash observation is rejected, and a
missing final answer or missing tool use receives zero.

For ANLS, both prediction and reference answers are lowercased, everything
other than ASCII letters and digits is replaced by a space, and whitespace is
collapsed. Let `d` be Levenshtein distance divided by the longer normalized
string length. The score is `1 - d` when `d < 0.5`, otherwise zero. If the
dataset supplies multiple accepted answers, evaluation uses the maximum ANLS
over them. Threshold accuracy is strictly `1` only when `ANLS > 0.5`.

`ANLS Mean4` averages the continuous ANLS of all four samples. `ANLS Pass4`
averages each question's best ANLS among its four samples. `Acc Mean4` averages
the thresholded sample accuracy, while `Acc Pass4` marks a question correct if
any of its four samples has `ANLS > 0.5`. The canonical implementation is in
`algorithms/verl_trace2skill/docvqa_protocol.py`,
`algorithms/verl_trace2skill/reward.py`, and
`docvqa-train-time/envs/docvqa.py`.

This branch provides a portable four-GPU Qwen3.5-4B text-backbone GRPO
experiment for DocVQA. It uses the historical Bash Action protocol and a
Bubblewrap OCR sandbox. The training reward is continuous ANLS in `[0, 1]` and
is forced to zero when Bash was not used; threshold accuracy is diagnostic only.

Use a Python environment with the bundled VERL dependencies (the project
`grpo` environment is the reference environment). First download the pinned
official model and DocVQA validation split, convert the model to its text
backbone, and align the evaluation order against the historical result:

```bash
PY=/path/to/grpo/bin/python
"$PY" scripts/docvqa/prepare_experiment_assets.py \
  --historical-jsonl /path/to/docvqa.jsonl \
  --python "$PY"
```

The alignment step checks `task_id`, accepted answers, and image basename. The
historical JSONL can also be scored independently:

```bash
"$PY" docvqa-train-time/scripts/validate_docvqa_jsonl.py /path/to/docvqa.jsonl \
  --expected-count 2000 \
  --expected-mean-anls 0.4277013246890895 \
  --expected-mean-accuracy 0.4515
```

Run the complete pre-evaluation, training, checkpoint validation,
post-evaluation, and report pipeline:

```bash
PY="$PY" scripts/docvqa/run_four_gpu_experiment_pipeline.sh
```

By default the launcher selects the four GPUs with the highest indices reported
by `nvidia-smi`, records both physical indices and UUIDs, and requires PyTorch
and Ray to see exactly four devices. Override selection with
`DOCVQA_PHYSICAL_GPU_IDS=1,3,5,7`; optionally provide the corresponding stable
UUIDs in `DOCVQA_GPU_UUIDS`.

Training uses 50 questions, batch size 4, eight rollouts per prompt, mini-batch
4, micro-batch 1/GPU, and 15 epochs. Dropping the final two records gives 12
updates per epoch, 180 updates, and 5760 trajectories. A full resumable
checkpoint is written every update. Rolling retention keeps the latest
non-milestone checkpoint plus permanent steps 60, 120, and 180. Exact resume is
supported only with the same four-GPU world size.

Both evaluations use the fixed first 100 held-out questions with four samples,
four TP=1 model replicas, and 400 required error-free records. Preflight starts
at total concurrency 8 and falls back to 4 on failure; the selected value is
pinned for post-training evaluation. The final report includes ANLS, threshold
accuracy, tool use, errors, turns, latency, token usage, deltas, and a paired
question-level bootstrap 95% confidence interval.
