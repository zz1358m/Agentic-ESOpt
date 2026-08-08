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
This workflow intentionally does not use the legacy direct-image HTTP prompt.
Configure machine paths through `scripts/settings.local.env`; see
`scripts/es_skill_workflow.example.env` and
`scripts/README_ES_SKILL_WORKFLOW.md`. Completed reference artifacts are under
`docvqa-train-time/results/`.

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
