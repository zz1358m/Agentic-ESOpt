# DocVQA

The maintained DocVQA setting supports Dynamic-Agent, multi-turn GRPO,
Trace2Skill, and Trace2Skill + Dynamic-Agent. Rewards use ANLS over the accepted
answer list.

Prepare the full validation split and verify that held-out data is nonempty:

```bash
python trace2skill-settings/scripts/prepare_data.py --setting docvqa
python scripts/check_data.py --task docvqa --strict
```

Run Dynamic-Agent against a vision-capable OpenAI-compatible endpoint (the
launcher defaults to `openai_vision_chat`):

```bash
DOCVQA_MODEL_PATH=/path/to/vision-language-model \
scripts/docvqa/start_vision_server.sh

DOCVQA_ENDPOINT_MODE=openai_vision_chat \
DOCVQA_ES_ENDPOINTS=http://127.0.0.1:11013 \
DOCVQA_ES_SIGMA_START=5e-4 DOCVQA_ES_SIGMA_END=1e-4 \
DOCVQA_ES_SIGMA_SCHEDULE=linear scripts/docvqa/run.sh
```

The included server accepts OpenAI image data URLs and exposes the shared
`/es/*` protocol over a live Hugging Face vision-language model. Start one
process per GPU/port to use multiple endpoints. It requires `torch`,
`transformers`, `accelerate`, `Pillow`, `Flask`, and `flask-cors`.

History and Trace2Skill-compatible logs are written under
`runs/docvqa_es/<run-id>/`.

Run multi-turn GRPO:

```bash
VERL_ROOT=/path/to/verl scripts/docvqa/run_grpo.sh
```

Run Trace2Skill alone or followed by Dynamic-Agent:

```bash
TRACE_LOGS=/path/to/trace_logs RUN_ID=docvqa_t2s scripts/docvqa/run_trace2skill.sh
TRACE_LOGS=/path/to/trace_logs RUN_ID=docvqa_combo scripts/docvqa/run_trace2skill_es.sh
```

The VERL converter deliberately rejects an empty held-out split.
