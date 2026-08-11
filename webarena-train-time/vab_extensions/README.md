# VAB runtime extensions

The WebArena experiments use the VAB-WebArena-Lite runtime at VisualAgentBench
commit `9055fc2`, plus two project-owned additions that are not present in that
upstream checkout:

- `p_webrl_chat_qwen_action.json`, the released Qwen action prompt;
- `local_completion.py`, the provider adapter for the local policy-model
  `/completions` and `/v1/chat/completions` endpoints.

Install both additions, together with the small VAB integration patch, from the
repository root:

```bash
python webarena-train-time/scripts/install_vab_extensions.py
```

The installer is idempotent and supports `--check`. It targets
`data/webarena/vab-lite` by default; pass `--vab-root` or set `VAB_ROOT` when
the external runtime lives elsewhere.
