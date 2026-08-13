# VAB runtime extensions

The WebArena experiments use the VAB-WebArena-Lite runtime at VisualAgentBench
commit `9055fc2`, plus project-owned additions that are not present in that
upstream checkout:

- `p_webrl_chat_qwen_action.json`, the released Qwen action prompt;
- `local_completion.py`, the provider adapter for the local policy-model
  `/completions` and `/v1/chat/completions` endpoints.
- `evaluation_judge.patch`, which makes the 40 fuzzy-match tasks use the
  released, hard-coded `gpt-4.1-mini` judge instead of upstream
  `gpt-4-1106-preview`.

Install the additions, together with the small VAB integration patches, from the
repository root:

```bash
python webarena-train-time/scripts/install_vab_extensions.py
```

The installer is idempotent and supports `--check`. It targets
`data/webarena/vab-lite` by default; pass `--vab-root` or set `VAB_ROOT` when
the external runtime lives elsewhere.

The judge is used only for benchmark grading. It does not generate browser
actions. Its model has no environment or CLI override, so compared checkpoints
cannot accidentally use different judges. An unavailable judge is an
evaluation error and must not be counted as an incorrect answer.
