# Dynamic-Agent VERL integration

This directory vendors the VERL source used by the Math and DocVQA multi-turn
GRPO baselines. It is based on upstream commit
`9e2072d1204daa57a5848d0e775d35075f6f7db4` (`0.5.0.dev`) and retains the local
SGLang, Qwen3.5, and multi-turn agent-loop compatibility changes required by
the maintained launchers.

VERL remains licensed under Apache-2.0; see `LICENSE` and `Notice.txt` in this
directory. Dynamic-Agent-specific tools, parser registration, and reward
functions live in the sibling `verl_trace2skill/` package.

From the repository root, install this exact tree with:

```bash
python -m pip install -e ./verl
```

The launchers select it automatically. Set `VERL_ROOT` only when intentionally
testing a different VERL checkout.
