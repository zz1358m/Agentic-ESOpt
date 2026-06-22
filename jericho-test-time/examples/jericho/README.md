# Jericho EvoTest + Horizontal ES

Current active runner:

```text
runs/qwen3_14b_parallel/run_jericho_evotest_standard.py
```

The file name is historical. The current aligned tests use local `Qwen3-32B`
servers.

## Runner Settings

```text
evotest
evotest_es
```

Definitions:

- `evotest`: unperturbed EvoTest control rollout, with official EvoTest `our`
  agent logic and memory retained across episodes inside one run only.
- `evotest_es`: horizontal LoRA parameter ES using the EvoTest rollout scaffold
  as its environment. The EvoTest attribution/evolution model must be the base
  Qwen endpoint without LoRA.

Direct-policy variants are not part of the packaged Jericho setting. The
top-level launcher runs EvoTest over `detective`, `library`, `ludicorp`, and
`balances`.

## Alignment Check

Run a compact local check through the top-level launcher:

```bash
RUNS=1 STEPS=30 POLICY_COMPLETIONS_URL=http://127.0.0.1:11013/completions \
  scripts/jericho/run.sh library our
```

It matches the official EvoTest source run at:

```text
runs/original_evotest_qwen32b_library_compare30_temp0_20260603/
```

All 30 actions, rewards, and cumulative scores match exactly.

## ES Defaults

```text
episodes: 50 per run
runs: 3 independent runs
horizon: 110 environment steps
es_interval: 10 steps
es_reward_baseline_decay: 0.8
es_sigma: start low, sweep upward
es_lr: start low, sweep upward
parameter_scope: lora
```

ES update is once per episode. Segment rewards are compared against an EMA
control value at the same segment position, normalized, then sent to
`/es/update`.
