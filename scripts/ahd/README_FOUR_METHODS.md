# AHD four-method experiments

This directory contains the launch and archive scripts migrated from the
Dynamic-Agent AHD experiments. The four canonical methods are:

| Launcher mode | AHD runner method | Meaning |
| --- | --- | --- |
| `eoh` | `eoh` | Original EoH with a fixed language model |
| `sample` | `sample` | Independent `i1` sampling with a fixed language model |
| `dynamic-eoh` | `es` | EoH with model-parameter ES updates |
| `dynamic-sample` | `sample_es` | Independent sampling with one ES update per batch |

All four methods use the repository's canonical runner
`ahd-test-time/scripts/run_eoh_ahd.py` and the single runtime under
`ahd-test-time/methods/eoh/`. The older
`ahd-test-time/scripts/run_ahd_four_methods.py` path is only a compatibility
forwarder for migrated launch scripts.

## ES compatibility

The unified AHD loop does not carry a second model-update implementation.
Both dynamic modes call the repository-level `es.ModelESClient`, and the
model servers execute `es.SeedReplayModelES`. The adapter also uses
`es.run_state` for the `sigma_start -> sigma_end` schedule, atomic
`history.json` writes, and deterministic update replay.

The AHD-specific layer only owns candidate generation and reward construction:
it installs one perturbation on an engine, generates one candidate, always
reverts the perturbation, evaluates candidates after every engine is clean,
then broadcasts the same seed/reward update to every engine. Per-engine locks
make this valid with the stricter active-perturbation checks in
`SeedReplayModelES`.

## Quick start

Start four local model servers:

```bash
MODEL=/path/to/Llama-3.1-8B-Instruct \
PY=/path/to/python \
bash scripts/ahd/start_llama31_8b_servers.sh
```

Then run one method. The default task set contains all six AHD tasks, with
three repetitions and a budget of 2,000 model generations:

```bash
PY=/path/to/python bash scripts/ahd/run_four_method_ahd.sh eoh
PY=/path/to/python bash scripts/ahd/run_four_method_ahd.sh sample
PY=/path/to/python bash scripts/ahd/run_four_method_ahd.sh dynamic-eoh
PY=/path/to/python bash scripts/ahd/run_four_method_ahd.sh dynamic-sample
```

All settings can be overridden through environment variables. For example:

```bash
TASKS="construct_tsp construct_kp" REPS="1" BUDGET=1000 \
PY=/path/to/python bash scripts/ahd/run_four_method_ahd.sh dynamic-sample
```

Important variables are `TASKS`, `REPS`, `BUDGET`, `BATCH_SIZE`, `EOH_K`,
`LLM_LOCAL_URL`, `ES_ENGINE_URLS`, `ES_SIGMA_START`, `ES_SIGMA_END`,
`ES_ALPHA`, `ES_SEED`, `ES_SIGMA_SCHEDULE`, `ES_SIGMA_WARMUP_STEPS`, and
`ES_INVALID_REWARD_STRATEGY`. `ES_SIGMA` remains a compatibility alias for
the old migrated launchers. Without an explicit `ES_SIGMA_END`, the runner
uses the start value for a constant schedule and zero for a decay schedule,
which preserves the historical cosine-to-zero experiments.

For EoH modes, the operator budget is
`2 * POP_SIZE * GENERATIONS * (1 + EOH_K)`, excluding initial population
construction. If `EOH_K` is omitted, the launcher derives it from `BUDGET`,
`AHD_POP_SIZE` (default 10), and `AHD_GENERATIONS` (default 25). The standard
budgets therefore select `k=1` for 1000 and `k=3` for 2000. Set `EOH_K`
explicitly for a non-standard combination.

Dynamic runs write their replayable model history to
`<run>/results/es/history.json`. Use `ES_HISTORY_FILE` to choose another
destination or `ES_RESUME_HISTORY=/path/to/history.json` to initialize fresh
servers by replaying a previous run. EoH population continuation and model-ES
history are separate mechanisms.

The CPU state-machine regression test uses tiny Torch models and the real
repository ES implementation:

```bash
PYTHONPATH=.:ahd-test-time/methods/eoh/original/eoh/src \
python -m unittest -v ahd-test-time/tests/test_four_method_es_adapter.py
```

The other scripts in this directory are the original experiment launchers,
queues, reruns, archive utilities, and evaluation helpers. Some preserve the
original machine defaults for provenance; override `PY`, `MODEL`, `GPUS`, and
`PORTS` when running them elsewhere.
