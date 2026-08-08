# AHD (test-time)

This setting maintains four comparable heuristic-design methods across six
constructive and ACO tasks. All four share
`scripts/run_eoh_ahd.py` and the single EoH runtime under `methods/eoh/`.

| `--method` | Meaning |
| --- | --- |
| `eoh` | Fixed-model EoH population search. |
| `sample` | Fixed-model independent `i1` sampling. |
| `es` | EoH plus Agentic ESOpt model-weight updates. |
| `sample_es` | Independent `i1` batches plus Agentic ESOpt updates. |

Run EoH with different offspring multiplier `k` values:

```bash
EOH_K=1 scripts/ahd/run.sh construct_tsp train eoh
EOH_K=3 scripts/ahd/run.sh construct_tsp train eoh
```

Run Dynamic-Agent with an explicit sigma schedule:

```bash
EOH_K=1 ES_SIGMA_START=1e-3 ES_SIGMA_END=1e-4 \
ES_SIGMA_SCHEDULE=cosine scripts/ahd/run.sh construct_tsp train es
```

Run the two sampling methods:

```bash
SAMPLE_TOTAL=1000 SAMPLE_BATCH_SIZE=20 \
scripts/ahd/run.sh construct_tsp train sample

SAMPLE_BATCH_SIZE=20 SAMPLE_GENERATIONS=50 \
ES_SIGMA_START=1e-3 ES_SIGMA_END=0 ES_SIGMA_SCHEDULE=cosine \
scripts/ahd/run.sh construct_tsp train sample_es
```

The reference four-endpoint server launcher is:

```bash
MODEL=/path/to/model scripts/ahd/start_llama31_8b_servers.sh
```

`AHD_POP_SIZE` and `AHD_GENERATIONS` control EoH population and horizon. The
supported task names are `construct_tsp`, `construct_kp`, `construct_asp`,
`aco_tsp`, `aco_cvrp`, and `aco_bpp`.

To run all six tasks with three repetitions, use:

```bash
BUDGET=1000 bash scripts/ahd/run_four_method_ahd.sh eoh
BUDGET=1000 bash scripts/ahd/run_four_method_ahd.sh sample
BUDGET=1000 bash scripts/ahd/run_four_method_ahd.sh dynamic-eoh
BUDGET=1000 bash scripts/ahd/run_four_method_ahd.sh dynamic-sample
```

The batch launcher infers EoH `k=1` for budget 1000 and `k=3` for budget 2000
with the default population 10 and 25 generations. See the root `README.md`
for the budget formula, exact result-folder meanings, and evaluation commands.

Dynamic-Agent model history is saved at `<run>/results/es/history.json`.
Population continuation is independent:

```text
ES_RESUME_HISTORY   model update history
AHD_CONTINUE_PATH   population_generation_N.json
AHD_CONTINUE_ID     N
```

Supply both states to continue a complete run.

`results/` currently contains only `EoH1000`, `EoH2000`, `Sample1000`,
`Sample2000`, `EoH+AgenticESOpt1000`, `EoH+AgenticESOpt2000`,
`Sample+AgenticESOpt1000`, and `Sample+AgenticESOpt2000`, plus the constructive
and ACO evaluator scripts.
