# AHD (test-time)

This setting maintains the EoH baseline and Dynamic-Agent + EoH across
constructive and ACO heuristic-design tasks.

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

The reference four-endpoint server launcher is:

```bash
MODEL=/path/to/model scripts/ahd/start_llama31_8b_servers.sh
```

`AHD_POP_SIZE` and `AHD_GENERATIONS` control EoH population and horizon. The
supported task names are `construct_tsp`, `construct_kp`, `construct_asp`,
`aco_tsp`, `aco_cvrp`, and `aco_bpp`.

Dynamic-Agent model history is saved at `<run>/results/es/history.json`.
Population continuation is independent:

```text
ES_RESUME_HISTORY   model update history
AHD_CONTINUE_PATH   population_generation_N.json
AHD_CONTINUE_ID     N
```

Supply both states to continue a complete run.
