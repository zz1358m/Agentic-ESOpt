# AHD test-time experiments

This directory implements six AHD tasks with four comparable methods:

| `--method` | Meaning |
| --- | --- |
| `eoh` | fixed-model EoH population search |
| `sample` | fixed-model independent `i1` sampling |
| `es` | EoH + Agentic-ESOpt |
| `sample_es` | independent sampling + Agentic-ESOpt |

The supported tasks are `construct_tsp`, `construct_kp`, `construct_asp`,
`aco_tsp`, `aco_cvrp`, and `aco_bpp`.

For the complete six-task, three-run experiments, use the fixed-budget launchers:

```bash
bash scripts/ahd/run_ahd_1000.sh eoh
bash scripts/ahd/run_ahd_1000.sh sample
bash scripts/ahd/run_ahd_1000.sh agentic-esopt-eoh
bash scripts/ahd/run_ahd_1000.sh agentic-esopt-sample

bash scripts/ahd/run_ahd_2000.sh eoh
bash scripts/ahd/run_ahd_2000.sh sample
bash scripts/ahd/run_ahd_2000.sh agentic-esopt-eoh
bash scripts/ahd/run_ahd_2000.sh agentic-esopt-sample
```

With population 10 and 25 generations, budget 1000 uses `EOH_K=1` and
budget 2000 uses `EOH_K=3`. Thus `m1` and `m2` are each run once or three
times per generation, respectively.

Use `scripts/ahd/run.sh <task> <split> <method>` for a single task. Agentic-ESOpt
history is written to `<run>/results/es/history.json`.

Results are organized under `results/` by method and budget.
