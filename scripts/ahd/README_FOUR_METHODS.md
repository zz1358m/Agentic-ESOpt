# AHD scripts

The AHD experiments cover six tasks and four methods:

| Launcher mode | Method |
| --- | --- |
| `eoh` | fixed-model EoH |
| `sample` | fixed-model independent sampling |
| `agentic-esopt-eoh` | EoH + Agentic-ESOpt |
| `agentic-esopt-sample` | independent sampling + Agentic-ESOpt |

Start four Llama-3.1-8B-Instruct model servers:

```bash
MODEL=/path/to/Llama-3.1-8B-Instruct \
bash scripts/ahd/start_llama31_8b_servers.sh
```

Run a method at either supported budget:

```bash
bash scripts/ahd/run_ahd_1000.sh eoh
bash scripts/ahd/run_ahd_1000.sh agentic-esopt-eoh

bash scripts/ahd/run_ahd_2000.sh eoh
bash scripts/ahd/run_ahd_2000.sh agentic-esopt-eoh
```

The same two launchers also accept `sample` and `agentic-esopt-sample`.
Every command runs all six tasks with `REPS="1 2 3"` by default.

## Default hyperparameters

- EoH: population 10, 25 generations.
- Budget 1000: `EOH_K=1`; `m1` and `m2` each generate one population per generation.
- Budget 2000: `EOH_K=3`; `m1` and `m2` each generate three populations per generation.
- Sampling: batch size 20; 50 generations at budget 1000 and 100 generations at budget 2000.
- Agentic-ESOpt: operators `m1,m2`, 10 ES directions, sigma `1e-3 -> 0` with cosine decay, alpha `5e-4`, seed 2024.
- Generation: temperature 1.0, top-p 0.98, no top-k cutoff, maximum 768 new tokens.

Use `run.sh` for one task. The `*_train_*.sh` and `*_test_*.sh` files are thin task-specific wrappers around it.
