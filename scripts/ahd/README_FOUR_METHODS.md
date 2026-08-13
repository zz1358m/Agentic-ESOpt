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

## ACO-TSP numerical and validation gate

ACO-TSP search and final evaluation call the same implementation in
`algorithms/ahd/aco_tsp_evaluator.py`. It freezes the distance calculation,
float32 conversion, heuristic flooring, ACO update order, and all Python,
NumPy, and PyTorch random streams. A candidate is invalid if conversion to
float32 overflows, if a weight is non-finite, or if a categorical row cannot
form a finite positive simplex. Finite values below the original `1e-9` floor
retain the published evaluator's flooring behavior; invalid values are not
counted as valid objectives.

After every `aco_tsp` `agentic-esopt-sample` search, the launcher copies the
selected `code` field byte-for-byte from the final population JSON and runs
`scripts/ahd/aco_tsp_validation_gate.py`. The gate is fixed to the repository's
N=20/50/100 `val` datasets, 100 ACO iterations, 30 ants, seed 1234, and two
identical seeded passes. It verifies frozen validation/test file SHA-256 values
and zero instance-fingerprint overlap, but evaluates objectives only on the
validation split. The launcher exits nonzero before `[done]` if any instance is
invalid or the two validation reports differ. Final `test` evaluation may start
only from the exact candidate SHA recorded by a `PASS` validation report.
The search-side seed is exposed as `AHD_EVALUATION_SEED` and defaults to the
same frozen value, 1234; formal runs must not override it.

Do not edit a generated heuristic to make this gate pass. A failed candidate
remains failed; a new search repeat must produce a valid candidate.
