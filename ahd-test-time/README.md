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
budget 2000 uses `EOH_K=3`. Each `m1`/`m2` call therefore generates 10 or
30 candidates, respectively. Each operator still applies one model update;
`k=3` increases the number of ES directions instead of performing three
sequential updates.

Use `scripts/ahd/run.sh <task> <split> <method>` for a single task. Agentic-ESOpt
history is written to `<run>/results/es/history.json`.

Results are organized under `results/` by method and budget.

## Agentic-ESOpt rewards

All AHD objectives are minimized. EoH + Agentic-ESOpt therefore rewards
improvement over a parent, while Sample + Agentic-ESOpt rewards the negative
absolute objective.

| Method | Raw reward |
| --- | --- |
| EoH + Agentic-ESOpt | `parent_objective - child_objective` |
| Sample + Agentic-ESOpt | `-child_objective` |

The formal launchers use `ES_INVALID_REWARD_STRATEGY=current` and z-score
normalization with `ddof=0` and epsilon `1e-8`.

### EoH + Agentic-ESOpt

Only `m1` and `m2` use perturbed models and trigger Agentic-ESOpt updates.
`e1` and `e2` remain part of EoH search but do not update model weights.

Let `P` be the selected parent's objective, `C` the child's objective, and
`W` the worst finite objective in the current population. A valid child with
`C <= W` receives

```text
r = P - C
```

For an invalid child or a child with `C > W`, the implementation uses

```text
C_effective = min(W, P + 1)
r = P - C_effective = max(P - W, -1)
```

For example, with `P=10` and `W=13`:

| Child result | Raw reward |
| --- | ---: |
| `C=8` | `2` |
| `C=10` | `0` |
| `C=12` | `-2` |
| `C=15` | `-1` after fallback |
| invalid code | `-1` after fallback |

This table exposes an exact implementation detail: a valid but poor child can
receive `-2`, while an invalid child receives the clipped fallback `-1`.

The rewards from each operator batch are then normalized as

```text
z_i = (r_i - mean(r)) / (std(r) + 1e-8)
```

At budget 1000, each `m1` and `m2` update has 10 directions. At budget 2000,
each update has 30 directions. Both budgets perform two model updates per EoH
generation, hence 50 updates over 25 generations.

### Sample + Agentic-ESOpt

Each batch contains 20 independent `i1` samples and has no parent-dependent
reward. A valid candidate receives

```text
r_i = -C_i
```

An invalid candidate is initially assigned `-1e30`, then replaced before
z-score normalization by a finite batch-relative floor:

```text
spread = std(valid_rewards, ddof=0)
invalid_reward = min(valid_rewards) - spread
```

The default margin multiplying `spread` is 1. If `spread <= 1e-8`, the
fallback is

```text
spread = max(0.01 * abs(mean(valid_rewards)), 1)
```

If the whole batch is invalid, every reward becomes zero and that update has
no direction signal. Otherwise, valid and adjusted-invalid rewards are
z-scored together.

For example, objectives `[4, 5, 7, invalid]` first produce valid rewards
`[-4, -5, -7]`. Their standard deviation is approximately `1.247`, so the
invalid reward becomes approximately `-8.247` before the complete batch is
z-scored.

Budget 1000 uses 50 batches and therefore 50 model updates. Budget 2000 uses
100 batches and therefore 100 model updates.

### Model update

For each seed, the candidate is generated under a one-sided perturbation

```text
theta_i = theta + sigma * epsilon_i
```

The perturbation is reverted before evaluation. After reward normalization,
the shared model update is

```text
theta <- theta + (alpha / N) * sum_i(z_i * epsilon_i)
```

The formal settings use `alpha=5e-4` and cosine sigma decay from `1e-3` to
zero. The update does not divide by sigma and does not use antithetic
`+epsilon/-epsilon` pairs.

The `rewards` stored in `history.json` are the finite rewards before z-score
normalization. The model server reconstructs the normalized weights during
the update or history replay.
