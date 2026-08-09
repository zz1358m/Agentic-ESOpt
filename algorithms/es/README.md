# Shared Dynamic-Agent ES layer

`es/` contains task-independent model-weight evolution code used by Sudoku,
Math, DocVQA, WebArena, and AHD.

```text
model_es_client.py   HTTP client for /es/* routes
seeded_model_es.py   deterministic noise and model update implementation
run_state.py         sigma schedules, atomic history, validation, replay
registry.py          method metadata
test_run_state.py    schedule and history unit tests
```

## Sigma contract

Task runners supply an explicit start and end:

```python
from es.run_state import sigma_at_step

sigma = sigma_at_step(
    sigma_start=1e-3,
    sigma_end=1e-4,
    step=generation,
    total_steps=20,
    schedule="cosine",  # constant, linear, or cosine
    warmup_steps=0,
)
```

For linear and cosine schedules, the first scheduled step is exactly
`sigma_start` and the last is exactly `sigma_end`. Warmup keeps the start value
fixed before the decay. A decay needs at least two generations; with one
generation, the sole perturbation uses `sigma_start`.

## Endpoint protocol

Initialize the selected parameter scope:

```json
POST /es/init
{"parameter_scope": "full", "target_modules": null, "verbose": true}
```

Calling `/es/init` again first restores all tracked updates and any active
perturbation, so replay on a reused server starts from the same base weights.

Apply and revert deterministic Gaussian noise:

```json
POST /es/apply
{"seed": 123, "sigma": 0.001}

POST /es/revert
{"seed": 123, "sigma": 0.001}
```

Apply the seed-replay update:

```json
POST /es/update
{
  "seeds": [123, 456],
  "rewards": [0.4, 0.8],
  "alpha": 0.001,
  "reward_normalization": "zscore",
  "reward_normalization_ddof": 0,
  "reward_normalization_eps": 1e-8
}
```

The server regenerates each noise direction and applies:

```text
theta <- theta + (alpha / N) * sum_i weight_i * epsilon(seed_i)
```

`/es/reset` undoes the ES state accumulated after initialization, and
`/es/status` reports the selected parameter set and active state.

Supported scopes are `full`, `all_linear`, and `lora`. LoRA scope requires the
server model to have LoRA parameters before `/es/init`.

## Durable history

Every active task stores the seeds, raw rewards, alpha, normalization settings,
schedule metadata, and server update responses in `history.json`. Writes use a
temporary sibling followed by `os.replace`, so an interrupted process does not
leave half-written JSON.

On resume, runners:

1. initialize every fresh model endpoint;
2. select completed update records;
3. validate the deterministic seed stream and population;
4. replay every update on every endpoint; and
5. continue from the next generation.

AHD additionally records the EoH generation and operator for each model update.
Its model history is separate from EoH's saved population JSON.

Run the unit tests with:

```bash
python -m unittest es.test_run_state -v
```
