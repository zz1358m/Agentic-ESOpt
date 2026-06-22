# ES Module

`es/` contains the shared evolution-strategy interface used by the active
settings:

- `ahd-test-time`
- `jericho-test-time`
- `webarena-train-time`

The setting directories decide what base method to evaluate. The ES module
defines how a model endpoint is initialized, perturbed, reverted, and updated.

## Files

```text
es/
  __init__.py
  model_es_client.py
  registry.py
  seeded_model_es.py
```

- `model_es_client.py`: HTTP client for model-server ES endpoints.
- `seeded_model_es.py`: executable seed-replay ES implementation for live
  torch model parameters.
- `registry.py`: method metadata for the shared Dynamic-Agent ES method.

The model server owns HTTP routing and generation:

```text
ahd-test-time/methods/eoh/original/eoh/src/eoh/llm_local_server/llama31_instruct_server.py
```

That server exposes `/es/*` endpoints and delegates the actual parameter
selection, seeded perturbation, ES update, and reset to `SeedReplayModelES`.

## Parameter Scopes

Supported scopes are:

```text
full
all_linear
lora
```

- `full`: perturb all selected model parameters.
- `all_linear`: perturb linear-layer parameters.
- `lora`: perturb only LoRA adapter parameters. The server must be started with
  LoRA enabled before `/es/init` can use this scope.

## Endpoint Protocol

### `/es/init`

Initializes the perturbation parameter set.

Payload:

```json
{
  "parameter_scope": "full",
  "target_modules": null,
  "verbose": true
}
```

Response includes the selected scope and parameter count.

### `/es/apply`

Applies deterministic Gaussian noise generated from a seed.

Payload:

```json
{
  "seed": 123,
  "sigma": 0.001
}
```

The server adds:

```text
theta <- theta + sigma * epsilon(seed)
```

### `/es/revert`

Reverts the exact perturbation from the same seed and sigma.

Payload:

```json
{
  "seed": 123,
  "sigma": 0.001
}
```

The server applies:

```text
theta <- theta - sigma * epsilon(seed)
```

### `/es/update`

Applies a seed-replay ES update from evaluated rewards.

Payload:

```json
{
  "seeds": [123, 456],
  "rewards": [0.4, 0.8],
  "alpha": 0.001,
  "reward_normalization": "zscore",
  "reward_normalization_ddof": 0,
  "reward_normalization_eps": 1e-8
}
```

The server normalizes rewards, regenerates each seed's noise, and updates:

```text
theta <- theta + (alpha / N) * sum_i normalized_reward_i * epsilon(seed_i)
```

### `/es/reset`

Restores the server-side ES state to the initialized method weights.

### `/es/status`

Returns current ES server state, including selected parameter scope where
available.

## Setting Flows

### AHD

1. EoH proposes candidate heuristic code.
2. A seed is sampled.
3. The model server applies `/es/apply`.
4. The candidate is evaluated by the AHD problem evaluator.
5. The same seed is reverted with `/es/revert`.
6. Candidate rewards are sent to `/es/update`.

### Jericho

1. The policy endpoint initializes LoRA ES with `parameter_scope=lora`.
2. Each rollout segment applies one seed.
3. Segment reward is measured from environment score change.
4. The seed is reverted at segment end.
5. Episode segment advantages are sent to `/es/update`.

### WebArena

1. A training batch of WebArena/WebRL tasks is selected.
2. Each population sample applies one seed.
3. The task batch is evaluated through the WebArena runner.
4. The seed is reverted.
5. Batch rewards are normalized and sent to `/es/update`.

## Client Usage

```python
from es import ModelESClient

client = ModelESClient("http://127.0.0.1:11013/completions")
client.init(parameter_scope="lora")
client.apply_perturbation(seed=123, sigma=0.001)
client.revert_perturbation(seed=123, sigma=0.001)
client.update(seeds=[123], rewards=[1.0], alpha=0.001)
```
