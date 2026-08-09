# Sudoku

This setting maintains Agentic-ESOpt and the repository-integrated
Qwen3.5/Accelerate multi-turn GRPO implementation used by the committed
comparison curves. A model emits one
`set <row> <col> <value>` action per turn; the
environment updates the board and gives sparse terminal reward for a complete,
legal board that preserves all givens.

Prepare controlled-mask JSONL data:

```bash
python sudoku-train-time/scripts/generate_sudoku_data.py \
  --output-dir data/sudoku --train-size 128 --eval-size 128 \
  --mask-counts 5,10,15,20
```

Run Agentic-ESOpt:

```bash
SUDOKU_TARGET_MASK_COUNT=15 \
SUDOKU_ES_SIGMA_START=1e-3 SUDOKU_ES_SIGMA_END=1e-4 \
SUDOKU_ES_SIGMA_SCHEDULE=cosine scripts/sudoku/run_es.sh
```

The history defaults to `runs/sudoku_es/<run-id>/history.json`. Set
`SUDOKU_ES_RESUME_HISTORY` to replay a prior run.

The exact population-32 ES comparison profiles are explicit scripts.
`vanilla-es32` keeps sigma constant (`1e-3` for masks 5/10 and `5e-4` for mask
15). `agentic-esopt-es32` uses the maintained schedule (`1e-3` to zero for
masks 5/10 and `7e-4` to `5e-4` for mask 15). Both use 100 generations, case
batch 32, `alpha=5e-4`, and evaluation every 10 generations:

```bash
sudoku-train-time/scripts/run_es_hyperparams.sh vanilla-es32 15
sudoku-train-time/scripts/run_es_hyperparams.sh agentic-esopt-es32 15
```

Run the GRPO comparison profile (100 steps, batch 32, 8 generations per task,
raw rollout-policy log probabilities, and evaluation every 20 steps) through
the canonical launcher:

```bash
SUDOKU_GRPO_MODEL=/path/to/Qwen3.5-4B \
  SUDOKU_TARGET_MASK_COUNT=15 scripts/sudoku/run_grpo.sh
```

The equivalent direct profile invocation is:

```bash
SUDOKU_GRPO_MODEL=/path/to/Qwen3.5-4B \
  sudoku-train-time/scripts/run_grpo_hyperparams.sh 15
```

GRPO has no global policy-action sample cap (`0` means unlimited; only an
incomplete distributed-alignment remainder may be dropped).
The default sampling profile is temperature 0.7, top-p 0.8, top-k 20. The
additional unfiltered profile is:

```bash
SUDOKU_GRPO_MODEL=/path/to/Qwen3.5-4B \
  sudoku-train-time/scripts/run_grpo_hyperparams_t1.sh 15
```

The archived three-repeat evaluation means for the two GRPO sampling profiles
are:

| GRPO rollout sampling | Eval sampling | Mask 5 | Mask 10 | Mask 15 |
|---|---|---:|---:|---:|
| temperature 0.7, top-p 0.8, top-k 20 | temperature 0.7, top-p 0.8, top-k 20 | 80.2083% | 44.7917% | 30.2083% |
| temperature 1.0, top-p 1.0, top-k -1 | temperature 0.7, top-p 0.8, top-k 20 | 85.4167% | 67.7083% | 40.6250% |

The second row's archived runs used a 512-example policy cap. The maintained
launch scripts use `0` (unlimited), as described above. Complete step-by-step
training and evaluation logs for both rows are under `results/training/`.

The old single-turn TRL trainer is preserved in `deprecated/sudoku/`.

Original 5/10/15-horizon heatmaps and the corresponding Vanilla ES G=32,
Agentic-ESOpt G=32, and GRPO results are under `results/`.
