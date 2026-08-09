# Sudoku 🧩

Multi-turn Sudoku training with Qwen3.5-4B. The maintained experiments cover
masks 5, 10, and 15 with Vanilla ES, Agentic-ESOpt, and two GRPO sampling
profiles.

## Files

- `envs/base.py`: shared environment interface.
- `envs/sudoku.py`: Sudoku state transitions, action parsing, and rewards.
- `scripts/generate_sudoku_data.py`: generates controlled-mask train/eval data.
- `scripts/run_sudoku_es_train.py`: core full-parameter ES trainer.
- `scripts/run_es_hyperparams.sh`: fixed Vanilla-ES32 and Agentic-ESOpt-ES32
  experiment launcher.
- `scripts/run_sudoku_multiturn_grpo_train.py`: core multi-turn GRPO trainer.
- `scripts/run_grpo_hyperparams.sh`: GRPO launcher using training sampling
  `T=0.7, top-p=0.8, top-k=20`.
- `scripts/run_grpo_hyperparams_t1.sh`: GRPO launcher using training sampling
  `T=1, top-p=1, top-k=-1`; evaluation remains at `0.7/0.8/20`.
- `results/heatmaps/`: Sudoku landscape heatmaps.
- `results/training/`: training and evaluation logs grouped by method.

Repository-level wrappers:

- `scripts/sudoku/run_es.sh`: configurable ES smoke launcher.
- `scripts/sudoku/run_grpo.sh`: standard GRPO experiment wrapper.
- `scripts/sudoku/run_grpo_t1.sh`: unfiltered-sampling GRPO wrapper.

## Run

```bash
# Vanilla ES or Agentic-ESOpt; mask is 5, 10, or 15.
sudoku-train-time/scripts/run_es_hyperparams.sh vanilla-es32 15
sudoku-train-time/scripts/run_es_hyperparams.sh agentic-esopt-es32 15

# GRPO. Set SUDOKU_TARGET_MASK_COUNT to 5, 10, or 15.
SUDOKU_GRPO_MODEL=/path/to/Qwen3.5-4B \
  SUDOKU_TARGET_MASK_COUNT=15 scripts/sudoku/run_grpo.sh
SUDOKU_GRPO_MODEL=/path/to/Qwen3.5-4B \
  SUDOKU_TARGET_MASK_COUNT=15 scripts/sudoku/run_grpo_t1.sh
```

The formal GRPO launcher uses policy minibatches of 512 and processes every
complete minibatch; 512 is not a total-example cap.
