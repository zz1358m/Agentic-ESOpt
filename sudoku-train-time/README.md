# Sudoku

This setting maintains Dynamic-Agent and an asynchronous multi-turn GRPO
baseline. A model emits one `set <row> <col> <value>` action per turn; the
environment updates the board and gives sparse terminal reward for a complete,
legal board that preserves all givens.

Prepare controlled-mask JSONL data:

```bash
python sudoku-train-time/scripts/generate_sudoku_data.py \
  --output-dir data/sudoku --train-size 128 --eval-size 128 \
  --mask-counts 5,10,15,20
```

Run Dynamic-Agent:

```bash
SUDOKU_TARGET_MASK_COUNT=15 \
SUDOKU_ES_SIGMA_START=1e-3 SUDOKU_ES_SIGMA_END=1e-4 \
SUDOKU_ES_SIGMA_SCHEDULE=cosine scripts/sudoku/run_es.sh
```

The history defaults to `runs/sudoku_es/<run-id>/history.json`. Set
`SUDOKU_ES_RESUME_HISTORY` to replay a prior run.

Run the maintained multi-turn GRPO baseline:

```bash
VERL_TOOL_ROOT=/path/to/verl-tool \
SUDOKU_TARGET_MASK_COUNT=15 scripts/sudoku/run_grpo.sh
```

`prepare_verl_data.py`, `install_verl_tool_adapter.py`, and `verl_tool/` provide
the dataset, tool, and binary reward integration. The old single-turn TRL
trainer is preserved in `deprecated/sudoku/`.
