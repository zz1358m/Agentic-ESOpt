# Sudoku Setting

Sudoku is a controllable reasoning setting for Llama-3.1-8B-Instruct. Code
lives here; generated datasets live under `data/sudoku/`.

Generate legal solved boards and masked puzzles:

```bash
python sudoku-train-time/scripts/generate_sudoku_data.py \
  --output-dir data/sudoku \
  --train-size 192 \
  --eval-size 192 \
  --mask-counts 10,20,30,40,50,60
```

`mask_count` is the number of hidden cells. Larger values make the puzzle
harder. The default split has 32 train and 32 eval puzzles per mask count.
Each JSONL row contains the solved board, masked puzzle, and metadata.

Run Dynamic-Agent ES against OpenAI-compatible local Llama-3.1-8B endpoints.
Evaluation is an agentic action loop: each model call must return one action,
`set <row> <col> <value>`. The environment updates the board after valid
actions and stops immediately when the board is full.

```bash
scripts/sudoku/run_es.sh
```

The GRPO entrypoint is kept for the next training step, but full interactive
multi-turn GRPO wiring is intentionally separate from the current ES eval path:

```bash
scripts/sudoku/run_grpo.sh
```

Training uses one fixed difficulty per run. Set `SUDOKU_TARGET_MASK_COUNT` to
one of `10,20,30,40,50,60`; the default is `50`.

```bash
SUDOKU_TARGET_MASK_COUNT=30 scripts/sudoku/run_es.sh
SUDOKU_TARGET_MASK_COUNT=30 scripts/sudoku/run_grpo.sh
```

The verifier gives full reward when the filled board is a legal Sudoku solution
that preserves all givens. It does not require matching the generator's
reference solution if the masked puzzle admits multiple solutions.
