# Agentic Sudoku Results

## ES agents converge faster on longer-horizon Sudoku

We evaluate agentic Sudoku as a sparse-reward multi-turn setting. Each episode
requires a sequence of `set <row> <col> <value>` actions and receives reward
`1.0` only when the final board is complete, legal, and preserves all givens;
all other final states receive `0.0`. The environment rejects malformed actions,
out-of-range indices, edits to fixed givens, and overwrites of already-filled
cells. Valid writes to empty cells are applied immediately, while Sudoku row,
column, and box consistency is checked by the terminal verifier.

![Agentic Sudoku ES vs GRPO](../figures/sudoku_agentic_es_vs_grpo.svg)

The 5-horizon, 10-horizon, and 15-horizon settings correspond to masking 5, 10,
and 15 cells. On these longer-horizon action tasks, ES agents show both faster
training progress and stronger final performance than GRPO. With population 32,
ES exceeds GRPO's final test success early on 10-horizon and 15-horizon tasks,
while also reaching the highest best-test success across all three horizons.

### Takeaways

- ES is naturally aligned with agentic credit assignment: it evaluates the
  complete trajectory and updates from end-to-end success, avoiding brittle
  token-level attribution across long action chains.
- Larger ES populations improve reliability in sparse binary-reward settings:
  population 32 is consistently stronger than population 16 on 5-, 10-, and
  15-horizon Sudoku.
- GRPO can improve with repeated updates, but its signal becomes noisier as the
  horizon grows because one failed action can invalidate an otherwise plausible
  trajectory.
- The benefit is clearest on 15-horizon Sudoku: ES population 32 reaches its
  best test success by early updates, while GRPO needs substantially more
  training and still trails the best ES result.
- Agentic ES therefore provides a practical alternative when environment
  feedback is sparse, delayed, and trajectory-level rather than token-local.
