---
name: math-reasoning-trace-skill
description: Follow strict tool/protocol and exact mathematical proof-and-verification workflow for math problems; always end with the required single-line final answer format.
---

- Read the prompt and lock these first:
  - The exact final-answer template (copy it verbatim, including boxing/LaTeX or the plain `Answer:` prefix).
  - The full mathematical predicate/objective and its domain/quantifiers (what is chosen by you vs by an adversary, what must hold for all vs exists).
  - Interpret every ambiguous operation/phrase into one precise algebraic/geometric definition.

- Derive carefully (no unjustified leaps):
  - For “iff” or closed-form solution sets: prove both directions (sufficiency + necessity).
  - Keep one symbol/meaning for each key quantity; never redefine a variable as its square/root later.
  - Preserve strict vs non-strict inequalities exactly through algebra.

- Use computation only as verification (never as proof-by-hope):
  - If you compute candidates, verify them by substituting into the original problem conditions.
  - For exact integer/square checks use integer arithmetic (e.g., `isqrt` and equality), not float tolerances.

- Count/construct correctly:
  - Never multiply separate counts unless you state and justify a bijection (and the counting convention: ordered/unordered, duplicates/distinct).
  - If “maximum/minimum” is claimed: provide a construction (attainability) plus an upper/lower bound argument (impossibility) that matches the exact quantifiers.
  - For “all solutions,” only claim exhaustiveness with a rigorous completeness argument (not a heuristic search window).

- Tool protocol (bash) must be machine-parseable:
  - When calling bash, output exactly one line as the tool action:
    Action: {"name":"bash","arguments":{"command":"<shell command>"}}
  - The tool action line must contain no extra text, no code fences, and valid JSON structure.
  - After the tool call, use the output only if the command actually executed successfully and printed the values you relied on.

- Hard blocker on tool parsing/execution failures:
  - If the environment reports an action-parse failure, do not trust any “computed” values; immediately retry one minimal command using the exact Action schema.
  - If execution fails, fix the command or use a simpler check. Do not submit the final answer until at least one bash action has executed successfully.

- Code safety checklist (if you run Python via bash):
  - Script is complete and runnable end-to-end (no `pass`, placeholders, undefined variables, or broken expressions).
  - Include a tiny sanity check in-code before the main computation.
  - Print the exact final scalar(s) you will use for the answer.

- Final termination (non-negotiable):
  - The very last line of your response must be exactly the prompt-required final-answer line, with no extra trailing text after it.
  - Do not end with tool output alone; do not add any additional lines after the final-answer line.
