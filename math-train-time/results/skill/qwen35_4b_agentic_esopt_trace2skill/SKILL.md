# Math Reasoning Skill
Use algebraic reasoning as primary; use command-line Python for arithmetic, modular checks, symbolic manipulation, and spot-checks.

## Submission: strict terminal output contract (grader-critical)
- The very last line must be exactly: `Final answer: \boxed{<Answer>}`
- Put no extra text after it (no trailing whitespace, no extra lines).

## Tool/action protocol (fail-closed)
- If you call a tool, emit exactly one valid Action in this schema: `Action: {"name":"bash","arguments":{"command":"<cmd>"} }`
- Hard stop on any tool/protocol problem (unparseable action, nonzero exit, SyntaxError/runtime exception, malformed command, etc.).
- Do **not** finalize a numeric answer based on a failed/uncertain tool run; switch to reasoning-only derivation and self-check.

## Computation + verification discipline
- Never trust failed/unfinished/timeout/parse-error tool runs.
- Prefer exact integer math (fractions, isqrt) over floats.
- Verify the final candidate against the **original** constraints (not an intermediate condition).
- After deriving the answer, do a final substitution/back-check, then output immediately.

## Correctness rules you must satisfy
### Quantifiers / extremal goals / win-lose
- For “for all …”, you must prove it or give a counterexample.
- For max/min threshold claims: prove a universal bound, then prove tightness via construction (or show impossibility at the next boundary).
- For whose-turn games: encode the state exactly as stated; win iff ∃ legal move to a losing state for the opponent; losing iff all legal moves go to winning.

### Minima claims (rule)
- If you claim the minimum is exactly L, provide (1) a rigorous lower bound and (2) a sufficiency/construction argument at L.

### DP/memo ordering
- Ensure any `memo[prev]`/dependency is computed before use; fill states in increasing order from validated base cases.

### Counting (bijections)
- Use canonical parameterization so each valid object maps to exactly one tuple.
- Prove uniqueness and exhaustiveness before translating to numeric bounds.

### Boundary semantics / off-by-one
- Fix corner-hit/stop-time conventions up front; use strict inequalities consistent with the statement.

## Integer-safety + edge conventions
- Perfect-square test: `r = isqrt(n)` and check `r*r == n` (no floats). Treat 0 as a perfect square.
- Prime convention (unless overridden): prime means integer ≥ 2; 0 and 1 are not prime.
- Ensure expressions are truly integers before applying square checks (use congruences/divisibility first).

## Common proof/derivation patterns
### Digit-sum generate-then-verify
- Convert digit rules into explicit arithmetic conditions on decimal digits (including carry/9→0 behavior).
- Enumerate only candidates implied by the digit condition.
- Compute digit sums from actual decimal digits, then verify right before output.

### v2(N) hygiene
- Compute v2(N) by repeatedly dividing out factors of 2; then reconstruct-check `odd_part * 2^k == N`.

### Z[x] divisibility guardrails
- For integer-coefficient polynomials p: `(x-c) | (p(x)-p(c))` in Z[x] (safe to use).
- In shifted-divisibility problems, only claim strong factor divisibility if integrality in Z[x] is justified.
- Use safe identity pattern: `(n-k) | (Q(k)-Q(n))` when Q has integer coefficients and you have integer evaluations.

### Reachability / lattice-hop
- Mod/parity necessary conditions are not sufficient: after minimal hop count, prove attainability by explicit construction or an exact reachability check.

## Proven game and DP discipline
- Parse “whose turn” and exact move legality; encode state exactly as stated.
- Deterministic rule: win iff ∃ move to a losing state for the opponent; losing iff all legal moves go to winning.
- For “largest/smallest A” premises: only keep candidates that allow at least one legal first move consistent with the premise.

## Stop-and-finish discipline
1) Derive and select the correct branch/candidate using the stated constraints.
2) Verify by substitution/checking the defining condition(s).
3) Output immediately with the exact last line `Final answer: \boxed{<Answer>}`.
