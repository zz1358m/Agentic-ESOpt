# Agentic ESOpt Explorer

An English, interactive research website built from the original Agentic ESOpt logs, the paper's reported results, and explicitly labeled Stage 3 capability rechecks that passed formal acceptance. It is a static multi-page React/Vite build: the browser loads compact JSON and image assets, and no API, database, model server, or user account is required.

Published site: <https://zz1358m.github.io/Agentic-ESOpt/>

## Local development

```bash
npm install
npm run data
npm run dev
```

The data command reads the parent `Agentic-ESOpt` repository by default. The
accepted recheck archive and manuscript are external inputs and must be named
explicitly:

```bash
ESOPT_SOURCE_ROOT=/path/to/Agentic-ESOpt \
ESOPT_RECHECK_ROOT=/path/to/20260809_ESOpt_Recheck \
ESOPT_MANUSCRIPT_ROOT=/path/to/manuscript \
npm run data
```

## Verify and build

```bash
python -m unittest discover -s tests -p 'test_*.py'
npm test
npm run typecheck
npm run build
```

The deployable site is emitted to `dist/`. The repository workflow in
`.github/workflows/pages.yml` tests and builds this directory, then publishes
it with GitHub Pages whenever `website/` changes on `main`. Each task has a
physical `index.html`, so deep links do not depend on server-side route
rewriting. Relative asset and data URLs keep the build valid at the project
subpath `/Agentic-ESOpt/`.

Before the first deployment, a repository administrator must open
**Settings → Pages** and set **Source** to **GitHub Actions**. GitHub does not
allow the workflow's ordinary `GITHUB_TOKEN` to perform this one-time
enablement. Subsequent pushes that change `website/` deploy automatically.

## Public-data boundary

- Only selected cases and compact curves are published.
- Absolute server paths and common email/phone patterns are removed or rejected at build time.
- Sudoku exposes full linked prediction boards only for the accepted mask-5 Stage 3 recheck.
- WebArena exposes linked task outcomes and the retained final text, not an invented browser replay or recycled earlier output.
- WebArena has periodic evaluation scores but no separately retained training-split score; the interface states this limitation.
- AHD exposes original objectives/final heuristics and clearly labeled intermediate code from the accepted Stage 3 recheck.
- Model-size scaling compares 4B/9B at fixed G; ES population scaling compares G=8/16 at fixed backbone. The site defines G as perturbation directions per update, not physical compute nodes.
- The interface contains no raw-log or dataset download action.
- `data_audit.json` records payload counts, private-path/local-endpoint/contact-pattern scans, source-curve checks, and manuscript-value checks.

See [PLAN_COVERAGE.md](PLAN_COVERAGE.md) for the plan-to-implementation matrix and explicit source-retention boundaries.
