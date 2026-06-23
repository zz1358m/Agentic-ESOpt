# Project Layout

The tracked repository is the paper-facing Dynamic-Agent codebase.

- `data/ahd/settings/`: AHD cfg and prompt files used by the reported tables.
- `ahd-test-time/`: EoH-based AHD runner and task environments for TSP, KP,
  ASP, and CVRP.
- `webarena-train-time/`: WebArena-Lite environment wrapper, Trace2Skill
  adapter, ES runners, and the generated Trace2Skill skill.
- `es/`: shared ES client, registry, and seeded perturbation helpers.
- `scripts/`: top-level launchers plus `settings.example.env`.

The following are intentionally not tracked:

- `data/ahd/datasets/`
- `data/webarena/`
- `webarena-train-time/methods/trace2skill/source/`
- run outputs under `runs/`, `cache/`, and `logs/`
- Jericho/JITRL/SkillOpt historical code
- paper source files

Machine-specific configuration belongs in `scripts/settings.local.env`, which is
ignored by git.
