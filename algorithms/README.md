# Algorithms

Shared optimization and training components live here:

- `es/`: seeded full-parameter Agentic-ESOpt implementation and replay state.
- `trace2skill-settings/`: Trace2Skill prompts, configs, skills, and preparation tools.
- `verl_trace2skill/`: multi-turn VERL tools, parsers, rewards, and sandbox support.
- `verl/`: bundled upstream VERL source used by the GRPO launchers.

Task entry points remain under `scripts/` and the corresponding
`*-train-time/` directories. Stable task data remains at the repository root
under `data/`.
