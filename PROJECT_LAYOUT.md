# Project Layout

The active experiment code is organized by setting:

- `data/`: vendored source snapshots, cached datasets, and prepared split data.
- `ahd-test-time/`: AHD test-time experiments with EoH plus model ES.
  - supported construct problems: `tsp_constructive`, `kp_constructive`, `asp_constructive`.
  - supported ACO problems: `tsp_aco`, `cvrp_aco`, `bpp_offline_aco`.
- `jericho-test-time/`: Jericho test-time agent wrappers and experiment notes.
- `webarena-train-time/`: WebArena/WebRL train-time harness for SkillOpt, Trace2Skill, and ES.
- `es/`: shared ES client/registry and model-server ES endpoints, including LoRA-capable flow.
- `scripts/`: compact shell launchers plus `scripts/settings.example.env`.
  Per-machine paths belong in an untracked `scripts/settings.local.env`, not in
  individual launch scripts.

Environment, setting, scenario, and method code is not kept at the repository
root. It lives inside the corresponding setting directory.
