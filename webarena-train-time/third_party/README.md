# WebArena third-party runtime

The maintained WebArena Trace2Skill runner imports the WebArena SFT rollout
environment from a local SkillOpt checkout at:

```text
webarena-train-time/third_party/skillopt
```

This checkout is a runtime dependency, not an active SkillOpt baseline, and is
ignored by Git because it is an external repository. Historical SkillOpt
training launchers are not included in the core repository.
