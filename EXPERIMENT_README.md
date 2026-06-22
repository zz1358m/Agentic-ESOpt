# Experiment Status

This document records the experiments that are finished, currently running, or
deprecated in `/home/zhi/Dynamic-Agent`.

Project overview and ES mechanics are in `README.md`. Directory layout is in
`PROJECT_LAYOUT.md`.

## Current Active Experiment

### WebRL-SFT WebArena-Lite Eval

Started on `2026-06-05 07:36 UTC`.

Purpose: align the WebArena-Lite harness to WebRL-style SFT behavior before
running ES. The active run is evaluation only: no fixed skill file, no ES noise,
and no parameter update.

```text
run: runs/webrl_lite_full_es/webrl_sft_eval_20260605_0738/
model: /data0/zhi/meta-llama/webrl-sft-llama-3.1-8b
endpoints: 11013, 11014, 11015, 11016
prompt: cache/external/VAB-WebArena-Lite-work/agent/prompts/jsons/p_webrl.json
action_set_tag: webrl_id
observation_type: webrl
temperature: 0.0
max_tokens: 2048
viewport: 1280x720
skill_file: disabled
```

Important implementation fixes for this run:

```text
ahd-test-time/methods/eoh/original/eoh/src/eoh/llm_local_server/llama31_instruct_server.py
- temperature <= 0 now switches to greedy decode instead of passing temperature=0
  to transformers sampling.

webarena-train-time/scripts/run_webrl_lite_full_es_train.py
- result directories are resolved to absolute paths before calling VAB run.py.
```

Current observed status:

```text
completed task action files: at least 4/165 when this section was written
early scores: task 0-3 all 0.0
```

The early `0.0` scores are real evaluator outputs, not missing result files.
Task 0 reached an answer but emitted `ANSWER:` in the exit message, which the
exact evaluator did not accept.

### WebRL / WebArena-Lite Data Splits

Use the WebRL SFT experience data as the original training source. Do not treat
the raw WebArena task-id remainder as the WebRL training split.

WebRL SFT training source:

```text
manifest: cache/webrl_sft_split/manifest.json
sft data: cache/external/WebRL/scripts/webarena_lite_sft.pt
task info: cache/external/WebRL/WebArena-Lite_info.json
task_info_count: 1123
trajectory_count: 1124
unique_task_text_count: 1113
duplicated_task_text_count: 10
step_total: 9469
step_min: 1
step_max: 52
step_avg: 8.424377224199288
```

WebRL SFT first-site category counts:

```text
reddit: 329
shopping: 255
map: 233
shopping_admin: 219
gitlab: 87
```

WebArena-Lite eval/test split currently used for full eval:

```text
path: cache/jitrl_webarena_lite/items.json
count: 165
map: 63
shopping_admin: 45
shopping: 34
gitlab: 14
reddit: 9
```

Custom raw WebArena remainder, kept only as a non-official candidate pool:

```text
path: cache/jitrl_webarena_train_excluding_lite/items.json
count: 647
gitlab: 182
shopping: 158
shopping_admin: 137
reddit: 105
map: 49
wikipedia: 16
```

This 647-task file is not used as a default ES training split anymore. ES
training must explicitly choose its training source; the current intended
alignment is to derive the original split from WebRL SFT/experience data.

## Historical Experiment

### Qwen32B Jericho Library

Started on `2026-06-03 12:26 UTC`.

Control rollouts:

```text
runs/library_qwen32b_direct_policy_20260603_123338/
runs/library_qwen32b_evotest_20260603_123338/
```

ES rollouts:

```text
runs/library_qwen32b_direct_policy_es_20260603_123338/
runs/library_qwen32b_evotest_es_20260603_123338/
```

Settings:

```text
game: library
model: Qwen3-32B
horizon: 110 environment steps
runs: 3 independent runs per setting
episodes: 50 per run
temperature: 0.4
prompt style: EvoTest ACTION format
official env probes: enabled, logged only, not injected into prompt
```

Control settings:

```text
direct_policy
evotest
```

ES settings:

```text
direct_policy_es
evotest_es
```

ES parameters:

```text
parameter_scope: lora
LoRA tensors: 896
LoRA ES params: 67,108,864
sigma: 0.003
lr / alpha: 0.001
interval: 10 environment steps
reward EMA decay: 0.8
reward normalization: zscore
update cadence: once per episode
```

Model routing:

```text
11013: base Qwen3-32B, no LoRA, used by direct controls and EvoTest attribution/evolution
11014: Qwen3-32B + LoRA policy endpoint
11015: Qwen3-32B + LoRA policy endpoint
11016: Qwen3-32B + LoRA policy endpoint
```

Status at document rewrite time:

- Four runner processes are alive, one per setting.
- The ES endpoints have successfully initialized with `parameter_scope=lora`.
- Early episodes show `evotest` and `evotest_es` can reach `5/30`; direct-policy
  settings are still at `0/30` in early episodes.
- No final scores are available yet; do not interpret this run until
  `summary.tsv`, `run_averages.tsv`, and `overall_averages.tsv` are complete.

Useful monitor commands:

```bash
tail -n 80 runs/library_qwen32b_evotest_20260603_123338/runner.log
tail -n 100 runs/library_qwen32b_evotest_es_20260603_123338/runner.log
find runs/library_qwen32b_*_20260603_123338 -name 'episode_*.json' | wc -l
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

## Completed Alignment Checks

### EvoTest Library 30-Step Deterministic Check

Purpose: verify that the local Jericho runner matches the official EvoTest
environment, prompt format, action parser, and parser-feedback loop.

Official source run:

```text
runs/original_evotest_qwen32b_library_compare30_temp0_20260603/
```

Local aligned run:

```text
runs/aligned_evotest_qwen32b_library_compare30_temp0_afterprompt_20260603/
```

Setting:

```text
game: library
model: Qwen3-32B
temperature: 0.0
episodes: 1
steps: 30
setting: evotest
```

Result:

- All 30 actions match exactly.
- All 30 rewards and cumulative scores match exactly.
- Final score is `0/30` in both runs.

Interpretation:

This is an environment/protocol alignment result, not a performance claim. It
means the local runner is now deterministic-equivalent to the official EvoTest
repo for this short `library` check.

## Completed AHD Results

### TSP Construct

Task:

```text
problem: tsp_construct
TSP size: 50
training instances: 64
population size: 10
generations: 25
seed: 1234
metric: final objective, lower is better
```

Unperturbed EoH control:

```text
mean final objective over 3 runs: 6.75471
```

Best completed Dynamic-Agent ES setting:

```text
sigma: 1e-3
alpha: 5e-4
final objectives: 6.53436, 6.51985, 6.44209
mean final objective: 6.49877
```

Other completed or partial settings:

```text
sigma=3e-3, alpha=1e-3: complete, mean 6.73115
sigma=1e-3, alpha=3e-3: partial, rep1 complete, rep2/rep3 stopped at generation 12
sigma=3e-3, alpha=3e-3: not started
```

Summary files:

```text
cache/tsp_construct_results_summary.md
cache/plots/tsp_construct_convergence.png
cache/plots/tsp_construct_convergence_summary.csv
```

### KP Construct

Task:

```text
problem: kp_constructive
items: 100
instances: 64
seed: 3333
capacity: 25
metric: -average_selected_value, lower is better
```

Completed safe comparison:

```text
unperturbed EoH control run2 objective: -40.15308
Dynamic-Agent ES run1 objective:       -40.15641
```

Interpretation:

Because the KP objective is negative selected value, `-40.15641` is slightly
better than `-40.15308`, corresponding to an average selected-value improvement
of about `0.00333`.

Invalid old result:

```text
-49.93006
```

This value came from generated code mutating evaluator inputs in place. It is a
reward-hacking artifact and must not be counted.

Summary file:

```text
cache/kp_construct_results_summary.md
```

## Deprecated Or Non-Interpretable Runs

The following Jericho runs should not be used as paper-aligned evidence:

```text
runs/jericho_library_zork1_gpu0_baseline_evotest_3run50ep_T110_20260603_070726/
runs/jericho_library_zork1_lora_es_sigma005_lr002_3run50ep_T110_20260603_070931/
```

Reason:

- They used valid-action or inventory oracle information.
- Their Library scores were inflated relative to EvoTest-style observation-only
  evaluation.

Many older `qwen3_14b` Jericho directories are smoke tests, failed environment
alignment attempts, or parameter probes. Prefer `qwen32b` and explicitly named
alignment/current-run directories when interpreting Jericho results.

## Implemented ES Mechanics

### AHD ES

Scope:

```text
parameter_scope: full or all_linear
```

Flow:

1. EoH proposes or mutates candidate heuristic code.
2. The local LLM server applies temporary parameter noise via `/es/apply`.
3. The candidate is evaluated by the optimization environment.
4. The same seed is reverted via `/es/revert`.
5. Candidate rewards are normalized and sent to `/es/update`.

Reward:

- TSP/KP reward is based on candidate objective quality and improvement.
- Lower objective is better.
- Rewards are normalized before model update.

### Jericho Horizontal LoRA ES

Scope:

```text
parameter_scope: lora
```

Flow:

1. At the start of each independent run, the policy endpoint calls `/es/reset`.
2. During an episode, every `K=10` environment steps starts a new segment.
3. Each segment samples one seed and applies LoRA noise with `/es/apply`.
4. At segment end, the seed is reverted with `/es/revert`.
5. At episode end, all segment seeds and segment advantages are sent to
   `/es/update`.

Reward:

```text
raw_segment_reward = score_after_segment - score_before_segment
old_ema = ema[segment_index]
advantage = raw_segment_reward - old_ema
ema[segment_index] = 0.8 * old_ema + 0.2 * raw_segment_reward
```

The episode's segment advantages are then z-score normalized by the ES server.

Update:

```text
theta <- theta + (alpha / N) * sum_i normalized_advantage_i * epsilon(seed_i)
```

For `evotest_es`, the EvoTest attribution/evolution model is the base Qwen
endpoint and is never perturbed. Only the policy endpoint's LoRA adapter is
updated.

## Current Entry Points

Jericho:

```text
runs/qwen3_14b_parallel/run_jericho_evotest_standard.py
examples/jericho/README.md
```

AHD:

```text
examples/tsp_construct/runEoH_llama31_maxeval1000.py
examples/tsp_construct/runEoH_llama31_model_es_maxeval1000.py
ahd-test-time/methods/eoh/original/eoh/src/eoh/methods/eoh/eoh_interface_EC.py
ahd-test-time/methods/eoh/original/eoh/src/eoh/llm/model_es_client.py
ahd-test-time/methods/eoh/original/eoh/src/eoh/llm_local_server/llama31_instruct_server.py
```

## WebAgent-Lite Llama-3.1-8B Status, 2026-06-04 20:20 UTC

Model servers:

```text
GPU0 port 11013: Llama-3.1-8B-Instruct
GPU1 port 11014: Llama-3.1-8B-Instruct
GPU2 port 11015: Llama-3.1-8B-Instruct
GPU3 port 11016: Llama-3.1-8B-Instruct
```

Environment:

```text
model_path: /data0/zhi/meta-llama/Llama-3.1-8B-Instruct
train_split: cache/webarena_lite_skillopt_split_available/train/items.json
train_size: 79
test_split: cache/webarena_lite_skillopt_split_available/test/items.json
test_size: 26
VAB/WebArena root: cache/external/VAB-WebArena-Lite-work
```

Baseline train probe:

- All 79 train cases were evaluated with the VAB WebArena-Lite runner.
- Only task `120` received a positive score: `1.0`.
- All other train cases scored `0.0`.
- Task `120` is a shopping task: `Add this product to my wishlist`.

Anchor ES batch:

```text
cache/webarena_lite_skillopt_split_es_anchor120/items.json
case_batch: [120, 114, 118, 14]
```

The anchor batch intentionally includes the only observed positive train task
plus three currently zero-score tasks so that ES can distinguish perturbations
that preserve or break the known successful behavior.

Current full-parameter ES runs:

```text
runs/webrl_lite_full_es/base_full_es_anchor120_pop32_batch4_sigma5e-4_alpha1e-3_20260604
runs/webrl_lite_full_es/base_full_es_anchor120_pop32_batch4_sigma2e-3_alpha1e-3_20260604
runs/webrl_lite_full_es/skillopt_full_es_anchor120_pop32_batch4_sigma5e-4_alpha1e-3_20260604
runs/webrl_lite_full_es/base_full_es_anchor120_pop32_batch4_sigma2e-3_alpha2e-3_seed20260606_20260604
runs/webrl_lite_full_es/base_full_es_anchor120_pop32_batch4_sigma2e-3_alpha2e-3_seed20260607_20260604
runs/webrl_lite_full_es/base_full_es_anchor120_pop32_batch4_sigma2e-3_alpha5e-3_seed20260608_20260604
```

Preliminary parameter read:

- `sigma=1e-4` was stopped and reverted because complete rewards stayed around
  `0.25`, producing too little ranking signal.
- `sigma=5e-4, alpha=1e-3` is the best current default for Base+ES. It produced
  complete rewards including `0.0`, `0.225`, and `0.25`, so it has usable
  ranking signal without uniformly destroying task `120`.
- `sigma=1e-3, alpha=1e-3` for Base+ES was stopped and reverted after early
  samples did not produce better ranking signal than `5e-4`.
- `sigma=2e-3, alpha=1e-3` for Base+ES is running as a higher-noise comparison.
  Early complete samples are mostly `0.25`, with several `0.225` samples from
  zero-task regressions.
- SkillOpt+ES at `sigma=5e-4, alpha=1e-3` currently has the clearest SkillOpt
  ranking signal: complete rewards include `0.0` and `0.25`, mostly driven by
  perturbations that break or preserve task `120`.
- SkillOpt+ES at `sigma=1e-3, alpha=1e-3` was stopped and manually reverted at
  active seed `665180348` after six complete samples all scored `0.25`, giving
  no useful rank diversity.
- SkillOpt+ES at `sigma=7e-4, alpha=2e-3` was started as the replacement:
  slightly more noise than the current `5e-4` run, higher update step size, and
  a different RNG seed so it does not duplicate the active `5e-4` sample order.
  The first background launch was reverted at seed `1748601613` after the shell
  dropped the process; it was restarted as `20260604b` in a held long session.
  It was later stopped and reverted at active seed `588993925` after six
  complete samples all scored `0.25`, giving no rank diversity.
- A Base+ES replication run at `sigma=2e-3, alpha=2e-3` was started on GPU2 with
  seed `20260606` because the original Base+ES `2e-3` run produced repeated
  high-reward samples.

Monitor snapshot, 2026-06-04 21:12 UTC:

```text
Base+ES sigma=5e-4:     16/32 complete, rewards [0.0, 0.25 x15], no history.json yet
Base+ES sigma=2e-3:      6/32 complete, rewards include 0.225 and 0.25, no history.json yet
SkillOpt+ES sigma=5e-4: 12/32 complete, rewards include 0.0 and 0.25, no history.json yet
SkillOpt+ES sigma=1e-3:  stopped after 6/32 complete, all rewards 0.25
SkillOpt+ES sigma=7e-4:  restarted as 20260604b on GPU2/port 11015
```

No ES update has been applied yet for these four active runs because each run
updates only after all `population=32` samples in the generation finish.

Monitor update, 2026-06-04 21:26 UTC:

```text
Base+ES sigma=2e-3: observed first positive ES sample above anchor baseline.
  gen_000_sample_08_seed_1068411708 scored reward 0.5:
  task_120=1.0, task_114=1.0, task_118=0.0, task_14=0.0.
Base+ES sigma=5e-4: 18/32 complete, rewards still at 0.0/0.225/0.25.
SkillOpt+ES sigma=5e-4: 14/32 complete, rewards at 0.0/0.25.
SkillOpt+ES sigma=7e-4 alpha=2e-3: restarted as 20260604b; sample 0 active.
```

This makes `Base+ES sigma=2e-3` worth keeping despite earlier concerns about
higher noise: it has produced the first sample that improves a previously zero
task while preserving the known positive anchor task.

Monitor update, 2026-06-04 21:34 UTC:

```text
Base+ES sigma=2e-3: 12/32 complete, two high-reward samples:
  sample_08 seed 1068411708 reward 0.5
  sample_10 seed 1709331676 reward 0.5
Both high samples keep task_120=1.0 and raise task_114=1.0.
SkillOpt+ES sigma=7e-4 alpha=2e-3: sample_00 and sample_01 both reward 0.25.
```

Monitor update, 2026-06-04 21:45 UTC:

```text
Base+ES sigma=2e-3: 14/32 complete, three high-reward samples:
  sample_08 seed 1068411708 reward 0.5
  sample_10 seed 1709331676 reward 0.5
  sample_13 seed 1863683177 reward 0.5
All three preserve task_120=1.0 and raise task_114=1.0.
Base+ES sigma=5e-4: 20/32 complete, no complete reward above 0.25 yet.
SkillOpt+ES sigma=5e-4: 17/32 complete, no complete reward above 0.25 yet.
SkillOpt+ES sigma=7e-4 alpha=2e-3: 2/32 complete, both reward 0.25.
```

Monitor update, 2026-06-04 21:54 UTC:

```text
Base+ES sigma=2e-3: 16/32 complete, five samples above 0.25:
  reward 0.5:   sample_08, sample_10, sample_13, sample_14
  reward 0.475: sample_15
All high samples raise task_114=1.0; sample_15 has task_14=-0.1, so reward is
slightly below 0.5. This is now the clear lead setting.
Base+ES sigma=5e-4: 21/32 complete, no complete reward above 0.25.
SkillOpt+ES sigma=5e-4: 18/32 complete, no complete reward above 0.25.
SkillOpt+ES sigma=7e-4 alpha=2e-3: 3/32 complete, all reward 0.25.
```

Monitor update, 2026-06-04 22:03 UTC:

```text
Base+ES sigma=2e-3: 18/32 complete, six samples above 0.25, mean reward 0.329.
  High samples consistently make task_114=1.0 while preserving task_120=1.0.
Base+ES sigma=5e-4: 23/32 complete, still no complete reward above 0.25.
SkillOpt+ES sigma=5e-4: 19/32 complete, still no complete reward above 0.25.
SkillOpt+ES sigma=7e-4 alpha=2e-3: 4/32 complete, all reward 0.25.
```

Monitor update, 2026-06-04 22:13 UTC:

```text
SkillOpt+ES sigma=7e-4 alpha=2e-3: stopped after 6/32 complete, all reward 0.25.
  Active seed 588993925 was manually reverted.
GPU2 replacement: Base+ES sigma=2e-3 alpha=2e-3, seed 20260606.
```

Monitor update, 2026-06-04 22:29 UTC:

```text
Base+ES sigma=2e-3 alpha=1e-3: 24/32 complete, 7 high-reward samples.
Base+ES sigma=2e-3 alpha=2e-3 seed=20260606: 2/32 complete, sample_01 reward 0.5.
  This independently confirms that sigma=2e-3 can raise task_114=1.0 while
  preserving task_120=1.0.
Base+ES sigma=5e-4: 27/32 complete, still no complete reward above 0.25.
SkillOpt+ES sigma=5e-4: 23/32 complete, still no complete reward above 0.25.
```

Monitor update, 2026-06-04 22:38 UTC:

```text
Base+ES sigma=2e-3 alpha=1e-3: 26/32 complete, 8 high-reward samples.
Base+ES sigma=2e-3 alpha=2e-3 seed=20260606: 3/32 complete, one high sample
  complete and another partial high sample observed.
Base+ES sigma=5e-4: 28/32 complete, still no complete reward above 0.25.
SkillOpt+ES sigma=5e-4: 24/32 complete, still no complete reward above 0.25.
```

Current read: `sigma=2e-3` is the only setting that repeatedly raises task 114
while preserving task 120. `sigma=5e-4` can occasionally move one of these
tasks, but not both in a complete high-reward sample so far.

Monitor update, 2026-06-04 22:47 UTC:

```text
Base+ES sigma=2e-3 alpha=1e-3: 27/32 complete, 9 high-reward samples.
Base+ES sigma=2e-3 alpha=2e-3 seed=20260606: 5/32 complete, two high samples
  (0.5 and 0.475), mean reward 0.34.
Base+ES sigma=5e-4: 29/32 complete, still no complete reward above 0.25.
SkillOpt+ES sigma=5e-4: 25/32 complete, still no complete reward above 0.25.
```

Monitor update, 2026-06-04 22:56 UTC:

```text
Base+ES sigma=2e-3 alpha=1e-3: 29/32 complete, 10 high-reward samples.
Base+ES sigma=2e-3 alpha=2e-3 seed=20260606: 6/32 complete, two high samples
  plus another partial high trajectory.
Base+ES sigma=5e-4: 30/32 complete, still no complete reward above 0.25.
SkillOpt+ES sigma=5e-4: 26/32 complete, still no complete reward above 0.25.
```

Monitor update, 2026-06-04 23:26 UTC:

```text
SkillOpt+ES sigma=5e-4: 32/32 complete, no reward above 0.25, mean reward 0.2266.
  It is now in update/eval; history.json not written yet.
Base+ES sigma=2e-3 alpha=2e-3 seed=20260606: 16/32 complete, 3 high samples
  at the previous poll, then sample_16 produced another reward 0.5.
Base+ES sigma=2e-3 alpha=2e-3 seed=20260607: early samples include a 0.0
  failure, so this seed is currently weak.
Base+ES sigma=2e-3 alpha=5e-3 seed=20260608: early samples include a poor
  sample, so this high-alpha setting is not yet convincing.
```

Monitor update, 2026-06-04 23:34 UTC:

```text
SkillOpt+ES sigma=5e-4 completed:
  reward_mean: 0.2265625
  reward_best: 0.25
  eval_average: 0.125
  eval positive task: task_17=1.0
```

Stopped and reverted weak Base+ES branches:

```text
base sigma=2e-3 alpha=2e-3 seed=20260607:
  stopped at active seed 74670887; early rewards included multiple 0.0 samples.
base sigma=2e-3 alpha=5e-3 seed=20260608:
  stopped at active seed 1225640092; early rewards were weak.
```

New active sweep:

```text
GPU0 port 11013: base sigma=1.5e-3 alpha=2e-3 seed=20260609
GPU1 port 11014: base sigma=2.5e-3 alpha=2e-3 seed=20260610
GPU2 port 11015: base sigma=2e-3 alpha=2e-3 seed=20260606, continuing
GPU3 port 11016: base sigma=2e-3 alpha=2e-3 seed=20260611
```

Monitor update, 2026-06-04 23:50 UTC:

```text
Base sigma=1.5e-3 seed=20260609: first 3 complete samples all reward 0.25.
Base sigma=2e-3 seed=20260611: early high sample reward 0.475.
Base sigma=2.5e-3 seed=20260610: early high sample reward 0.5, but earlier
  samples showed task_114=1.0 with task_120=0.0, so this noise level may be
  stronger but less stable.
Base sigma=2e-3 seed=20260606: continuing, still produces high samples.
```

Monitor update, 2026-06-05 00:03 UTC:

```text
Base sigma=1.5e-3 seed=20260609: 4/32 complete, all reward 0.25.
Base sigma=2e-3 seed=20260611: 7/32 complete, high samples include 0.475 and 0.5.
Base sigma=2.5e-3 seed=20260610: 4/32 complete, two high samples at 0.5.
Base sigma=2e-3 seed=20260606: 25/32 complete, six high samples, mean around 0.294.
```

Current parameter read: `1.5e-3` looks too weak so far; `2e-3` remains robust;
`2.5e-3` is promising but needs more samples to see whether it breaks the anchor
too often.

Monitor update, 2026-06-05 00:12 UTC:

```text
Base sigma=1.5e-3 seed=20260609: stopped after 8 complete samples, all reward 0.25.
  Active seed 1081108893 was manually reverted.
Replacement on GPU0: base sigma=3e-3 alpha=2e-3 seed=20260612.
```

Monitor update, 2026-06-05 00:22 UTC:

```text
Base sigma=2e-3 seed=20260606: population complete, reward_mean around 0.2898,
  8 high samples; now in eval.
Base sigma=2.5e-3 seed=20260610: 8/32 complete, 5 high samples, mean around 0.406.
Base sigma=3e-3 seed=20260612: first complete sample reward 0.5.
Base sigma=2e-3 seed=20260611: 16/32 complete, 3 high samples.
```

Updated parameter read: `2.5e-3` is now the strongest early sweep. `3e-3`
needs more samples but is not immediately destructive.

Completed run, 2026-06-05 00:31 UTC:

```text
Base sigma=2e-3 alpha=2e-3 seed=20260606:
  population complete: 32/32
  reward_mean: 0.28984373807907104
  reward_best: 0.5
  update: ok, weight_norm 5.656853675842285
  eval_average: 0.125
  eval positive task: task_17=1.0
```

Current active read:

```text
Base sigma=2.5e-3 seed=20260610: 10/32 complete, 7 high samples, mean ~0.4225.
Base sigma=3e-3 seed=20260612: 2/32 complete, both reward 0.5.
Base sigma=2e-3 seed=20260611: 19/32 complete, 4 high samples.
```

GPU2/port 11015 was restarted from original weights after the completed
`seed=20260606` run, then reused for:

```text
Base sigma=3.5e-3 alpha=2e-3 seed=20260613
```

Monitor update, 2026-06-05 00:41 UTC:

```text
Base sigma=3e-3 seed=20260612: 5/32 complete, all five reward 0.5.
Base sigma=2.5e-3 seed=20260610: 11/32 complete, 8 high samples, mean ~0.432.
Base sigma=2e-3 seed=20260611: 21/32 complete, 4 high samples, later samples weaker.
Base sigma=3.5e-3 seed=20260613: sample 0 still running; no complete sample yet.
```

Current best early setting is `sigma=3e-3, alpha=2e-3`; `2.5e-3` is close and
has more samples.

Monitor update, 2026-06-05 00:49 UTC:

```text
Base sigma=3e-3 seed=20260612: 6/32 complete, all 6 reward 0.5.
Base sigma=2.5e-3 seed=20260610: 14/32 complete, 11 high samples.
Base sigma=3.5e-3 seed=20260613: 1/32 complete, reward 0.5.
Base sigma=2e-3 seed=20260611: 25/32 complete, weaker than higher-sigma runs.
```

The promising noise region has moved upward: `2.5e-3` to `3.5e-3` is currently
better than `2e-3`, while `1.5e-3` was too weak.

Monitor update, 2026-06-05 00:55 UTC:

```text
Stopped Base sigma=2e-3 seed=20260611 after 27 complete samples; it was weaker
than the higher-sigma sweeps. Active seed 1603290677 was reverted.
Replacement on GPU3: Base sigma=4e-3 alpha=2e-3 seed=20260614.
```

Monitor update, 2026-06-05 01:02 UTC:

```text
Base sigma=2.5e-3 seed=20260610: 19 samples observed, 16 high rewards around 0.5.
Base sigma=3e-3 seed=20260612: 8/8 high rewards, all 0.5.
Base sigma=3.5e-3 seed=20260613: 3/3 high rewards, all 0.5.
Base sigma=4e-3 seed=20260614: first sample reward 0.5.
```

Current read: the best region is now at least `3e-3` to `4e-3`; these
perturbations consistently make task 114 succeed while preserving task 120.

Monitor update, 2026-06-05 01:10 UTC:

```text
Base sigma=2.5e-3 seed=20260610: 21 samples observed, 18 high rewards.
Base sigma=3e-3 seed=20260612: 9/9 high rewards.
Base sigma=3.5e-3 seed=20260613: 4/4 high rewards.
Base sigma=4e-3 seed=20260614: 2/2 high rewards.
```

No upper noise boundary has appeared yet in this anchor batch; higher sigma is
still preserving task 120 while raising task 114.

Completed run, 2026-06-04 23:15 UTC:

```text
Base+ES sigma=2e-3 alpha=1e-3:
  population complete: 32/32
  reward_mean: 0.3257812261581421
  reward_best: 0.5
  high_reward_samples: 11
  update: ok, weight_norm 5.656854152679443
  eval_limit: 8
  eval_average: 0.125
  eval_max: 1.0
  eval positive task: task_17=1.0
```

Training reward clearly improves over `sigma=5e-4`, but the current 8-task eval
slice is unchanged: both completed Base+ES runs get only `task_17=1.0`.
GPU0/port 11013 is therefore being restarted from original weights for a higher
learning-rate replication at `sigma=2e-3, alpha=5e-3, seed=20260608`.
The corresponding run id is
`base_full_es_anchor120_pop32_batch4_sigma2e-3_alpha5e-3_seed20260608_20260604`.

Completed run, 2026-06-04 23:05 UTC:

```text
Base+ES sigma=5e-4 alpha=1e-3:
  population complete: 32/32
  reward_mean: 0.234375
  reward_best: 0.25
  update: ok, weight_norm 5.656852722167969
  eval_limit: 8
  eval_average: 0.125
  eval_max: 1.0
  eval positive task: task_17=1.0
```

This run did not discover a complete high-reward training sample above `0.25`,
so GPU1/port 11014 is being recycled for another `sigma=2e-3` Base+ES
replication after restarting the model server to restore original weights.
The replacement run is
`base_full_es_anchor120_pop32_batch4_sigma2e-3_alpha2e-3_seed20260607_20260604`.

Original SkillOpt run:

```text
runs/skillopt_webagent_lite/skillopt_epoch_eval_llama8b_v3_20260604
```

Observed status:

- Steps `1`, `2`, `3`, and `4` each generated a failure patch, ran selection eval,
  and were rejected.
- `best_score` remains `0.0`.
- The run was stopped during step `5` and GPU2 was reassigned to
  `skillopt_full_es_anchor120_pop32_batch4_sigma1e-3_alpha1e-3_20260604`.

Monitor update, 2026-06-05 03:44 UTC:

Direct Llama-8B WebArena-Lite eval snapshots:

```text
base_available100_base_evalfix_run1:
  completed cases: 56
  average score: 0.0339
  total score: 1.9
  positive cases: 2
  max score: 1.0

skillopt_available100_skillopt_evalfix_run1:
  completed cases: 35
  average score: 0.0571
  total score: 2.0
  positive cases: 2
  max score: 1.0
```

These are incomplete slices of the available-100 split, so they are useful for
sanity checking but not yet a final base-vs-skillopt comparison.

Completed ES evals after one full population update:

```text
Base+ES sigma=5e-4 alpha=1e-3: eval_average=0.125
SkillOpt+ES sigma=5e-4 alpha=1e-3: eval_average=0.125
Base+ES sigma=2e-3 alpha=1e-3: eval_average=0.125
Base+ES sigma=2e-3 alpha=2e-3 seed=20260606: eval_average=0.125
```

All completed ES evals above use the same 8-task eval slice and currently only
solve `task_17`, so the eval average remains `1/8 = 0.125` even when training
anchor reward improves.

Active Base+ES anchor-batch runs:

```text
sigma=2.5e-3 alpha=2e-3 seed=20260610:
  complete population samples: 24/32
  train anchor mean over complete samples: 0.4677
  high samples: 21/24

sigma=3e-3 alpha=2e-3 seed=20260612:
  complete population samples: 11/32
  train anchor mean over complete samples: 0.5000
  high samples: 11/11

sigma=3.5e-3 alpha=2e-3 seed=20260613:
  complete population samples: 7/32
  train anchor mean over complete samples: 0.5000
  high samples: 7/7

sigma=4e-3 alpha=2e-3 seed=20260614:
  complete population samples: 4/32
  train anchor mean over complete samples: 0.5000
  high samples: 4/4
```

The high-sigma runs are still active. GPU utilization was observed around
90-96% on all four cards during generation; later spot checks show transient
low utilization on GPUs waiting between browser tasks, but the driver processes
continue advancing.

Correction, 2026-06-05 03:56 UTC:

The ES driver default eval split has been changed from the 26-task
`cache/webarena_lite_skillopt_split_available/test/items.json` smoke split to
the full WebArena-Lite test cache:

```text
eval_split: cache/webrl_lite/test/items.json
local task count: 165
sites: shopping, shopping_admin, reddit, gitlab, wikipedia, map
```

The user-facing final score table should therefore report full WebArena-Lite
evals for these four conditions:

```text
Base
PromptSkill
Base+ES
SkillOpt+ES
```

The older `eval_average=0.125` results above are retained only as 8-task smoke
checks and should not be treated as final test scores. The Base+ES runs already
in flight were launched with `--eval-limit 8`; after their ES update completes,
run an additional full eval on the updated endpoint before restarting that
model server.

Important SkillOpt reproduction note:

The original SkillOpt code was downloaded under `cache/external/SkillOpt` and
was run with `configs/webagent_lite/default.yaml`. In the current
`runs/skillopt_webagent_lite/skillopt_epoch_eval_llama8b_v3_20260604` attempt,
steps 1-4 generated candidate edits but all candidates were rejected by
selection gating:

```text
rollout_hard: 0.0
selection_hard: 0.0
candidate_gate_score: 0.0
action: reject
best_score: 0.0
best_origin: initial_skill
```

The no-ES skill-prompt baseline is labeled `PromptSkill` and uses
`webarena-train-time/skills/webrl_lite_skillopt_v1.md` as the skill file injected into the
WebRL/VAB prompt. The ES run with the same skill file is labeled `SkillOpt+ES`.

GPU3/port 11016 was recycled from the unstable `sigma=4e-3` Base+ES run after
manual revert of active seed `364504264`. A new SkillOpt+ES run is active:

```text
run_id: skillopt_full_es_anchor120_pop32_batch4_sigma3e-3_alpha2e-3_seed20260615_20260605
method: SkillOpt+ES
skill_file: webarena-train-time/skills/webrl_lite_skillopt_v1.md
sigma: 3e-3
alpha: 2e-3
population: 32
case_batch_size: 4
train_anchor_batch: [120, 114, 118, 14]
eval_split: cache/webrl_lite/test/items.json
eval_limit: none
```

Initial SkillOpt+ES output confirms the skill prompt is being accepted by the
VAB runner; sample 0 has begun on `task_120`.
