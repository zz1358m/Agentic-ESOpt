# 运行超参数一览 🧪 / Run hyperparameters

本页列出所有维护中的用户入口脚本的实际默认值。模型路径、endpoint、GPU 编号和
输出目录等机器相关配置不重复列出。覆盖优先级为：脚本默认值 <
`scripts/settings.local.env` < 显式环境变量 < 命令行尾部参数。

Agentic-ESOpt 会把真实更新序列写入 `history.json`；Math/DocVQA 的 VERL
训练还会在日志旁保存 `experiment_config.json`。

## Sudoku

固定 ES 对比入口：
`sudoku-train-time/scripts/run_es_hyperparams.sh <profile> <mask>`。

| Profile | Generations / population / case batch | Sigma | Alpha | Max turns |
| --- | --- | --- | --- | --- |
| `vanilla-es32`, mask 5/10 | 100 / 32 / 32 | `1e-3` constant | `5e-4` | `mask × 3` |
| `vanilla-es32`, mask 15 | 100 / 32 / 32 | `5e-4` constant | `5e-4` | 45 |
| `agentic-esopt-es32`, mask 5/10 | 100 / 32 / 32 | `1e-3 → 2.5e-4` cosine | `5e-4` | `mask × 3` |
| `agentic-esopt-es32`, mask 15 | 100 / 32 / 32 | `7e-4 → 5e-4` cosine | `5e-4` | 45 |

全部使用全参数更新、z-score normalization、4 case workers、每 turn 64
tokens、endpoint batch 32；每 10 代评测并重复 3 次。采样为 `T=0.7`、
`top_p=0.8`、`top_k=20`、`min_p=0`、presence penalty `1.5`、repetition
penalty `1.0`。

`scripts/sudoku/run_es.sh` 是 smoke/configurable 入口，默认 1 代、population
8、case batch 8、sigma `5e-4` constant、alpha `5e-4`、90 turns。复现正式
对比应使用上面的固定 profile。

两套固定 GRPO 入口：

```bash
scripts/sudoku/run_grpo.sh      # T=0.7, top-p=0.8, top-k=20
scripts/sudoku/run_grpo_t1.sh   # T=1.0, top-p=1.0, top-k=-1
```

| Parameter | `run_grpo.sh` | `run_grpo_t1.sh` |
| --- | --- | --- |
| Steps / global batch / generations per prompt | 100 / 32 / 8 | 100 / 32 / 8 |
| Rollout / train micro-batch | 8 / 2 | 8 / 2 |
| Policy batch size / total cap | `512`（循环所有完整 batch）/ 无总量上限 | 相同 |
| LR / KL beta / clip epsilon | `1e-6` / `1e-3` / `0.2` | 相同 |
| Train sampling | `T=0.7`, `p=0.8`, `k=20` | `T=1`, `p=1`, `k=-1` |
| Eval sampling | `T=0.7`, `p=0.8`, `k=20` | `T=0.7`, `p=0.8`, `k=20` |

两组均使用 raw rollout-policy log probability、4 个 Accelerate 进程、
`mask × 3` turns；训练前评测，此后每 20 steps 评测并重复 3 次。

## Math

标准 Agentic-ESOpt/skill 入口：
`scripts/es_skill_workflow.sh math <es-train|eval|distill-skill|skill-eval>`。

| Group | Effective defaults |
| --- | --- |
| ES | 25 generations, population 16, case batch 16, sigma `1e-3 → 5e-4` cosine, alpha `5e-4`, full parameters, z-score (`ddof=0`, epsilon `1e-8`), seed `20260627` |
| Rollout | 1 train sample, 50 turns, 4096 tokens/turn, no total-token cap, `T=1`, `p=1`, `k=40`, presence penalty 2, repetition penalty 1 |
| Runtime | 4 vLLM engines, inference batch 16, context 131072, GPU memory utilization 0.85, bfloat16, eager mode |
| Eval | every 10 generations; 1 sample during training; 4 final raw/skill samples; DAPO 100; AIME 30 |
| Trace2Skill | all 25 ES generations and all candidate trajectories scanned; at most one failure per training problem (400 maximum), 32 workers, 80 skill lines, evolution temperature 1 |

底层 `scripts/math/run_vllm_es_4gpu.sh` 默认是 1 代、population 8、case
batch 8、sigma `5e-4` constant、alpha `5e-4`、全参数、z-score、1 train
sample、16 eval samples、`T=1`、`p=1`、`k=40`、presence penalty 2；另使用
4 engines、context 32768、GPU memory utilization 0.85。

异步 VERL GRPO 训练入口：`scripts/math/run_react_verl_grpo.sh train`
（等价于 `scripts/math/run_grpo.sh`）。

| Group | Effective defaults |
| --- | --- |
| Optimizer | GRPO, LR `1e-6`, mini-batch 20, micro-batch 1/GPU, KL loss `0.001`, low-variance KL |
| Data/run | train batch 20, 8 rollouts/prompt, 15 epochs, shuffle, seed 1, 4 GPUs |
| Rollout | async SGLang, TP=1, 100 user + 100 assistant turns, 512 tokens/turn, 8192 response tokens, 6000 tool tokens |
| Sampling | `T=1`, `p=1`, `k=40`, presence penalty 2, repetition penalty 1 |
| Runtime | context 40960, GPU memory utilization 0.50, max 16 sequences, 8 agent-loop workers |
| Eval/checkpoint | eval before training and every 5 steps; save every 20 steps |

`scripts/math/run_react_verl_grpo.sh eval` 默认 4 个 TP=1 replicas、每题 4
samples、`repo-react-v1-50x4096`、50 turns、每次 assistant 请求 4096
tokens、seed `20260629`、并发 8（失败降到 4）、context 262144。

`scripts/math/run_trace2skill.sh` 默认 seed `20260627`、8 workers、
`gpt-5.4-nano`、20 行 skill；与 Agentic-ESOpt 的组合统一使用
`scripts/es_skill_workflow.sh math <distill-skill|skill-eval>`。

## DocVQA

标准 Agentic-ESOpt/skill 入口：
`scripts/es_skill_workflow.sh docvqa <es-train|eval|distill-skill|skill-eval>`。

| Group | Effective defaults |
| --- | --- |
| ES | 40 generations, population 16, case batch 16, sigma `1e-3 → 5e-4` cosine, alpha `5e-4`, full parameters, z-score (`ddof=0`, epsilon `1e-8`), seed `20260627` |
| Rollout | 1 train sample, 50 turns, 512 tokens/turn, 32768 total tokens, `T=1`, `p=1`, `k=40`, presence penalty 2, repetition penalty 1 |
| Runtime | 4 vLLM engines, inference batch 16, context 131072, GPU memory utilization 0.85, bfloat16, eager mode |
| Eval | every 10 generations; 1 sample during training; 4 final raw/skill samples; 100 held-out documents |
| Trace2Skill | final 50 task occurrences, at most one success + one failure per task, 32 workers, 80 skill lines, evolution temperature 1 |

异步 VERL GRPO 训练入口：`scripts/docvqa/run_react_verl_grpo.sh train`
（等价于 `scripts/docvqa/run_grpo.sh`）。

| Group | Effective defaults |
| --- | --- |
| Optimizer | GRPO, LR `1e-6`, mini-batch 4, micro-batch 1/GPU, KL loss `0.001`, low-variance KL |
| Data/run | 50 documents, train batch 4, 8 rollouts/prompt, 15 epochs, seed 42, 4 GPUs |
| Rollout | async SGLang, TP=1, 50 user + 50 assistant turns, 512 tokens/turn, 32768 response tokens, 6000 tool tokens |
| Sampling | `T=1`, `p=1`, `k=40`, presence penalty 2, repetition penalty 1 |
| Runtime | context 131072, GPU memory utilization 0.50, max 64 sequences, 4 agent-loop workers |
| Eval/checkpoint | no validation before/during training; save every step; protect steps 60/120/180 |

`scripts/docvqa/run_react_verl_grpo.sh eval` 默认 held-out 前 100 题、每题 4
samples、4 个 TP=1 replicas、seed 42、context 131072、并发 8（失败降到 4）、
GPU memory utilization 0.82。

`scripts/docvqa/run_trace2skill.sh` 与 Math 的 standalone Trace2Skill 默认值
相同；与 Agentic-ESOpt 的组合统一使用
`scripts/es_skill_workflow.sh docvqa <distill-skill|skill-eval>`。

## WebArena

入口：`scripts/webarena/run.sh <method> <stage>`。

| Method | Effective launcher defaults |
| --- | --- |
| `noskill_agentic_esopt train` | 70 generations, population 8, case batch 8, 8 case workers/sample, cosine sigma `1.5e-3 → 1.5e-3`, zero warmup, alpha `2.5e-4`, full parameters, z-score, seed `20260605`, eval every 10 generations |
| `trace2skill_no-finetune distill` | fixed base weights; 70 skill steps, 8 tasks/step, 8 rollouts/task, at most one positive + one usable negative per task, skill update every step, eval every 10 steps, 32 rollout workers, 16 analysis workers, empty initial skill, seed `20260605` |
| `trace2skill_agentic_esopt distill` | all completed generations and all trajectories from the NoSkill ES run, HTML limit 12000, empty initial skill, committed WebArena success/error prompts, `gpt-5.4-nano`, 16 analysis workers, medium analysis/skill/consolidation effort, seed `20260721`, unlimited skill lines/tokens and zero references |
| `trace2skill_agentic_esopt test` | 重放同一份 NoSkill Agentic-ESOpt history，只在最终评测时注入蒸馏后的 `SKILL.md`；蒸馏后不再执行 ES update |
| all `test` stages | reset/init clean replicas, replay zero or all selected ES updates, 3 repeats over all 165 tasks, 8 workers/replica, 30 turns, 2048 tokens/turn; 40 fuzzy tasks use `gpt-4.1-mini` judge at temperature 0, evaluator failures abort the incomplete repeat and never become score 0 |

训练和最终评测统一使用 `T=0.7`、top-p `0.8`、top-k `20`、min-p
`0.0`、presence penalty `1.5`、repetition penalty `1.0`。训练使用从原始
812 题中按 Lite `old_task_id` 排除 165 题后生成的 582 题 non-Lite split；
另外 65 题用于 Trace2Skill-No-Finetune validation；最终评测使用完整 165 题
WebArena-Lite。
`reddit,gitlab,wikipedia,map,shopping,shopping_admin` 六个 site
都启用。最终噪声虽然用统一的 cosine 起点/终点接口表达，但起点与终点都为
`1.5e-3`，所以整个训练过程数值恒定。

## AHD

正式入口是 `scripts/ahd/run_ahd_1000.sh` 和
`scripts/ahd/run_ahd_2000.sh`；二者默认都跑 6 tasks × 3 repeats。

- EoH：population 10、25 generations；1000 budget 使用 `k=1`，2000
  budget 使用 `k=3`，也就是每代的 `m1/m2` 分别各跑一次或三次。
- Sample：batch 20；1000/2000 budget 分别为 50/100 generations。
- Agentic-ESOpt：operators `m1,m2`，directions 10，sigma `1e-3 → 0`
  cosine，alpha `5e-4`，seed 2024。
