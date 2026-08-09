# Agentic-ESOpt 🚀

[English](README.md)

Agentic-ESOpt 使用带随机种子、可重放的进化策略，根据智能体 rollout
奖励优化语言模型权重。本仓库包含共享优化器、任务运行脚本、对比基线、
Trace2Skill 流程，以及五类智能体任务中保留的实验日志。

## 包含内容 🧭

| 任务 | 维护中的流程 |
| --- | --- |
| Sudoku | Agentic-ESOpt、多轮 GRPO |
| Math | Agentic-ESOpt、多轮 GRPO、Trace2Skill、Trace2Skill + Agentic-ESOpt |
| DocVQA | Agentic-ESOpt、多轮 GRPO、Trace2Skill、Trace2Skill + Agentic-ESOpt |
| WebArena | Agentic-ESOpt、Trace2Skill、Trace2Skill + Agentic-ESOpt |
| AHD | EoH、独立采样及其 Agentic-ESOpt 版本 |

共享实现位于 [`algorithms/es/`](algorithms/es/)。每个维护中的 Agentic-ESOpt 运行都会把带种子的
扰动、奖励、调度和更新记录到 `history.json`，可以在新启动的模型服务上重放。

## 仓库结构 📦

```text
algorithms/                 优化算法与训练集成
  es/                       Agentic-ESOpt 共享实现
  trace2skill-settings/     prompt、配置、脚本和 skill
  verl/                     GRPO 使用的内置 VERL 源码
  verl_trace2skill/         多轮工具、解析器、奖励和测试
scripts/                    用户入口脚本和数据检查
sudoku-train-time/          Sudoku 环境、运行脚本和日志
math-train-time/            Math 环境、运行脚本和日志
docvqa-train-time/          DocVQA 环境、运行脚本和日志
webarena-train-time/        WebArena 环境、集成和日志
ahd-test-time/              EoH/AHD 运行时、评测器和程序
data/                       稳定的数据约定和小型源数据
```

简要入口映射见 [`PROJECT_LAYOUT.md`](PROJECT_LAYOUT.md)，任务的完整参数见各任务
目录下的 README。

## 环境要求 🧰

- 训练和模型服务需要 Linux 与 NVIDIA GPU。
- Python `>=3.10`，推荐 Python 3.10 或 3.11。
- CUDA 12.x，以及与 CUDA 匹配的 PyTorch。
- DocVQA 的 bash/OCR 环境需要 `bubblewrap` 和 `tesseract-ocr`。
- 本仓库不提交本地模型权重和完整任务数据集。

下面是已经验证可运行的版本组合，并非唯一支持的版本：

| 流程 | Python | PyTorch | Transformers | vLLM / VERL |
| --- | --- | --- | --- | --- |
| Qwen3.5 Agentic-ESOpt 与评测 | 3.10 | 2.10.0 | 4.57.6 | vLLM 0.19.1 |
| 多轮 GRPO | 3.11 | 2.6.0 | 4.51.1 | 仓库内置 VERL |
| CPU 测试与 AHD 工具 | >=3.10 | 可选 | 兼容的近期版本 | 不需要 |

建议为 Qwen3.5 推理和 GRPO 分别建立环境：内置 VERL 与新版 Qwen3.5 vLLM
的依赖约束不同。先安装与本机 CUDA 匹配的 PyTorch，再安装其余依赖。

```bash
# Agentic-ESOpt / 评测环境
python3.10 -m venv .venv-es
source .venv-es/bin/activate
python -m pip install --upgrade pip
python -m pip install transformers==4.57.6 vllm==0.19.1 \
  accelerate datasets pillow pandas pyarrow math-verify
python -m pip install -e 'ahd-test-time/methods/eoh/original/eoh[all]'

# GRPO 环境（另建并激活一个 Python 3.11 环境）
python -m pip install torch==2.6.0 transformers==4.51.1
python -m pip install -e ./algorithms/verl
```

CUDA 对应的 VERL 镜像及可选后端见
[`algorithms/verl/docker/`](algorithms/verl/docker/) 和
[`algorithms/verl/README.md`](algorithms/verl/README.md)。

## 快速开始 ⚡

```bash
git clone https://github.com/zz1358m/Agentic-ESOpt.git
cd Agentic-ESOpt
cp scripts/settings.example.env scripts/settings.local.env
```

在 `scripts/settings.local.env` 中填写模型路径、GPU、端口和服务地址。按照
[`data/README.md`](data/README.md) 准备数据，并在启动实验前检查数据约定：

```bash
python scripts/check_data.py
python scripts/check_data.py --task math --strict
```

生成的 checkpoint 和普通运行目录默认不会进入 git。每次实验请使用唯一的
`RUN_ID`，并随运行目录保存完整命令和本机配置。

## 复现实验流程 🧪

### Sudoku

缺少默认 controlled-mask 数据时，ES 入口会自动生成。两个入口都会读取
`scripts/settings.local.env`。

```bash
SUDOKU_TARGET_MASK_COUNT=15 RUN_ID=sudoku_es_m15 \
  scripts/sudoku/run_es.sh

SUDOKU_TARGET_MASK_COUNT=15 SUDOKU_GRPO_MODEL=/path/to/Qwen3.5-4B \
  scripts/sudoku/run_grpo.sh
```

两套 GRPO 采样参数以及 ES/GRPO 超参数见
[`sudoku-train-time/README.md`](sudoku-train-time/README.md)。

### Math 与 DocVQA

两个任务使用相同的四阶段标准流程：

```bash
scripts/es_skill_workflow.sh math es-train
scripts/es_skill_workflow.sh math eval
scripts/es_skill_workflow.sh math distill-skill
scripts/es_skill_workflow.sh math skill-eval

# DocVQA 流程把 math 替换为 docvqa。
```

在 `scripts/settings.local.env` 中设置 `MODEL_PATH`、`TRAIN_RUN_ID`、数据路径和
GPU 参数。完整变量见
[`scripts/README_ES_SKILL_WORKFLOW.md`](scripts/README_ES_SKILL_WORKFLOW.md)。

React-VERL 封装脚本提供合并后的 GRPO 训练和四副本评测入口：

```bash
scripts/math/run_react_verl_grpo.sh train
MATH_GRPO_EVAL_MODEL_PATH=/path/to/hf_checkpoint \
  scripts/math/run_react_verl_grpo.sh eval

scripts/docvqa/run_react_verl_grpo.sh train
DOCVQA_GRPO_EVAL_MODEL_PATH=/path/to/hf_checkpoint \
  scripts/docvqa/run_react_verl_grpo.sh eval
```

训练 checkpoint、原始 trajectory、验证 trajectory 和日志统一写入
`runs/multiturn_grpo/`。Math 评测默认使用四样本
`repo-react-v1-50x4096` profile。通过 `*_PHYSICAL_GPU_IDS`、`*_EVAL_OUT`
和 `*_EVAL_SAMPLES` 可以适配其他服务器布局。

若要运行固定 DAPO-400 Math 守护流程，并依次续跑初始评测、训练、训练后评测、
trajectory 导出和报告，可执行：

```bash
python scripts/math/run_experiment_until_complete.py
```

### WebArena

先准备配置好的训练集和 held-out 评测集：

```bash
python webarena-train-time/scripts/prepare_webarena_nonlite_split.py
python webarena-train-time/scripts/prepare_vab_webarena_lite_split.py
```

然后选择方法和阶段：

```bash
scripts/webarena/run.sh trace2skill train
scripts/webarena/run.sh trace2skill test
scripts/webarena/run.sh no_skill_es train
scripts/webarena/run.sh no_skill_es test
scripts/webarena/run.sh trace2skill_es train
scripts/webarena/run.sh trace2skill_es test
```

WebArena 所需的外部代码和服务见 [`data/README.md`](data/README.md) 与
[`webarena-train-time/README.md`](webarena-train-time/README.md)。

### AHD

安装 EoH 运行时、启动模型服务，然后运行单个任务：

```bash
MODEL=/path/to/Llama-3.1-8B-Instruct \
  scripts/ahd/start_llama31_8b_servers.sh

EOH_K=1 AHD_POP_SIZE=10 AHD_GENERATIONS=25 \
  scripts/ahd/run.sh construct_tsp train eoh

EOH_K=1 AHD_POP_SIZE=10 AHD_GENERATIONS=25 \
  scripts/ahd/run.sh construct_tsp train es
```

六个任务、采样流程、预算和续跑参数见
[`ahd-test-time/README.md`](ahd-test-time/README.md)。

## 续跑与输出 ♻️

Agentic-ESOpt 会原子写入 `history.json`。使用相应的 `*_RESUME_HISTORY`
环境变量，可以在继续训练前重放已经完成的更新。例如：

```bash
SUDOKU_ES_RESUME_HISTORY=/path/to/history.json \
RUN_ID=sudoku_resumed scripts/sudoku/run_es.sh
```

新运行写入 `runs/` 或 `cache/active_runs/`。筛选保留的日志和评测文件位于各任务
的 `*-train-time/results/`。这些目录只保存实验依据，不重复维护结果汇总表；
核对结果时请同时查看任务 README 与原始日志。

## 检查 ✅

下面的快速检查不需要启动模型服务：

```bash
python -m unittest es.test_run_state -v
python -m unittest es.test_seeded_model_es -v
python -m unittest discover math-train-time/tests -v
python -m unittest discover docvqa-train-time/tests -v
python -m unittest algorithms.verl_trace2skill.test_reward -v
python scripts/check_data.py
```

## 许可证

见 [`LICENSE`](LICENSE)。
