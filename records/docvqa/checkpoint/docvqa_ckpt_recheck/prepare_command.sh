#!/usr/bin/env bash
set -euo pipefail
cd /home/bayp/Agentic-ESOpt

# Data contract: pinned DocVQA validation revision and historical fixed-eval order.
python algorithms/trace2skill-settings/scripts/prepare_data.py \
  --setting docvqa \
  --output-dir /home/bayp/Agentic-ESOpt/data/trace2skill/docvqa \
  --seed 42 \
  --docvqa-evolve-count 50 \
  --docvqa-revision 539088ef8a8ada01ac8e2e6d4e372586748a265e \
  --docvqa-order-reference /home/bayp/Agentic-ESOpt/docvqa-train-time/results/eval/qwen35_4b_grpo_noskill/docvqa.jsonl

python scripts/check_data.py --task docvqa --strict

# Published checkpoint: direct checkpoint / zero history replay.
python -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='zz1358m/Qwen3.5-4B-DocVQA-ReAct-Agentic-ESOpt', revision='a7970c6ff80accc4c224c81fec5b502412b844e6', local_dir='/home/bayp/models/Qwen3.5-4B-DocVQA-ReAct-Agentic-ESOpt-a7970c6'))"
