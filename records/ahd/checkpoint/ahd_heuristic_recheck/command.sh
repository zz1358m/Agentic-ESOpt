#!/usr/bin/env bash
set -euo pipefail
cd /home/bayp/Agentic-ESOpt

# Stage 2 only: evaluate the already-curated final heuristic programs.
# This does not invoke run_ahd_*.sh, an LLM endpoint, search, or training.
python ahd-test-time/results/eval_construct_results.py \
  --tasks tsp,kp,asp \
  --tsp-sizes 50,100 \
  --kp-settings 100:25,200:25 \
  --asp-settings 12:7,15:10,21:15 \
  --max-instances 0 \
  --output records/ahd/checkpoint/ahd_heuristic_recheck/construct_eval_results.json \
  --csv-output records/ahd/checkpoint/ahd_heuristic_recheck/construct_eval_results.csv

python ahd-test-time/results/eval_aco_results.py \
  --tasks tsp,cvrp,bpp \
  --split test \
  --tsp-sizes 20,50,100 \
  --cvrp-sizes 20,50,100 \
  --bpp-sizes 500,1000 \
  --tsp-iterations 100 --tsp-ants 30 \
  --cvrp-iterations 100 --cvrp-ants 30 --cvrp-capacity 50 \
  --bpp-mode sample --bpp-sample-count 200 --bpp-capacity 150 \
  --max-instances 0 --keep-going \
  --output records/ahd/checkpoint/ahd_heuristic_recheck/aco_eval_results.json \
  --csv-output records/ahd/checkpoint/ahd_heuristic_recheck/aco_eval_results.csv
