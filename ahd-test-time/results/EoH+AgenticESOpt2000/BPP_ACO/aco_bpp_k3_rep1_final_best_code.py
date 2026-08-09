# Archived from /home/zhi/Agentic-ESOpt/cache/active_runs/aco_bpp_train_es_sigma0.001_alpha0.0005_aco_bpp_train_es_full_reload_sigma1e-3_alpha5e-4_rep1_k3_decay_full_all6_3rep_reload8_20260705_121329/results/pops_best/population_generation_25.json
# run_id: aco_bpp_train_es_full_reload_sigma1e-3_alpha5e-4_rep1_k3_decay_full_all6_3rep_reload8_20260705_121329
# train_objective: 202.0
# m1m2_multiplier: 3.0
# sigma_schedule: cosine
# final_model_es_sigma: 0.0009957224306869053

import numpy as np

def heuristics_v3(demand, capacity):
    n = len(demand)
    heuristics = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            total_capacity = capacity
            total_demand = demand[i] + demand[j]
            if total_demand > capacity:
                heuristics[i, j] = 0
            else:
                used_capacity_i = (total_capacity - demand[i])
                used_capacity_j = (total_capacity - demand[j])
                capacity_sum = used_capacity_i + used_capacity_j
                ratio_sum = (used_capacity_i / demand[i]) + (used_capacity_j / demand[j])
                heuristics[i, j] = 1 / (1 + ratio_sum / capacity_sum)
                heuristics[j, i] = heuristics[i, j]
    np.fill_diagonal(heuristics, 0)
    return heuristics
