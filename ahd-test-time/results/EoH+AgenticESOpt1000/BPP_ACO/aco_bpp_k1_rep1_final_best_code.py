# Archived from /home/zhi/Agentic-ESOpt/cache/active_runs/aco_bpp_train_es_sigma0.001_alpha0.0005_aco_bpp_train_es_full_reload_sigma1e-3_alpha5e-4_rep1_k1_decay_full_all6_3rep_reload8_20260706_153046/results/pops_best/population_generation_25.json
# run_id: aco_bpp_train_es_full_reload_sigma1e-3_alpha5e-4_rep1_k1_decay_full_all6_3rep_reload8_20260706_153046
# train_objective: 202.4
# m1m2_multiplier: 1.0
# sigma_schedule: cosine
# final_model_es_sigma: None

import numpy as np

def heuristics_v2(demand, capacity):
    n = len(demand)
    heuristics = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                heuristics[i, j] = 1
            else:
                l2_distance = np.sqrt((demand[i] - demand[j])**2 + (capacity - demand[i])**2 + (capacity - demand[j])**2)
                heuristics[i, j] = 1 / l2_distance
                heuristics[j, i] = heuristics[i, j]  # Make the matrix symmetric
    rowsum = heuristics.sum(axis=1, keepdims=True)
    heuristics = heuristics / (rowsum * (rowsum - 1))  # Normalize the heuristics matrix
    return heuristics
