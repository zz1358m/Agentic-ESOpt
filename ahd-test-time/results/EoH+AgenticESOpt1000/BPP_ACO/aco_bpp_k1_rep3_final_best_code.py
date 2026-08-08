# Archived from /home/zhi/Dynamic-Agent/cache/active_runs/aco_bpp_train_es_sigma0.001_alpha0.0005_aco_bpp_train_es_full_reload_sigma1e-3_alpha5e-4_rep3_k1_decay_full_all6_3rep_reload8_20260706_153046/results/pops_best/population_generation_25.json
# run_id: aco_bpp_train_es_full_reload_sigma1e-3_alpha5e-4_rep3_k1_decay_full_all6_3rep_reload8_20260706_153046
# train_objective: 202.6
# m1m2_multiplier: 1.0
# sigma_schedule: cosine
# final_model_es_sigma: None

import numpy as np

def heuristics_v2(demand, capacity):
    n = len(demand)
    heuristics_matrix = np.zeros((n, n))
    forward_densities = np.zeros((n, n))
    selected = [False] * n
    
    for i in range(n):
        max_density = 0
        max_index = 0
        for j in range(i):
            if not selected[j]:
                density = demand[j] / (capacity - demand[j])
                if density > max_density:
                    max_density = density
                    max_index = j
        selected[max_index] = True
        for j in range(i+1, n):
            if not selected[j]:
                complementary_density = (demand[max_index] + demand[j]) / capacity
                if heuristics_matrix[max_index, j] == 0:
                    heuristics_matrix[max_index, j] = complementary_density
                elif heuristics_matrix[max_index, j] < complementary_density:
                    heuristics_matrix[max_index, j] = complementary_density
                if heuristics_matrix[j, max_index] == 0:
                    heuristics_matrix[j, max_index] = complementary_density
                elif heuristics_matrix[j, max_index] < complementary_density:
                    heuristics_matrix[j, max_index] = complementary_density
        heuristics_matrix[i, i] = 1
    
    return heuristics_matrix
