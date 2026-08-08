# source: /home/zhi/Dynamic-Agent/cache/active_runs/aco_tsp_train_es_sigma0.001_alpha0.0005_aco_tsp_train_es_full_reload_sigma1e-3_alpha5e-4_rep1_rerun_dynamic_k3_kp_tspaco_3rep_8gpu_20260716_120821/results/pops_best/population_generation_25.json
# run_id: aco_tsp_train_es_full_reload_sigma1e-3_alpha5e-4_rep1_rerun_dynamic_k3_kp_tspaco_3rep_8gpu_20260716_120821
# train_objective: 5.84502
# m1m2_multiplier: 3.0

import numpy as np

def heuristics_v3(distance_matrix, non_zero_distances_mean=0.5):
    # Get the number of cities
    n = len(distance_matrix)

    # Calculate a matrix that is the inverse of the distance
    heuristics_matrix = np.copy(distance_matrix)
    heuristics_matrix = np.where(heuristics_matrix == 0, 1e10, heuristics_matrix)

    # Generate the prior matrix based on the inverse of the distance and other terms
    for i in range(n):
        for j in range(n):
            if i!= j:
                heuristics_matrix[i, j] = (1 / heuristics_matrix[i, j] ** 4) - 1 / heuristics_matrix[i, j]

    # Ensure the diagonal of the prior matrix is 0
    np.fill_diagonal(heuristics_matrix, 0)

    # Normalize each row by the sum of the terms for non-zero distances with a threshold
    for i in range(n):
        non_zero_terms = [term for j, term in enumerate(heuristics_matrix[i]) if term > 1e-5]
        if non_zero_terms:
            heuristics_matrix[i] = heuristics_matrix[i] / np.mean(non_zero_terms) * non_zero_distances_mean

    return heuristics_matrix
