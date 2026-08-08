# Archived from /home/zhi/Dynamic-Agent/cache/active_runs/aco_tsp_train_es_sigma0.001_alpha0.0005_aco_tsp_train_es_full_reload_sigma1e-3_alpha5e-4_rep3_k1_decay_full_all6_3rep_reload8_20260706_153046/results/pops_best/population_generation_25.json
# run_id: aco_tsp_train_es_full_reload_sigma1e-3_alpha5e-4_rep3_k1_decay_full_all6_3rep_reload8_20260706_153046
# train_objective: 5.83881
# m1m2_multiplier: 1.0
# sigma_schedule: cosine
# final_model_es_sigma: 0.0006294095225512603

import numpy as np

def heuristics(distance_matrix, alpha=1.5, beta=3.0, gamma=0.0, delta=0.0):
    num_cities = len(distance_matrix)
    heuristics_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i!= j:
                heuristics_matrix[i, j] = (1 / (distance_matrix[i, j] + 1)) ** alpha * (1 + np.random.normal(delta)) + (1 / distance_matrix[i, j]) ** beta
    return heuristics_matrix / heuristics_matrix.max()

def heuristics_v3(distance_matrix):
    num_cities = len(distance_matrix)
    max_iter = 100
    heuristics_matrix = heuristics(distance_matrix, alpha=1.5, beta=2.0, gamma=0.0, delta=0.0)
    visited = [0]
    current_city = 0
    unvisited = list(range(1, num_cities))
    for _ in range(max_iter):
        next_city = max(unvisited, key=lambda x: max(heuristics_matrix[current_city][x], heuristics_matrix[x][current_city]))
        unvisited.remove(next_city)
        visited.append(next_city)
        current_city = next_city
        for j in unvisited:
            heuristics_matrix[current_city][j] = 0
            heuristics_matrix[j][current_city] = 0
        heuristics_matrix[current_city] /= (num_cities - len(visited))
    for i in range(num_cities):
        for j in range(num_cities):
            if i!= j:
                heuristics_matrix[i, j] /= (num_cities - 1)
    return heuristics_matrix
