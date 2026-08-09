# Archived from /home/zhi/Agentic-ESOpt/cache/active_runs/aco_tsp_train_eoh_aco_tsp_train_eoh_rep3_unfinished_true8_aco_tsp_rep3_20260715_053235/results/pops_best/population_generation_25.json
# run_id: aco_tsp_train_eoh_rep3_unfinished_true8_aco_tsp_rep3_20260715_053235
# train_objective: 5.81681
# method: original EoH, population=10, generations=25, k=3 replicates

import numpy as np

def heuristics_v3(distance_matrix):
    num_nodes = len(distance_matrix)
    min_distance = np.min(distance_matrix)
    max_distance = np.max(distance_matrix)
    mean_distance = np.mean(distance_matrix[distance_matrix!= 0])
    degree_bias = 1.5  # Degree bias can be adjusted
    degree_matrix = np.sum(distance_matrix, axis=1, dtype=float)
    degree_matrix[degree_matrix == 0] = 1  # Avoid division by zero
    
    heuristics_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        dists = distance_matrix[i]
        dists = dists[dists!= 0]  # Remove 0 distances
        mean_dist_i = np.mean(dists)
        std_dev = np.std(dists)
        for j in range(num_nodes):
            if i!= j:
                dist = distance_matrix[i, j]
                if dist == 0:
                    heuristics_matrix[i, j] = 0
                    heuristics_matrix[j, i] = 0
                    continue
                heuristics_matrix[i, j] = (dist ** -3) * degree_matrix[j]**2 / np.sum(degree_matrix) * (1 + (dist - mean_dist_i) / (std_dev * mean_distance))

    # Normalizing the matrix to ensure that each row sums to 1
    for i in range(num_nodes):
        heuristics_matrix[i] /= np.sum(heuristics_matrix[i])

    return heuristics_matrix
