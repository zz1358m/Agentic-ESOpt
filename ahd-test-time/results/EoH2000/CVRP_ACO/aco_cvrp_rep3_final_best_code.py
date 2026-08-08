# Archived from /home/zhi/Dynamic-Agent/cache/active_runs/aco_cvrp_train_eoh_aco_cvrp_train_eoh_rep3_unfinished_true8_cvrp_bpp_20260715_053235/results/pops_best/population_generation_25.json
# run_id: aco_cvrp_train_eoh_rep3_unfinished_true8_cvrp_bpp_20260715_053235
# train_objective: 9.05117
# method: original EoH, population=10, generations=25, k=3 replicates

import numpy as np

def heuristics_v3(distance_matrix, coordinates, demands, capacity):
    n = len(coordinates)
    depot_coordinates = coordinates[0]

    proximity = np.zeros((n, n))
    for i in range(1, n):
        for j in range(1, n):
            if i!= j:
                proximity[i, j] = 1 / (distance_matrix[i, j] ** 2)
                proximity[j, i] = proximity[i, j]

    remaining_capacity = np.zeros((n, n))
    for i in range(1, n):
        for j in range(1, n):
            if i!= j:
                remaining_capacity[i, j] = 1 / max(1 + demands[i] / capacity, 1) * 1 / max(1 + demands[j] / capacity, 1)
                remaining_capacity[j, i] = remaining_capacity[i, j]

    node_degree = np.sum(remaining_capacity, axis=0)

    potential_impact = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i!= j:
                potential_impact[i, j] = (demands[i] + demands[j]) * proximity[i, j] / (node_degree[i] ** 0.5)
                potential_impact[j, i] = potential_impact[i, j]

    score_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i!= j:
                score_matrix[i, j] = (potential_impact[i, j] / distance_matrix[i, j]) * (remaining_capacity[i, j] ** 1.5) * ((1 - demands[i] / capacity) ** 0.9) * ((1 - demands[j] / capacity) ** 0.1)
                score_matrix[j, i] = score_matrix[i, j]

    # Assign high priority to edges from the depot to other nodes with a temperature-dependent factor
    temperature = 100
    for i in range(1, n):
        score_matrix[0, i] = 1 / (distance_matrix[0, i] ** 1.5) * np.exp(-distance_matrix[0, i] / temperature)

    return score_matrix
