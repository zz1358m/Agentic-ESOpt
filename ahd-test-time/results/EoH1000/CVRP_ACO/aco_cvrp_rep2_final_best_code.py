# source: /home/zhi/Dynamic-Agent/cache/active_runs/aco_cvrp_train_eoh_aco_cvrp_train_eoh_rep2_eoh_cvrp_bpp_rerun_eva30_20260610_1050/results/pops_best/population_generation_25.json

import numpy as np
import random

def heuristics_v2(distance_matrix, coordinates, demands, capacity):
    n = distance_matrix.shape[0]
    heuristics = np.zeros((n, n))

    for i in range(1, n):
        for j in range(i + 1, n):
            edge_demand = min(demands[i], demands[j])
            heuristics[i, j] = demands[i] * demands[j] / (distance_matrix[i, j] ** 2) / (capacity ** 2) / (distance_matrix[i, j] ** 2)
            heuristics[j, i] = heuristics[i, j]

        for j in range(n):
            heuristics[i, j] = np.maximum(0, heuristics[i, j])

    # For the depot node, consider the total demand of all nodes as the demand on the edge from depot
    heuristics[0, 1:] = (np.sum(demands[1:]) / (capacity ** 2) / (distance_matrix[0, 1:] ** 2))

    return heuristics
