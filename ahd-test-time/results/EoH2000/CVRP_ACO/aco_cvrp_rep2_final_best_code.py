# Archived from /home/zhi/Agentic-ESOpt/cache/active_runs/aco_cvrp_train_eoh_aco_cvrp_train_eoh_rep2_unfinished_true8_cvrp_bpp_20260715_053235/results/pops_best/population_generation_25.json
# run_id: aco_cvrp_train_eoh_rep2_unfinished_true8_cvrp_bpp_20260715_053235
# train_objective: 9.20741
# method: original EoH, population=10, generations=25, k=3 replicates

import numpy as np

def heuristics_v3(distance_matrix, coordinates, demands, capacity):
    n = len(coordinates)
    heuristics_matrix = np.zeros((n, n))

    # Calculate the Euclidean distances
    distances = np.linalg.norm(coordinates[:, None] - coordinates, axis=2)

    # Calculate edge scores based on a weighted power combination of demand, distance, and capacity
    for i in range(n):
        for j in range(n):
            if i!= j:
                score = (1 / np.power(np.sqrt(distances[i, j]), 2)) * 0.5 + np.power(np.log(1 + 1/demands[j]), 5) * 0.2 + np.power(demands[j] / capacity, 3) * 0.3
                if (demands[j]!= 0):
                    heuristics_matrix[i, j] = score / (max(distances[i, j], 1e-9) * (1 + 1/demands[j]))
                else:
                    heuristics_matrix[i, j] = 0

    # Remove self-loops and diagonal elements
    np.fill_diagonal(heuristics_matrix, 0)

    return heuristics_matrix
