# Archived from /home/zhi/Agentic-ESOpt/cache/active_runs/aco_cvrp_train_eoh_aco_cvrp_train_eoh_rep1_unfinished_true8_cvrp_bpp_20260715_053235/results/pops_best/population_generation_25.json
# run_id: aco_cvrp_train_eoh_rep1_unfinished_true8_cvrp_bpp_20260715_053235
# train_objective: 9.12083
# method: original EoH, population=10, generations=25, k=3 replicates

import numpy as np
from scipy.spatial import distance

def heuristics_v3(distance_matrix, coordinates, demands, capacity):
    n = len(coordinates)
    heuristics_matrix = np.zeros((n, n))

    # Initialize the heuristics for the depot node
    for i in range(1, n):
        x1, y1 = coordinates[0]
        x2, y2 = coordinates[i]
        dist = distance.euclidean((x1, y1), (x2, y2))
        heuristics_matrix[0, i] = (demands[i] + 1) / (dist**3) * np.exp(-demands[i] / capacity)

    # Calculate the heuristics for edges between nodes
    for i in range(n):
        for j in range(n):
            # Skip self-loops, the case where the vehicle is at the destination node, and the case where the vehicle is at the source node
            if i == j or i == 0 or j == 0:
                continue
                
            # Node i is the source node
            x1, y1 = coordinates[i]
            # Node j is the destination node
            x2, y2 = coordinates[j]
            dist = distance.euclidean((x1, y1), (x2, y2))
            num_vehicles = 1
            if i == 0:
                num_vehicles = 0
            remaining_capacity = capacity - demands[i] * num_vehicles
            if remaining_capacity < demands[j]:
                adjustment_factor = 1 + demands[j] / remaining_capacity
            else:
                adjustment_factor = 1

            heuristics_matrix[i, j] = (demands[i] + 1) / (dist**3) * (demands[j] + 1) / (dist**3) * adjustment_factor / (capacity - demands[j])

    return heuristics_matrix
