# source: /home/zhi/Dynamic-Agent/cache/active_runs/aco_cvrp_train_eoh_aco_cvrp_train_eoh_rep3_eoh_cvrp_bpp_esaligned_20260609_173550/results/pops_best/population_generation_25.json
# objective: 9.13332

import numpy as np

def heuristics_v2(distance_matrix, coordinates, demands, capacity):
    n = len(demands)
    demands_with_depot = np.append(demands, 0)  # add depot demand as 0
    coordinates_with_depot = np.vstack((coordinates, np.array([0, 0])))
    
    def euclidean_distance(node1, node2):
        return np.linalg.norm(coordinates_with_depot[node1] - coordinates_with_depot[node2])
    
    # Calculate prior indicators (probabilities) for edges
    heuristics_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i!= j:  # do not include diagonal elements
                heuristics_matrix[i, j] = (demands[j] / (demands[j] + 1)) * (euclidean_distance(i, j) ** -1) * (euclidean_distance(i, j) ** -2) / (distance_matrix[i, j] ** 1)
    
    # Add an indicator for edges with the depot node
    for i in range(n):
        heuristics_matrix[i, i] = 0  # setting to 0, assuming no importance for staying at depot
    
    return heuristics_matrix
