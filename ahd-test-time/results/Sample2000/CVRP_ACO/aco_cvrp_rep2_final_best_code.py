# source: /home/zhi/Agentic-ESOpt/cache/active_runs/aco_cvrp_train_sample_t2000_aco_cvrp_sample_t2000_from_rep2_20260718_145101/results/pops_best/population_generation_100.json
# method: sample, prefix=2000, batch_size=20
# task: aco_cvrp, rep: 2
# train_objective: 9.45441

import numpy as np
from scipy.spatial import distance

def heuristics(distance_matrix, coordinates, demands, capacity):
    # Initialize prior indicators
    n = len(coordinates)
    heuristics_matrix = np.zeros((n, n))
    
    # Calculate the demand-weighted distance for each possible edge
    for i in range(n):
        for j in range(n):
            if i!= 0 and j!= 0:
                dij = distance_matrix[i, j] / (demands[i] + demands[j])
                if dij > 0:
                    heuristics_matrix[i, j] = 1 / dij ** 2
    
    # Consider vehicle capacity
    for i in range(n):
        for j in range(n):
            if i == 0 and j!= 0:
                d0j = distance_matrix[i, j]
                djk = distance_matrix[j, 0]
                h0j = distance_matrix[i, j] + djk
                heuristics_matrix[j, 0] += 1 / (h0j ** 2 * (demands[j] + capacity))
                
    for i in range(n):
        for j in range(n):
            if j == 0 and i!= 0:
                dij = distance_matrix[i, j]
                d0i = distance_matrix[0, i]
                h0i = dij + d0i
                heuristics_matrix[0, i] += 1 / (h0i ** 2 * (demands[i] + capacity))
    
    return heuristics_matrix
