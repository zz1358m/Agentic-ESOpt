# source: /home/zhi/Agentic-ESOpt/cache/active_runs/aco_cvrp_train_eoh_aco_cvrp_train_eoh_rep1_eoh_cvrp_bpp_rerun_eva30_20260610_1050/results/pops_best/population_generation_25.json

import numpy as np

def heuristics_v3(distance_matrix, coordinates, demands, capacity):
    n = len(coordinates)
    heuristics_matrix = np.ones((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j or i == 0 or j == 0:
                heuristics_matrix[i, j] = 0
            else:
                # Get neighboring edges
                neighbors = np.delete(distance_matrix[i], i)
                # Calculate the edge weight and the edge gradient
                edge_weight = distance_matrix[i, j]
                edge_gradient = 2 * np.log(len(neighbors))
                # Calculate the average neighboring edge weight
                avg_neighbor_weight = np.mean(neighbors)
                # Calculate the demand-to-capacity ratio and its inverse
                demand_ratio = demands[j] / capacity
                # Linearly interpolate between demand-to-capacity ratio and average neighboring edge weight
                safety_factor = demand_ratio + (1 - demand_ratio) * (avg_neighbor_weight / edge_weight)
                # Ensure safety factor is within valid range
                safety_factor = np.clip(safety_factor, 0, 1)
                # Calculate the fitness score as a power-law decay of the safety factor and an exponential increase of the edge weight
                fitness = 1 / (edge_weight ** edge_gradient) * np.exp(-safety_factor)
                heuristics_matrix[i, j] = fitness
    
    return heuristics_matrix
