# source: /home/zhi/Dynamic-Agent/cache/active_runs/aco_cvrp_train_es_sigma0.001_alpha0.0005_aco_cvrp_train_es_m1m2_sigma1e-3_alpha5e-4_rep2_m1m2_fixed_rerun_cvrp_bpp_initfix_20260609_110120/results/pops_best/population_generation_25.json

import numpy as np

def heuristics_v2(distance_matrix, coordinates, demands, capacity):
    n = len(coordinates)
    
    # Calculate the population density based on Euclidean distances
    population_density = np.exp(-np.sqrt((coordinates[:, None] - coordinates[:, :n]) ** 2).sum(axis=-1))
    
    # Calculate the crowdedness score
    crowdedness = 1 + np.divide(demands, (capacity - demands)) * population_density
    
    # Calculate the proximity score
    proximity = np.divide(1, (distance_matrix ** 2)) * (1 + np.max(distance_matrix))
    
    # Calculate the regularity score
    regularity = np.divide(1, (distance_matrix ** 2)) * (np.max(distance_matrix) - np.sum(distance_matrix, axis=1))
    
    # Calculate the temporal penalty
    time_penalty = distance_matrix / (1 + (crowdedness * proximity) + (1 - proximity) * regularity)
    
    # Calculate the spatial relationship indicator
    spatial_indicator = 1 / (distance_matrix + np.mean(population_density))
    
    # Combine indicators to calculate the edge heuristic value
    heuristics_value = np.divide(spatial_indicator / time_penalty, (1 + distance_matrix))
    
    # Normalize the heuristic values
    denominator = time_penalty + np.max(distance_matrix)
    heuristics_matrix = np.divide(heuristics_value, denominator, out=np.zeros_like(distance_matrix), where=denominator!=0)
    
    # Ensure the first row and column are all zeros for routing starting and ending at the depot.
    heuristics_matrix[:, 0] = np.zeros(n)
    heuristics_matrix[0, :] = np.zeros(n)
    
    return heuristics_matrix
