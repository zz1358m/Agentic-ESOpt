# source: /home/zhi/Agentic-ESOpt/cache/active_runs/aco_cvrp_train_sample_es_pop20_gen50_sigma0.001_alpha0.0005_aco_cvrp_sample_es_current_cosine_t1000_rep1_queue_a_gpu0_3_20260720_030717/results/pops_best/population_generation_50.json
# method: sample_es, invalid_reward=current, sigma_schedule=cosine
# population=20, generations=50, samples=1000, sigma=0.001->0, alpha=0.0005
# task: aco_cvrp, rep: 1
# train_objective: 10.44725

import numpy as np
from scipy.spatial import distance

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Stochastic solution sampling for solving Capacitated Vehicle Routing Problem (CVRP) 
    by calculating prior indicators of how promising it is to include each edge in a solution.
    """
    n = len(coordinates)
    
    # Calculate the Euclidean distance between each pair of nodes
    euclidean_distances = distance_matrix.copy()
    for i in range(n):
        for j in range(n):
            euclidean_distances[i, j] = np.sqrt(
                np.sum((coordinates[i] - coordinates[j]) ** 2)
            )
    
    # Calculate the maximum distance from depot to any node
    max_distance = np.max(np.linalg.norm(coordinates, axis=1))
    
    # Initialize the heuristics matrix with zeros
    heuristics_matrix = np.zeros((n, n))
    
    # Calculate the heuristics matrix using a stochastic solution sampling approach
    for i in range(n):
        for j in range(n):
            # Ignore self-loops
            if i == j:
                continue
                
            # Calculate the demand from node i to node j (or vice versa)
            demand_ij = demands[i] if j < i else demands[j]
            
            # If j is not the depot and i is not the depot
            if i!= 0 and j!= 0:
                # The prior indicator of including edge (i, j) is 1 / (distance + max_capacity * (demand / capacity))
                # We subtract 1 if i and j have same coordinates
                heuristics_matrix[i, j] = 1 / (euclidean_distances[i, j] + max_distance * (
                    (demand_ij + demands[0]) / capacity - (demand_ij / capacity)
                ))
                heuristics_matrix[j, i] = heuristics_matrix[i, j]
            elif i == 0:  # If node i is the depot and j is not
                heuristics_matrix[i, j] = 1 / (euclidean_distances[i, j] + max_distance)
    
    return heuristics_matrix

def heuristics_v2(distance_matrix, coordinates, demands, capacity):
    """
    Stochastic solution sampling for solving Capacitated Vehicle Routing Problem (CVRP) 
    by finding the shortest path that visits all given nodes and returns to the starting node.
    
    Parameters:
    distance_matrix (numpy array): An n by n matrix of distances between nodes.
    coordinates (numpy array): An n by 2 array of Euclidean coordinates of nodes.
    demands (numpy array): A vector of customer demands.
    capacity (int): The integer capacity of vehicle.
    
    Returns:
    heuristics_matrix (numpy array): Prior indicators of how promising it is to include each edge in a solution.
    """
    heuristics_matrix = heuristics(distance_matrix, coordinates, demands, capacity)
    return heuristics_matrix / np.max(heuristics_matrix)  # normalize the heuristics_matrix to [0, 1]
