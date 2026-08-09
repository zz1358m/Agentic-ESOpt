# source: /home/zhi/Agentic-ESOpt/cache/active_runs/aco_cvrp_train_sample_es_pop20_gen50_sigma0.001_alpha0.0005_aco_cvrp_sample_es_current_cosine_t1000_rep2_queue_b_gpu4_7_20260720_030717/results/pops_best/population_generation_50.json
# method: sample_es, invalid_reward=current, sigma_schedule=cosine
# population=20, generations=50, samples=1000, sigma=0.001->0, alpha=0.0005
# task: aco_cvrp, rep: 2
# train_objective: 9.73154

import numpy as np

def heuristics_v2(distance_matrix, coordinates, demands, capacity):
    n = len(coordinates)
    
    # Calculate the maximum travel distance from any node to any other node (using triangle inequality)
    for i in range(n):
        for j in range(i+1, n):
            distance_matrix[i, j] = distance_matrix[j, i] = min(distance_matrix[i, j], 
                                                               distance_matrix[i, 0] + distance_matrix[0, j], 
                                                               distance_matrix[j, 0] + distance_matrix[0, i])
    
    # Calculate the Euclidean distance between nodes
    euclidean_distances = np.array([[np.linalg.norm(coordinates[i] - coordinates[j]) if i!= j else 0 for j in range(n)] for i in range(n)])
    distance_matrix = euclidean_distances + distance_matrix
    
    # Calculate the probability of a new route including the edge (i, j)
    def probability(edge_value, edge_distance):
        if edge_distance == 0:
            return 0
        if edge_value == 1:
            return 1
        if demands[i] + demands[j] <= capacity:
            return (demands[i] + demands[j]) / (capacity * (edge_distance / 200)**2)
        else:
            return 0.001
    
    # Initialize the heuristics matrix
    heuristics_matrix = np.zeros((n, n))
    
    # Compute the heuristics matrix
    for i in range(1, n):
        for j in range(i, n):
            if i!= j:
                heuristics_matrix[i, j] = heuristics_matrix[j, i] = probability(distance_matrix[i, j], distance_matrix[i, j])
            
    return heuristics_matrix
