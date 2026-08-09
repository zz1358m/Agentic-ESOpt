# Archived from /home/zhi/Agentic-ESOpt/cache/active_runs/construct_tsp_train_eoh_construct_tsp_train_eoh_rep2_orig_eoh_all6_k3_8gpu_20260713_142341/results/pops_best/population_generation_25.json
# run_id: construct_tsp_train_eoh_rep2_orig_eoh_all6_k3_8gpu_20260713_142341
# train_objective: 6.44101
# method: original EoH, population=10, generations=25, k=3 replicates

import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Calculate distances from current node to all unvisited nodes
    distances_to_unvisited = distance_matrix[current_node, unvisited_nodes]
    
    # Initialize score values
    scores = np.zeros(len(unvisited_nodes))
    
    # Calculate the barycenter (centroid) of each unvisited node's distances to the other unvisited nodes
    centroids = np.mean(distance_matrix[unvisited_nodes[:, None], unvisited_nodes], axis=1)
    
    # Calculate score for each unvisited node, which is the distance to the current node minus the distance to the centroid
    scores = distances_to_unvisited - centroids
    
    # Select the node with the minimum negative score (i.e., closest centroid)
    next_node = unvisited_nodes[np.argmin(scores)]
    
    return next_node
