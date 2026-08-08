# Archived from /home/zhi/Dynamic-Agent/cache/active_runs/construct_tsp_train_eoh_construct_tsp_train_eoh_rep3_orig_eoh_all6_k3_8gpu_20260713_142341/results/pops_best/population_generation_25.json
# run_id: construct_tsp_train_eoh_rep3_orig_eoh_all6_k3_8gpu_20260713_142341
# train_objective: 6.56232
# method: original EoH, population=10, generations=25, k=3 replicates

import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Ensure unvisited_nodes is a numpy array
    unvisited_nodes = np.array(unvisited_nodes)
    
    # Calculate the distances from the current node to all unvisited nodes
    distances_to_curr = distance_matrix[current_node, unvisited_nodes]
    
    # Calculate the distances from the unvisited nodes to the destination node
    distances_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Calculate the variance of distances from the current node to all unvisited nodes
    variances = np.var(distances_to_curr)
    
    # Calculate the relative importance of each unvisited node to the current node
    rel_importance = 1 / (1 + distances_to_curr)
    
    # Calculate the relative importance to the destination node
    rel_dest_importance = distances_to_dest / (1 + distances_to_dest)
    
    # Calculate the score as the product of variance and the sum of relative importance and relative destination importance
    score = variances * (rel_importance + rel_dest_importance)
    
    # Select the node with the maximum score
    next_node = unvisited_nodes[np.argmax(score)]
    
    return next_node
