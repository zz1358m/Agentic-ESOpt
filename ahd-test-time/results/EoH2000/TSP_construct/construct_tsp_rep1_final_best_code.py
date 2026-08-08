# Archived from /home/zhi/Dynamic-Agent/cache/active_runs/construct_tsp_train_eoh_construct_tsp_train_eoh_rep1_orig_eoh_all6_k3_8gpu_20260713_142341/results/pops_best/population_generation_25.json
# run_id: construct_tsp_train_eoh_rep1_orig_eoh_all6_k3_8gpu_20260713_142341
# train_objective: 6.49214
# method: original EoH, population=10, generations=25, k=3 replicates

import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Calculate the distances from the current node to all unvisited nodes and the destination node in reverse direction
    distances_from_current_node = distance_matrix[:, current_node]
    distances_to_destination = distance_matrix[destination_node, :]
    
    # Calculate the median distance from the current node to all unvisited nodes
    mid_index = len(unvisited_nodes) // 2
    median_distance_unvisited = np.median(distances_from_current_node[unvisited_nodes])
    
    # Calculate the median distance from the current node to the destination node and all unvisited nodes
    median_distance = np.median([median_distance_unvisited, distances_to_destination[current_node]])
    
    # Calculate the proximity penalty to the destination node
    distance_to_destination = distances_to_destination[current_node]
    proximity_penalty = 1 / (1 + np.exp(-(distance_to_destination / 10)))  # a simple exponential penalty
    
    # Calculate the score for each unvisited node
    scores = []
    for node in unvisited_nodes:
        distance_to_node = distances_from_current_node[node]
        distance_to_destination_node = distances_to_destination[node]
        score = median_distance + (distance_to_node - median_distance_unvisited) + proximity_penalty * (1 - distance_to_destination_node)
        scores.append(score)
    
    # Select the unvisited node with the minimum score as the next node
    next_node = unvisited_nodes[np.argmin(scores)]

    return next_node
