# source: /home/zhi/Agentic-ESOpt/cache/active_runs/construct_tsp_train_sample_t2000_construct_tsp_sample_t2000_from_rep2_20260718_145101/results/pops_best/population_generation_100.json
# method: sample, prefix=2000, batch_size=20
# task: construct_tsp, rep: 2
# train_objective: 6.66389

import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Initialize the current minimum distance to the destination
    min_dist_to_dst = np.inf
    # Initialize the index of the next node
    next_node_idx = 0
    # Initialize the weight of the best node
    best_node_weight = 0

    for i, node in enumerate(unvisited_nodes):
        # Calculate the distance to the destination
        dist_to_dst = distance_matrix[node, destination_node]
        # Calculate the distance to the current node
        dist_to_curr = distance_matrix[node, current_node]
        
        # Calculate the weight of the current node
        weight = dist_to_dst / (dist_to_dst + dist_to_curr)
        
        # If the weight is better than the best node, update the best node
        if weight > best_node_weight:
            best_node_weight = weight
            next_node_idx = i

    # Get the ID of the next node
    next_node = unvisited_nodes[next_node_idx]
    
    return next_node
