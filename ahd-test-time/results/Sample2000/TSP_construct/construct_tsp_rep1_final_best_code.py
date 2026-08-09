# source: /home/zhi/Agentic-ESOpt/cache/active_runs/construct_tsp_train_sample_t2000_construct_tsp_sample_t2000_from_rep1_20260718_145101/results/pops_best/population_generation_100.json
# method: sample, prefix=2000, batch_size=20
# task: construct_tsp, rep: 1
# train_objective: 6.66389

import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Calculate differences between distances from current node to all unvisited nodes and 
    # distances from each unvisited node to the destination node
    differences = np.zeros(len(unvisited_nodes))
    for i, node in enumerate(unvisited_nodes):
        if node == current_node:
            differences[i] = np.inf
            continue
        differences[i] = (distance_matrix[current_node, node] - 
                         distance_matrix[node, destination_node]) / distance_matrix[current_node, node]
    
    # Choose the node with the minimum difference
    next_node = unvisited_nodes[np.argmin(differences)]
    
    return next_node
