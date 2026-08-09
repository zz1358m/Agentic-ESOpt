# source: /home/zhi/Agentic-ESOpt/cache/active_runs/construct_tsp_train_sample_es_pop20_gen50_sigma0.001_alpha0.0005_construct_tsp_sample_es_reload_cosine_current_pop20_gen50_rep1_20260719_150222/results/pops_best/population_generation_50.json
# method: sample_es, invalid_reward=current, sigma_schedule=cosine
# population=20, generations=50, samples=1000, sigma=0.001->0, alpha=0.0005
# task: construct_tsp, rep: 1
# train_objective: 6.44101

import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node in the Modified Greedy Routing algorithm.
    
    Args:
    current_node (int): The ID of the current node.
    destination_node (int): The ID of the destination node.
    unvisited_nodes (numpy.array): The IDs of the unvisited nodes.
    distance_matrix (numpy.array): The distance matrix of nodes.
    
    Returns:
    next_node (int): The ID of the next node.
    """
    # Initialize the next node and maximum average distance
    next_node = None
    max_avg_dist = float('-inf')
    
    # Calculate the distance from the current node to all unvisited nodes
    dist_to_unvisited = distance_matrix[current_node, unvisited_nodes]
    
    # For each unvisited node
    for node in unvisited_nodes:
        # Calculate the average distance from the current node to this node and all other unvisited nodes
        avg_dist = np.mean(dist_to_unvisited + distance_matrix[node, unvisited_nodes])
        
        # Add a penalty for the distance from the current node to this node
        penalty = distance_matrix[current_node, node]
        avg_dist -= penalty
        
        # Update the next node if the average distance is higher
        if avg_dist > max_avg_dist:
            max_avg_dist = avg_dist
            next_node = node
    
    return next_node
