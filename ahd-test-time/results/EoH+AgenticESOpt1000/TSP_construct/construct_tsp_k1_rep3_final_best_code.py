# Selected 2026-07-29 as a mean-nearby EoH+Agentic-ESOpt-k1 representative
# canonical_slot: 3
# candidate_id: agentic_esopt_construct_tsp_round3_rep2
# source: /home/zhi/Agentic-ESOpt/cache/active_runs/construct_tsp_train_es_sigma0.001_alpha0.0005_construct_tsp_train_es_full_reload_sigma1e-3_alpha5e-4_rep2_agentic_esopt_k1_construct_tsp_3rep_round3_20260727_015750/results/pops_best/population_generation_25.json
# train_objective: 6.42779
# standardized_rms_distance_to_agentic_esopt_mean: 0.7561096068268649
# test_N=20: 4.1953646925810455
# test_N=50: 6.41836241184237
# agentic_esopt_mean_N=20: 4.216999687809493
# agentic_esopt_mean_N=50: 6.463097960386238

# source: /home/zhi/Agentic-ESOpt/cache/active_runs/construct_tsp_train_es_sigma0.001_alpha0.0005_construct_tsp_train_es_full_reload_sigma1e-3_alpha5e-4_rep2_agentic_esopt_k1_construct_tsp_3rep_round3_20260727_015750/results/pops_best/population_generation_25.json
# train_objective: 6.42779
# method: Agentic-ESOpt; round: 3; original_rep: 2

import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node based on prioritized neighborhood expansion and 
    weighted least squares optimization.
    
    Parameters:
    current_node (int): The current node's ID.
    destination_node (int): The destination node's ID.
    unvisited_nodes (array): An array of unvisited nodes' IDs.
    distance_matrix (array): The distance matrix of nodes.
    
    Returns:
    next_node (int): The ID of the selected next node.
    """

    # Initialize the weights for the weighted least squares optimization
    weights = np.linspace(0, 1, len(unvisited_nodes))  # assign decreasing weights to the nodes
    
    # Initialize the minimum score and the next node
    min_diff = np.inf
    next_node = -1

    # Iterate over all unvisited nodes
    for node in unvisited_nodes:
        # Calculate the distances to the current node
        distance_to_current = distance_matrix[current_node, node]

        # Calculate the distances to the remaining unvisited nodes
        distance_to_unvisited = np.exp(-(distance_matrix[node, unvisited_nodes[unvisited_nodes!= node]] / np.max(distance_matrix)))  # use exponential function to penalize large distances
        distance_to_remaining_unvisited = np.mean(distance_to_unvisited)

        # Calculate the score of the node
        # Use weighted least squares optimization to minimize the weighted sum of the distance to the current node and the distance to the remaining unvisited nodes
        score = np.sum(weights * (distance_to_current + distance_to_remaining_unvisited))  # add weights
        
        # Update the minimum difference and the next node if the score is smaller
        if score < min_diff:
            min_diff = score
            next_node = node
            
    return next_node
