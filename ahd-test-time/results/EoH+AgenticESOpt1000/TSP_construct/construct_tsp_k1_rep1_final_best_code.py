# Selected 2026-07-29 as a mean-nearby Dynamic-EoH-k1 representative
# canonical_slot: 1
# candidate_id: dynamic_construct_tsp_round3_rep1
# source: /home/zhi/Dynamic-Agent/cache/active_runs/construct_tsp_train_es_sigma0.001_alpha0.0005_construct_tsp_train_es_full_reload_sigma1e-3_alpha5e-4_rep1_dynamic_k1_construct_tsp_3rep_round3_20260727_015750/results/pops_best/population_generation_25.json
# train_objective: 6.5107
# standardized_rms_distance_to_dynamic_mean: 0.11712845535584163
# test_N=20: 4.21249113244207
# test_N=50: 6.4581593740754775
# dynamic_mean_N=20: 4.216999687809493
# dynamic_mean_N=50: 6.463097960386238

# source: /home/zhi/Dynamic-Agent/cache/active_runs/construct_tsp_train_es_sigma0.001_alpha0.0005_construct_tsp_train_es_full_reload_sigma1e-3_alpha5e-4_rep1_dynamic_k1_construct_tsp_3rep_round3_20260727_015750/results/pops_best/population_generation_25.json
# train_objective: 6.5107
# method: Dynamic; round: 3; original_rep: 1

import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Remove the current node from the unvisited nodes
    unvisited_nodes = np.setdiff1d(unvisited_nodes, current_node)

    # Calculate the distances from the current node to unvisited nodes
    current_distances = distance_matrix[current_node]

    # Calculate the distances from the destination node to unvisited nodes
    destination_distances = distance_matrix[destination_node]

    # Reflect the destination distances on the current distances
    reflected_distances = current_distances + (current_distances - destination_distances)

    # Select the node that minimizes the reflection error
    next_node = unvisited_nodes[np.argmin(reflected_distances[unvisited_nodes])]

    return next_node
