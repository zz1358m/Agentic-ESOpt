# Selected 2026-07-29 as a mean-nearby Dynamic-EoH-k1 representative
# canonical_slot: 2
# candidate_id: dynamic_construct_tsp_round2_rep1
# source: /home/zhi/Dynamic-Agent/cache/active_runs/construct_tsp_train_es_sigma0.001_alpha0.0005_construct_tsp_train_es_full_reload_sigma1e-3_alpha5e-4_rep1_dynamic_k1_construct_tsp_3rep_round2_20260727_015750/results/pops_best/population_generation_25.json
# train_objective: 6.50444
# standardized_rms_distance_to_dynamic_mean: 0.5329529915986309
# test_N=20: 4.192534291736646
# test_N=50: 6.456528515982292
# dynamic_mean_N=20: 4.216999687809493
# dynamic_mean_N=50: 6.463097960386238

# source: /home/zhi/Dynamic-Agent/cache/active_runs/construct_tsp_train_es_sigma0.001_alpha0.0005_construct_tsp_train_es_full_reload_sigma1e-3_alpha5e-4_rep1_dynamic_k1_construct_tsp_3rep_round2_20260727_015750/results/pops_best/population_generation_25.json
# train_objective: 6.50444
# method: Dynamic; round: 2; original_rep: 1

import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Remove current node from unvisited nodes
    unvisited_nodes = np.setdiff1d(unvisited_nodes, [current_node])

    # Initialize variables to keep track of the best node
    best_node = None
    max_clustering = -np.inf

    # Calculate the distance to the node with the maximum distance to the current node
    max_dist_node = np.argmax(distance_matrix[current_node, unvisited_nodes])
    max_dist_nodes = unvisited_nodes[max_dist_node]

    # Iterate over the unvisited nodes
    for node in unvisited_nodes:
        # Calculate the distance to the current node
        dist_to_current = distance_matrix[current_node, node]
        
        # If the distance to the current node is infinity, skip this node
        if dist_to_current == np.inf:
            continue
        
        # Calculate the clustering coefficient of the node (average distance to its neighbors)
        neighbors = [n for n in unvisited_nodes if n!= node]
        dist_to_neighbors = [distance_matrix[node, neighbor] for neighbor in neighbors]
        clustering = np.mean([d for d in dist_to_neighbors if d!= np.inf])

        # Calculate the distance-weighted clustering coefficient of the node
        weight = clustering * np.sum([distance_matrix[node, max_dist_nodes] for n in neighbors if distance_matrix[node, n]!= np.inf])

        # Calculate the ratio of the weighted centrality of the node to the distance to the current node
        ratio = weight / dist_to_current
        
        # Check if this node is the best so far
        if ratio > max_clustering:
            max_clustering = ratio
            best_node = node

    return best_node
