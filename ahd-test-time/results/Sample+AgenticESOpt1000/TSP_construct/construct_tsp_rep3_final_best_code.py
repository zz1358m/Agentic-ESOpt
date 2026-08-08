# source: /home/zhi/Dynamic-Agent/cache/active_runs/construct_tsp_train_sample_es_pop20_gen50_sigma0.001_alpha0.0005_construct_tsp_sample_es_reload_cosine_current_pop20_gen50_rep3_20260719_150222/results/pops_best/population_generation_50.json
# method: sample_es, invalid_reward=current, sigma_schedule=cosine
# population=20, generations=50, samples=1000, sigma=0.001->0, alpha=0.0005
# task: construct_tsp, rep: 3
# train_objective: 6.66389

import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Calculate distances to the destination node and from the current node to all unvisited nodes
    distances_to_destination = distance_matrix[destination_node, unvisited_nodes]
    distances_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Calculate the ratio of distances for each unvisited node
    ratios = distances_to_destination / distances_from_current
    
    # Find the node with the maximum ratio
    next_node = unvisited_nodes[np.argmax(ratios)]
    
    return next_node
