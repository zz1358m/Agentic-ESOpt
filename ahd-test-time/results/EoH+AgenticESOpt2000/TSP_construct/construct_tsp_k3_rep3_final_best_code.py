# Archived from /home/zhi/Agentic-ESOpt/cache/active_runs/construct_tsp_train_es_sigma0.001_alpha0.0005_construct_tsp_train_es_full_reload_sigma1e-3_alpha5e-4_rep3_k3_decay_full_all6_3rep_reload8_20260705_121329/results/pops_best/population_generation_25.json
# run_id: construct_tsp_train_es_full_reload_sigma1e-3_alpha5e-4_rep3_k3_decay_full_all6_3rep_reload8_20260705_121329
# train_objective: 6.48154
# m1m2_multiplier: 3.0
# sigma_schedule: cosine
# final_model_es_sigma: 0.00019561928549563967

import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    distance_factor = 1.8  # Factor that increases with the distance from the current node to the destination node
    inverse_distance_factor = 0.8  # Factor that is inversely proportional to the distance from the current node to the next node
    exponential_factor = 2.2  # Tension factor for the exponential
    exponent = 2.0  # Power for the exponential
    unvisited_distances = distance_matrix[unvisited_nodes, current_node]
    remaining_distances = distance_matrix[unvisited_nodes, destination_node] - unvisited_distances
    exponential_scores = np.exp(exponent * exponential_factor * np.exp(remaining_distances))
    inverse_distance_scores = 1 / unvisited_distances ** inverse_distance_factor
    scores = exponential_scores * inverse_distance_scores
    next_node = unvisited_nodes[np.argmax(scores)]
    return next_node
