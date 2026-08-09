# Archived from /home/zhi/Agentic-ESOpt/cache/active_runs/construct_tsp_train_es_sigma0.001_alpha0.0005_construct_tsp_train_es_full_reload_sigma1e-3_alpha5e-4_rep1_k3_decay_full_all6_3rep_reload8_20260705_121329/results/pops_best/population_generation_25.json
# run_id: construct_tsp_train_es_full_reload_sigma1e-3_alpha5e-4_rep1_k3_decay_full_all6_3rep_reload8_20260705_121329
# train_objective: 6.43195
# m1m2_multiplier: 3.0
# sigma_schedule: cosine
# final_model_es_sigma: 6.698729810778065e-05

import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    def angle_between_vectors(v1, v2):
        return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    next_node = None
    min_ratio = np.inf
    prev_node = current_node
    for next_possible_node in unvisited_nodes:
        if next_possible_node == current_node or next_possible_node == destination_node:
            continue
        current_vector = distance_matrix[current_node] - distance_matrix[current_node, current_node]
        next_vector = distance_matrix[next_possible_node] - distance_matrix[current_node, current_node]
        angle = angle_between_vectors(current_vector, next_vector)
        toroidal_distance = 1 / (np.linalg.norm(next_vector) + 0.01)
        distance_var = np.var(np.linalg.norm(distance_matrix[current_node] - distance_matrix[unvisited_nodes], axis=1))
        ratio = distance_var * 0.6 + np.sin(angle / 2) * 0.3 + 0.1 * toroidal_distance
        prev_distance = np.linalg.norm(distance_matrix[prev_node] - distance_matrix[next_possible_node])
        ratio *= prev_distance / (np.linalg.norm(distance_matrix[prev_node] - distance_matrix[current_node]) + np.linalg.norm(distance_matrix[current_node] - distance_matrix[next_possible_node]))
        if ratio < min_ratio:
            min_ratio = ratio
            next_node = next_possible_node
        elif ratio == min_ratio:
            next_node_distance_prev = np.linalg.norm(distance_matrix[prev_node] - distance_matrix[next_possible_node])
            next_node_distance_current = np.linalg.norm(distance_matrix[current_node] - distance_matrix[next_possible_node])
            if next_node_distance_current < next_node_distance_prev:
                next_node = next_possible_node
    if next_node is None:
        next_node = unvisited_nodes[0]
    return next_node
