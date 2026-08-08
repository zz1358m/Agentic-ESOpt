# Archived from /home/zhi/Dynamic-Agent/cache/active_runs/construct_tsp_train_es_sigma0.001_alpha0.0005_construct_tsp_train_es_full_reload_sigma1e-3_alpha5e-4_rep2_k3_decay_full_all6_3rep_reload8_20260705_121329/results/pops_best/population_generation_25.json
# run_id: construct_tsp_train_es_full_reload_sigma1e-3_alpha5e-4_rep2_k3_decay_full_all6_3rep_reload8_20260705_121329
# train_objective: 6.51895
# m1m2_multiplier: 3.0
# sigma_schedule: cosine
# final_model_es_sigma: 0.0

import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    remaining_nodes = np.setdiff1d(unvisited_nodes, current_node)

    divergence_measure = np.exp(-distance_matrix[current_node, remaining_nodes])
    convergence_measure = np.exp(-distance_matrix[current_node, remaining_nodes] + distance_matrix[destination_node, remaining_nodes])
    interaction_measure = np.exp((divergence_measure + convergence_measure) / 2)

    weighted_divergence_measure = divergence_measure / (divergence_measure + convergence_measure + interaction_measure)
    weighted_convergence_measure = convergence_measure / (divergence_measure + convergence_measure + interaction_measure)
    weighted_interaction_measure = interaction_measure / (divergence_measure + convergence_measure + interaction_measure)

    scores = weighted_divergence_measure + weighted_convergence_measure + weighted_interaction_measure
    
    # Introduce a bias towards nodes with higher divergence and convergence measures
    scores *= (1 + divergence_measure + convergence_measure) / (np.max(divergence_measure + convergence_measure + interaction_measure) + 1e-6)
    
    next_node = remaining_nodes[np.argmax(scores)]
    
    return next_node
