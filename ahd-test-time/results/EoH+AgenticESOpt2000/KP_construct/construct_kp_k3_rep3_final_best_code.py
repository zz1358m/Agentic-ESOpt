# source: /home/zhi/Agentic-ESOpt/cache/active_runs/construct_kp_train_es_sigma0.001_alpha0.0005_construct_kp_train_es_full_reload_sigma1e-3_alpha5e-4_rep3_rerun_agentic_esopt_k3_kp_tspaco_3rep_8gpu_20260716_120821/results/pops_best/population_generation_25.json
# run_id: construct_kp_train_es_full_reload_sigma1e-3_alpha5e-4_rep3_rerun_agentic_esopt_k3_kp_tspaco_3rep_8gpu_20260716_120821
# train_objective: -40.15841
# m1m2_multiplier: 3.0

import numpy as np

def select_next_item(remaining_capacity, weights, values):
    if len(weights) == 0 or remaining_capacity == 0:
        return -1

    # Calculate the initial selected value
    selected_values = np.sum(values)

    # Calculate the current total weight
    selected_weights = np.sum(weights)

    # Calculate the target value
    def calculate_target_value(selected_values, values):
        return (selected_values + np.max(values)) / 2

    # Initialize target value and smoothing factor
    target_value = calculate_target_value(selected_values, values)
    smoothing_factor = 0.2

    # Calculate scores for each item based on value-to-weight ratio and proximity to the target value
    def calculate_scores(values, weights, target_value):
        scores = (values / weights) / abs(values - target_value)
        return scores

    scores = calculate_scores(values, weights, target_value)

    # Exponentially smooth the target value
    def smooth_target_value(target_value, selected_values, values):
        max_value = np.max(values)
        return target_value * (1 - smoothing_factor) + selected_values / (2 * max_value) * smoothing_factor

    max_score = -np.inf
    next_item_index = 0

    for i in range(len(scores)):
        if weights[i] <= remaining_capacity and scores[i] > max_score:
            max_score = scores[i]
            next_item_index = i

    # Update the target value for the next iteration
    target_value = smooth_target_value(target_value, selected_values, values)

    return next_item_index
