# source: /home/zhi/Dynamic-Agent/cache/active_runs/construct_kp_train_es_sigma0.001_alpha0.0005_construct_kp_train_es_full_reload_sigma1e-3_alpha5e-4_rep1_rerun_dynamic_k3_kp_tspaco_3rep_8gpu_20260716_120821/results/pops_best/population_generation_25.json
# run_id: construct_kp_train_es_full_reload_sigma1e-3_alpha5e-4_rep1_rerun_dynamic_k3_kp_tspaco_3rep_8gpu_20260716_120821
# train_objective: -40.16268
# m1m2_multiplier: 3.0

import numpy as np

def select_next_item(remaining_capacity, weights, values):
    # Calculate the fraction of the total capacity that each item occupies
    occupation_factors = weights / np.sum(weights)
    
    # Initialize the best item index and score
    next_item = -1
    best_score = 0
    
    # Iterate through the items, giving preference to items with lower weights first and higher values
    for i in np.argsort(values)[::-1]:
        item_weight = weights[i]
        item_value = values[i]
        # Check if the item can be selected
        if item_weight <= remaining_capacity:
            # Calculate the score with a downward priority based on remaining capacity
            if np.log2(item_weight + 1) > 0:
                score = (item_value / (item_weight + 0.1)) + 0.3 * (1 / (item_weight + 1)) + 0.3 * occupation_factors[i] + 0.3 * remaining_capacity / (np.sum(weights) - item_weight) ** 0.2
            else:
                score = (item_value / (item_weight + 0.1)) + 0.3 * occupation_factors[i] + 0.3 * remaining_capacity / (np.sum(weights) - item_weight) ** 0.2
            # Update the best item and score if necessary
            if score > best_score:
                next_item = i
                best_score = score

    return next_item
