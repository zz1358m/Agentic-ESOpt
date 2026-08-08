# source: /home/zhi/Dynamic-Agent/cache/active_runs/construct_kp_train_es_sigma0.001_alpha0.0005_construct_kp_train_es_full_reload_sigma1e-3_alpha5e-4_rep1_rerun_dynamic_k1_kp_asp_3rep_8gpu_20260716_120821/results/pops_best/population_generation_25.json
# run_id: construct_kp_train_es_full_reload_sigma1e-3_alpha5e-4_rep1_rerun_dynamic_k1_kp_asp_3rep_8gpu_20260716_120821
# train_objective: -40.15671
# m1m2_multiplier: 1.0

import numpy as np

def select_next_item(remaining_capacity, weights, values):
    unselected_items = np.argwhere((weights <= remaining_capacity)).flatten()

    if len(unselected_items) == 0:
        return None

    # Calculate the total value of remaining unselected items
    total_value = np.sum(values[unselected_items])
    
    # Calculate the item value percentage based on its contribution to the total value
    item_percentages = values[unselected_items] / total_value
    
    # Calculate the discount rates based on item percentages
    discount_rates = 1 - (item_percentages * total_value / np.sum(values[unselected_items]))
    
    # Calculate the undiscounted value-to-weight ratios
    undiscounted_ratios = values[unselected_items] / (weights[unselected_items] * discount_rates)
    
    next_item_idx = unselected_items[np.argmax(undiscounted_ratios)]  # Select the item with the highest undiscounted ratio
    
    remaining_capacity -= weights[next_item_idx]
    
    return next_item_idx
