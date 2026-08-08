# source: /home/zhi/Dynamic-Agent/cache/active_runs/construct_kp_train_es_sigma0.001_alpha0.0005_construct_kp_train_es_full_reload_sigma1e-3_alpha5e-4_rep2_rerun_dynamic_k3_kp_tspaco_3rep_8gpu_20260716_120821/results/pops_best/population_generation_25.json
# run_id: construct_kp_train_es_full_reload_sigma1e-3_alpha5e-4_rep2_rerun_dynamic_k3_kp_tspaco_3rep_8gpu_20260716_120821
# train_objective: -40.15798
# m1m2_multiplier: 3.0

import numpy as np

def select_next_item(remaining_capacity, weights, values):
    # Calculate the value-to-weight ratio for each item
    ratios = values / weights
    
    # Calculate the capacity usage for each item and divide by the value
    capacity_usage = weights / (values / (remaining_capacity / weights))
    
    # Calculate the priority score of each item based on the capacity usage
    priority_scores = 1 / (capacity_usage + 1)
    
    # Calculate a score for each item considering the ratio and priority
    scores = (ratios * 0.7) + (priority_scores * 0.3)
    
    # Filter out items that are too heavy
    eligible_items = np.where(weights <= remaining_capacity)[0]
    
    # Select the item that maximizes the score, if there are eligible items
    if len(eligible_items) > 0:
        next_item = eligible_items[np.argmax(scores[eligible_items])]
    else:
        next_item = None
    
    return next_item
