# Archived from /home/zhi/Agentic-ESOpt/cache/active_runs/construct_kp_train_eoh_construct_kp_train_eoh_rep1_orig_eoh_all6_k3_8gpu_20260713_142341/results/pops_best/population_generation_25.json
# run_id: construct_kp_train_eoh_rep1_orig_eoh_all6_k3_8gpu_20260713_142341
# train_objective: -40.15772
# method: original EoH, population=10, generations=25, k=3 replicates

import numpy as np

def select_next_item(remaining_capacity, weights, values):
    # Find indices of items with non-zero weights
    valid_items = np.where(weights > 0)[0]
    
    # Calculate the value-to-weight ratio for each item
    ratio = values[valid_items] / weights[valid_items]
    
    # Calculate the fraction of remaining capacity for each item if it is selected
    fractions = weights[valid_items] / (remaining_capacity + weights[valid_items])
    
    # Calculate the meta-score for each item
    average_ratio = np.mean(ratio)
    average_fraction = np.mean(fractions)
    meta_score = average_ratio * average_fraction + 0.5 * (average_ratio - average_ratio)
    
    # Calculate the weighted score for each item
    scores = 0.55 * ratio + 0.25 * fractions + 0.20 * (ratio - meta_score)
    
    # Sort the valid items by their scores in descending order
    sorted_items = np.argsort(-scores)
    
    # Iterate over the sorted items to find the best next item
    for item in sorted_items:
        if weights[item] <= remaining_capacity:
            return item
    
    # If no valid items are found, return -1
    return -1
