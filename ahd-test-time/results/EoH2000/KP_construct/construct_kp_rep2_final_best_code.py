# Archived from /home/zhi/Dynamic-Agent/cache/active_runs/construct_kp_train_eoh_construct_kp_train_eoh_rep2_orig_eoh_all6_k3_8gpu_20260713_142341/results/pops_best/population_generation_25.json
# run_id: construct_kp_train_eoh_rep2_orig_eoh_all6_k3_8gpu_20260713_142341
# train_objective: -40.15901
# method: original EoH, population=10, generations=25, k=3 replicates

import numpy as np

def select_next_item(remaining_capacity, weights, values):
    next_item = None
    max_weighted_median_ratio = -np.inf

    if len(values) == 0:
        return next_item

    sorted_weights = np.sort(weights)
    median_weight = np.median(sorted_weights)
    remaining_weights = np.delete(weights, np.argmin(weights))
    sorted_remaining_weights = np.sort(remaining_weights)
    median_remaining_weight = np.median(sorted_remaining_weights)

    for i in range(len(values)):
        if weights[i] <= remaining_capacity:
            # Calculate the weighted median ratio
            weighted_median_ratio = values[i] / (weights[i] * np.mean([median_weight, median_remaining_weight]) + 1 / (len(weights) + 1))
            # Update the item with the maximum weighted median ratio
            if weighted_median_ratio > max_weighted_median_ratio:
                max_weighted_median_ratio = weighted_median_ratio
                next_item = i

    return next_item
