# source: /home/zhi/Agentic-ESOpt/cache/active_runs/construct_kp_train_sample_es_pop20_gen50_sigma0.001_alpha0.0005_construct_kp_sample_es_reload_cosine_current_pop20_gen50_rep3_20260719_150222/results/pops_best/population_generation_50.json
# method: sample_es, invalid_reward=current, sigma_schedule=cosine
# population=20, generations=50, samples=1000, sigma=0.001->0, alpha=0.0005
# task: construct_kp, rep: 3
# train_objective: -40.15177

import numpy as np

def select_next_item(remaining_capacity, weights, values):
    item_value_ratios = values / weights  # Calculate value-to-weight ratios for all items
    max_ratio = np.max(item_value_ratios)  # Find the maximum ratio
    next_item = np.where(item_value_ratios == max_ratio)  # Find the indices of items with the maximum ratio
    for item in next_item[0]:  # Check if any item with the maximum ratio does not exceed capacity
        if weights[item] <= remaining_capacity:
            return item
    # If no such item exists, find the item with the greatest value among items that do not exceed capacity
    max_value = 0
    next_item = None
    for item in range(len(values)):
        if weights[item] <= remaining_capacity:
            if values[item] > max_value:
                max_value = values[item]
                next_item = item
    return next_item
