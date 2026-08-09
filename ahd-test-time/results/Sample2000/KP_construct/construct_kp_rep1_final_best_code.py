# source: /home/zhi/Agentic-ESOpt/cache/active_runs/construct_kp_train_sample_t2000_construct_kp_sample_t2000_from_rep1_20260718_145101/results/pops_best/population_generation_100.json
# method: sample, prefix=2000, batch_size=20
# task: construct_kp, rep: 1
# train_objective: -40.15177

import numpy as np

def select_next_item(remaining_capacity, weights, values):
    # {Greedy algorithm for 0-1 Knapsack Problem: select the next item as the one with the highest value-to-weight ratio from the unselected items}
    
    # Calculate the value-to-weight ratios of all unselected items
    ratios = values / weights
    
    # Find the maximum ratio, the index of the item with the maximum ratio, and its weight
    max_ratio = np.max(ratios)
    max_index = np.argmax(ratios)
    max_weight = weights[max_index]
    
    # If the maximum weight is less than or equal to the remaining capacity, select the corresponding item
    if max_weight <= remaining_capacity:
        next_item = max_index
    else:
        # If the maximum weight is more than the remaining capacity, find the maximum value from the unselected items with weights less than or equal to the remaining capacity
        valid_indices = np.where(weights <= remaining_capacity)[0]
        if len(valid_indices) > 0:
            max_value = np.max(values[valid_indices])
            max_index = np.where(values == max_value)[0][0]
            next_item = max_index
        else:
            next_item = -1  # Return -1 if no valid item is found
    return next_item
