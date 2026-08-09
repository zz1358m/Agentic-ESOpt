# source: /home/zhi/Agentic-ESOpt/cache/active_runs/construct_kp_train_sample_t1000_construct_kp_sample_t1000_rep1_20260718_060041/results/pops_best/population_generation_50.json
# method: sample, prefix=1000, batch_size=20
# task: construct_kp, rep: 1
# train_objective: -40.14595

import numpy as np

def select_next_item(remaining_capacity, weights, values):
    if remaining_capacity == 0 or len(values) == 0:
        return None
    
    # Calculate the value-to-weight ratios for each item
    ratios = values / weights
    
    # Initialize variables to track the item with the highest ratio and its index
    max_ratio = -np.inf
    next_item_idx = -1
    
    # Iterate over the items to find the one with the highest ratio
    for i in range(len(ratios)):
        if weights[i] <= remaining_capacity and ratios[i] > max_ratio:
            max_ratio = ratios[i]
            next_item_idx = i
    
    if next_item_idx == -1:
        return None
    
    return next_item_idx
