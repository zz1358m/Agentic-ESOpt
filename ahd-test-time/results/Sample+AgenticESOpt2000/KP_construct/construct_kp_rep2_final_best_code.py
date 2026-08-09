# source: /home/zhi/Agentic-ESOpt/cache/active_runs/construct_kp_train_sample_es_pop20_gen100_sigma0.001_alpha0.0005_construct_kp_sample_es_current_cosine_t2000_rep2_queue_b_gpu4_7_20260720_030717/results/pops_best/population_generation_100.json
# method: sample_es, invalid_reward=current, sigma_schedule=cosine
# population=20, generations=100, samples=2000, sigma=0.001->0, alpha=0.0005
# task: construct_kp, rep: 2
# train_objective: -40.15177

import numpy as np

def select_next_item(remaining_capacity, weights, values):
    """
    Select the next item based on the maximum value-to-weight ratio from the unselected items.

    Parameters:
    remaining_capacity (float): The remaining knapsack capacity.
    weights (numpy array): An array of weights for the currently unselected items.
    values (numpy array): An array of values for the currently unselected items.

    Returns:
    next_item (int): The index of the selected item.
    """
    # Calculate the value-to-weight ratio for each unselected item
    value_to_weight_ratios = values / weights
    
    # Find the item with the maximum value-to-weight ratio
    max_ratio_index = np.argmax(value_to_weight_ratios)
    
    # Select the item if its weight is within the remaining capacity
    if weights[max_ratio_index] <= remaining_capacity:
        next_item = max_ratio_index
    else:
        # Otherwise, select the item that brings the most value up to the remaining capacity
        best_item_index = 0
        max_value_brought = 0
        for i in range(weights.shape[0]):
            if weights[i] <= remaining_capacity and values[i] > max_value_brought:
                best_item_index = i
                max_value_brought = values[i]
        next_item = best_item_index
    
    return next_item
