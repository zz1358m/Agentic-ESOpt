# source: /home/zhi/Agentic-ESOpt/cache/active_runs/construct_kp_train_sample_es_pop20_gen50_sigma0.001_alpha0.0005_construct_kp_sample_es_reload_cosine_current_pop20_gen50_rep2_20260719_150222/results/pops_best/population_generation_50.json
# method: sample_es, invalid_reward=current, sigma_schedule=cosine
# population=20, generations=50, samples=1000, sigma=0.001->0, alpha=0.0005
# task: construct_kp, rep: 2
# train_objective: -40.15177

import numpy as np

def select_next_item(remaining_capacity, weights, values):
    """Select the next item that maximizes the total value while not exceeding the remaining capacity.

    Parameters:
    remaining_capacity (float): The remaining knapsack capacity.
    weights (numpy array): The weights of the currently unselected items.
    values (numpy array): The values of the currently unselected items.

    Returns:
    next_item (int): The integer index of the selected item in the weights and values arrays.
    """
    value_to_weight_ratios = values / weights  # Calculate the value-to-weight ratio for each item
    max_ratio = np.max(value_to_weight_ratios)  # Find the maximum value-to-weight ratio
    next_item = np.where(value_to_weight_ratios == max_ratio)[0][0]  # Find the item with the maximum value-to-weight ratio
    if weights[next_item] <= remaining_capacity:  # If the item's weight does not exceed the remaining capacity
        return next_item
    else:
        items_not_exceeding_capacity = np.where(weights <= remaining_capacity)[0]
        max_value = 0
        next_item = -1
        for i in items_not_exceeding_capacity:
            if values[i] > max_value:  # If the current item's value is greater than the max value
                max_value = values[i]
                next_item = i
        return next_item
