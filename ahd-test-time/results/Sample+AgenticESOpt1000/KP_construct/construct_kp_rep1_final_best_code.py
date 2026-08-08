# source: /home/zhi/Dynamic-Agent/cache/active_runs/construct_kp_train_sample_es_pop20_gen50_sigma0.001_alpha0.0005_construct_kp_sample_es_reload_cosine_current_pop20_gen50_rep1_20260719_150222/results/pops_best/population_generation_50.json
# method: sample_es, invalid_reward=current, sigma_schedule=cosine
# population=20, generations=50, samples=1000, sigma=0.001->0, alpha=0.0005
# task: construct_kp, rep: 1
# train_objective: -40.15177

import numpy as np

def select_next_item(remaining_capacity, weights, values):
    if len(weights) == 0:
        return -1  # all items have been selected or capacity is reached

    # calculate the value-to-weight ratio for each item
    ratios = values / weights

    # find the maximum ratio
    max_ratio_idx = np.argmax(ratios)

    # check if the item with maximum ratio does not exceed the capacity
    if weights[max_ratio_idx] <= remaining_capacity:
        # select the item
        next_item = max_ratio_idx
    else:
        # find the item with the highest value among items that do not exceed the capacity
        max_value = 0
        next_item = -1
        for i in range(len(weights)):
            if weights[i] <= remaining_capacity and values[i] > max_value:
                max_value = values[i]
                next_item = i

    return next_item
