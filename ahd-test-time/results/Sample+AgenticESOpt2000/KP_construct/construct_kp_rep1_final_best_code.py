# source: /home/zhi/Dynamic-Agent/cache/active_runs/construct_kp_train_sample_es_pop20_gen100_sigma0.001_alpha0.0005_construct_kp_sample_es_current_cosine_t2000_rep1_queue_a_gpu0_3_20260720_030717/results/pops_best/population_generation_100.json
# method: sample_es, invalid_reward=current, sigma_schedule=cosine
# population=20, generations=100, samples=2000, sigma=0.001->0, alpha=0.0005
# task: construct_kp, rep: 1
# train_objective: -40.15177

import numpy as np

def select_next_item(remaining_capacity, weights, values):
    if len(weights) == 0:
        return None

    # Calculate the item with the highest value-to-weight ratio
    ratios = values / weights
    item_index = np.argmax(ratios)
    if weights[item_index] <= remaining_capacity:
        remaining_capacity -= weights[item_index]
        return item_index

    # Select the item with the highest value that fits the remaining capacity
    max_value, next_item = -float('inf'), -1
    for i in range(len(weights)):
        if weights[i] <= remaining_capacity:
            value = values[i]
            if value > max_value:
                max_value, next_item = value, i

    return next_item
