# source: /home/zhi/Agentic-ESOpt/cache/active_runs/construct_kp_train_sample_t1000_construct_kp_sample_t1000_rep2_20260718_144918/results/pops_best/population_generation_50.json
# method: sample, prefix=1000, batch_size=20
# task: construct_kp, rep: 2
# train_objective: -40.15177

import numpy as np

def select_next_item(remaining_capacity, weights, values):
    if np.any(values == 0):
        return -1  # No items left

    ratios = values / weights
    max_ratio_index = np.argmax(ratios)
    if weights[max_ratio_index] > remaining_capacity:
        remaining_max_value = 0
        max_ratio_index = -1
        for i in range(len(values)):
            if weights[i] <= remaining_capacity and values[i] > remaining_max_value:
                remaining_max_value = values[i]
                max_ratio_index = i
    return max_ratio_index
