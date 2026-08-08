# source: /home/zhi/Dynamic-Agent/cache/active_runs/construct_kp_train_sample_t1000_construct_kp_sample_t1000_rep3_20260718_144918/results/pops_best/population_generation_50.json
# method: sample, prefix=1000, batch_size=20
# task: construct_kp, rep: 3
# train_objective: -40.15313

import numpy as np

def select_next_item(remaining_capacity, weights, values):
    next_item = None
    max_ratio = -np.inf

    for i in np.argsort(-values) if remaining_capacity < np.max(weights) else np.argsort(-values/weights):
        if weights[i] <= remaining_capacity:
            if values[i] / weights[i] > max_ratio:
                next_item = i
                max_ratio = values[i] / weights[i]
                remaining_capacity -= weights[i]

    return next_item
