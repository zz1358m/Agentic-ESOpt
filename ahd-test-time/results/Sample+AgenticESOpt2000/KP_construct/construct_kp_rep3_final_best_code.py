# source: /home/zhi/Dynamic-Agent/cache/active_runs/construct_kp_train_sample_es_pop20_gen100_sigma0.001_alpha0.0005_construct_kp_sample_es_current_cosine_t2000_rep3_queue_a_gpu0_3_20260720_030717/results/pops_best/population_generation_100.json
# method: sample_es, invalid_reward=current, sigma_schedule=cosine
# population=20, generations=100, samples=2000, sigma=0.001->0, alpha=0.0005
# task: construct_kp, rep: 3
# train_objective: -40.15313

import numpy as np

def select_next_item(remaining_capacity, weights, values):
    if np.max(weights) <= remaining_capacity:
        max_ratio = -np.inf
        next_item = np.argmin(weights)
        for i in range(len(values)):
            if weights[i] <= remaining_capacity and values[i] / weights[i] > max_ratio:
                max_ratio = values[i] / weights[i]
                next_item = i
    else:
        max_value = 0
        next_item = np.argmin(weights)
        for i in range(len(values)):
            if weights[i] <= remaining_capacity:
                if values[i] > max_value:
                    max_value = values[i]
                    next_item = i
    return next_item
