# source: /home/zhi/Dynamic-Agent/cache/active_runs/construct_kp_train_es_sigma0.001_alpha0.0005_construct_kp_train_es_full_reload_sigma1e-3_alpha5e-4_rep2_rerun_dynamic_k1_kp_asp_3rep_8gpu_20260716_120821/results/pops_best/population_generation_25.json
# run_id: construct_kp_train_es_full_reload_sigma1e-3_alpha5e-4_rep2_rerun_dynamic_k1_kp_asp_3rep_8gpu_20260716_120821
# train_objective: -40.15744
# m1m2_multiplier: 1.0

import numpy as np

def select_next_item(remaining_capacity, weights, values):
    if np.any(remaining_capacity == 0):
        return -1

    # calculate value-weight ratio for each item
    ratios = values / weights

    # calculate normalized densities for each item
    densities = values / np.sum(values)

    # calculate distances to capacity for each item
    distances = np.minimum(remaining_capacity / weights, 1)

    best_next_item = None
    best_score = float('-inf')

    for i in range(len(values)):
        weight, ratio, density, dist = weights[i], ratios[i], densities[i], distances[i]
        score = 0.4 * density + 0.3 * ratio + 0.3 * dist

        # check if the current item is the best so far
        if weight <= remaining_capacity and score > best_score:
            best_next_item = i
            best_score = score

    if best_next_item is None:
        return -1
    return best_next_item
