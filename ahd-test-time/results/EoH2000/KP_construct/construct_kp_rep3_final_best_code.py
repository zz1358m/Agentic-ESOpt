# Archived from /home/zhi/Dynamic-Agent/cache/active_runs/construct_kp_train_eoh_construct_kp_train_eoh_rep3_orig_eoh_all6_k3_8gpu_20260713_142341/results/pops_best/population_generation_25.json
# run_id: construct_kp_train_eoh_rep3_orig_eoh_all6_k3_8gpu_20260713_142341
# train_objective: -40.15778
# method: original EoH, population=10, generations=25, k=3 replicates

import numpy as np

def select_next_item(remaining_capacity, weights, values):
    unselected_item_indices = np.arange(len(values))
    unselected = (weights[unselected_item_indices] > 0)

    scores = np.zeros(len(values))
    for i, (weight, value) in enumerate(zip(weights[unselected], values[unselected])):
        ratio = np.log(value / weight)
        proportion = remaining_capacity / (weight + remaining_capacity)
        scores[i] = ratio * proportion

    max_score = np.max(scores[weights[unselected] <= remaining_capacity])
    for i in reversed(np.argsort(scores)):
        if weights[unselected][i] <= remaining_capacity:
            if scores[i] == max_score:
                next_item = unselected_item_indices[unselected][i]
                break
    return next_item
