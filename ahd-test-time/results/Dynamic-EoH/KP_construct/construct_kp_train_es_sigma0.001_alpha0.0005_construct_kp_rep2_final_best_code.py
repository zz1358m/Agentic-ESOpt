import numpy as np

def select_next_item(remaining_capacity, weights, values):
    ratio = values / weights
    best_score = -np.inf
    next_best_item = -1
    for i in np.argsort(-ratio)[::-1]:  
        if weights[i] > remaining_capacity:
            continue
        penalty_val = 1 - np.exp(-values[i] / (np.max(values) + 1e-9))  # penalty for small values
        penalty_ratio = 1 - np.exp(-ratio[i] / (np.max(ratio) + 1e-9))  # penalty for low ratios
        penalty_weight = 1 - np.exp(-ratio[i] / (np.max(ratio) + 1e-9))  # penalty for low ratios
        score = ratio[i] * (1 + penalty_val * (1 + np.exp(-2 * (values[i] - 1e-9))) * (1 - penalty_weight))  # modified score function
        if score > best_score:
            best_score = score
            next_best_item = i
    return next_best_item
