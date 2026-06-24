import numpy as np

def select_next_item(remaining_capacity, weights, values):
    if np.all(remaining_capacity == 0) or np.all(weights == 0):
        return None
    gamma = 0.1  # penalty coefficient
    n = len(values)
    scores = []
    for i in range(n):
        if weights[i] > remaining_capacity:
            continue
        v = values[i]
        w = weights[i]
        score = v / (w ** (1 + gamma * np.log(np.minimum(1, w))) + v / np.max(values))
        scores.append((score, i))
    if len(scores) == 0:
        return None
    max_score = max(scores, key=lambda x: x[0])[0]
    next_item = [i for s, i in sorted(scores) if s == max_score][0]
    return next_item
