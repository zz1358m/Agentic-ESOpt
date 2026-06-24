import numpy as np

def select_next_item(remaining_capacity, weights, values):
    n = len(weights)
    selected_mask = np.zeros(n, dtype=bool)
    total_value = np.sum(values * (1 - selected_mask))

    # Define new algorithm parameters
    alpha = 0.3  # Weight for value-to-weight ratio
    beta = 0.7   # Weight for total-value-to-selected-value ratio
    threshold = 1e-6  # Threshold for a "boundary" case
    divisor = np.linalg.norm(weights) + 1e-9  # Normalization divisor

    # Implement new scoring function
    next_item = None
    best_score = 0

    for i in range(n):
        if weights[i] > remaining_capacity:
            continue
        score = (alpha * values[i] / weights[i] * (1 + divisor / (total_value + 1))) + beta * (values[i] / (total_value + values[i]))

        # Handle a "boundary" case
        if total_value + values[i] < threshold:
            score = (alpha * values[i] / (weights[i] + 1e-9))

        if score > best_score:
            best_score = score
            next_item = i

    return next_item
