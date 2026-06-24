import numpy as np

def select_next_item(remaining_capacity, weights, values):
    valid_indices = []

    # Calculate the average value-to-weight ratio
    average_value_per_weight = np.sum(values) / np.sum(weights)

    for i in range(len(values)):
        if weights[i] <= remaining_capacity:
            # Calculate the probscore using the new formula
            probscore = average_value_per_weight * (values[i] / weights[i]) + remaining_capacity / np.sum(weights) - (np.sum(weights) - weights[i]) / np.sum(weights)
            valid_indices.append((probscore, i))

    # Get the index with the maximum prob score
    max_probscore, next_item = max(valid_indices, key=lambda x: x[0])

    # Update the remaining capacity
    remaining_capacity -= weights[next_item]

    return next_item
